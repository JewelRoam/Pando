"""End to end: the swarm notices it is too slow and grows itself.

Starts a root whose adaptive quorum is pinned by a small ceiling, plus two
serialized replicas. Under concurrent load the decisions are slower than the
target and there is no checking power left to spend - the one condition where
growing is a capacity answer rather than a policy question.

The supervisor reads the root's `/status`, decides, and calls the supplied
grow callback, which launches another replica (a stand-in for the
reproduction path verified end to end in round 108). The loop repeats until
the target is met or the member ceiling is reached, and the trajectory is
printed.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.autonomy import Journal  # noqa: E402
from pandosi.elastic import ElasticSupervisor  # noqa: E402

REPLICA_MS = 30.0
TASKS = 24


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_root(url: str, deadline: float = 30.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=2) as resp:
                if json.loads(resp.read()).get("required"):
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise SystemExit("rootd did not become ready")


def wait_nodes(url: str, token: str, expected: int,
               deadline: float = 60.0) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    end = time.monotonic() + deadline
    seen: dict = {}
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"{url}/nodes", headers=headers),
                    timeout=3) as resp:
                seen = json.loads(resp.read()).get("nodes", {})
            if len(seen) >= expected:
                return seen
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.2)
    raise SystemExit(f"wanted {expected} nodes, saw {sorted(seen)}")


def measure(url: str, token: str, concurrency: int) -> dict:
    def one(index: int) -> float:
        body = json.dumps({"state": f"elastic probe {index}",
                           "question": "approve?",
                           "candidates": ["approve", "reject"]}).encode()
        request = urllib.request.Request(
            f"{url}/decide", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as resp:
            resp.read()
        return (time.perf_counter() - started) * 1000

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = sorted(pool.map(one, range(TASKS)))
    return {"p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(latencies[int(0.95 * len(latencies)) - 1], 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    # The ceiling on a single decision is one replica's latency (30 ms here),
    # so a 50 ms target is reachable by adding members but not by tuning the
    # quorum - which is exactly the condition that should trigger growth.
    parser.add_argument("--target-ms", type=float, default=50.0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-members", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/stemd processes; pass --yes")

    work = Path(tempfile.mkdtemp(prefix="pandosi-elastic-"))
    token = secrets.token_urlsafe(18)
    token_file = work / "pairing.token"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    procs: list[subprocess.Popen] = []
    started = 0

    def start_replica() -> str:
        nonlocal started
        node_id = f"replica-{started}"
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pandosi.stemd",
             "--node-id", node_id, "--root", url,
             "--pairing-token-file", str(token_file),
             "--backend", "mock", "--checkpoint", "mock",
             "--delay-ms", str(REPLICA_MS), "--serialize"],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))
        started += 1
        return node_id

    trajectory = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pandosi.rootd", "--port", str(port),
             "--pairing-token-file", str(token_file),
             "--adaptive-quorum", "--quorum-k", "2", "--quorum-floor", "1",
             "--latency-target-ms", str(args.target_ms),
             "--rollout-dir", str(work / "rollouts")],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_root(url)
        start_replica()
        start_replica()
        wait_nodes(url, token, 2)
        time.sleep(1.0)

        supervisor = ElasticSupervisor(url, token,
                                       max_members=args.max_members)
        journal = Journal(work / "supervision.jsonl")

        def grow() -> str:
            node_id = start_replica()
            wait_nodes(url, token, started)
            time.sleep(0.5)
            return node_id

        for round_index in range(args.rounds):
            stats = measure(url, token, args.concurrency)
            status = supervisor.status()
            supervision = supervisor.step(grow=grow, journal=journal)
            row = {"round": round_index,
                   "members": status.get("members"),
                   "quorum_k": status.get("quorum_k"),
                   "decision_ewma_ms": status.get("decision_ewma_ms"),
                   **stats,
                   "action": supervision.action,
                   "reason": supervision.reason}
            trajectory.append(row)
            print(f"  round {round_index}: p50={stats['p50_ms']:6.1f} ms "
                  f"members={row['members']} k={row['quorum_k']} "
                  f"-> {supervision.action}", flush=True)
            if supervision.action == "hold" and \
                    stats["p50_ms"] <= args.target_ms:
                break
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    grew = [row for row in trajectory if row["action"] == "grow"]
    final = trajectory[-1]
    summary = {"target_ms": args.target_ms,
               "concurrency": args.concurrency,
               "replicas_started": started,
               "growths": len(grew),
               "final_p50_ms": final["p50_ms"],
               "final_members": final["members"],
               "trajectory": trajectory}
    out_dir = ROOT / "metrics" / "elastic-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"elastic-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "trajectory"}, indent=2))
    print(f"report: {report}")
    shutil.rmtree(work, ignore_errors=True)

    if not grew:
        raise SystemExit("the supervisor never grew despite being over target")
    if final["p50_ms"] > args.target_ms * 1.5:
        raise SystemExit(f"growth did not bring latency near the target: "
                         f"{final['p50_ms']} ms")
    print("ELASTIC LOOP CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
