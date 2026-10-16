"""Measure the scaling law of a voting quorum.

The swarm is not a serving fleet. Every decision is answered by the selected
quorum *together*, and each voter's latency is on the critical path, so
adding replicas may raise latency without raising throughput at all. That
question decides whether an elastic policy should add replicas for capacity
or only to replace slow members - so it is measured rather than assumed.

Replicas are deterministic mock stems with a simulated per-decision delay
(``stemd --delay-ms``), because a real 1.7 GB model per replica would make
the sweep take hours and would measure model loading rather than the
architecture.

Usage:
    python tools/scaling_law_check.py --yes --delay-ms 50
"""

from __future__ import annotations

import argparse
import json
import secrets
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
            if len(seen) == expected:
                return seen
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.2)
    raise SystemExit(f"wanted {expected} nodes, saw {sorted(seen)}")


def decide(url: str, token: str, index: int, question: str) -> dict:
    body = json.dumps({"state": f"scaling probe {index}",
                       "question": question,
                       "candidates": ["approve", "reject"]}).encode()
    request = urllib.request.Request(
        f"{url}/decide", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as resp:
        payload = json.loads(resp.read())
    return {"ms": (time.perf_counter() - started) * 1000,
            "responded": payload.get("responded"),
            "quorum_size": payload.get("quorum_size")}


def sweep(url: str, token: str, concurrency: int, tasks: int,
          question: str) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda i: decide(url, token, i, question),
                                range(tasks)))
    elapsed = time.perf_counter() - started
    latencies = sorted(row["ms"] for row in results)
    return {
        "concurrency": concurrency,
        "throughput_rps": round(tasks / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[min(len(latencies) - 1,
                                      int(0.95 * len(latencies)))], 1),
        "quorum_size": results[-1]["quorum_size"],
        "responded": results[-1]["responded"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--replicas", default="1,2,3,5")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--quorum-k", type=int, default=None,
                        help="voters per decision (rootd --quorum-k); leave "
                             "unset to let the quorum grow with membership")
    parser.add_argument("--label", default=None,
                        help="recorded in the report so sweeps can be compared")
    parser.add_argument("--serialize", action="store_true",
                        help="give every replica a device-like serialization "
                             "gate (one decision at a time)")
    parser.add_argument("--adaptive-quorum", action="store_true",
                        help="let the root trade quorum size against latency")
    parser.add_argument("--quorum-floor", type=int, default=1)
    parser.add_argument("--latency-target-ms", type=float, default=None)
    parser.add_argument("--question", default="approve?",
                        help="question text; the playbook's category floors "
                             "are chosen from it")
    parser.add_argument("--playbook", default=None,
                        help="playbook whose gating carries the quorum floors")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/stemd processes; pass --yes")

    replica_counts = [int(v) for v in args.replicas.split(",") if v.strip()]
    concurrencies = [int(v) for v in args.concurrency.split(",") if v.strip()]
    token = secrets.token_urlsafe(18)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    work = Path(tempfile.mkdtemp(prefix="pando-scaling-"))
    token_file = work / "pairing.token"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"

    procs: list[subprocess.Popen] = []
    rows: list[dict] = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pando.rootd", "--port", str(port),
             "--pairing-token-file", str(token_file),
             "--rollout-dir", str(work / "rollouts"),
             *(["--quorum-k", str(args.quorum_k)]
               if args.quorum_k is not None else []),
             *(["--adaptive-quorum", "--quorum-floor",
                str(args.quorum_floor), "--latency-target-ms",
                str(args.latency_target_ms)]
               if args.adaptive_quorum and args.latency_target_ms
               else []),
             *(["--playbook", str(args.playbook)] if args.playbook else [])],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_root(url)

        started = 0
        for target in replica_counts:
            while started < target:
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "pando.stemd",
                     "--node-id", f"replica-{started}",
                     "--root", url,
                     "--pairing-token-file", str(token_file),
                     "--backend", "mock", "--checkpoint", "mock",
                     "--delay-ms", str(args.delay_ms),
                     *(["--serialize"] if args.serialize else [])],
                    cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=creationflags))
                started += 1
            wait_nodes(url, token, target)
            time.sleep(0.5)
            for concurrency in concurrencies:
                row = sweep(url, token, concurrency, args.tasks,
                            args.question)
                row["replicas"] = target
                rows.append(row)
                print(f"  replicas={target} concurrency={concurrency} "
                      f"throughput={row['throughput_rps']:6.1f} req/s "
                      f"p50={row['p50_ms']:6.1f} ms "
                      f"quorum={row['quorum_size']}", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    by_key = {(row["replicas"], row["concurrency"]): row for row in rows}
    growth = []
    for concurrency in concurrencies:
        first = by_key.get((replica_counts[0], concurrency))
        last = by_key.get((replica_counts[-1], concurrency))
        if first and last:
            growth.append({
                "concurrency": concurrency,
                "throughput_gain": round(
                    last["throughput_rps"] / max(first["throughput_rps"], 1e-9),
                    2),
                "latency_cost_ms": round(last["p50_ms"] - first["p50_ms"], 1),
            })
    if args.adaptive_quorum:
        default_label = f"adaptive-k<={args.quorum_k or 5}@{args.latency_target_ms}ms"
    elif args.quorum_k:
        default_label = f"quorum-k={args.quorum_k}"
    else:
        default_label = "default"
    summary = {"label": args.label or default_label,
               "question": args.question,
               "playbook": args.playbook,
               "adaptive": args.adaptive_quorum,
               "quorum_k": args.quorum_k,
               "delay_ms": args.delay_ms, "tasks": args.tasks,
               "replicas": replica_counts, "concurrency": concurrencies,
               "rows": rows, "growth": growth}
    out_dir = ROOT / "metrics" / "scaling-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"scaling-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary if args.json else {"growth": growth}, indent=2))
    print(f"report: {report}")
    print("SCALING LAW CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
