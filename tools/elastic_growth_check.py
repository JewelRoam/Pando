"""End to end: the swarm grows itself by replicating a real spore.

The supervisor's ``grow`` action is wired to ``pando.reproduce`` - the real
path, with a served artifact, a declared lineage and digest, attestation and
the ``reproduce`` grant - instead of a local launcher. A reproduced node has
to transfer an artifact, start and register before it can vote, which is why
the supervisor needs a cooldown rather than re-deciding every round.

Two deliberately slow serialized members keep the swarm over target at its
quorum floor; the swarm grows; the joined node is a real ONNX worker.

Usage:
    python tools/elastic_growth_check.py --yes
"""

from __future__ import annotations

import argparse
import datetime as dt
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

from pando.autonomy import Journal  # noqa: E402
from pando.elastic import ElasticSupervisor, Retirer  # noqa: E402
from pando.playbook import load_playbook, resolve_admitted_lineages, \
    sign_playbook  # noqa: E402
from pando.reproduce import Target, reproduce  # noqa: E402
from pando.spore import artifact_digest, generate_signing_key  # noqa: E402
from pando.grants import save_grant  # noqa: E402

LINEAGE = "english"
SLOW_MS = 120.0
REPLICAS = 2
TARGET_MS = 80.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_root(url: str, deadline: float = 60.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=2) as resp:
                if json.loads(resp.read()).get("required"):
                    return
        except OSError:
            pass
        time.sleep(0.3)
    raise SystemExit("rootd did not become ready")


def wait_nodes(url: str, token: str, expected: int,
               deadline: float = 300.0) -> dict:
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
        time.sleep(0.5)
    raise SystemExit(f"wanted {expected} nodes, saw {sorted(seen)}")


def measure(url: str, token: str, concurrency: int, tasks: int) -> dict:
    def one(index: int) -> float:
        body = json.dumps({"state": f"growth probe {index}",
                           "question": "approve?",
                           "candidates": ["approve", "reject"]}).encode()
        request = urllib.request.Request(
            f"{url}/decide", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as resp:
            resp.read()
        return (time.perf_counter() - started) * 1000

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = sorted(pool.map(one, range(tasks)))
    return {"p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(latencies[int(0.95 * len(latencies)) - 1], 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--payload",
                        default=str(ROOT / "artifacts" / "laya-en.int8.onnx"))
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--interval-s", type=float, default=10.0,
                        help="gap between rounds, so a cooldown is meaningful")
    parser.add_argument("--cooldown-s", type=float, default=20.0)
    parser.add_argument("--max-per-host", type=int, default=1,
                        help="same-lineage members allowed per host; raise it "
                             "to let growth make the mistake rollback is for")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/stemd/join processes; pass --yes")
    payload = Path(args.payload)
    if not payload.exists():
        raise SystemExit(f"missing payload: {payload}")

    digest = artifact_digest(payload)
    print(f"payload {payload.name} digest {digest[:16]}...", flush=True)
    work = Path(tempfile.mkdtemp(prefix="pando-growth-"))
    token = secrets.token_urlsafe(18)
    token_file = work / "pairing.token"
    token_file.write_text(token, encoding="utf-8")
    grants = work / "grants.toml"
    expires = (dt.datetime.now(dt.timezone.utc)
               + dt.timedelta(hours=1)).isoformat()
    save_grant("reproduce", "localhost", "operator-host", "check", expires,
               path=grants)
    key = generate_signing_key()
    playbook_path = work / "playbook.toml"
    playbook_path.write_text(
        '[meta]\nname = "elastic-growth"\nversion = 1\n\n'
        f'[lineages]\nadmit = ["{LINEAGE}", "mock"]\n\n'
        f'[lineages.artifacts]\n{LINEAGE} = "{digest}"\n',
        encoding="utf-8")
    playbook = load_playbook(playbook_path)
    (work / (playbook_path.name + ".sig")).write_text(
        json.dumps(sign_playbook(playbook, key)), encoding="utf-8")
    admitted, _ = resolve_admitted_lineages(playbook)

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    # Everything in this check lives on one machine, which is the point: the
    # supervisor has to notice that before it adds a second same-lineage
    # member there.
    target_host = "127.0.0.1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    procs: list[subprocess.Popen] = []
    children: list[subprocess.Popen] = []
    trajectory: list[dict] = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pando.rootd", "--port", str(port),
             "--pairing-token-file", str(token_file),
             "--playbook", str(playbook_path),
             "--playbook-key", key.verify_key.encode().hex(),
             "--serve-spore", str(payload), "--chunking", "cdc",
             "--adaptive-quorum", "--quorum-k", "2", "--quorum-floor", "1",
             "--latency-target-ms", str(TARGET_MS),
             "--rollout-dir", str(work / "rollouts")],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_root(url)
        for index in range(REPLICAS):
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "pando.stemd",
                 "--node-id", f"slow-{index}", "--root", url,
                 "--pairing-token-file", str(token_file),
                 "--backend", "mock", "--checkpoint", "mock",
                 "--delay-ms", str(SLOW_MS), "--serialize"],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_nodes(url, token, REPLICAS)
        time.sleep(1.0)

        # This check is about growth; retirement has its own check
        # (tools/retirement_check.py), so the member ceiling is out of reach
        # here and the trajectory shows one mechanism at a time.
        supervisor = ElasticSupervisor(
            url, token, max_members=4, keep=2,
            max_member_latency_ms=10 ** 9, cooldown_s=args.cooldown_s,
            max_per_host=args.max_per_host)
        journal = Journal(work / "supervision.jsonl")
        grown = 0

        def grow() -> str:
            nonlocal grown
            grown += 1
            node_id = f"reproduced-{grown}"
            target = Target(
                node_id=node_id, host="localhost", cwd=str(ROOT),
                command=[sys.executable, "-m", "pando.join",
                         "--node-id", node_id,
                         "--spore", "--root", url, "--sandbox-confirmed",
                         "--pairing-token", token,
                         "--checkpoint", LINEAGE])
            outcome, child = reproduce(
                payload, LINEAGE, target, url, token,
                declared_lineages=sorted(admitted),
                declared_artifacts={LINEAGE: digest},
                grants_path=grants, sandbox_confirmed=True,
                log_path=work / f"{node_id}.log", timeout=600.0)
            if child is not None:
                children.append(child)
            if not outcome.registered:
                tail = (work / f"{node_id}.log").read_text(
                    encoding="utf-8", errors="replace").splitlines()[-8:]
                print("join log tail:", *tail, sep="\n  ")
            return f"{node_id} registered={outcome.registered}"

        retirer = Retirer(url, token, grants_path=grants, journal=journal)

        def retire(node_id: str, reason: str, grant_action: str = "retire"):
            return retirer.retire_with_grant(node_id, reason, grant_action)

        for round_index in range(args.rounds):
            stats = measure(url, token, args.concurrency, args.tasks)
            status = supervisor.status()
            supervision = supervisor.step(
                grow=grow, journal=journal, retire=retire,
                candidate_host=target_host,
                candidate_checkpoint=LINEAGE)
            row = {"round": round_index, "members": status.get("members"),
                   "quorum_k": status.get("quorum_k"),
                   "decision_ewma_ms": status.get("decision_ewma_ms"),
                   **stats, "action": supervision.action,
                   "reason": supervision.reason}
            trajectory.append(row)
            print(f"  round {round_index}: p50={stats['p50_ms']:6.1f} ms "
                  f"members={row['members']} k={row['quorum_k']} "
                  f"-> {supervision.action}: {supervision.reason}",
                  flush=True)
            time.sleep(args.interval_s)
    finally:
        for proc in children + procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in children + procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for index in range(1, grown + 1):
            shutil.rmtree(ROOT / "spore-cache" / f"reproduced-{index}",
                          ignore_errors=True)

    grew = [row for row in trajectory if row["action"] == "grow"]
    final = trajectory[-1]
    summary = {"target_ms": TARGET_MS, "concurrency": args.concurrency,
               "spore_reproductions": grown,
               "growths": len(grew),
               "final_p50_ms": final["p50_ms"],
               "final_members": final["members"],
               "trajectory": trajectory}
    out_dir = ROOT / "metrics" / "elastic-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"elastic-growth-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "trajectory"}, indent=2))
    print(f"report: {report}")
    shutil.rmtree(work, ignore_errors=True)
    # What this check can assert is the mechanism: the supervisor decided to
    # grow, the growth went through the real reproduction path, and the joined
    # node registered as a worker. Whether growth *helps* is a separate
    # question with a measured answer - it depends on whether the new member
    # brings new capacity (round 114).
    if not grew:
        raise SystemExit("the supervisor never decided to grow")
    if grown < 1:
        raise SystemExit("growth did not happen through reproduction")
    registered = sum(1 for row in trajectory
                     if "registered=True" in row["reason"])
    if registered < 1:
        raise SystemExit("no reproduced node registered with the root")
    first = trajectory[0]["p50_ms"]
    summary["registered_reproductions"] = registered
    summary["latency_ratio"] = round(final["p50_ms"] / first, 2)
    print(f"{grown} real reproduction(s), {registered} registered; latency "
          f"{first} -> {final['p50_ms']} ms "
          f"({summary['latency_ratio']}x). Growth helps only when the new "
          f"member brings capacity the swarm did not have - on one host, a "
          f"CPU-bound member shares the CPU and the loop's own measurement "
          f"says so.")
    if final["p50_ms"] > first:
        print("note: latency got worse here, which the supervisor then "
              "reported as a per-decision-cost problem at the member ceiling")
    print("ELASTIC GROWTH CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
