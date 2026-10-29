"""Does a fleet pick the cheap unit when it has the choice?

Round 134 measured what each compute unit costs the machine it lives on: the
CPU stem spends 791-1647 ms of host CPU per decision, the Intel iGPU 6.6-5.1
ms and the Intel NPU 1.4-1.8 ms, and only the CPU stem loses throughput when
the host is loaded. Round 132 had already found the reason a fleet still
reached for the expensive member: `pick_quorum` ordered candidates by
in-flight calls and a latency band, and nothing in that ordering knew what a
member costs the machine.

`pick_quorum` now sorts within a checkpoint group by in-flight calls and then
by the cost the stem reports for itself, so a fleet with an idle and a cheap
member reaches for the cheap one. This measures the fleet on all three units
at k=1, idle and loaded, and records which member served each decision - the
only way to see whether the rule took effect.

Usage:
    python tools/cost_aware_quorum_check.py [--clients 4] [--decisions 30]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pando.common import DecisionRequest  # noqa: E402
from pando.coordinator import Coordinator, Root  # noqa: E402

UNITS = [
    ("en-cpu", "CPU", "artifacts/laya-en.onnx"),
    ("en-gpu", "GPU", "artifacts/laya-en-static-fixed.onnx"),
    ("en-npu", "NPU", "artifacts/laya-en-static-nonisnan.onnx"),
]
REPO = "convaiinnovations/laya"


def spawn_unit(name: str, device: str, path: str, root_url: str,
               log_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "pando.worker",
        "--node-id", name, "--checkpoint", "english",
        "--backend", "onnx", "--onnx-path", str(ROOT_DIR / path),
        "--repo", REPO, "--device", device, "--root", root_url,
        "--host", "127.0.0.1",
    ]
    if device != "CPU":
        command += ["--pad-to", "256"]
    log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
    print(f"  starting {name} ({device}) ...", flush=True)
    return subprocess.Popen(command, cwd=str(ROOT_DIR), stdout=log,
                            stderr=subprocess.STDOUT)


def wait_for_units(coordinator: Coordinator, expected: int,
                   timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(coordinator.live_nodes()) >= expected:
            return
        time.sleep(1.0)
    raise SystemExit("units did not register in time")


def drive(root: Root, payloads: list[bytes], clients: int,
          total: int) -> tuple[float, list[float], dict]:
    latencies: list[float] = []
    servers: dict[str, int] = {}
    lock = threading.Lock()
    counter = {"sent": 0}
    refusals = [0]
    last_error = [None]

    def worker() -> None:
        while True:
            with lock:
                if counter["sent"] >= total:
                    return
                index = counter["sent"]
                counter["sent"] += 1
            started = time.perf_counter()
            body = None
            for _ in range(20):
                try:
                    with urllib.request.urlopen(
                            urllib.request.Request(
                                f"{root.url()}/decide",
                                data=payloads[index % len(payloads)],
                                headers={"Content-Type":
                                         "application/json"}),
                            timeout=300) as response:
                        body = json.loads(response.read())
                    break
                except Exception as exc:
                    with lock:
                        refusals[0] += 1
                        last_error[0] = f"{type(exc).__name__}: {exc}"[:200]
                    time.sleep(0.02)
            if body is None:
                continue
            if body.get("votes"):
                served = body["votes"][0].get("node_id", "unknown")
            elif body.get("unanswered"):
                served = "silent:" + ",".join(body["unanswered"])
            else:
                served = "no-quorum"
            with lock:
                latencies.append((time.perf_counter() - started) * 1000)
                servers[served] = servers.get(served, 0) + 1

    threads = [threading.Thread(target=worker) for _ in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    return elapsed, sorted(latencies), {"served_by": servers,
                                        "refusals": refusals[0],
                                        "last_error": last_error[0]}


def start_burners(count: int) -> list[subprocess.Popen]:
    code = "x = 0\nwhile True:\n    x = (x * 3 + 1) % 1000003\n"
    return [subprocess.Popen([sys.executable, "-c", code],
                             creationflags=0x08000000)
            for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--burn", type=int, default=8)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "heldout_test.jsonl"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    payloads = [DecisionRequest(state=task["state"],
                                question=task["question"],
                                candidates=task["candidates"]).to_json()
                for task in tasks]
    total = args.clients * args.decisions

    root = Root(coordinator=Coordinator(
        quorum_k=1, call_timeout=300.0, initial_stem_timeout=240.0,
        timeout_multiplier=6.0, min_stem_timeout=30.0))
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = [spawn_unit(*unit, root_url=root.url(), log_dir=log_dir)
                 for unit in UNITS]
    burners: list[subprocess.Popen] = []
    results: dict[str, dict] = {}
    try:
        wait_for_units(root.coordinator, len(UNITS))
        # Let the stems serve a few decisions so each can report a cost.
        drive(root, payloads, 1, 5)
        time.sleep(6.0)
        costs = {name: root.coordinator.nodes[name]["capability"]
                 .get("host_ms_per_decision") for name, _, _ in UNITS}
        print(f"  reported host cost per decision: {costs}", flush=True)

        for phase in ("idle", "busy"):
            if phase == "busy":
                burners = start_burners(args.burn)
                time.sleep(3.0)
            elapsed, latencies, shape = drive(root, payloads, args.clients,
                                              total)
            if not latencies:
                print(f"  {phase:4s} no decisions completed; "
                      f"{shape}", flush=True)
                results[phase] = {"phase": phase, "completed": 0, **shape}
                continue
            entry = {
                "phase": phase,
                "completed": len(latencies),
                "throughput_per_s": round(len(latencies) / elapsed, 2),
                "p50_ms": round(statistics.median(latencies), 1),
                "served_by": shape["served_by"],
                "refusals": shape["refusals"],
            }
            results[phase] = entry
            print(f"  {phase:4s} {entry['completed']:4d} decisions in "
                  f"{elapsed:6.2f}s  {entry['throughput_per_s']:6.2f}/s  "
                  f"p50 {entry['p50_ms']:6.1f} ms  "
                  f"served by {entry['served_by']}  "
                  f"refusals {entry['refusals']}", flush=True)
    finally:
        for process in burners + processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        root.stop()

    results["reported_host_ms_per_decision"] = costs
    out = ROOT_DIR / "metrics" / "ensemble-checks" / "cost-aware-quorum.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
