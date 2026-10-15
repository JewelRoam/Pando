"""Does a slower compute unit help the fleet's throughput, or cap it?

Rounds 130-131 established that the CPU, the Intel iGPU and the Intel NPU
answer identically on every task, at 17.6, 10.0 and 8.7 decisions per second
respectively. Because the answers are identical, a fleet of them can be
measured for *capacity* alone - no accuracy question is entangled with it.

The quorum waits for the members it picked, so a slow unit in a quorum caps
that decision's latency, while a fast one raises the ceiling. Round 109 found
that replicas alone buy no capacity unless the quorum spreads across them;
round 110 made the spreading load-aware. This measures what those rules do on
real heterogeneous silicon: one unit, two, then all three, under a fixed
concurrent load.

Usage:
    python tools/hetero_capacity_check.py [--clients 4] [--decisions 40]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import OnnxLayaBackend  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402
from pandosi.stemd import Stem, heartbeat_loop  # noqa: E402

UNITS = [
    ("en-cpu", "CPU", "artifacts/laya-en.onnx"),
    ("en-gpu", "GPU", "artifacts/laya-en-static-fixed.onnx"),
    ("en-npu", "NPU", "artifacts/laya-en-static-nonisnan.onnx"),
]
REPO = "convaiinnovations/laya"
PAD_TO = 256


def start_unit(name: str, device: str, path: str) -> Stem:
    backend = OnnxLayaBackend(name, str(ROOT / path), repo=REPO,
                              device=device,
                              pad_to=PAD_TO if device != "CPU" else None,
                              checkpoint="english")
    stem = Stem(name, backend)
    stem.start()
    return stem


def drive(root: Root, task: dict, clients: int,
          total: int) -> tuple[float, list[float]]:
    """Send `total` decisions through `clients` concurrent callers."""
    payload = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"]).to_json()
    latencies: list[float] = []
    lock = threading.Lock()
    counter = {"sent": 0}

    def worker() -> None:
        while True:
            with lock:
                if counter["sent"] >= total:
                    return
                counter["sent"] += 1
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(
                            f"{root.url()}/decide", data=payload,
                            headers={"Content-Type": "application/json"}),
                        timeout=300) as response:
                    response.read()
            except Exception:
                continue
            with lock:
                latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - started, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=40,
                        help="decisions per client per configuration")
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    print(f"{len(tasks)} tasks available; "
          f"{args.clients} clients x {args.decisions} decisions per config",
          flush=True)

    print("loading units ...", flush=True)
    stems = {name: start_unit(name, device, path)
             for name, device, path in UNITS}

    # The matrix that separates the two things a fleet can change: how many
    # units it holds, and how many of them one decision waits for. With a
    # single lineage the answers are identical in every cell (rounds 130-131
    # measured 240/240 across units), so every difference here is capacity.
    configurations = [
        (f"{len(units)}u-k{k}", units[:len(units)], k)
        for units in (["en-cpu"], ["en-cpu", "en-gpu"],
                      ["en-cpu", "en-gpu", "en-npu"])
        for k in range(1, len(units) + 1)
    ]
    results = {}
    for label, members, quorum_k in configurations:
        root = Root(coordinator=Coordinator(
            quorum_k=quorum_k, call_timeout=300.0,
            initial_stem_timeout=180.0, timeout_multiplier=6.0,
            min_stem_timeout=30.0))
        root.start()
        for name in members:
            threading.Thread(target=heartbeat_loop,
                             args=(stems[name], root.url()),
                             daemon=True).start()
        time.sleep(1.0)
        try:
            elapsed, latencies = drive(root, tasks[0], args.clients,
                                       args.clients * args.decisions)
        finally:
            root.stop()
        latencies.sort()
        entry = {
            "members": members,
            "quorum_k": quorum_k,
            "completed": len(latencies),
            "seconds": round(elapsed, 2),
            "throughput_per_s": round(len(latencies) / elapsed, 2),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
        }
        results[label] = entry
        print(f"  {label:18s} {entry['completed']:4d} decisions in "
              f"{entry['seconds']:6.2f}s  {entry['throughput_per_s']:6.2f}/s  "
              f"p50 {entry['p50_ms']:6.1f} ms  p95 {entry['p95_ms']:6.1f} ms",
              flush=True)
    for stem in stems.values():
        stem.stop()

    baseline = results["1u-k1"]["throughput_per_s"]
    for label, entry in results.items():
        entry["speedup_vs_cpu"] = round(entry["throughput_per_s"] / baseline, 2)
    out = ROOT / "metrics" / "ensemble-checks" / "hetero-capacity.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
