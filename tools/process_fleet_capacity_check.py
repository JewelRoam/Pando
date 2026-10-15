"""Does the capacity ceiling survive when the stems are real processes?

Round 132 measured a ceiling on one host: two units added capacity at k=1 and
the third did not, and the stated reason was that all three stems shared one
process - one GIL, one scheduler, one interpreter per request. That is a
confound in the project's own measurement, because the shape a fleet actually
deploys is separate `stemd` processes.

This runs the same units-by-quorum phases against stems started the way
production starts them: `python -m pandosi.stemd --root <url>` per unit, each
with its own interpreter, its own ONNX/OpenVINO session and its own HTTP
server, heartbeating to a root that is the only shared thing in the picture.

If the third unit adds capacity here and not in-process, round 132's ceiling
was an artifact of the harness. If it does not, the ceiling is the host.

Usage:
    python tools/process_fleet_capacity_check.py [--clients 4] [--decisions 40]
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

from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402

UNITS = [
    ("en-cpu", "CPU", "artifacts/laya-en.onnx"),
    ("en-gpu", "GPU", "artifacts/laya-en-static-fixed.onnx"),
    ("en-npu", "NPU", "artifacts/laya-en-static-nonisnan.onnx"),
]
REPO = "convaiinnovations/laya"


def spawn_unit(name: str, device: str, path: str, root_url: str,
               log_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "pandosi.stemd",
        "--node-id", name,
        "--checkpoint", "english",
        "--backend", "onnx",
        "--onnx-path", str(ROOT_DIR / path),
        "--repo", REPO,
        "--device", device,
        "--root", root_url,
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
    raise SystemExit(f"only {len(coordinator.live_nodes())} of {expected} "
                     "units registered in time")


def drive(root: Root, payload: bytes, clients: int,
          total: int) -> tuple[float, list[float]]:
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
    return time.perf_counter() - started, sorted(latencies)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--sweep-clients", default=None,
                        help="comma-separated client counts to run against a "
                             "single CPU unit, to find the host's ceiling")
    parser.add_argument("--solo-units", action="store_true",
                        help="measure each unit alone at the given client "
                             "count, to see whether the accelerator matters")
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "heldout_test.jsonl"))
    args = parser.parse_args()

    task = json.loads(Path(args.suite).read_text(encoding="utf-8")
                      .splitlines()[0])
    payload = DecisionRequest(state=task["state"],
                              question=task["question"],
                              candidates=task["candidates"]).to_json()
    tasks_total = args.clients * args.decisions

    root = Root(coordinator=Coordinator(
        quorum_k=1, call_timeout=300.0, initial_stem_timeout=240.0,
        timeout_multiplier=6.0, min_stem_timeout=30.0))
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for name, device, path in UNITS:
            processes.append(spawn_unit(name, device, path, root.url(),
                                        log_dir))
        wait_for_units(root.coordinator, len(UNITS))
        print(f"  {len(root.coordinator.live_nodes())} units registered",
              flush=True)

        coordinator = root.coordinator
        names = [name for name, _, _ in UNITS]
        phases = []
        for blocked_count in (2, 1, 0):
            for quorum_k in (1,) if blocked_count else (1, 3):
                available = names[:len(names) - blocked_count]
                label = f"{len(available)}u-k{quorum_k}"
                phases.append((label, available, quorum_k))
        if args.sweep_clients:
            phases = [(f"1u-k1-c{count}", [names[0]], 1)
                      for count in
                      (int(value) for value in args.sweep_clients.split(","))]
        if args.solo_units:
            phases = [(f"{name}-c{args.clients}", [name], 1)
                      for name in names]

        results = {}
        for label, available, quorum_k in phases:
            coordinator.blocked = {
                name: "out of this phase" for name in names
                if name not in available}
            coordinator.quorum_k = quorum_k
            time.sleep(0.5)
            clients = (int(label.rsplit("-c", 1)[1]) if "-c" in label
                       else args.clients)
            elapsed, latencies = drive(root, payload, clients,
                                       clients * args.decisions)
            entry = {
                "units": available,
                "quorum_k": quorum_k,
                "completed": len(latencies),
                "seconds": round(elapsed, 2),
                "throughput_per_s": round(len(latencies) / elapsed, 2),
                "p50_ms": round(statistics.median(latencies), 1),
                "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))],
                                1),
            }
            results[label] = entry
            print(f"  {label:10s} {entry['completed']:4d} decisions in "
                  f"{entry['seconds']:6.2f}s  "
                  f"{entry['throughput_per_s']:6.2f}/s  "
                  f"p50 {entry['p50_ms']:6.1f} ms  "
                  f"p95 {entry['p95_ms']:6.1f} ms", flush=True)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        root.stop()

    out = ROOT_DIR / "metrics" / "ensemble-checks" / "process-capacity.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
