"""Which silicon is still available when the CPU is busy?

Round 133 concluded that the host CPU is the ceiling: at four concurrent
clients one stem of any kind reaches ~25-33 decisions per second and adding
more units changes nothing. That measurement assumed an *idle machine*, and a
personal computer is not idle - the CPU is the part that is always busy, with
the browser, the editor and this agent, while the GPU and the NPU sit unused.

So the useful question is not "which unit is fastest on an empty machine" but
"what does each unit cost the host, and what survives when the host is
loaded". This measures, for each unit alone and for a GPU+NPU fleet with no
CPU stem at all:

  - throughput and latency with the host idle;
  - the same with eight of the machine's logical cores burned by background
    load, i.e. a machine that is somebody's daily driver;
  - the host CPU-seconds each unit spends per decision, which is what the
    unit actually costs the machine it lives on.

Usage:
    python tools/idle_silicon_check.py [--clients 4] [--decisions 40]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
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
from pando.rootd import Coordinator, Root  # noqa: E402

UNITS = [
    ("en-cpu", "CPU", "artifacts/laya-en.onnx"),
    ("en-gpu", "GPU", "artifacts/laya-en-static-fixed.onnx"),
    ("en-npu", "NPU", "artifacts/laya-en-static-nonisnan.onnx"),
]
REPO = "convaiinnovations/laya"


def process_cpu_seconds(pid: int) -> float:
    """Total processor time of one process, in seconds (Windows)."""
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return 0.0
    try:
        creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
        ok = ctypes.windll.kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_),
            ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return 0.0
        ticks = ((kernel.dwHighDateTime << 32) | kernel.dwLowDateTime) + \
                ((user.dwHighDateTime << 32) | user.dwLowDateTime)
        return ticks / 1e7
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def host_cpu_load() -> float:
    """Share of the machine's processors in use since the previous call."""
    if not hasattr(host_cpu_load, "_previous"):
        host_cpu_load._previous = None
    idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
    ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    values = tuple(((filetime.dwHighDateTime << 32) | filetime.dwLowDateTime)
                   for filetime in (idle, kernel, user))
    previous = host_cpu_load._previous
    host_cpu_load._previous = values
    if previous is None:
        time.sleep(0.2)
        return host_cpu_load()
    idle_delta = values[0] - previous[0]
    total_delta = (values[1] - previous[1]) + (values[2] - previous[2])
    if total_delta <= 0:
        return 0.0
    return round(max(0.0, 1.0 - idle_delta / total_delta), 4)


def spawn_unit(name: str, device: str, path: str, root_url: str,
               log_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "pando.stemd",
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
    raise SystemExit(f"only {len(coordinator.live_nodes())} of {expected} "
                     "units registered in time")


def drive(root: Root, payloads: list[bytes], clients: int,
          total: int) -> tuple[float, list[float], float, dict]:
    latencies: list[float] = []
    observed: list[dict] = []
    trace: list[tuple] = []
    load_samples: list[float] = []
    lock = threading.Lock()
    counter = {"sent": 0}
    refusals = [0]
    done = threading.Event()

    def sampler() -> None:
        while not done.is_set():
            load_samples.append(host_cpu_load())
            time.sleep(0.5)

    def worker() -> None:
        while True:
            with lock:
                if counter["sent"] >= total:
                    return
                counter["sent"] += 1
            started = time.perf_counter()
            payload = payloads[counter["sent"] % len(payloads)]
            body = None
            for _ in range(20):
                try:
                    with urllib.request.urlopen(
                            urllib.request.Request(
                                f"{root.url()}/decide", data=payload,
                                headers={"Content-Type":
                                         "application/json"}),
                            timeout=300) as response:
                        body = json.loads(response.read())
                    break
                except Exception:
                    # A refused connection is not a served decision: retry it
                    # and count it, so a starved server cannot look fast.
                    with lock:
                        refusals[0] += 1
                    time.sleep(0.02)
            if body is None:
                continue
            with lock:
                observed.append(body)
                trace.append((round((time.perf_counter() - started) * 1000, 1),
                              body.get("tier"), body.get("quorum_size"),
                              body.get("responded")))
            with lock:
                latencies.append((time.perf_counter() - started) * 1000)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    done.set()
    sampler_thread.join(timeout=2)
    answered = sum(1 for body in observed if body.get("quorum_size"))
    return (elapsed, sorted(latencies),
            statistics.mean(load_samples) if load_samples else 0.0,
            {"answered_with_quorum": answered,
             "empty_quorum": len(observed) - answered,
             "connection_refusals": refusals[0],
             "first_responses": [{"ms": entry[0], "tier": entry[1],
                                  "quorum": entry[2], "responded": entry[3]}
                                 for entry in trace[:10]],
             "tiers": {tier: sum(1 for body in observed
                                 if body.get("tier") == tier)
                       for tier in ("auto", "verify", "escalate")}})


def start_burners(count: int) -> list[subprocess.Popen]:
    code = "x = 0\nwhile True:\n    x = (x * 3 + 1) % 1000003\n"
    # 0x08000000 = CREATE_NO_WINDOW, so the load does not open consoles.
    return [subprocess.Popen([sys.executable, "-c", code],
                             creationflags=0x08000000)
            for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--burn", type=int, default=8,
                        help="background busy processes for the loaded phase")
    parser.add_argument("--units", default=None,
                        help="comma-separated units to start (default: all). "
                             "Isolation by construction: `blocked` is a "
                             "retirement bookkeeping dict and pick_quorum "
                             "ignores it, so a unit is only out of the fleet "
                             "if it was never started (round 134)")
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "heldout_test.jsonl"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    # Distinct states, cycled: a loader that repeats one task measures the
    # per-request overhead, not the model (round 134).
    payloads = [DecisionRequest(state=task["state"],
                                question=task["question"],
                                candidates=task["candidates"]).to_json()
                for task in tasks]
    total = args.clients * args.decisions

    wanted = ([name.strip() for name in args.units.split(",")]
              if args.units else [name for name, _, _ in UNITS])
    unknown = [name for name in wanted
               if name not in {n for n, _, _ in UNITS}]
    if unknown:
        raise SystemExit(f"unknown unit(s) {unknown}")

    root = Root(coordinator=Coordinator(
        quorum_k=1, call_timeout=300.0, initial_stem_timeout=240.0,
        timeout_multiplier=6.0, min_stem_timeout=30.0))
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen] = {}
    burners: list[subprocess.Popen] = []
    results: dict[str, dict] = {}
    try:
        for name, device, path in UNITS:
            if name not in wanted:
                continue
            processes[name] = spawn_unit(name, device, path, root.url(),
                                         log_dir)
        wait_for_units(root.coordinator, len(wanted))
        print(f"  {len(root.coordinator.live_nodes())} units registered; "
              f"host idle at {host_cpu_load():.0%}", flush=True)
        names = wanted
        label = "+".join(name.replace("en-", "") for name in names)
        fleets = [(label, list(names))]

        for phase in ("idle", "busy"):
            if phase == "busy":
                burners = start_burners(args.burn)
                time.sleep(3.0)
                print(f"  background load: {args.burn} busy processes, host "
                      f"at {host_cpu_load():.0%}", flush=True)
            for label, members in fleets:
                root.coordinator.quorum_k = len(members)
                time.sleep(0.5)
                before = {name: process_cpu_seconds(processes[name].pid)
                          for name in members}
                elapsed, latencies, load, shape = drive(
                    root, payloads, args.clients, total)
                cpu_seconds = sum(process_cpu_seconds(processes[name].pid)
                                  - before[name] for name in members)
                if not latencies:
                    print(f"  {phase:4s} {label:8s} no decisions completed")
                    continue
                entry = {
                    "phase": phase,
                    "units": members,
                    "completed": len(latencies),
                    "throughput_per_s": round(len(latencies) / elapsed, 2),
                    "p50_ms": round(statistics.median(latencies), 1),
                    "host_load_during": round(load, 3),
                    "stem_cpu_ms_per_decision": round(
                        1000 * cpu_seconds / len(latencies), 2),
                    "quorum_shape": shape,
                }
                results[f"{phase}:{label}"] = entry
                errors = {name: root.coordinator.nodes.get(name, {})
                          .get("stats", {}).get("last_error")
                          for name in members}
                failures = {name: root.coordinator.nodes.get(name, {})
                            .get("stats", {}).get("failures")
                            for name in members}
                entry["stem_last_error"] = errors
                entry["stem_failures"] = failures
                print(f"  {phase:4s} {label:8s} {entry['completed']:4d} dec "
                      f"in {elapsed:6.2f}s  "
                      f"{entry['throughput_per_s']:6.2f}/s  "
                      f"p50 {entry['p50_ms']:6.1f} ms  "
                      f"host {entry['host_load_during']:.0%}  "
                      f"stem cpu {entry['stem_cpu_ms_per_decision']:6.1f} "
                      f"ms/dec  empty-quorum "
                      f"{shape['empty_quorum']}/{len(latencies)}  refusals "
                      f"{shape['connection_refusals']}  stem-errors "
                      f"{ {k: v for k, v in errors.items() if v} }",
                      flush=True)
    finally:
        for process in burners + list(processes.values()):
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        root.stop()

    out = ROOT_DIR / "metrics" / "ensemble-checks" / "idle-silicon.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
