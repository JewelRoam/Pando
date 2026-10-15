"""Can the fleet run with the CPU left alone?

Round 133 found the host CPU is the fleet's ceiling, and round 134 priced it:
a CPU stem spends ~2.9 CPU-seconds per decision while the iGPU spends ~36
CPU-milliseconds and the NPU ~4. On a personal computer that difference is
the whole game - the CPU is the part that is *always* busy (the browser, the
editor, this agent) while the accelerators sit idle. Round 130 sketched a
GPU+NPU fleet with no CPU stem in it, but the recorded run ever only
measured the NPU alone.

This runs the fleet a daily-driver machine actually has: the iGPU (OpenVINO
GPU) and the NPU (OpenVINO NPU) as the two cheap english-lineage replicas,
and the discrete RTX GPU carrying the member this project trained itself
(`ft-v3`, torch/CUDA) - the first live member outside the OpenVINO
ecosystem, reached over the same HTTP contract. No CPU inference anywhere.

It drives the whole fourth wording split (set D, 240 rows - nothing in the
fleet or its combiner was fitted on any of it) through a real root, sampling
the machine's CPU load throughout, and reads each stem's own CPU bill for
the window. It reports what the all-accelerator fleet answered, what it cost
the CPU, and how much CPU it left for its owner.

Usage:
    python tools/cpu_free_fleet_check.py [--limit N] [--quorum 3]
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
sys.path.insert(0, str(ROOT_DIR / "tools"))

from idle_silicon_check import host_cpu_load, process_cpu_seconds  # noqa: E402
from pandosi.combiner import Combiner  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402

VENV_CUDA_PYTHON = ROOT_DIR / ".venv-cuda" / "Scripts" / "python.exe"
REPO = "convaiinnovations/laya"


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * 260),
    )


def process_tree_cpu_seconds(pid: int) -> float:
    """CPU seconds of a process plus all its descendants.

    A venv's Scripts\\python.exe is a redirector that runs the real
    interpreter as a child (verified: 255 KB against python.exe's ~100 KB),
    so the pid Popen hands back can burn ~0.00 CPU-s while its child serves
    every decision - the first run of this tool measured exactly that for the
    cuda stem and had to be corrected. Measurement follows the tree.
    """
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD,
                                                  wintypes.DWORD)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32First.argtypes = (ctypes.c_void_p,
                                        ctypes.POINTER(_PROCESSENTRY32))
    kernel32.Process32Next.argtypes = (ctypes.c_void_p,
                                       ctypes.POINTER(_PROCESSENTRY32))
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return process_cpu_seconds(pid)
    children: dict[int, list[int]] = {}
    try:
        entry = _PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32)
        ok = kernel32.Process32First(snapshot, ctypes.byref(entry))
        while ok:
            children.setdefault(entry.th32ParentProcessID,
                                []).append(entry.th32ProcessID)
            ok = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snapshot))
    total, stack, seen = 0.0, [pid], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += process_cpu_seconds(current)
        stack.extend(children.get(current, ()))
    return total

# One entry per accelerator. The two OpenVINO units carry the cheap english
# lineage; the discrete GPU carries the project's own fine-tuned member.
STEMS = {
    "igpu": {"device": "GPU", "backend": "onnx", "checkpoint": "english",
             "path": "artifacts/laya-en-static-fixed.onnx"},
    "npu": {"device": "NPU", "backend": "onnx", "checkpoint": "english",
            "path": "artifacts/laya-en-static-nonisnan.onnx"},
    "dgpu": {"device": "cuda", "backend": "laya", "checkpoint": "ft-v3",
             "path": "artifacts/rlcd-fi-v3"},
}


def stem_command(name: str, spec: dict, root_url: str) -> list[str]:
    interpreter = (sys.executable if spec["backend"] == "onnx"
                   else str(VENV_CUDA_PYTHON))
    command = [interpreter, "-m", "pandosi.stemd",
               "--node-id", name, "--checkpoint", spec["checkpoint"],
               "--backend", spec["backend"],
               "--device", spec["device"], "--root", root_url]
    if spec["backend"] == "onnx":
        command += ["--onnx-path", str(ROOT_DIR / spec["path"]),
                    "--repo", REPO, "--pad-to", "256"]
    else:
        command += ["--model", str(ROOT_DIR / spec["path"])]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets"
                                    / "heldout3_test.jsonl"))
    parser.add_argument("--combiner",
                        default=str(ROOT_DIR / "assets" / "combiner.json"))
    parser.add_argument("--quorum", type=int, default=len(STEMS),
                        help="votes required per row (default: all stems)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=360.0,
                        help="seconds to wait for every stem to register")
    args = parser.parse_args()
    if any(spec["backend"] != "onnx" for spec in STEMS.values()) \
            and not VENV_CUDA_PYTHON.exists():
        raise SystemExit(f"missing {VENV_CUDA_PYTHON}; the cuda stem needs it")

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    combiner = Combiner.load(args.combiner) if Path(args.combiner).exists() \
        else None
    root = Root(coordinator=Coordinator(
        quorum_k=args.quorum, call_timeout=300.0,
        initial_stem_timeout=args.timeout, timeout_multiplier=6.0,
        min_stem_timeout=30.0, combiner=combiner))
    root.start()
    print(f"root up with combiner={bool(combiner)}, quorum {args.quorum}",
          flush=True)
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    logs, processes = {}, {}
    try:
        for name, spec in STEMS.items():
            print(f"  starting {name} ({spec['device']}) ...", flush=True)
            log = open(log_dir / f"cpu-free-{name}.log", "w",
                       encoding="utf-8")
            logs[name] = log
            # 0x08000000 = CREATE_NO_WINDOW: device processes, not consoles.
            processes[name] = subprocess.Popen(
                stem_command(name, spec, root.url()), cwd=str(ROOT_DIR),
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=0x08000000)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and \
                len(root.coordinator.live_nodes()) < len(STEMS):
            time.sleep(1.0)
        live = root.coordinator.live_nodes()
        if len(live) < len(STEMS):
            missing = [name for name in STEMS if name not in live]
            raise SystemExit(
                f"only {len(live)}/{len(STEMS)} stems registered; missing "
                f"{missing}; see metrics/ensemble-checks/cpu-free-*.log")
        for name in STEMS:
            print(f"  live: {name} checkpoint="
                  f"{live[name].get('checkpoint')} url={live[name].get('url')}",
                  flush=True)
        host_cpu_load()  # prime the delta
        baseline_load = host_cpu_load()
        print(f"  host CPU before the run: {baseline_load:.0%} busy",
              flush=True)

        samples: list[float] = []
        stop = threading.Event()

        def sampler() -> None:
            while not stop.is_set():
                samples.append(host_cpu_load())
                time.sleep(0.5)

        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()

        before = {name: process_tree_cpu_seconds(process.pid)
                  for name, process in processes.items()}
        driver_before = time.process_time()
        rows: list[dict] = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"])
            row_started = time.perf_counter()
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(
                            f"{root.url()}/decide", data=request.to_json(),
                            headers={"Content-Type": "application/json"}),
                        timeout=300) as response:
                    body = json.loads(response.read())
                rows.append({"task": task, "result": body,
                             "ms": round(
                                 (time.perf_counter() - row_started) * 1000,
                                 1)})
            except Exception as exc:  # a lost row is data, not a crash
                rows.append({"task": task, "error": str(exc),
                             "ms": round(
                                 (time.perf_counter() - row_started) * 1000,
                                 1)})
            if index % 40 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        elapsed = time.perf_counter() - started
        driver_cpu = time.process_time() - driver_before
        stop.set()
        sampler_thread.join(timeout=2)
        cpu_seconds = {name: process_tree_cpu_seconds(process.pid)
                       - before[name]
                       for name, process in processes.items()}
        health = {}
        for name in STEMS:
            try:
                with urllib.request.urlopen(
                        f"{live[name]['url']}/health", timeout=5) as response:
                    health[name] = json.loads(response.read())
            except Exception as exc:
                health[name] = {"error": str(exc)}
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in logs.values():
            log.close()
        root.stop()

    answered = [row for row in rows if "result" in row]
    auto = [row for row in answered if row["result"]["tier"] == "auto"]
    declined = [row for row in answered if row["result"]["tier"] != "auto"]
    auto_hits = sum(1 for row in auto
                    if row["result"]["choice"] == row["task"]["expected"])
    suggested = [row for row in declined
                 if row["result"].get("suggested_choice")]
    suggest_hits = sum(1 for row in suggested
                       if row["result"]["suggested_choice"]
                       == row["task"]["expected"])
    decided = auto_hits + suggest_hits
    served = {name: 0 for name in STEMS}
    for row in answered:
        for vote in row["result"].get("votes", []):
            if vote.get("node_id") in served:
                served[vote["node_id"]] += 1
    latencies = sorted(row["ms"] for row in rows)
    load_during = statistics.mean(samples) if samples else None
    per_stem = {}
    for name in STEMS:
        per_stem[name] = {
            "served": served[name],
            "cpu_s": round(cpu_seconds[name], 2),
            "cpu_ms_per_decision": round(
                1000 * cpu_seconds[name] / max(1, served[name]), 2),
            "health_host_ms_per_decision":
                health.get(name, {}).get("host_ms_per_decision"),
        }
    fleet_cpu_ms_per_row = round(
        1000 * (sum(cpu_seconds.values()) + driver_cpu) / max(1, len(rows)), 2)

    print(f"\n{len(rows)} rows through an accelerator-only live fleet, "
          f"quorum {args.quorum}")
    print(f"  auto-executed {len(auto)} at "
          f"{auto_hits / max(1, len(auto)):.4f}")
    print(f"  declined      {len(declined)}, suggestions {len(suggested)} "
          f"at {suggest_hits / max(1, len(suggested)):.4f}")
    print(f"  a decided answer in {decided}/{len(rows)} = "
          f"{decided / len(rows):.4f}")
    print(f"  wall {elapsed:.1f} s, {len(rows) / elapsed:.2f} rows/s, "
          f"p50 {statistics.median(latencies):.0f} ms")
    if load_during is not None:
        print(f"  host CPU: {baseline_load:.0%} busy before, "
              f"{load_during:.0%} during -> {1 - load_during:.0%} left free")
    for name, entry in per_stem.items():
        print(f"  {name:4s} served {entry['served']:3d}: "
              f"{entry['cpu_s']:6.2f} CPU-s = "
              f"{entry['cpu_ms_per_decision']:7.2f} ms CPU/decision "
              f"(stem self-report "
              f"{entry['health_host_ms_per_decision']} ms)")
    print(f"  fleet + root cost the CPU {fleet_cpu_ms_per_row:.2f} ms per row "
          f"(one CPU stem alone: ~2856 ms/decision, round 134)")

    report = {"rows": len(rows), "quorum": args.quorum,
              "members": {name: {"checkpoint": STEMS[name]["checkpoint"],
                                 "device": STEMS[name]["device"],
                                 "backend": STEMS[name]["backend"]}
                          for name in STEMS},
              "auto": len(auto),
              "auto_accuracy": round(auto_hits / max(1, len(auto)), 4),
              "declined": len(declined), "suggestions": len(suggested),
              "suggestion_accuracy": round(
                  suggest_hits / max(1, len(suggested)), 4),
              "decided_accuracy": round(decided / max(1, len(rows)), 4),
              "wall_s": round(elapsed, 1), "rows_per_s": round(
                  len(rows) / elapsed, 2),
              "p50_ms": round(statistics.median(latencies), 1),
              "host_load_before": baseline_load,
              "host_load_during": round(load_during, 4)
              if load_during is not None else None,
              "stems": per_stem,
              "fleet_cpu_ms_per_row": fleet_cpu_ms_per_row}
    out = log_dir / "cpu-free-fleet.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
