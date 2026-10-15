"""Live three-tier verification (round 160): Laya iGPU + fine-tune + LAN LLM.

The offline matrix (round 159) predicted: with members {english, ft-v3,
mac-llm} and `--assent-checkpoint ft-v3`, split D decides ~0.9833 with ~199
auto-executed rows all correct, the Mac answering at B-prompt speed. This
runs exactly that on real hardware: onnx GPU english stem, torch/CUDA ft-v3
stem, and a stemd `--backend openai` stem that calls the Mac's Qwen3-4B over
the LAN (Windows -> Mac).

Prediction written first; the report lands in metrics/ensemble-checks/.

Usage:
    python tools/tiered_live_check.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "tools"))

from cpu_free_fleet_check import process_tree_cpu_seconds  # noqa: E402
from pandosi.combiner import Combiner  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402

MEMBERS = ["english", "ft-v3", "mac-llm"]


def stem_commands(root_url: str) -> dict[str, list[str]]:
    python = sys.executable
    venv = str(ROOT_DIR / ".venv-cuda" / "Scripts" / "python.exe")
    key_file = ROOT_DIR / "metrics" / "lan-check" / "llm-api.token"
    return {
        "igpu": [python, "-m", "pandosi.stemd", "--node-id", "igpu",
                 "--checkpoint", "english", "--backend", "onnx",
                 "--onnx-path", "artifacts/laya-en-static-fixed.onnx",
                 "--repo", "convaiinnovations/laya", "--device", "GPU",
                 "--pad-to", "256", "--root", root_url],
        "dgpu": [venv, "-m", "pandosi.stemd", "--node-id", "dgpu",
                 "--checkpoint", "ft-v3", "--backend", "laya",
                 "--model", "artifacts/rlcd-fi-v3", "--device", "cuda",
                 "--root", root_url],
        "macllm": [python, "-m", "pandosi.stemd", "--node-id", "mac-llm",
                   "--checkpoint", "mac-llm", "--backend", "openai",
                   "--openai-url", "http://192.168.31.41:8080/v1",
                   "--openai-key-file", str(key_file),
                   "--openai-model", "qwen3-4b-instruct-2507",
                   "--openai-probability", "0.5", "--root", root_url],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT_DIR / "assets"
                                               / "heldout3_test.jsonl"))
    parser.add_argument("--combiner", default=str(ROOT_DIR / "assets"
                                                  / "combiner.json"))
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--assent-fallback", default="advisor",
                        choices=["advisor", "veto"])
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    print(f"prediction (round 159, offline): decided ~0.9833, auto ~199 "
          f"rows all correct, mac tier ~139 ms/row. {len(tasks)} rows.",
          flush=True)

    coordinator = Coordinator(
        quorum_k=len(MEMBERS), call_timeout=120.0,
        initial_stem_timeout=args.timeout, timeout_multiplier=6.0,
        min_stem_timeout=30.0, combiner=Combiner.load(args.combiner),
        assent_checkpoint="ft-v3",
        assent_fallback=args.assent_fallback)
    root = Root(coordinator=coordinator)
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    logs, processes = {}, {}
    try:
        for name, command in stem_commands(root.url()).items():
            print(f"  starting {name} ...", flush=True)
            log = open(log_dir / f"tiered-{name}.log", "w", encoding="utf-8")
            logs[name] = log
            processes[name] = subprocess.Popen(
                command, cwd=str(ROOT_DIR), stdout=log,
                stderr=subprocess.STDOUT, creationflags=0x08000000)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and \
                len(coordinator.live_nodes()) < len(MEMBERS):
            time.sleep(1.0)
        if len(coordinator.live_nodes()) < len(MEMBERS):
            raise SystemExit("stems did not all register in time")
        for node_id, info in sorted(coordinator.live_nodes().items()):
            print(f"  live: {node_id} checkpoint={info.get('checkpoint')}")

        before = {name: process_tree_cpu_seconds(process.pid)
                  for name, process in processes.items()}
        auto = auto_hits = decided = 0
        suggestion_total = suggestion_hits = 0
        latencies = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"])
            row_started = time.perf_counter()
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{root.url()}/decide", data=request.to_json(),
                        headers={"Content-Type": "application/json"}),
                    timeout=120) as response:
                body = json.loads(response.read())
            latencies.append((time.perf_counter() - row_started) * 1000)
            if body.get("tier") == "auto":
                auto += 1
                hit = body.get("choice") == task["expected"]
                auto_hits += hit
                decided += hit
            elif body.get("veto"):
                # veto fallback: the acting answer is the trusted member's
                # choice, carried in `choice`; the suggestion is advisory
                # only (round 161).
                decided += body.get("choice") == task["expected"]
            elif body.get("suggested_choice"):
                suggestion_total += 1
                hit = body["suggested_choice"] == task["expected"]
                suggestion_hits += hit
                decided += hit
            if index % 60 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        elapsed = time.perf_counter() - started
        cpu_seconds = {name: process_tree_cpu_seconds(process.pid)
                       - before[name] for name, process in processes.items()}
        health = {}
        for name in MEMBERS:
            url = coordinator.nodes.get(name, {}).get("url")
            if not url:
                continue
            try:
                with urllib.request.urlopen(f"{url}/health",
                                            timeout=5) as response:
                    health[name] = json.loads(response.read())
            except Exception as exc:
                health[name] = {"error": str(exc)}
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except Exception:
                process.kill()
        for log in logs.values():
            log.close()
        root.stop()

    print(f"\n{len(tasks)} rows through the live three-tier fleet "
          f"(assent=ft-v3)")
    print(f"  auto {auto} @ {auto_hits / max(1, auto):.4f}; suggestions "
          f"{suggestion_total} @ {suggestion_hits / max(1, suggestion_total):.4f}; "
          f"decided {decided}/{len(tasks)} = {decided / len(tasks):.4f}")
    print(f"  wall {elapsed:.1f}s, {len(tasks) / elapsed:.2f} rows/s, "
          f"p50 {statistics.median(latencies):.0f} ms")
    stems_of = {"english": "igpu", "ft-v3": "dgpu", "mac-llm": "macllm"}
    for member, stem in stems_of.items():
        asked = health.get(stem, {}).get("decisions")
        print(f"  {member:7s} asked {asked} times, "
              f"{1000 * cpu_seconds[stem] / max(1, asked or 0):7.2f} "
              f"ms CPU/decision")
    report = {"rows": len(tasks), "assent": "ft-v3",
              "assent_fallback": args.assent_fallback,
              "members": MEMBERS, "auto": auto,
              "auto_accuracy": round(auto_hits / max(1, auto), 4),
              "suggestions": suggestion_total,
              "suggestion_accuracy": round(suggestion_hits
                                           / max(1, suggestion_total), 4),
              "decided": round(decided / len(tasks), 4),
              "rows_per_s": round(len(tasks) / elapsed, 2),
              "p50_ms": round(statistics.median(latencies), 1),
              "asked": {member: health.get(stem, {}).get("decisions")
                        for member, stem in stems_of.items()}}
    out = log_dir / "tiered-live.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
