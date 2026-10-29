"""Elastic growth, priced: what does each added device buy?

Rounds 138-146 built growth and retirement and measured them by opinions: a
new opinion helps, a replica does not (round 146: auto 50 -> 15 rows at
0.800 -> 1.000 with a second opinion; a replica changed nothing). Round 154
changed the arithmetic underneath: opinions are counted
(`--lineage-quorum`), the advisor sees the counted set, and the member that
arrived by fine-tuning decides 0.9417 alone on split D.

So the ladder deserves its second measurement - live, on the real
accelerators, over the full 240 rows of split D, with the offline prediction
written down *first*:

  stage 1  english alone (iGPU)          k=1   one opinion
  stage 2  + an english replica (NPU)    k=2   still one opinion
  stage 3  + the ft-v3 opinion (3070)    k=3   a new opinion arrives
  stage 4  retire the replica            k=2   same opinions as stage 3

Zero CPU inference throughout. What is measured per stage: the system's
answers (gate + advisor), latency, and what each device cost the host CPU.

Usage:
    python tools/growth_ladder_check.py
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
sys.path.insert(0, str(ROOT_DIR / "tools"))

from calibrated_gate_check import decide_lineage  # noqa: E402
from cpu_free_fleet_check import (STEMS, process_tree_cpu_seconds,  # noqa: E402
                                  stem_command)
from idle_silicon_check import host_cpu_load  # noqa: E402
from pando.calibrate import load_calibration  # noqa: E402
from pando.combiner import Combiner  # noqa: E402
from pando.common import DecisionRequest  # noqa: E402
from pando.coordinator import Coordinator, Root  # noqa: E402

NEW = "ft-v3"


def load_rows(suite_path: Path, votes_dir: Path) -> list[dict]:
    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in suite_path.read_text(encoding="utf-8").splitlines()
              if line.strip())}

    def load(name: str) -> dict:
        payload = json.loads((votes_dir / f"votes-d-{name}.json")
                             .read_text(encoding="utf-8"))
        return {row["task_id"]: row for row in payload["rows"]}

    english, new_votes = load("seated-english"), load(NEW)
    rows = []
    for task_id in sorted(tasks):
        rows.append({
            "task_id": task_id,
            "expected": english[task_id]["expected"],
            "candidates": tasks[task_id]["candidates"],
            "state": tasks[task_id]["state"],
            "question": tasks[task_id]["question"],
            "choices": {"english": english[task_id]["choice"],
                        NEW: new_votes[task_id]["choice"]},
            "confidences": {"english": english[task_id]["probability"],
                            NEW: new_votes[task_id]["probability"]},
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT_DIR / "assets"
                                               / "heldout3_test.jsonl"))
    parser.add_argument("--votes-dir", default=str(ROOT_DIR / "metrics"
                                                  / "ensemble-checks"))
    parser.add_argument("--combiner", default=str(ROOT_DIR / "assets"
                                                  / "combiner.json"))
    parser.add_argument("--timeout", type=float, default=360.0)
    args = parser.parse_args()

    combiner = Combiner.load(args.combiner)
    temperatures = {name: load_calibration(name)["temperature"]
                    for name in ("english", NEW)}
    rows = load_rows(Path(args.suite), Path(args.votes_dir))
    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]

    stages = [
        {"label": "1. english alone (iGPU)", "stems": ["igpu"],
         "quorum": 1, "opinions": ["english"]},
        {"label": "2. + english replica (NPU)", "stems": ["igpu", "npu"],
         "quorum": 2, "opinions": ["english", "english"]},
        {"label": "3. + ft-v3 opinion (3070)", "stems": ["igpu", "npu",
                                                         "dgpu"],
         "quorum": 3, "opinions": ["english", "english", NEW]},
        {"label": "4. retire the replica", "stems": ["igpu", "dgpu"],
         "quorum": 2, "opinions": ["english", NEW]},
    ]

    print("prediction, written before anything is started "
          "(recorded split-D votes, live semantics):")
    print(f"  {'stage':34s} {'auto':>5s} {'auto acc':>9s} {'decided':>8s}")
    predictions = {}
    for stage in stages:
        entry = decide_lineage(rows, stage["opinions"], temperatures,
                               combiner, combiner_sees_opinions=True)
        predictions[stage["label"]] = entry
        print(f"  {stage['label']:34s} {entry['auto']:>5d} "
              f"{entry['auto_accuracy']:>9.4f} {entry['decided']:>8.4f}")

    root = Root(coordinator=Coordinator(
        quorum_k=1, call_timeout=300.0,
        initial_stem_timeout=args.timeout, timeout_multiplier=6.0,
        min_stem_timeout=30.0, combiner=combiner, count_lineages=True))
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    logs, processes = {}, {}

    def start_stem(name: str) -> None:
        print(f"  starting {name} ({STEMS[name]['device']}) ...", flush=True)
        log = open(log_dir / f"ladder-{name}.log", "w", encoding="utf-8")
        logs[name] = log
        processes[name] = subprocess.Popen(
            stem_command(name, STEMS[name], root.url()), cwd=str(ROOT_DIR),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=0x08000000)

    def wait_live(expected: int) -> None:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and \
                len(root.coordinator.live_nodes()) < expected:
            time.sleep(1.0)
        live = len(root.coordinator.live_nodes())
        if live < expected:
            raise SystemExit(f"only {live}/{expected} stems registered")

    def drive() -> dict:
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
        served = {name: 0 for name in processes}
        auto = auto_hits = 0
        suggestion_hits = suggestion_total = 0
        decided = 0
        latencies = []
        started = time.perf_counter()
        for task in tasks:
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"])
            row_started = time.perf_counter()
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{root.url()}/decide", data=request.to_json(),
                        headers={"Content-Type": "application/json"}),
                    timeout=300) as response:
                body = json.loads(response.read())
            latencies.append((time.perf_counter() - row_started) * 1000)
            hit = body.get("choice") == task["expected"] \
                if body.get("tier") == "auto" else False
            if body.get("tier") == "auto":
                auto += 1
                auto_hits += hit
            if body.get("suggested_choice"):
                suggestion_total += 1
                suggestion_hits += body["suggested_choice"] \
                    == task["expected"]
            decided += hit or (body.get("suggested_choice")
                               == task["expected"])
            for vote in body.get("votes", []):
                if vote.get("node_id") in served:
                    served[vote["node_id"]] += 1
        elapsed = time.perf_counter() - started
        stop.set()
        sampler_thread.join(timeout=2)
        cpu_seconds = {name: process_tree_cpu_seconds(process.pid)
                       - before[name] for name, process in processes.items()}
        return {
            "auto": auto,
            "auto_accuracy": round(auto_hits / max(1, auto), 4),
            "declined": len(tasks) - auto,
            "suggestions": suggestion_total,
            "suggestion_accuracy": round(suggestion_hits
                                         / max(1, suggestion_total), 4),
            "decided": round(decided / len(tasks), 4),
            "wall_s": round(elapsed, 1),
            "rows_per_s": round(len(tasks) / elapsed, 2),
            "p50_ms": round(statistics.median(latencies), 1),
            "host_load_during": round(statistics.mean(samples), 4)
            if samples else None,
            "stems": {name: {
                "served": served[name],
                "cpu_ms_per_decision": round(
                    1000 * cpu_seconds[name] / max(1, served[name]), 2)}
                for name in processes},
        }

    results = {}
    try:
        print("live ladder:")
        for stage in stages:
            for name in stage["stems"]:
                if name not in processes:
                    start_stem(name)
                    wait_live(len(processes))
            if stage["label"].startswith("4"):
                # Durable removal: retire() drops the node and blocks its
                # re-registration, so the heartbeat cannot bring it back.
                root.coordinator.retire("npu", "replica not needed")
                processes["npu"].terminate()
                del processes["npu"]
            root.coordinator.quorum_k = stage["quorum"]
            time.sleep(1.0)
            entry = drive()
            results[stage["label"]] = entry
            print(f"  {stage['label']:34s} auto {entry['auto']:3d} "
                  f"@{entry['auto_accuracy']:.4f}  sugg "
                  f"{entry['suggestions']:3d}@{entry['suggestion_accuracy']:.4f}"
                  f"  decided {entry['decided']:.4f}  "
                  f"{entry['rows_per_s']:5.2f} rows/s  "
                  f"host {entry['host_load_during']:.0%}", flush=True)
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

    print("\nprediction vs live (decided; predictions from recorded votes):")
    for stage in stages:
        predicted = predictions[stage["label"]]["decided"]
        live = results[stage["label"]]["decided"]
        print(f"  {stage['label']:34s} predicted {predicted:.4f}  "
              f"live {live:.4f}  delta {live - predicted:+.4f}")

    out = log_dir / "growth-ladder.json"
    out.write_text(json.dumps({"predictions": predictions,
                               "stages": results}, indent=2),
                   encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
