"""What does each kind of growth buy, measured while the fleet serves?

The growth machinery was built in rounds 109-116, before the rules that
decide what a member *means*: a replica is one opinion (121), a correlated
family is one opinion (129), the cheapest member serves (136), and a governor
declares all of it (145). Nobody has watched a fleet grow and shrink under
those rules.

This runs four stages against one running root, driving the same labelled rows
through each and recording what changed:

  1. one leaf        - english on the iGPU
  2. + a new opinion - the 19 KB text member joins
  3. + a replica     - english again, on the NPU
  4. - the replica   - the NPU stem is retired

For every stage: throughput, latency, how many opinions the quorum reported,
what it auto-executed and how often it produced a decided answer.

Usage:
    python tools/elastic_fleet_check.py [--rows 60]
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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pando.backend import OnnxLayaBackend  # noqa: E402
from pando.combiner import Combiner  # noqa: E402
from pando.common import DecisionRequest  # noqa: E402
from pando.coordinator import Coordinator, Root  # noqa: E402
from pando.worker import Stem, heartbeat_loop  # noqa: E402
from pando.textmodel import TextModel  # noqa: E402

ACCEL = "artifacts/laya-en-static-fixed.onnx"
NPU_GRAPH = "artifacts/laya-en-static-nonisnan.onnx"


def ask(root: Root, task: dict, token: str | None) -> dict:
    request = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"])
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Pairing-Token"] = token
    with urllib.request.urlopen(
            urllib.request.Request(f"{root.url()}/decide",
                                   data=request.to_json(), headers=headers),
            timeout=120) as response:
        return json.loads(response.read())


def stage(label: str, root: Root, tasks: list[dict]) -> dict:
    latencies, rows = [], []
    started = time.perf_counter()
    for task in tasks:
        began = time.perf_counter()
        body = ask(root, task, None)
        latencies.append((time.perf_counter() - began) * 1000)
        rows.append({"task": task, "result": body})
    elapsed = time.perf_counter() - started
    latencies.sort()
    auto = [row for row in rows if row["result"]["tier"] == "auto"]
    auto_hits = sum(1 for row in auto
                    if row["result"]["choice"] == row["task"]["expected"])
    decided = []
    for row in rows:
        result = row["result"]
        answer = result.get("choice") if result["tier"] == "auto" \
            else result.get("suggested_choice")
        decided.append(answer == row["task"]["expected"])
    opinions = sorted({row["result"].get("opinions") for row in rows
                       if row["result"].get("opinions") is not None})
    entry = {
        "rows": len(rows),
        "seconds": round(elapsed, 2),
        "decisions_per_s": round(len(rows) / elapsed, 2),
        "p50_ms": round(statistics.median(latencies), 1),
        "opinions_reported": opinions,
        "auto": len(auto),
        "auto_accuracy": round(auto_hits / max(1, len(auto)), 4),
        "decided": round(sum(decided) / len(decided), 4),
    }
    print(f"  {label:16s} {entry['decisions_per_s']:6.2f}/s  "
          f"p50 {entry['p50_ms']:6.1f} ms  opinions {opinions}  "
          f"auto {len(auto):3d} at {entry['auto_accuracy']:.4f}  "
          f"decided {entry['decided']:.4f}", flush=True)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets"
                                    / "heldout2_test.jsonl"))
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--text-model",
                        default=str(ROOT_DIR / "assets" / "textmodel.json"))
    parser.add_argument("--combiner",
                        default=str(ROOT_DIR / "assets" / "combiner.json"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()][:args.rows]
    combiner = Combiner.load(args.combiner) if Path(args.combiner).exists() \
        else None
    coordinator = Coordinator(quorum_k=1, count_lineages=True,
                              call_timeout=300.0, initial_stem_timeout=240.0,
                              timeout_multiplier=6.0, min_stem_timeout=30.0,
                              combiner=combiner)
    root = Root(coordinator=coordinator)
    root.start()

    leaf = Stem("leaf", OnnxLayaBackend("leaf", str(ROOT_DIR / ACCEL),
                                        repo="convaiinnovations/laya",
                                        device="GPU", pad_to=256,
                                        checkpoint="english"))
    small = Stem("small", TextModel.load(args.text_model))
    replica = Stem("replica", OnnxLayaBackend("replica",
                                              str(ROOT_DIR / NPU_GRAPH),
                                              repo="convaiinnovations/laya",
                                              device="NPU", pad_to=256,
                                              checkpoint="english"))
    results = {}
    try:
        leaf.start()
        threading.Thread(target=heartbeat_loop, args=(leaf, root.url()),
                         daemon=True).start()
        time.sleep(2.0)
        results["one leaf"] = stage("one leaf", root, tasks)

        coordinator.quorum_k = 2
        small.start()
        threading.Thread(target=heartbeat_loop, args=(small, root.url()),
                         daemon=True).start()
        time.sleep(2.0)
        results["+ new opinion"] = stage("+ new opinion", root, tasks)

        replica.start()
        threading.Thread(target=heartbeat_loop, args=(replica, root.url()),
                         daemon=True).start()
        time.sleep(2.0)
        results["+ replica"] = stage("+ replica", root, tasks)

        coordinator.retire("replica", "measured retirement")
        time.sleep(2.0)
        results["- replica"] = stage("- replica", root, tasks)
    finally:
        for stem in (leaf, small, replica):
            stem.stop()
        root.stop()

    out = ROOT_DIR / "metrics" / "ensemble-checks" / "elastic-fleet.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
