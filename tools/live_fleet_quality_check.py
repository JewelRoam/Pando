"""The smallest real fleet, deciding a labelled suite end to end.

Everything the last rounds measured was measured offline - recorded votes
replayed through the aggregator. The spec-3 combiner is membership-agnostic,
so a live fleet no longer has to match the trio it was fitted on, and the
obvious question is what a *real* deployment does: two stems, one accelerator
and one 19 KB member, served over HTTP, with the gate and the advisory
combiner both live.

This drives the whole wording-held-out split (set C, 240 rows) through a real
root and reports what the system answered: what it auto-executed, what it
declined, what the combiner suggested where it declined, and how all of that
compares with the offline numbers from rounds 141-142.

Usage:
    python tools/live_fleet_quality_check.py [--limit N]
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
from pando.rootd import Coordinator, Root  # noqa: E402
from pando.stemd import Stem, heartbeat_loop  # noqa: E402
from pando.textmodel import TextModel  # noqa: E402

ARTIFACT = "artifacts/laya-en-static-fixed.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets"
                                    / "heldout2_test.jsonl"))
    parser.add_argument("--combiner",
                        default=str(ROOT_DIR / "assets" / "combiner.json"))
    parser.add_argument("--text-model",
                        default=str(ROOT_DIR / "assets" / "textmodel.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    combiner = Combiner.load(args.combiner) if Path(args.combiner).exists() \
        else None
    root = Root(coordinator=Coordinator(
        quorum_k=2, call_timeout=300.0, initial_stem_timeout=240.0,
        timeout_multiplier=6.0, min_stem_timeout=30.0, combiner=combiner))
    root.start()
    print(f"root up with combiner={bool(combiner)}", flush=True)
    stems = []
    try:
        print("  starting the accelerator stem (english, iGPU) ...", flush=True)
        stems.append(Stem("accel", OnnxLayaBackend(
            "accel", str(ROOT_DIR / ARTIFACT), repo="convaiinnovations/laya",
            device="GPU", pad_to=256, checkpoint="english")))
        stems.append(Stem("small", TextModel.load(args.text_model)))
        for stem in stems:
            stem.start()
            threading.Thread(target=heartbeat_loop, args=(stem, root.url()),
                             daemon=True).start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and \
                len(root.coordinator.live_nodes()) < 2:
            time.sleep(1)
        print(f"  {len(root.coordinator.live_nodes())} stems live: "
              f"{[s.node_id for s in stems]}", flush=True)

        rows = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"])
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{root.url()}/decide", data=request.to_json(),
                        headers={"Content-Type": "application/json"}),
                    timeout=300) as response:
                body = json.loads(response.read())
            rows.append({"task": task, "result": body})
            if index % 40 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        for stem in stems:
            stem.stop()
        root.stop()

    auto = [row for row in rows if row["result"]["tier"] == "auto"]
    declined = [row for row in rows if row["result"]["tier"] != "auto"]
    auto_hits = sum(1 for row in auto
                    if row["result"]["choice"] == row["task"]["expected"])
    suggested = [row for row in declined
                 if row["result"].get("suggested_choice")]
    suggest_hits = sum(1 for row in suggested
                       if row["result"]["suggested_choice"]
                       == row["task"]["expected"])
    decided = auto_hits + suggest_hits
    print(f"\n{len(rows)} rows through a live two-stem fleet")
    print(f"  auto-executed {len(auto)} at "
          f"{auto_hits / max(1, len(auto)):.4f}")
    print(f"  declined      {len(declined)}, suggestions "
          f"{len(suggested)} at {suggest_hits / max(1, len(suggested)):.4f}")
    print(f"  a decided answer in {decided}/{len(rows)} = "
          f"{decided / len(rows):.4f}")
    oracle = sum(1 for row in rows
                 if row["task"]["expected"] in
                 {vote["choice"]
                  for vote in row["result"].get("votes", [])})
    print(f"  oracle (answer in the voting members) "
          f"{oracle / len(rows):.4f}")

    summary = {"rows": len(rows), "auto": len(auto),
               "auto_accuracy": round(auto_hits / max(1, len(auto)), 4),
               "declined": len(declined), "suggestions": len(suggested),
               "suggestion_accuracy": round(
                   suggest_hits / max(1, len(suggested)), 4),
               "decided_accuracy": round(decided / len(rows), 4)}
    out = ROOT_DIR / "metrics" / "ensemble-checks" / "live-fleet-quality.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
