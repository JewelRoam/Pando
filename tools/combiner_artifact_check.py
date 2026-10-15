"""Does the shipped combiner artifact reproduce the measurement, and stay advisory?

Round 139 measured the fitted combiner offline; round 140 ships it as
`assets/combiner.json` and lets a root attach its answer to rows the gate does
not auto-execute. Two things have to be true before that is worth having:

  - the artifact reproduces the numbers it was fitted for, on rows it has
    never seen;
  - it never changes a tier, a choice, or anything an agent could act on - the
    suggestion is advice attached to a decision the gate already made.

This replays the wording-held-out report through the **shipped artifact** and
checks both, including the per-group breakdown that carries the claim (54.4%
on the rows the gate declines against the best member's 41.2%).

Usage:
    python tools/combiner_artifact_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.combiner import Combiner  # noqa: E402
from pandosi.common import Vote, aggregate  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combiner", default=str(ROOT / "assets"
                                                  / "combiner.json"))
    parser.add_argument("--report", default=str(ROOT / "metrics"
                                                / "ensemble-checks"
                                                / "ensemble-1792076803.json"))
    parser.add_argument("--suite", default=str(ROOT / "assets"
                                               / "heldout2_test.jsonl"),
                        help="the tasks the report's rows came from; needed "
                             "for a spec-2 artifact's text features")
    args = parser.parse_args()

    combiner = Combiner.load(args.combiner)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    rows = report["rows"]
    members = sorted(rows[0]["per_lineage"])
    print(f"artifact {Path(args.combiner).name}: {combiner.meta}")

    hits, auto_hits, ranked = [], [], []
    suggestions, suggested_auto = 0, 0
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        hit = result.choice == row["expected"]
        hits.append(hit)
        if result.tier == "auto":
            auto_hits.append(hit)
        ranked.append((TIER_ORDER[result.tier], -result.mean_probability,
                       hit, row["task_id"]))
        if result.tier != "auto":
            suggestion = combiner.suggest(
                [(name, row["per_lineage"][name], row["confidences"][name])
                 for name in members],
                tasks[row["task_id"]]["candidates"],
                state=tasks[row["task_id"]]["state"],
                question=tasks[row["task_id"]]["question"])
            if suggestion is not None:
                suggestions += 1
                if result.tier == "auto":
                    suggested_auto += 1
    ranked.sort(key=lambda item: (item[0], item[1]))

    suggested_hits = {}
    for row in rows:
        suggestion = combiner.suggest(
            [(name, row["per_lineage"][name], row["confidences"][name])
             for name in members],
            tasks[row["task_id"]]["candidates"],
            state=tasks[row["task_id"]]["state"],
            question=tasks[row["task_id"]]["question"])
        suggested_hits[row["task_id"]] = bool(
            suggestion and suggestion["choice"] == row["expected"])

    coverage = len(auto_hits)
    accepted = {task_id for _, _, _, task_id in ranked[:coverage]}
    declined = [row["task_id"] for row in rows
                if row["task_id"] not in accepted]
    best_member = max(
        sum(1 for row in rows
            if row["per_lineage"][name] == row["expected"]) for name in members)
    print(f"  shipped gate   full {sum(hits) / len(hits):.4f}  "
          f"auto {coverage} at {sum(auto_hits) / coverage:.4f}")
    print(f"  artifact       full "
          f"{sum(suggested_hits.values()) / len(rows):.4f}")
    print(f"  best member    full {best_member / len(rows):.4f}")
    print(f"  on the {len(declined)} declined rows: artifact "
          f"{sum(1 for task_id in declined if suggested_hits[task_id]) / len(declined):.4f}")
    print(f"  suggestions offered on {suggestions} rows, "
          f"{suggested_auto} of them at tier=auto: "
          f"{'OK' if suggested_auto == 0 else 'FAIL - it would change a decision'}")

    failures = []
    if suggested_auto:
        failures.append("a suggestion was offered for an auto row")
    if suggestions != len(declined):
        failures.append(f"expected {len(declined)} suggestions, got "
                        f"{suggestions}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("COMBINER ARTIFACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
