"""Is the quorum invariant under replication?

Round 119 measured what a replica does to a quorum: adding a second english
dropped the weighted vote to exactly english's own accuracy, because a 2:1
majority of (english, english-copy, typed-decisions) *is* english. A swarm
whose whole growth story is "copy the lineage onto more devices" therefore
has to be invariant under that copy, or growth silently rewrites the answer.

This replays a recorded report - real votes, real calibrated confidences, no
models re-run - through the shipped aggregation and through the
`count_lineages` rule, with replicas added to each lineage in turn. What it
measures is the *policy*: the votes are held fixed, so a difference between
the two columns is caused by counting stems instead of lineages.

Usage:
    python tools/lineage_quorum_check.py --report <ensemble-*.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.common import Vote, aggregate  # noqa: E402


def quorum(names: list[str]) -> list[tuple[str, str]]:
    return [(name, name) for name in names]


def judge(rows: list[dict], members: list[tuple[str, str]],
          count_lineages: bool) -> dict:
    lineages = len({checkpoint for _, checkpoint in members})
    floor = min(3, lineages if count_lineages else len(members))
    accepted, correct = 0, 0
    auto_correct = 0
    for row in rows:
        votes = [Vote(node, checkpoint, row["per_lineage"][checkpoint],
                      row["confidences"][checkpoint])
                 for node, checkpoint in members]
        result = aggregate(votes, min_quorum=floor,
                           count_lineages=count_lineages)
        correct += result.choice == row["expected"]
        if result.tier == "auto":
            accepted += 1
            auto_correct += result.choice == row["expected"]
    return {
        "members": len(members),
        "lineages": lineages,
        "min_quorum": floor,
        "full_coverage_accuracy": round(correct / len(rows), 4),
        "accepted": accepted,
        "accepted_coverage": round(accepted / len(rows), 4),
        "accuracy_where_accepted": (round(auto_correct / accepted, 4)
                                    if accepted else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = report["rows"]
    lineages = sorted(rows[0]["per_lineage"])
    scenarios = [("no replica", quorum(lineages))]
    for target in lineages:
        for extra in (1, 2):
            scenarios.append(
                (f"+{extra} {target}", quorum(lineages + [target] * extra)))

    print(f"{len(rows)} recorded rows, lineages {lineages}")
    print(f"  {'scenario':22s} {'stems':>5s} {'lineages':>8s} "
          f"{'shipped full':>12s} {'collapse full':>13s} "
          f"{'shipped auto':>12s} {'collapse auto':>13s}")
    table = {}
    for label, members in scenarios:
        shipped = judge(rows, members, False)
        collapsed = judge(rows, members, True)
        table[label] = {"shipped": shipped, "collapsed": collapsed}
        print(f"  {label:22s} {shipped['members']:5d} "
              f"{shipped['lineages']:8d} "
              f"{shipped['full_coverage_accuracy']:12.3f} "
              f"{collapsed['full_coverage_accuracy']:13.3f} "
              f"{shipped['accepted_coverage']:12.3f} "
              f"{collapsed['accepted_coverage']:13.3f}")

    replicated = [label for label in table if label != "no replica"]
    drift = max(abs(table[label]["shipped"]["full_coverage_accuracy"]
                    - table["no replica"]["shipped"]["full_coverage_accuracy"])
                for label in replicated)
    collapse_drift = max(
        abs(table[label]["collapsed"]["full_coverage_accuracy"]
            - table["no replica"]["collapsed"]["full_coverage_accuracy"])
        for label in replicated)
    print(f"\nreplication moves the shipped answer by up to {drift:.3f} "
          f"and the collapsed answer by {collapse_drift:.3f}")

    out = ROOT / "metrics" / "ensemble-checks" / (
        f"lineage-quorum-{int(time.time())}.json")
    out.write_text(json.dumps({"report": str(args.report),
                               "scenarios": table}, indent=2),
                   encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
