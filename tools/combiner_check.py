"""Which way of combining three lineages actually earns the oracle?

Round 117 measured that the majority vote adds nothing over a single lineage
(both 51.85%), that the swarm's gate is beaten at matched coverage by one
lineage's own confidence (92.31% vs 100% on the same 13 tasks), and that the
correct answer is available somewhere in the trio 74.07% of the time.

That gap is the combiner's, not the members'. This scores candidate
combiners on the same recorded decisions - no models are re-run:

  gate          the shipped tier + mean-probability ordering
  best_lineage  one lineage's own calibrated confidence (the incumbent)
  max_conf      the most confident member's vote, ranked by its confidence
  min_conf      the least confident member as a veto, ranked by it
  margin        weighted vote, ranked by (top - runner-up) confidence mass
  agreement     unanimity first, then mean probability

Usage:
    python tools/combiner_check.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def newest_report() -> Path:
    reports = sorted(glob.glob(str(ROOT / "metrics" / "ensemble-checks"
                                   / "ensemble-*.json")))
    if not reports:
        raise SystemExit("run tools/ensemble_value_check.py first")
    return Path(reports[-1])


def vote_mass(row: dict) -> dict:
    mass: dict[str, float] = {}
    for name, choice in row["per_lineage"].items():
        mass[choice] = mass.get(choice, 0.0) + row["confidences"].get(name, 0.0)
    return mass


def margin(row: dict) -> float:
    mass = sorted(vote_mass(row).values(), reverse=True)
    if len(mass) < 2:
        return mass[0] if mass else 0.0
    return mass[0] - mass[1]


def weighted_choice(row: dict) -> str | None:
    mass = vote_mass(row)
    if not mass:
        return None
    ranked = sorted(mass.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None                       # a tie is not a decision
    return ranked[0][0]


def ranking(row: dict, name: str, lineage: str) -> float:
    confidences = row["confidences"]
    if name == "gate":
        order = {"auto": 0, "verify": 1, "escalate": 2}
        return -order[row["tier"]] * 1000.0 + row["mean_probability"]
    if name == "best_lineage":
        return confidences.get(lineage, 0.0)
    if name == "max_conf":
        return max(confidences.values(), default=0.0)
    if name == "min_conf":
        return min(confidences.values(), default=0.0)
    if name == "margin":
        return margin(row)
    if name == "agreement":
        return (1000.0 if row["distinct"] == 1 else 0.0) \
            + row["mean_probability"]
    raise ValueError(name)


def choice_for(row: dict, name: str, lineage: str) -> str | None:
    if name == "best_lineage":
        return row["per_lineage"].get(lineage)
    if name == "gate":
        return row.get("swarm_choice")
    if name in ("max_conf", "min_conf"):
        pick = max if name == "max_conf" else min
        member = pick(row["confidences"],
                      key=lambda key: row["confidences"][key])
        return row["per_lineage"][member]
    return weighted_choice(row)


def curve(rows: list[dict], name: str,
          lineage: str) -> list[tuple[float, float]]:
    ranked = sorted(rows, key=lambda row: -ranking(row, name, lineage))
    points = []
    for count in range(1, len(ranked) + 1):
        window = ranked[:count]
        hits = sum(1 for row in window
                   if choice_for(row, name, lineage) == row["expected"])
        points.append((round(count / len(ranked), 4), round(hits / count, 4)))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", default="multilingual",
                        help="the incumbent single model to beat")
    parser.add_argument("--target", type=float, default=0.9,
                        help="accuracy the curve must hold")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report_path = newest_report()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    total = len(rows)
    incumbent_point = report["swarm_auto_count"]
    print(f"{total} recorded decisions from {report_path.name}; "
          f"oracle {report['oracle_accuracy']}; incumbent coverage point "
          f"{incumbent_point}/{total}")

    names = ["best_lineage", "gate", "max_conf", "min_conf", "margin",
             "agreement"]
    table = {}
    for name in names:
        points = curve(rows, name, args.lineage)
        holds = [coverage for coverage, acc in points if acc >= args.target]
        table[name] = {
            "accuracy_at_incumbent_coverage": points[incumbent_point - 1][1],
            "accuracy_full": points[-1][1],
            "coverage_holding_target": max(holds) if holds else None,
            "curve": points,
        }

    ranked = sorted(table.items(),
                    key=lambda item: (-(item[1]["coverage_holding_target"] or 0),
                                      -item[1]["accuracy_at_incumbent_coverage"]))
    for name, _ in ranked:
        entry = table[name]
        print(f"  {name:14s} @{incumbent_point}={entry['accuracy_at_incumbent_coverage']:.3f}"
              f"  full={entry['accuracy_full']:.3f}"
              f"  coverage@>={args.target:.0%}="
              f"{entry['coverage_holding_target']}")

    best_name = ranked[0][0]
    summary = {
        "report": str(report_path),
        "lineage": args.lineage,
        "target": args.target,
        "incumbent_coverage": incumbent_point / total,
        "incumbent_holds": table["best_lineage"]["coverage_holding_target"],
        "best_combiner": best_name,
        "best_holds": table[best_name]["coverage_holding_target"],
        "table": {name: {k: v for k, v in entry.items() if k != "curve"}
                  for name, entry in table.items()},
    }
    print(f"\nincumbent ({args.lineage}) holds {args.target:.0%} to coverage "
          f"{summary['incumbent_holds']}")
    print(f"best combiner {best_name} holds it to "
          f"{summary['best_holds']}")
    if args.json:
        print(json.dumps(summary, indent=2))
    out = ROOT / "metrics" / "ensemble-checks" / (
        f"combiners-{report_path.stem.split('-')[-1]}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
