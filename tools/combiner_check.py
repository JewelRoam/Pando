"""Which way of combining three lineages actually earns the oracle?

Round 117 measured that the majority vote adds nothing over a single lineage
(both 51.85%), that the swarm's gate is beaten at matched coverage by one
lineage's own confidence (92.31% vs 100% on the same 13 tasks), and that the
correct answer is available somewhere in the trio 74.07% of the time.

That gap is the combiner's, not the members'. This scores candidate
combiners - and every member on its own - on the same recorded decisions.
No model is re-run:

  member:<name> one lineage's own calibrated confidence (the leaderboard
                needs these, otherwise "best_lineage" silently means "the
                incumbent", which in round 117 was the *second* best member)
  gate          the shipped tier + mean-probability ordering
  max_conf      the most confident member's vote, ranked by its confidence
  min_conf      the least confident member as a veto, ranked by it
  margin        weighted vote, ranked by (top - runner-up) confidence mass
  agreement     unanimity first, then mean probability

Two comparisons come out of it. The *matched-coverage* one asks what
accuracy each ranking reaches when it is allowed to answer the same
number of tasks. The *matched-set* one asks, on the very tasks the best
combiner chooses to answer, whether it beats the best member answering
those same tasks - a paired test, so it needs far less data.

Usage:
    python tools/combiner_check.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMBINERS = ["gate", "max_conf", "min_conf", "margin", "agreement"]
MEMBER_PREFIX = "member:"


def newest_report(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    reports = sorted(glob.glob(str(ROOT / "metrics" / "ensemble-checks"
                                   / "ensemble-*.json")))
    if not reports:
        raise SystemExit("run tools/ensemble_value_check.py first")
    return Path(reports[-1])


def member_of(name: str) -> str | None:
    if not name.startswith(MEMBER_PREFIX):
        return None
    return name[len(MEMBER_PREFIX):]


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
    member = member_of(name)
    if member is not None:
        return confidences.get(member, 0.0)
    if name == "gate":
        order = {"auto": 0, "verify": 1, "escalate": 2}
        return -order[row["tier"]] * 1000.0 + row["mean_probability"]
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
    member = member_of(name)
    if member is not None:
        return row["per_lineage"].get(member)
    if name == "gate":
        return row.get("swarm_choice")
    if name in ("max_conf", "min_conf"):
        pick = max if name == "max_conf" else min
        member = pick(row["confidences"],
                      key=lambda key: row["confidences"][key])
        return row["per_lineage"][member]
    return weighted_choice(row)


def ranked(rows: list[dict], name: str, lineage: str) -> list[dict]:
    return sorted(rows, key=lambda row: -ranking(row, name, lineage))


def hits(window: list[dict], name: str, lineage: str) -> list[bool]:
    return [choice_for(row, name, lineage) == row["expected"]
            for row in window]


def curve(rows: list[dict], name: str,
          lineage: str) -> list[tuple[float, float]]:
    order = ranked(rows, name, lineage)
    points, running = [], 0
    for count, row in enumerate(order, start=1):
        running += choice_for(row, name, lineage) == row["expected"]
        points.append((round(count / len(order), 4), round(running / count, 4)))
    return points


def mcnemar(a_hits: list[bool], b_hits: list[bool]) -> dict:
    """Exact two-sided paired test on the tasks the two disagree about."""
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    trials = a_only + b_only
    if trials == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    tail = sum(math.comb(trials, k) for k in
               range(min(a_only, b_only) + 1))
    return {"a_only": a_only, "b_only": b_only,
            "p_value": round(min(1.0, 2 * tail / 2 ** trials), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", default="multilingual",
                        help="the incumbent single model to beat")
    parser.add_argument("--target", type=float, default=0.9,
                        help="accuracy the curve must hold")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", default=None,
                        help="an ensemble-*.json to analyse "
                             "(default: the newest)")
    args = parser.parse_args()

    report_path = newest_report(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    total = len(rows)
    incumbent_point = report["swarm_auto_count"]
    incumbent = MEMBER_PREFIX + args.lineage
    members = [MEMBER_PREFIX + name for name in rows[0]["per_lineage"]]
    has_incumbent = incumbent in members
    print(f"{total} recorded decisions from {report_path.name}; "
          f"oracle {report['oracle_accuracy']}; incumbent coverage point "
          f"{incumbent_point}/{total}")
    if not has_incumbent:
        print(f"  (no {args.lineage} in this trio: {', '.join(members)})")

    table = {}
    for name in members + COMBINERS:
        points = curve(rows, name, args.lineage)
        holds = [coverage for coverage, acc in points if acc >= args.target]
        table[name] = {
            "accuracy_at_incumbent_coverage": points[incumbent_point - 1][1],
            "accuracy_full": points[-1][1],
            "coverage_holding_target": max(holds) if holds else None,
            "curve": points,
        }

    board = sorted(table.items(),
                   key=lambda item: (-(item[1]["coverage_holding_target"] or 0),
                                     -item[1]["accuracy_at_incumbent_coverage"]))
    for name, _ in board:
        entry = table[name]
        print(f"  {name:14s} @{incumbent_point}={entry['accuracy_at_incumbent_coverage']:.3f}"
              f"  full={entry['accuracy_full']:.3f}"
              f"  coverage@>={args.target:.0%}="
              f"{entry['coverage_holding_target']}")

    best_name = board[0][0]
    best_member = next(name for name, _ in board if member_of(name))
    best_combiner = next(name for name, _ in board if not member_of(name))

    # Matched-set: on the tasks the best combiner accepts, does it beat the
    # best member answering those same tasks? This is paired, so it is the
    # only comparison here with enough power at a few hundred tasks.
    matched = {}
    for point_label, count in (("incumbent", incumbent_point),
                               ("full", total)):
        window = ranked(rows, best_combiner, args.lineage)[:count]
        a_hits = hits(window, best_combiner, args.lineage)
        b_hits = hits(window, best_member, args.lineage)
        matched[point_label] = {
            "tasks": count,
            "combiner": round(sum(a_hits) / count, 4),
            "best_member": round(sum(b_hits) / count, 4),
            "mcnemar": mcnemar(a_hits, b_hits),
        }
        entry = matched[point_label]
        print(f"  matched-set @{point_label}: {best_combiner} "
              f"{entry['combiner']:.3f} vs {best_member} "
              f"{entry['best_member']:.3f} "
              f"(a-only {entry['mcnemar']['a_only']}, "
              f"b-only {entry['mcnemar']['b_only']}, "
              f"p={entry['mcnemar']['p_value']})")

    summary = {
        "report": str(report_path),
        "lineage": args.lineage,
        "target": args.target,
        "tasks": total,
        "incumbent_coverage": incumbent_point / total,
        "incumbent_holds": (table[incumbent]["coverage_holding_target"]
                            if has_incumbent else None),
        "leaderboard_winner": best_name,
        "best_combiner": best_combiner,
        "best_member": best_member,
        "best_holds": table[best_name]["coverage_holding_target"],
        "matched_set": matched,
        "table": {name: {k: v for k, v in entry.items() if k != "curve"}
                  for name, entry in table.items()},
    }
    if has_incumbent:
        print(f"\nincumbent ({args.lineage}) holds {args.target:.0%} to "
              f"coverage {summary['incumbent_holds']}")
    print(f"best member {best_member} holds it to "
          f"{table[best_member]['coverage_holding_target']}")
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
