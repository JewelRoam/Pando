"""Is the swarm's edge over its best member real, and what would settle it?

Round 118 put every combiner and every member on one leaderboard: over 240
labeled tasks the best single lineage (typed-decisions, 90.0%) ties the best
combiner (margin, 90.4%) - a one-task gap. Two questions follow.

  - A majority vote of three members each right 76-90% of the time would be
    far better than any of them *if their errors were independent*. Measuring
    the correlation of those errors says whether the premise holds, and how
    wrong the independent-error prediction is.
  - A paired bootstrap says how much of the observed gap could survive
    resampling, and the discordance rate says how many tasks it would take
    to resolve the rest. That number decides the next experiment: more
    tasks, or more diverse members.

Usage:
    python tools/ensemble_significance.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combiner_check import (  # noqa: E402
    MEMBER_PREFIX,
    hits,
    newest_report,
    ranked,
)

Z_ALPHA = 1.959964
Z_POWER = 0.8416


def plain_majority(row: dict) -> str | None:
    top = Counter(row["per_lineage"].values()).most_common()
    if not top:
        return None
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None                       # a tie is not a decision
    return top[0][0]


def errors(rows: list[dict], name: str) -> list[bool]:
    return [row["per_lineage"].get(name) != row["expected"] for row in rows]


def discordance(rows: list[dict], left: str, right: str,
                coverage: float = 1.0) -> float:
    """How often exactly one of the two answers the task correctly."""
    window = rows[:max(1, round(coverage * len(rows)))]
    left_hits = hits(window, left, "")
    right_hits = hits(window, right, "")
    return sum(1 for a, b in zip(left_hits, right_hits) if a != b) / len(window)


def correlation(left: list[bool], right: list[bool]) -> dict:
    both = sum(1 for a, b in zip(left, right) if a and b)
    neither = sum(1 for a, b in zip(left, right) if not a and not b)
    left_only = sum(1 for a, b in zip(left, right) if a and not b)
    right_only = sum(1 for a, b in zip(left, right) if b and not a)
    total = len(left)
    denominator = math.sqrt((neither + left_only) * (right_only + both)
                            * (neither + right_only) * (left_only + both))
    left_rate = (both + left_only) / total
    right_rate = (right_only + both) / total
    expected = left_rate * right_rate * total
    return {
        "disagreement": round((left_only + right_only) / total, 4),
        "phi": round((neither * both - left_only * right_only) / denominator, 4)
        if denominator else None,
        "both_wrong_over_expected": round(both / expected, 3)
        if expected else None,
    }


def independent_majority_error(rates: list[float]) -> float:
    """P(a majority of independent members is wrong) given their error rates."""
    total = 0.0
    for mask in range(1 << len(rates)):
        wrong = [bool(mask >> index & 1) for index in range(len(rates))]
        if sum(wrong) * 2 <= len(rates):
            continue
        probability = 1.0
        for rate, missed in zip(rates, wrong):
            probability *= rate if missed else 1.0 - rate
        total += probability
    return total


def bootstrap(rows: list[dict], left: str, right: str, coverage: float,
              trials: int, seed: int) -> dict:
    """Paired resample of the accuracy difference, re-ranking each draw."""
    rng = random.Random(seed)
    total = len(rows)
    count = max(1, round(coverage * total))
    differences = []
    for _ in range(trials):
        sample = [rows[rng.randrange(total)] for _ in range(total)]
        window = ranked(sample, left, "")[:count]
        differences.append((sum(hits(window, left, ""))
                            - sum(hits(window, right, ""))) / count)
    differences.sort()
    share = sum(1 for value in differences if value > 0) / trials
    return {
        "tasks": count,
        "mean": round(sum(differences) / trials, 4),
        "low": round(differences[int(0.025 * trials)], 4),
        "high": round(differences[int(0.975 * trials)], 4),
        "share_positive": round(share, 4),
    }


def required_tasks(discordance: float, effect: float) -> int | None:
    if effect <= 0 or discordance <= 0:
        return None
    return math.ceil((Z_ALPHA + Z_POWER) ** 2 * discordance / effect ** 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", default=None,
                        help="ensemble-*.json to analyse; several may be "
                             "pooled with commas (default: the newest)")
    parser.add_argument("--combiner", default=None,
                        help="override the combiner under test")
    parser.add_argument("--member", default=None,
                        help="override the member it must beat")
    args = parser.parse_args()

    paths = ([newest_report()] if not args.report else
             [Path(name.strip()) for name in args.report.split(",")
              if name.strip()])
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in paths]
    rows = [row for report in reports for row in report["rows"]]
    report = reports[-1]
    label = " + ".join(path.name for path in paths)
    names = sorted(rows[0]["per_lineage"])
    incumbent_coverage = (sum(1 for row in rows if row["tier"] == "auto")
                          / len(rows))
    print(f"{len(rows)} recorded decisions from {label}; members {names}")

    majority = [plain_majority(row) == row["expected"] for row in rows]
    print(f"plain majority vote, full coverage: "
          f"{round(sum(majority) / len(rows), 4)}")

    print("pairwise error structure (phi > 0 means errors come together):")
    per_pair = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            entry = correlation(errors(rows, left), errors(rows, right))
            per_pair[f"{left}|{right}"] = entry
            print(f"  {left:14s} x {right:14s} "
                  f"disagreement {entry['disagreement']:.3f}  "
                  f"both-wrong x{entry['both_wrong_over_expected']:.2f} "
                  f"of independence  phi {entry['phi']:+.3f}")

    rates = [sum(errors(rows, name)) / len(rows) for name in names]
    predicted = independent_majority_error(rates)
    observed = 1 - sum(majority) / len(rows)
    print(f"independent errors at these rates would leave a majority wrong "
          f"{predicted:.3f} of the time; observed {observed:.3f} "
          f"({observed / predicted:.1f}x)")

    combiner, member = args.combiner, args.member
    if not combiner or not member:
        board_path = paths[-1].with_name(
            paths[-1].name.replace("ensemble-", "combiners-"))
        if not board_path.exists():
            raise SystemExit("run tools/combiner_check.py first, or pass "
                             "--combiner and --member")
        board = json.loads(board_path.read_text(encoding="utf-8"))
        combiner = combiner or board["best_combiner"]
        member = member or board["best_member"]
    print(f"paired bootstrap, {combiner} vs {member} "
          f"({args.trials} resamples):")
    intervals = {}
    for label, coverage in (("full", 1.0),
                            ("incumbent", incumbent_coverage)):
        entry = bootstrap(rows, combiner, member, coverage, args.trials,
                          args.seed)
        intervals[label] = entry
        print(f"  {label:10s} coverage {entry['tasks']:3d}  "
              f"difference {entry['mean']:+.4f}  "
              f"95% CI [{entry['low']:+.4f}, {entry['high']:+.4f}]  "
              f"positive in {entry['share_positive']:.1%} of draws")

    gap = discordance(rows, combiner, member)
    print(f"{combiner} and {member} answer differently on {gap:.1%} of "
          f"tasks; tasks needed for 80% power at alpha=0.05:")
    print("            " + "".join(f"{rate:>10.0%}" for rate in
                                   (0.05, 0.10, 0.20, 0.30)) + "   discordance")
    for effect in (0.01, 0.02, 0.05):
        cells = "".join(f"{required_tasks(rate, effect):>10d}" for rate in
                        (0.05, 0.10, 0.20, 0.30))
        print(f"  {effect:>5.0%} gap {cells}")

    summary = {
        "report": [str(path) for path in paths],
        "combiner": combiner,
        "member": member,
        "tasks": len(rows),
        "plain_majority_full_coverage": round(sum(majority) / len(rows), 4),
        "pairwise": per_pair,
        "independent_majority_error": round(predicted, 4),
        "observed_majority_error": round(observed, 4),
        "bootstrap": intervals,
        "discordance": round(gap, 4),
        "discordance_at_incumbent": round(
            discordance(rows, combiner, member, incumbent_coverage), 4),
    }
    out = paths[-1].with_name(
        f"significance-pooled-{len(paths)}.json" if len(paths) > 1 else
        paths[-1].name.replace("ensemble-", "significance-"))
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
