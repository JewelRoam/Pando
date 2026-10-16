"""The admission rule, applied to real candidates.

Three rounds built it up. Rounds 122-137: a member earns a seat on
*independence* - a near-copy of a seated lineage is worth nothing, however
accurate. Round 150: on *capability* - the candidate has to be better on a
split nothing was trained on. Round 152: on *replication* - five runs of two
recipes spread across 5 to 13 points, so a recipe is judged by its samples'
mean, not by its best run.

This applies the rule to what actually exists: a seated fleet (the three Laya
lineages) and five fine-tuned candidates, all voted on the fourth wording
split, none of which any of them trained on. Every candidate is measured for
capability, independence and what the fleet's *system* accuracy becomes with
it seated.

Usage:
    python tools/admit_member_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from improved_fleet_check import decide  # noqa: E402
from pando.combiner import Combiner  # noqa: E402

# Two standard errors of a proportion at n=240 around p=0.9, the smallest
# difference this evaluation can resolve on its own.
EVALUATION_NOISE = 0.04
# A candidate whose errors correlate with a seated member this strongly is a
# near-copy (round 119's descendant measured +1.000; round 150's member +0.28).
REDUNDANCY_LIMIT = 0.6


def phi(rows: list[dict], left: str, right: str) -> float | None:
    left_errors = [row["choices"][left] != row["expected"] for row in rows]
    right_errors = [row["choices"][right] != row["expected"] for row in rows]
    both = sum(1 for a, b in zip(left_errors, right_errors) if a and b)
    neither = sum(1 for a, b in zip(left_errors, right_errors)
                  if not a and not b)
    left_only = sum(1 for a, b in zip(left_errors, right_errors) if a and not b)
    right_only = sum(1 for a, b in zip(left_errors, right_errors)
                     if b and not a)
    denominator = math.sqrt((neither + left_only) * (right_only + both)
                            * (neither + right_only) * (left_only + both))
    if not denominator:
        return None
    return round((neither * both - left_only * right_only) / denominator, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT / "assets"
                                               / "heldout3_test.jsonl"))
    parser.add_argument("--votes-dir", default=str(ROOT / "metrics"
                                                  / "ensemble-checks"))
    parser.add_argument("--combiner", default=str(ROOT / "assets"
                                                  / "combiner.json"))
    args = parser.parse_args()

    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    directory = Path(args.votes_dir)
    seated = ["english", "multilingual", "typed-decisions"]
    candidates = ["ft-v3", "ft-v3-s1", "ft-v3-s2", "ft-v4", "ft-v4-s1"]
    recipe = {"ft-v3": "720", "ft-v3-s1": "720", "ft-v3-s2": "720",
              "ft-v4": "960", "ft-v4-s1": "960"}

    def load(name: str) -> dict:
        payload = json.loads((directory / f"votes-d-{name}.json")
                             .read_text(encoding="utf-8"))
        return {row["task_id"]: row for row in payload["rows"]}

    seated_votes = {name: load("seated-" + ("typed" if name ==
                                            "typed-decisions" else name))
                    for name in seated}
    task_ids = sorted(seated_votes[seated[0]])
    rows = []
    for task_id in task_ids:
        rows.append({
            "task_id": task_id,
            "expected": seated_votes[seated[0]][task_id]["expected"],
            "candidates": tasks[task_id]["candidates"],
            "state": tasks[task_id]["state"],
            "question": tasks[task_id]["question"],
            "choices": {name: seated_votes[name][task_id]["choice"]
                        for name in seated},
            "confidences": {name: seated_votes[name][task_id]["probability"]
                            for name in seated},
        })
    combiner = Combiner.load(args.combiner)

    def accuracy(subset: list[str]) -> float:
        return round(sum(1 for row in rows
                         if row["choices"][subset[0]] == row["expected"])
                     / len(rows), 4)

    seated_accuracy = {name: round(sum(
        1 for row in rows if row["choices"][name] == row["expected"])
        / len(rows), 4) for name in seated}
    best_seated = max(seated_accuracy.values())
    baseline = decide(rows, seated, combiner)
    print(f"{len(rows)} rows on the fourth split, seen by nothing below")
    print(f"  seated: " + ", ".join(f"{name} {value:.4f}"
                                     for name, value in seated_accuracy.items()))
    print(f"  seated fleet (gate + combiner): gate "
          f"{baseline['full']:.4f}, decided {baseline['decided']:.4f}")
    print(f"  admission floors: capability > best seated + "
          f"{EVALUATION_NOISE:.0%}, mean phi < {REDUNDANCY_LIMIT}, "
          f"fleet decided must not fall")

    verdicts = {}
    print(f"\n  {'candidate':10s} {'recipe':>7s} {'accuracy':>9s} "
          f"{'mean phi':>9s} {'fleet decided':>14s} {'verdict':>10s}")
    for name in candidates:
        votes = load(name)
        for row in rows:
            row["choices"][name] = votes[row["task_id"]]["choice"]
            row["confidences"][name] = votes[row["task_id"]]["probability"]
    by_recipe: dict[str, list[str]] = {}
    for name in candidates:
        by_recipe.setdefault(recipe[name], []).append(name)
    for corpus, members in sorted(by_recipe.items()):
        values = [accuracy([name]) for name in members]
        print(f"  recipe {corpus} rows: {len(values)} sample(s) "
              f"{sorted(values)} mean {sum(values) / len(values):.4f}")

    for name in candidates:
        accuracy_here = accuracy([name])
        correlations = [phi(rows, seated_name, name)
                        for seated_name in seated]
        mean_phi = sum(value for value in correlations
                       if value is not None) / len(correlations)
        with_new = decide(rows, seated + [name], combiner)
        capable = accuracy_here > best_seated + EVALUATION_NOISE
        independent = mean_phi < REDUNDANCY_LIMIT
        helps = with_new["decided"] >= baseline["decided"]
        verdict = ("admit" if capable and independent and helps
                   else "refuse")
        verdicts[name] = {"accuracy": accuracy_here,
                          "mean_phi": round(mean_phi, 4),
                          "fleet_decided": with_new["decided"],
                          "capable": capable, "independent": independent,
                          "helps": helps, "verdict": verdict,
                          "recipe": recipe[name]}
        print(f"  {name:10s} {recipe[name]:>7s} {accuracy_here:>9.4f} "
              f"{mean_phi:>+9.3f} {with_new['decided']:>14.4f} "
              f"{verdict:>10s}   "
              f"(capable {capable}, independent {independent}, "
              f"helps {helps})")

    admitted = [name for name, entry in verdicts.items()
                if entry["verdict"] == "admit"]
    print(f"\n  admitted by the rule: {admitted or 'none'}")
    print(f"  the rule's point: a recipe is judged by its samples' mean, so "
          f"the best of five runs ({max(verdicts, key=lambda k: verdicts[k]['accuracy'])}) "
          f"is not admitted on the strength of being the best draw")
    out = ROOT / "metrics" / "ensemble-checks" / "admission.json"
    out.write_text(json.dumps({"seated": seated_accuracy,
                               "baseline_decided": baseline["decided"],
                               "floors": {"capability": EVALUATION_NOISE,
                                          "redundancy": REDUNDANCY_LIMIT},
                               "candidates": verdicts}, indent=2),
                   encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
