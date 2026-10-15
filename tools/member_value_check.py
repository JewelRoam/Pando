"""What does a member's accuracy and correlation say about what it is worth?

The project has measured, piece by piece, that a member's value to a swarm is
governed by how independent its errors are and not by how accurate it is: a
98.75%-accurate aligned voter was worth less than a 57.5% decorrelated one
(round 128), four mutually correlated members cost the swarm 11.6 points
(round 128), and a single decorrelated one was worth +3.4 (round 128) and
+1.3 through the live path (round 127). Each of those was one comparison.

This builds a portfolio of cheap members that vary in both quantities at
once - feature view, smoothing, and a different algorithm on the same
features - measures each member's accuracy and its mean error correlation
with the three neural lineages, and measures what each one actually adds to
the swarm. Then it asks whether the two numbers predict the third, which is
the design rule the elastic premise needs: a way to choose a member before
admitting it.

Usage:
    python tools/member_value_check.py --report <ensemble-*.json>
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

from member_portfolio_check import (  # noqa: E402
    TIER_ORDER,
    Centroid,
    fit_temperature,
    phi,
)
from pandosi.calibrate import apply_temperature  # noqa: E402
from pandosi.common import DecisionRequest, Vote, aggregate  # noqa: E402
from pandosi.textmodel import TextModel  # noqa: E402


def judge(rows: list[dict], members: list[str]) -> tuple[float, int, float]:
    """Full-coverage accuracy, auto coverage, and accuracy where auto."""
    hits, auto_hits = [], []
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        hit = result.choice == row["expected"]
        hits.append(hit)
        if result.tier == "auto":
            auto_hits.append(hit)
    return (sum(hits) / len(hits), len(auto_hits),
            sum(auto_hits) / len(auto_hits) if auto_hits else 0.0)


def trio_majority(row: dict, trio: list[str]) -> str | None:
    counts = Counter(row["per_lineage"][name] for name in trio)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def rescue_and_harm(row: dict, member: str, trio: list[str]) -> tuple[bool,
                                                                     bool]:
    """(right where the trio erred, wrong where the trio was right)."""
    majority = trio_majority(row, trio)
    if majority is None:
        return (False, False)
    member_right = row["per_lineage"][member] == row["expected"]
    return (majority != row["expected"] and member_right,
            majority == row["expected"] and not member_right)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train", default=str(ROOT / "assets"
                                              / "big_dev.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "assets"
                                              / "heldout_test.jsonl"))
    parser.add_argument("--holdout", type=int, default=48)
    args = parser.parse_args()

    def load(path):
        return [json.loads(line) for line
                in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    train = load(args.train)
    tasks = {task["task_id"]: task for task in load(args.test)}
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    trio = sorted(report["rows"][0]["per_lineage"])
    fitted, holdout = train[:-args.holdout], train[-args.holdout:]

    variants: dict[str, object] = {}
    for mode in ("words", "chars", "mixed"):
        for alpha in (0.1, 1.0, 3.0):
            model = TextModel(checkpoint=f"text-{mode}-a{alpha}",
                              alpha=alpha, feature_mode=mode)
            model.fit(fitted)
            variants[f"nb-{mode}-a{alpha}"] = model
    centroid = Centroid()
    centroid.fit(fitted)
    variants["centroid"] = centroid
    for model in variants.values():
        model._temperature = fit_temperature(model, holdout)

    rows = []
    for record in report["rows"]:
        task = tasks[record["task_id"]]
        row = {"expected": record["expected"],
               "per_lineage": dict(record["per_lineage"]),
               "confidences": dict(record["confidences"])}
        for name, model in variants.items():
            vote = model.decide(DecisionRequest(state=task["state"],
                                                question=task["question"],
                                                candidates=task["candidates"]))
            row["per_lineage"][name] = vote.choice
            row["confidences"][name] = apply_temperature(vote.probability,
                                                         model._temperature)
        rows.append(row)

    base_accuracy, base_auto, base_auto_accuracy = judge(rows, trio)
    print(f"{len(rows)} held-out rows; the trio scores "
          f"{base_accuracy:.4f} at full coverage, {base_auto} accepted at "
          f"{base_auto_accuracy:.4f}")
    print(f"  {'member':18s} {'accuracy':>9s} {'mean phi':>9s} "
          f"{'swarm':>7s} {'value':>7s} {'auto':>5s} {'rescue':>7s} "
          f"{'harm':>6s}")
    table = {}
    for name in variants:
        hits = sum(1 for row in rows
                   if row["per_lineage"][name] == row["expected"])
        accuracy = hits / len(rows)
        correlations = [phi(rows, lineage, name) for lineage in trio]
        mean_phi = sum(value for value in correlations
                       if value is not None) / len(correlations)
        swarm, auto, auto_accuracy = judge(rows, trio + [name])
        rescue = sum(1 for row in rows
                     if rescue_and_harm(row, name, trio)[0])
        harm = sum(1 for row in rows if rescue_and_harm(row, name, trio)[1])
        table[name] = {"accuracy": round(accuracy, 4),
                       "mean_phi": round(mean_phi, 4),
                       "swarm_accuracy": round(swarm, 4),
                       "value": round(swarm - base_accuracy, 4),
                       "auto": auto,
                       "rescue": rescue, "harm": harm,
                       "net": rescue - harm}
        print(f"  {name:18s} {accuracy:9.4f} {mean_phi:+9.3f} {swarm:7.4f} "
              f"{swarm - base_accuracy:+7.4f} {auto:5d} {rescue:7d} "
              f"{harm:6d}")

    def spearman(quantity: str) -> float:
        """Rank correlation between one column and the measured value."""
        ranked_value = {name: rank for rank, (name, _) in enumerate(
            sorted(table.items(), key=lambda item: item[1]["value"]), start=1)}
        ranked_other = {name: rank for rank, (name, _) in enumerate(
            sorted(table.items(), key=lambda item: item[1][quantity]),
            start=1)}
        n = len(table)
        total = sum((ranked_value[name] - ranked_other[name]) ** 2
                    for name in table)
        return round(1 - 6 * total / (n * (n * n - 1)), 3)

    accuracy_rank = spearman("accuracy")
    phi_rank = spearman("mean_phi")
    net_rank = spearman("net")
    print(f"\n  rank correlation with the measured value: accuracy "
          f"{accuracy_rank:+.3f}, mean phi {phi_rank:+.3f}, "
          f"rescues-minus-harms {net_rank:+.3f}")
    print("  (towards -1 for phi means less correlation is worth more)")

    summary = {"report": str(args.report), "rows": len(rows),
               "baseline": {"accuracy": round(base_accuracy, 4),
                            "auto": base_auto},
               "members": table,
               "rank_correlation": {"accuracy": accuracy_rank,
                                    "mean_phi": phi_rank,
                                    "net_rescues": net_rank}}
    out = ROOT / "metrics" / "ensemble-checks" / "member-value.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
