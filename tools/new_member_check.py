"""What does a strong, newly-trained member do to the swarm?

Round 150 fine-tuned the `english` lineage on 720 labelled rows with the
encoder unfrozen and produced a member that scores 0.9125 on a wording split
it never saw - against 0.7333 for its own parent, 0.7375 for the best existing
lineage, and an oracle of 0.8583 for the trio. It also disagrees with its
parent on 62 of 240 decisions, which no previous descendant did.

That is the first candidate for the one thing rounds 122-137 said the swarm
needed and could not get: a member that is both *better* and *different*.
This asks the two questions that decide whether it earns a seat - is it
decorrelated from the incumbent members, and does the swarm improve when it
joins - by replaying the recorded trio beside its recorded votes.

Usage:
    python tools/new_member_check.py
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

from pando.common import Vote, aggregate  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}
NAME = "ft-v3"


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


def gate(rows: list[dict], members: list[str]) -> dict:
    auto, auto_hits, hits, ranked = 0, 0, 0, []
    for row in rows:
        votes = [Vote(name, name, row["choices"][name],
                      row["confidences"].get(name, 0.5)) for name in members
                 if row["choices"].get(name)]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        hit = result.choice == row["expected"]
        hits += hit
        if result.tier == "auto":
            auto += 1
            auto_hits += hit
        ranked.append((TIER_ORDER[result.tier], -result.mean_probability, hit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return {"full": round(hits / len(rows), 4), "auto": auto,
            "auto_accuracy": round(auto_hits / max(1, auto), 4),
            "ranked": ranked}


def majority(rows: list[dict], members: list[str]) -> float:
    hits = 0
    for row in rows:
        counts = Counter(row["choices"][name] for name in members
                         if row["choices"].get(name))
        top = counts.most_common()
        if not top:
            continue
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue
        hits += top[0][0] == row["expected"]
    return round(hits / len(rows), 4)


def at_matched(entry: dict, coverage: int) -> float:
    window = entry["ranked"][:coverage]
    return round(sum(hit for _, _, hit in window) / len(window), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "metrics"
                                                / "ensemble-checks"
                                                / "ensemble-1792076803.json"))
    parser.add_argument("--votes", default=str(ROOT / "metrics"
                                               / "ensemble-checks"
                                               / "finetuned-votes.json"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    new = json.loads(Path(args.votes).read_text(encoding="utf-8"))["rows"]
    new_by_id = {row["task_id"]: row for row in new}
    trio = sorted(report["rows"][0]["per_lineage"])

    rows = []
    for record in report["rows"]:
        vote = new_by_id[record["task_id"]]
        rows.append({
            "task_id": record["task_id"], "expected": record["expected"],
            "choices": dict(record["per_lineage"], **{NAME: vote["choice"]}),
            "confidences": dict(record["confidences"],
                                **{NAME: vote["probability"]}),
        })
    members = trio + [NAME]

    print(f"{len(rows)} rows; the trio's recorded votes plus {NAME}")
    for name in members:
        hits = sum(1 for row in rows if row["choices"][name] == row["expected"])
        print(f"  {name:22s} accuracy {hits / len(rows):.4f}")
    print("  error correlation with the new member:")
    for name in trio:
        print(f"    {name:22s} phi {phi(rows, name, NAME):+.3f}  "
              f"disagreement "
              f"{sum(1 for row in rows if row['choices'][name] != row['choices'][NAME]) / len(rows):.3f}")

    trio_only = gate(rows, trio)
    with_new = gate(rows, members)
    print(f"\n  {'fleet':22s} {'majority':>9s} {'gate full':>10s} "
          f"{'auto':>6s} {'auto acc':>9s}")
    print(f"  {'trio':22s} {majority(rows, trio):>9.4f} "
          f"{trio_only['full']:>10.4f} {trio_only['auto']:>6d} "
          f"{trio_only['auto_accuracy']:>9.4f}")
    print(f"  {'trio + ' + NAME:22s} {majority(rows, members):>9.4f} "
          f"{with_new['full']:>10.4f} {with_new['auto']:>6d} "
          f"{with_new['auto_accuracy']:>9.4f}")
    print(f"  at the trio's coverage ({trio_only['auto']} rows): trio "
          f"{at_matched(trio_only, trio_only['auto']):.4f} -> "
          f"with the new member "
          f"{at_matched(with_new, trio_only['auto']):.4f}")
    oracle = sum(1 for row in rows
                 if row["expected"] in set(row["choices"].values()))
    print(f"  oracle (answer among the four) {oracle / len(rows):.4f}")

    summary = {"members": {name: round(sum(
        1 for row in rows if row["choices"][name] == row["expected"])
        / len(rows), 4) for name in members},
        "phi_vs_new": {name: phi(rows, name, NAME) for name in trio},
        "trio": {"majority": majority(rows, trio), **{k: v for k, v in
                                                      trio_only.items()
                                                      if k != "ranked"}},
        "with_new": {"majority": majority(rows, members),
                     **{k: v for k, v in with_new.items()
                        if k != "ranked"}},
        "matched_at_trio_coverage": {
            "trio": at_matched(trio_only, trio_only["auto"]),
            "with_new": at_matched(with_new, trio_only["auto"])},
        "oracle": round(oracle / len(rows), 4)}
    out = ROOT / "metrics" / "ensemble-checks" / "new-member.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
