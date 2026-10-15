"""Should a tied quorum answer, or stay silent?

Round 126 added the cheap independent member to the live swarm and it made
the swarm slightly *worse*: accepted coverage fell from 172 rows to 154 and
matched-coverage accuracy from 90.12% to 89.53%, against an offline
prediction of +3.8 points. The prediction used the confidence-weighted vote;
the shipped gate uses a plain count and refuses to decide a tie.

Three members cannot tie. Four can: a 2-2 split has no majority, the gate
returns no choice, and the row counts as wrong at full coverage - so adding
a weak-but-independent member converts some disagreements into non-decisions,
and the information the member brought is discarded rather than weighed.

This replays the recorded live votes through the shipped gate to confirm it
reproduces the live numbers, then through one variant: a tie is broken by
confidence mass and left at the `verify` tier, so the quorum still refuses to
act alone but stops withholding an answer it can rank.

Usage:
    python tools/tie_break_check.py --report <ensemble-*.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.calibrate import Calibrator  # noqa: E402
from pandosi.common import DecisionRequest, Vote, aggregate  # noqa: E402
from pandosi.textmodel import TextModel  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def votes_for(record: dict, text_choice: str, text_probability: float,
              with_text: bool) -> list[Vote]:
    votes = [Vote(name, name, choice, record["confidences"][name])
             for name, choice in record["per_lineage"].items()]
    if with_text:
        votes.append(Vote("text", "text-nb-v1", text_choice, text_probability))
    return votes


def decide(votes: list[Vote], tie_break: bool) -> tuple[str | None, str]:
    """The shipped aggregation, or the rule it replaced in round 127."""
    result = aggregate(votes, min_quorum=min(3, len(votes)))
    if tie_break:
        return result.choice, result.tier
    counts = Counter(vote.choice for vote in votes)
    top = counts.most_common()
    if len(top) < 2 or top[0][1] != top[1][1]:
        return result.choice, result.tier
    return None, "verify"      # the pre-round-127 rule: a tie was silence


def score(rows: list[dict], tie_break: bool, coverage: int | None) -> dict:
    answered, correct, auto = [], 0, 0
    ranked = []
    for row in rows:
        choice, tier = decide(row["votes"], tie_break)
        hit = choice == row["expected"]
        correct += hit
        auto += tier == "auto"
        if choice is not None:
            answered.append(hit)
        ranked.append((TIER_ORDER[tier], -row["mean_probability"], hit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    result = {
        "full_coverage_accuracy": round(correct / len(rows), 4),
        "answered": len(answered),
        "accuracy_where_answered": (round(sum(answered) / len(answered), 4)
                                    if answered else None),
        "auto": auto,
        "accuracy_where_auto": None,
    }
    auto_hits = [hit for tier, _, hit in ranked
                 if tier == TIER_ORDER["auto"]]
    if auto_hits:
        result["accuracy_where_auto"] = round(sum(auto_hits)
                                              / len(auto_hits), 4)
    if coverage:
        window = ranked[:coverage]
        result[f"accuracy_at_{coverage}"] = round(
            sum(hit for _, _, hit in window) / len(window), 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"))
    parser.add_argument("--text-model",
                        default=str(ROOT / "assets" / "textmodel.json"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    model = TextModel.load(args.text_model)
    calibrator = Calibrator()

    rows = []
    for record in report["rows"]:
        task = tasks[record["task_id"]]
        vote = model.decide(DecisionRequest(state=task["state"],
                                            question=task["question"],
                                            candidates=task["candidates"]))
        probability, _ = calibrator.calibrate(vote.checkpoint,
                                              vote.probability)
        rows.append({
            "expected": record["expected"],
            "mean_probability": record["mean_probability"],
            "votes": votes_for(record, vote.choice, probability, True),
            "votes_without": votes_for(record, vote.choice, probability,
                                       False),
        })

    without = [dict(row, votes=row["votes_without"]) for row in rows]
    neural = score(without, False, None)
    entries = (
        ("trio, before", score(without, False, None)),
        ("trio, shipped", score(without, True, None)),
        ("+text, before", score(rows, False, neural["auto"])),
        ("+text, shipped", score(rows, True, neural["auto"])),
    )
    print(f"{len(rows)} held-out rows; the neural trio auto-accepted "
          f"{neural['auto']}")
    for label, entry in entries:
        print(f"  {label:16s} full {entry['full_coverage_accuracy']:.4f}  "
              f"auto {entry['auto']} at {entry['accuracy_where_auto']}  "
              f"answered {entry['answered']}")

    summary = {"report": str(args.report), "rows": len(rows),
               **{label: entry for label, entry in entries}}
    out = ROOT / "metrics" / "ensemble-checks" / "tie-break.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
