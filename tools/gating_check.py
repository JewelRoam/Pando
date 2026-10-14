"""Risk-coverage of the swarm's own gate on the labeled graft suite.

Round 98 left "knows when it is guessing" as "partly". This measures it:
run the three exported lineages (en/ml/td) over the 27 labeled choice tasks,
aggregate them exactly as rootd does, and report how accuracy changes as the
gate is allowed to cover less of the suite.

Answers the operational question: if the autonomy loop acts only when the
swarm says `auto`, how much work does it cover and how much more accurate is
that work than the undifferentiated average?
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import OnnxLayaBackend  # noqa: E402
from pandosi.common import DecisionRequest, aggregate  # noqa: E402

LINEAGES = [
    ("en", "convaiinnovations/laya", "artifacts/laya-en.onnx"),
    ("ml", "convaiinnovations/laya-multilingual", "artifacts/laya-ml.onnx"),
    ("td", "convaiinnovations/laya-typed-decisions", "artifacts/laya-td.onnx"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in
             Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    tasks = [task for task in tasks if task.get("qtype") == "choice"]

    backends = []
    for name, repo, onnx in LINEAGES:
        path = ROOT / onnx
        if not path.exists():
            raise SystemExit(f"missing graph for {name}: {path}")
        print(f"loading {name} ...", flush=True)
        backends.append(OnnxLayaBackend(name, str(path), repo=repo,
                                        device="CPU", checkpoint=name))

    rows = []
    for task in tasks:
        request = DecisionRequest(state=task["state"],
                                  question=task["question"],
                                  candidates=task["candidates"])
        votes = [backend.decide(request) for backend in backends]
        result = aggregate(votes, min_quorum=3)
        correct = result.choice == task["expected"]
        rows.append({
            "task_id": task["task_id"],
            "tier": result.tier,
            "agreement": result.agreement,
            "mean_probability": round(result.mean_probability, 4),
            "choice": result.choice,
            "expected": task["expected"],
            "correct": correct,
            "votes": [vote.choice for vote in votes],
            "distinct_votes": len({vote.choice for vote in votes}),
        })
        print(f"  {task['task_id']}: {result.tier:8s} "
              f"agree={result.agreement:.2f} "
              f"conf={result.mean_probability:.3f} "
              f"{'ok' if correct else 'MISS'}", flush=True)

    total = len(rows)
    order = {"auto": 0, "verify": 1, "escalate": 2}
    ranked = sorted(rows, key=lambda row: (order[row["tier"]],
                                           -row["mean_probability"]))
    curve = []
    for covered in range(1, total + 1):
        window = ranked[:covered]
        accuracy = sum(row["correct"] for row in window) / covered
        curve.append({"coverage": round(covered / total, 4),
                      "accuracy": round(accuracy, 4)})

    auto = [row for row in rows if row["tier"] == "auto"]
    unanimous = [row for row in rows if row["distinct_votes"] == 1]
    summary = {
        "tasks": total,
        "overall_accuracy": round(sum(r["correct"] for r in rows) / total, 4),
        "tier_counts": {tier: sum(1 for row in rows if row["tier"] == tier)
                        for tier in order},
        "auto_coverage": round(len(auto) / total, 4),
        "auto_accuracy": round(sum(r["correct"] for r in auto) / len(auto), 4)
        if auto else None,
        "unanimous_coverage": round(len(unanimous) / total, 4),
        "unanimous_accuracy": round(
            sum(r["correct"] for r in unanimous) / len(unanimous), 4)
        if unanimous else None,
        "split_accuracy": round(
            sum(r["correct"] for r in rows if r["distinct_votes"] > 1)
            / max(1, total - len(unanimous)), 4),
        "mean_probability_correct": round(statistics.mean(
            row["mean_probability"] for row in rows if row["correct"]), 4),
        "mean_probability_wrong": round(statistics.mean(
            row["mean_probability"] for row in rows if not row["correct"]), 4),
        "risk_coverage_curve": curve,
        "rows": rows,
    }

    out_dir = ROOT / "metrics" / "gating-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"gating-{int(time.time())}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)

    if args.json:
        print(json.dumps({k: v for k, v in summary.items()
                          if k != "risk_coverage_curve"}, indent=2))
    else:
        print(f"overall accuracy   : {summary['overall_accuracy']:.1%}")
        print(f"auto coverage      : {summary['auto_coverage']:.1%} "
              f"at {summary['auto_accuracy']:.1%} accuracy")
        print(f"unanimous coverage : {summary['unanimous_coverage']:.1%} "
              f"at {summary['unanimous_accuracy']:.1%} accuracy")
        print(f"split agreement    : {summary['split_accuracy']:.1%} accuracy")
        print(f"confidence         : correct {summary['mean_probability_correct']}"
              f" vs wrong {summary['mean_probability_wrong']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
