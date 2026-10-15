"""Does the swarm decide better than its best member, at equal coverage?

The project has measured that a checkpoint-diverse quorum produces
independent evidence (round 57) and that its gate lifts accuracy on the work
it accepts (rounds 99-100). Neither answers the premise the whole design
rests on: with the *same amount of work accepted*, is the assembled swarm
better than simply trusting the strongest single lineage?

Three curves are computed on the 27 labeled choice tasks, all with the
production calibration applied:

  - each lineage alone, selecting its own most-confident tasks;
  - the swarm, selecting by its tier and agreement;
  - the swarm at full coverage (plain majority vote).

The comparison is made at matched coverage, because a gate that accepts half
the work and looks accurate is only interesting if a single model given the
same half does worse.

Usage:
    python tools/ensemble_value_check.py --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import OnnxLayaBackend  # noqa: E402
from pandosi.calibrate import Calibrator  # noqa: E402
from pandosi.common import DecisionRequest, Vote, aggregate  # noqa: E402

LINEAGES = {
    "english": ("convaiinnovations/laya", "artifacts/laya-en.onnx"),
    "multilingual": ("convaiinnovations/laya-multilingual",
                     "artifacts/laya-ml.onnx"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions",
                        "artifacts/laya-td.onnx"),
    # Trained in this repository (the child of laya-en.onnx in the delta
    # replication checks), so its weights differ from the three above rather
    # than re-rendering them - the membership the swarm premise needs
    # (round 119). Its tokenizer and config come from the base it was
    # fine-tuned out of, which is what the repo argument supplies.
    "rlcd-v7": ("convaiinnovations/laya", "artifacts/laya-rlcd-v7.onnx"),
}
DEFAULT_LINEAGES = "english,multilingual,typed-decisions"


def accuracy_at(ranked: list[dict], count: int,
                key: str = "correct") -> float | None:
    if count <= 0:
        return None
    window = ranked[:count]
    return round(sum(row[key] for row in window) / len(window), 4)


def curve(ranked: list[dict], step: int, key: str = "correct") -> list[dict]:
    points = []
    for count in range(step, len(ranked) + 1, step):
        points.append({"coverage": round(count / len(ranked), 4),
                       "accuracy": accuracy_at(ranked, count, key)})
    return points


def load_backends(names: list[str]) -> list[OnnxLayaBackend]:
    backends = []
    for name in names:
        repo, onnx = LINEAGES[name]
        path = ROOT / onnx
        if not path.exists():
            raise SystemExit(f"missing graph for {name}: {path}")
        print(f"loading {name} ...", flush=True)
        backends.append(OnnxLayaBackend(name, str(path), repo=repo,
                                        device="CPU", checkpoint=name))
    return backends


def run_task(backends: list[OnnxLayaBackend], calibrator: Calibrator,
             task: dict) -> dict:
    """One task through every lineage, recorded as a ruled-on row."""
    request = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"])
    votes = [backend.decide(request) for backend in backends]
    calibrated = []
    for vote in votes:
        probability, _ = calibrator.calibrate(vote.checkpoint,
                                              vote.probability)
        calibrated.append(Vote(vote.node_id, vote.checkpoint, vote.choice,
                               probability))
    result = aggregate(calibrated, min_quorum=len(backends))
    expected = task["expected"]
    choices = [vote.choice for vote in calibrated]
    top_votes = Counter(choices).most_common()
    majority = (top_votes[0][0] if top_votes
                and (len(top_votes) == 1
                     or top_votes[0][1] > top_votes[1][1]) else None)
    per_lineage = {vote.checkpoint: vote.choice for vote in calibrated}
    return {
        "task_id": task["task_id"],
        "expected": expected,
        "choices": choices,
        "per_lineage": per_lineage,
        "confidences": {vote.checkpoint: vote.probability
                        for vote in calibrated},
        "tier": result.tier,
        "agreement": result.agreement,
        "mean_probability": result.mean_probability,
        "swarm_choice": result.choice,
        "correct": result.choice == expected,
        "unanimous_correct": majority == expected,
        "distinct": len(set(choices)),
        # Per-lineage correctness: the swarm's own `correct` says nothing
        # about an individual member, and mixing them up made every
        # lineage look as accurate as the vote (found in round 117).
        "lineage_correct": {name: choice == expected
                            for name, choice in per_lineage.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--lineages", default=DEFAULT_LINEAGES,
                        help="comma-separated lineage names")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N tasks (smoke runs only)")
    args = parser.parse_args()

    chosen = [name.strip() for name in args.lineages.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in LINEAGES]
    if unknown:
        raise SystemExit(f"unknown lineage(s) {unknown}; "
                         f"known: {sorted(LINEAGES)}")

    tasks = [json.loads(line) for line in
             Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    tasks = [task for task in tasks if task.get("qtype") == "choice"]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"suite: {Path(args.suite).name}, {len(tasks)} labeled choice "
          f"tasks", flush=True)

    backends = load_backends(chosen)
    calibrator = Calibrator()

    rows = []
    for task in tasks:
        rows.append(run_task(backends, calibrator, task))
        print(f"  {task['task_id']}: {rows[-1]['tier']:8s} "
              f"{rows[-1]['choices']} -> "
              f"{'ok' if rows[-1]['correct'] else 'MISS'}",
              flush=True)

    total = len(rows)
    order = {"auto": 0, "verify": 1, "escalate": 2}
    swarm_ranked = sorted(rows, key=lambda row: (order[row["tier"]],
                                                 -row["mean_probability"]))
    single = {}
    for name in chosen:
        lane = [dict(row, correct=row["lineage_correct"][name])
                for row in rows]
        ranked = sorted(lane, key=lambda row, n=name:
                        -row["confidences"].get(n, 0.0))
        single[name] = {
            "accuracy_full": accuracy_at(ranked, total),
            "curve": curve(ranked, max(1, total // 8)),
        }

    auto = [row for row in rows if row["tier"] == "auto"]
    auto_coverage = len(auto) / total
    auto_count = len(auto)
    swarm_at_auto = accuracy_at(swarm_ranked, auto_count)
    best_single_full = max(entry["accuracy_full"] for entry in single.values())
    # The honest baseline: the strongest lineage, allowed to pick the same
    # number of tasks it is most confident about.
    matched = {}
    for name, entry in single.items():
        lane = [dict(row, correct=row["lineage_correct"][name])
                for row in rows]
        ranked = sorted(lane, key=lambda row, n=name:
                        -row["confidences"].get(n, 0.0))
        matched[name] = accuracy_at(ranked, auto_count)
    best_matched = max(matched.values())
    unanimous = [row for row in rows if row["distinct"] == 1]

    summary = {
        "suite": Path(args.suite).name,
        "lineages": chosen,
        "tasks": total,
        "per_lineage_full_coverage": {name: entry["accuracy_full"]
                                      for name, entry in single.items()},
        "per_lineage_curve": {name: entry["curve"]
                              for name, entry in single.items()},
        "best_single_full_coverage": best_single_full,
        "majority_full_coverage": accuracy_at(swarm_ranked, total),
        "swarm_auto_coverage": round(auto_coverage, 4),
        "swarm_auto_count": auto_count,
        "swarm_accuracy_at_auto_coverage": swarm_at_auto,
        "best_single_at_matched_coverage": best_matched,
        "per_lineage_at_matched_coverage": matched,
        "ensemble_lift_at_matched_coverage": (
            round(swarm_at_auto - best_matched, 4)
            if swarm_at_auto is not None and best_matched is not None
            else None),
        "unanimous_coverage": round(len(unanimous) / total, 4),
        "unanimous_accuracy": round(
            sum(row["correct"] for row in unanimous)
            / max(1, len(unanimous)), 4),
        "disagreement_accuracy": round(
            sum(row["correct"] for row in rows if row["distinct"] > 1)
            / max(1, total - len(unanimous)), 4),
        "oracle_accuracy": round(
            sum(1 for row in rows
                if row["expected"] in row["choices"]) / total, 4),
        "rows": rows,
    }

    out_dir = ROOT / "metrics" / "ensemble-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A truncated run must not become "the newest report" for the tools that
    # glob ensemble-*.json, so it is written under its own prefix.
    report = out_dir / (f"smoke-{int(time.time())}.json" if args.limit
                        else f"ensemble-{int(time.time())}.json")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("rows", "per_lineage_curve")},
                         indent=2))
    else:
        print(f"single lineage, full coverage : "
              f"{summary['per_lineage_full_coverage']}")
        print(f"swarm, full coverage          : "
              f"{summary['majority_full_coverage']}")
        print(f"swarm gate                    : {auto_count}/{total} "
              f"accepted at {swarm_at_auto}")
        print(f"best single at that coverage  : {best_matched} "
              f"({matched})")
        print(f"ensemble lift                 : "
              f"{summary['ensemble_lift_at_matched_coverage']}")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
