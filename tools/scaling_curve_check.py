"""How does a swarm scale - with member count, or with member competence?

Round 128's portfolio measurement found a cliff: one cheap independent member
is worth up to +3.4 points to the recorded three-lineage swarm, and four of
them leave it 11.6 points worse than no addition at all. Two reweightings of
the choice rule did not change the cliff, which leaves the question the
elastic premise needs answered: is the swarm limited by how many members it
has, or by how weak they are?

Real members cannot separate those - there are only four cheap ones and their
accuracies are close together. Synthetic voters can: add k voters of a fixed
accuracy to the recorded trio, only the added members' votes change, and the
curve is averaged over many independent draws so the answer is not one
split's noise.

The synthetic voter is deliberately simple: right with probability p,
otherwise a uniformly random wrong option, at a fixed confidence. That is a
worse model than any real member, and it is the right control - it carries no
knowledge of the task and no correlation with the trio, so whatever the curve
shows is caused by competence and count alone.

Usage:
    python tools/scaling_curve_check.py --report <ensemble-*.json>
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.common import Vote, aggregate  # noqa: E402

ACCURACIES = (0.5, 0.6, 0.7, 0.8, 0.9)
COUNTS = (0, 1, 2, 3, 4, 6)
SYNTHETIC_CONFIDENCE = 0.7


def synthetic_vote(row: dict, task: dict, accuracy: float,
                   rng: random.Random) -> Vote:
    if rng.random() < accuracy:
        choice = row["expected"]
    else:
        wrong = [name for name in task["candidates"]
                 if name != row["expected"]]
        choice = rng.choice(wrong)
    return Vote("synthetic", "synthetic", choice, SYNTHETIC_CONFIDENCE)


def run(rows: list[dict], tasks: dict, accuracy: float, count: int,
        rng: random.Random) -> tuple[float, int]:
    correct = 0
    auto = 0
    for row in rows:
        task = tasks[row["task_id"]]
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name])
                 for name in row["per_lineage"]]
        votes += [synthetic_vote(row, task, accuracy, rng)
                  for _ in range(count)]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        correct += result.choice == row["expected"]
        auto += result.tier == "auto"
    return correct / len(rows), auto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"))
    parser.add_argument("--draws", type=int, default=30,
                        help="independent vote draws per cell")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    rows = json.loads(Path(args.report).read_text(encoding="utf-8"))["rows"]
    print(f"{len(rows)} recorded rows, three lineages, "
          f"{args.draws} draws per cell")

    rng = random.Random(args.seed)
    curve = {}
    print(f"  {'accuracy':>9s} " + "".join(f"{count:>8d}" for count in COUNTS))
    for accuracy in ACCURACIES:
        cells = []
        for count in COUNTS:
            draws = [run(rows, tasks, accuracy, count, rng)
                     for _ in range(args.draws)]
            full = statistics.mean(value for value, _ in draws)
            auto = statistics.mean(value for _, value in draws)
            cells.append((round(full, 4), round(auto, 1)))
            curve[f"{accuracy}:{count}"] = {"full": round(full, 4),
                                            "auto": round(auto, 1)}
        print(f"  {accuracy:9.2f} "
              + "".join(f"{cell[0]:8.3f}" for cell in cells))
        print(f"  {'auto':>9s} "
              + "".join(f"{cell[1]:8.0f}" for cell in cells))

    baseline = curve[f"{ACCURACIES[0]}:0"]["full"]
    summary = {"report": str(args.report), "rows": len(rows),
               "baseline_full": baseline, "curve": curve,
               "draws": args.draws}
    out = ROOT / "metrics" / "ensemble-checks" / "scaling-curve.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
