"""Assemble a fine-tuning file from the labels this session produced.

Rounds 122-137 concluded that the swarm's ceiling is its members, that nothing
in the Laya family decorrelates from the rest of it, and that the one lever
left is a member that is genuinely different. The project has a fine-tuning
path (`tools/rlcd_cuda_train.py`, rounds 79-81) that failed to move a decision
when it trained on 22 contrast pairs; there are now 720 labelled rows.

This writes them in the format `laya.train.finetune` reads, and it makes the
contrast rows the interesting ones: where the recorded swarm was *wrong*, its
answer becomes the `rejected` field, so the objective is pushed away from the
mistake rather than from a random alternative. The third wording split (set C)
is deliberately not included - it is the evaluation.

Usage:
    python tools/build_finetune_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = [
    ("assets/big_dev.jsonl",
     "metrics/ensemble-checks/ensemble-1792062150.json"),
    ("assets/big_test.jsonl",
     "metrics/ensemble-checks/ensemble-1792062757.json"),
    ("assets/heldout_test.jsonl",
     "metrics/ensemble-checks/ensemble-1792064900.json"),
]
TRAINING_SET_C = ("assets/heldout2_test.jsonl",
                  "metrics/ensemble-checks/ensemble-1792076803.json")
INSTRUCTION = "Classify the intent of the message."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "rollouts"
                                             / "train_finetune_720.jsonl"))
    parser.add_argument("--with-c", action="store_true",
                        help="add the third wording split (960 rows). Its "
                             "evaluation role moves to a fourth split, which "
                             "is what round 152 measures the curve on")
    args = parser.parse_args()

    sources = SOURCES + ([TRAINING_SET_C] if args.with_c else [])
    rows, contrasted, skipped = [], 0, 0
    for suite_name, report_name in sources:
        tasks = [json.loads(line) for line
                 in (ROOT / suite_name).read_text(
                     encoding="utf-8").splitlines() if line.strip()]
        report_path = ROOT / report_name
        swarm = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            swarm = {row["task_id"]: row for row in report["rows"]}
        for task in tasks:
            record = swarm.get(task["task_id"])
            if record is None:
                skipped += 1
                continue
            expected = task["expected"]
            answer = record.get("swarm_choice")
            wrong = answer is not None and answer != expected
            alternative = answer if wrong else next(
                name for name in sorted(task["candidates"])
                if name != expected)
            if wrong:
                contrasted += 1
            rows.append({
                "state": task["state"],
                "questions": {"intent": {
                    "type": "choice",
                    "instructions": INSTRUCTION,
                    "criteria": list(task["candidates"])}},
                "expected": {"intent": expected},
                "rejected": {"intent": alternative},
                "source": Path(suite_name).name,
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                   encoding="utf-8")
    print(f"wrote {out.name}: {len(rows)} rows "
          f"({contrasted} of them contrast rows where the swarm was wrong, "
          f"{skipped} skipped for want of a recorded decision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
