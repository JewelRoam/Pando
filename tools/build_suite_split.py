"""Deterministically split the graft suite into train and held-out sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--train-out",
                        default=str(ROOT / "assets" / "graft_train.jsonl"))
    parser.add_argument("--holdout-out",
                        default=str(ROOT / "assets" / "graft_holdout.jsonl"))
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--seed", default="pando-2026")
    args = parser.parse_args()

    tasks = [
        json.loads(line)
        for line in Path(args.suite).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train, holdout = [], []
    for task in tasks:
        digest = hashlib.sha256(
            f"{args.seed}:{task['task_id']}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2 ** 64
        (holdout if value < args.holdout_frac else train).append(task)

    for path, rows in ((Path(args.train_out), train),
                       (Path(args.holdout_out), holdout)):
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            + "\n",
            encoding="utf-8")
    print(f"train={len(train)} holdout={len(holdout)} "
          f"holdout_ids={[task['task_id'] for task in holdout]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
