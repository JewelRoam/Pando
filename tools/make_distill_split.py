"""Split the dual-teacher distillation corpus for the ft-v5 fine-tune.

Every 10th row (deterministic) becomes the teacher-labeled holdout; the
rest trains ft-v5. The holdout's `expected` is the 8B label both teachers
agreed on, so agreement with it is the honest relative measure here.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    rows = [json.loads(line) for line in
            (ROOT / "rollouts" / "distill_train.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    holdout = [row for index, row in enumerate(rows) if index % 10 == 9]
    train = [row for index, row in enumerate(rows) if index % 10 != 9]
    for name, data in (("distill_ft5_train.jsonl", train),
                       ("distill_ft5_holdout.jsonl", holdout)):
        path = ROOT / "rollouts" / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                  for row in data) + "\n", encoding="utf-8")
        print(f"{name}: {len(data)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
