"""Convert a suite-format corpus into the trainer's row format.

Mirrors tools/build_finetune_data.py's mapping, with the label coming from
the corpus itself (for the distillation corpus: the dual-teacher label).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Classify the intent of the message."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in
             (ROOT / args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    rows = []
    for task in tasks:
        expected = task["expected"]
        alternative = next(name for name in sorted(task["candidates"])
                           if name != expected)
        rows.append({
            "state": task["state"],
            "questions": {"intent": {
                "type": "choice",
                "instructions": INSTRUCTION,
                "criteria": list(task["candidates"])}},
            "expected": {"intent": expected},
            "rejected": {"intent": alternative},
            "source": Path(args.suite).name})
    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                             for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
