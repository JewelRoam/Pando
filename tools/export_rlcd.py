"""Export accumulated live rollouts into RLCD training rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.rollout import export_training_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts",
                        default=str(ROOT / "rollouts" / "rollouts.jsonl"))
    parser.add_argument("--out",
                        default=str(ROOT / "rollouts" / "train_rlcd_all.jsonl"))
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.append:
        out_path.unlink()
    rows = export_training_rows(Path(args.rollouts), out_path,
                                min_votes=args.min_votes)
    print(f"exported {rows} contrast rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
