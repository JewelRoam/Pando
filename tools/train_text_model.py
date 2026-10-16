"""Fit the cheap independent member, and calibrate it like any other lineage.

Trains `pando.textmodel.TextModel` on a labeled suite and writes the two
artifacts a swarm needs to admit it: the model itself (`assets/textmodel.json`,
a few hundred kilobytes) and its calibration table
(`assets/calibration/<checkpoint>.json`), fitted on rows the counts did not
see, using the same temperature transform every neural lineage uses - a vote
on an unknown confidence scale cannot justify an autonomous action.

Usage:
    python tools/train_text_model.py --checkpoint text-nb-v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.calibrate import apply_temperature  # noqa: E402
from pando.textmodel import TextModel  # noqa: E402

TEMPERATURES = [0.25, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.0,
                4.0, 5.0, 6.0, 8.0]


def log_loss(entries: list[tuple[float, bool]], temperature: float) -> float:
    total = 0.0
    for probability, correct in entries:
        calibrated = apply_temperature(probability, temperature)
        calibrated = min(max(calibrated, 1e-7), 1 - 1e-7)
        total -= math.log(calibrated if correct else 1 - calibrated)
    return total / max(1, len(entries))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT / "assets"
                                               / "big_dev.jsonl"))
    parser.add_argument("--checkpoint", default="text-nb-v1")
    parser.add_argument("--holdout", type=int, default=48,
                        help="rows set aside for the temperature fit")
    parser.add_argument("--out", default=str(ROOT / "assets"
                                             / "textmodel.json"))
    parser.add_argument("--calibration-dir",
                        default=str(ROOT / "assets" / "calibration"))
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in Path(args.suite).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.holdout >= len(rows):
        raise SystemExit("holdout must be smaller than the suite")
    held = rows[-args.holdout:]
    trained = rows[:-args.holdout]

    model = TextModel(checkpoint=args.checkpoint)
    stats = model.fit(trained)
    print(f"fitted on {len(trained)} rows: {stats}")

    entries = []
    for row in held:
        vote = model.decide(_request(row))
        entries.append((vote.probability, vote.choice == row["expected"]))
    raw = sum(1 for _, correct in entries if correct) / len(entries)
    raw_loss = log_loss(entries, 1.0)
    scored = [(log_loss(entries, temperature), temperature)
              for temperature in TEMPERATURES]
    best_loss, temperature = min(scored)
    accuracy = raw
    print(f"holdout {len(held)} rows: accuracy {accuracy:.4f}, "
          f"raw log-loss {raw_loss:.4f} -> {best_loss:.4f} at T={temperature}")

    model_path = model.save(args.out)
    calibration = {
        "checkpoint": args.checkpoint,
        "temperature": temperature,
        "source": "tools/train_text_model.py holdout fit",
        "raw_ece": None,
        "n_tasks": len(held),
        "accuracy": round(accuracy, 4),
        "raw_log_loss": round(raw_loss, 4),
        "calibrated_log_loss": round(best_loss, 4),
    }
    calibration_path = Path(args.calibration_dir) / f"{args.checkpoint}.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration, indent=2),
                                encoding="utf-8")
    print(f"model: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"calibration: {calibration_path}")
    return 0


def _request(row: dict):
    from pando.common import DecisionRequest
    return DecisionRequest(state=row.get("state", ""),
                           question=row.get("question", ""),
                           candidates=row.get("candidates", []))


if __name__ == "__main__":
    raise SystemExit(main())
