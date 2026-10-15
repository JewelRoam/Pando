"""Fit `pandosi.combiner.Combiner` and write it as a shippable artifact.

Round 139 measured the fitted combiner's value offline; this turns it into a
file a root can load (`assets/combiner.json`), so the same rule that reached
54.4% on the rows the gate declines can start answering where the gate has
nothing to say. The regularisation is chosen by cross-validation on the
training split alone, so the held-out evaluation stays untouched.

Usage:
    python tools/train_combiner.py --reports <report.json>[,<report.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.combiner import Combiner, features  # noqa: E402


def votes_of(row: dict) -> list[tuple[str, str, float]]:
    return [(name, choice, row["confidences"][name])
            for name, choice in row["per_lineage"].items()]


def fit(train_x: np.ndarray, train_y: np.ndarray, classes: list[str],
        l2: float, steps: int = 600, rate: float = 0.5) -> np.ndarray:
    weights = np.zeros((train_x.shape[1], len(classes)))
    onehot = np.zeros((len(train_y), len(classes)))
    onehot[np.arange(len(train_y)), train_y] = 1.0
    for _ in range(steps):
        logits = train_x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights -= rate * (train_x.T @ (probabilities - onehot) / len(train_y)
                           + l2 * weights)
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--out", default=str(ROOT / "assets"
                                             / "combiner.json"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    paths = [Path(name.strip()) for name in args.reports.split(",")
             if name.strip()]
    rows = [row for path in paths
            for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    classes = sorted({row["expected"] for row in rows})
    lineages = sorted(rows[0]["per_lineage"])
    index = {name: position for position, name in enumerate(classes)}
    train_x = np.array([features(votes_of(row), classes, lineages)
                        for row in rows])
    train_y = np.array([index[row["expected"]] for row in rows])
    print(f"{len(rows)} rows, {train_x.shape[1]} features, "
          f"{len(classes)} classes, lineages {lineages}")

    order = np.arange(len(train_y))
    best = None
    for l2 in (0.003, 0.01, 0.03, 0.1):
        scores = []
        for fold in range(args.folds):
            mask = order % args.folds == fold
            weights = fit(train_x[~mask], train_y[~mask], classes, l2)
            guesses = (train_x[mask] @ weights).argmax(axis=1)
            scores.append((guesses == train_y[mask]).mean())
        mean = float(sum(scores) / args.folds)
        print(f"  {args.folds}-fold CV: l2={l2:<6} {mean:.4f}")
        if best is None or mean > best[1]:
            best = (l2, mean)
    l2, cv = best
    print(f"  chosen: l2={l2} (cross-validated {cv:.4f})")

    weights = fit(train_x, train_y, classes, l2)
    in_sample = float((((train_x @ weights).argmax(axis=1))
                       == train_y).mean())
    combiner = Combiner(classes, lineages, weights.tolist(),
                        {"trained_on": [path.name for path in paths],
                         "rows": len(rows), "l2": l2,
                         "cross_validated_accuracy": round(cv, 4),
                         "in_sample_accuracy": round(in_sample, 4)})
    out = combiner.save(args.out)
    print(f"  in-sample {in_sample:.4f}; wrote {out} "
          f"({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
