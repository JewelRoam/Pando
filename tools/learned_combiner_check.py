"""Can a *learned* combiner earn the gap the hand rules cannot?

Round 118 measured where the swarm's headroom is: on the disagreements, the
correct answer is present somewhere in the trio 91.7% of the time, while the
best hand-designed rule over the same votes captures 74.5%. Round 120 tried
the other cheap move - a narrower ballot - and got 1.3 points. Every combiner
tested so far has been a rule a human wrote: majority, weighted vote,
unanimity-first, most- or least-confident member.

This trains one instead. The features are the votes themselves - how many
members chose each option, and how much calibrated confidence stands behind
each option - and the target is the label. It is fitted on the dev split's
recorded votes and scored on the wording-held-out split, so the evaluation
shares no rows and no wording with the fit, and the bar is the shipped gate
on the same rows.

Usage:
    python tools/learned_combiner_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.common import Vote, aggregate  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def features(row: dict, candidates: list[str],
             members: list[str]) -> list[float]:
    """What the quorum's votes say: counts and confidence mass per candidate,
    the size of the majority, who dissents, and what each member believed."""
    votes = Counter(row["per_lineage"].values())
    confidence = {}
    for name, choice in row["per_lineage"].items():
        confidence[choice] = (confidence.get(choice, 0.0)
                              + row["confidences"][name])
    top = votes.most_common(1)
    majority = top[0][0] if top else None
    return ([votes.get(name, 0) / 3 for name in candidates]
            + [confidence.get(name, 0.0) / 3 for name in candidates]
            + [(top[0][1] / 3) if top else 0.0]
            + [1.0 if row["per_lineage"][name] != majority else 0.0
               for name in members]
            + [row["confidences"][name] for name in members]
            + [1.0])


def softmax_regression(train_x: np.ndarray, train_y: np.ndarray,
                       classes: list[str], steps: int = 400,
                       rate: float = 0.5, l2: float = 1e-2) -> np.ndarray:
    """Multinomial logistic regression, trained by gradient descent."""
    width = train_x.shape[1]
    weights = np.zeros((width, len(classes)))
    onehot = np.zeros((len(train_y), len(classes)))
    onehot[np.arange(len(train_y)), train_y] = 1.0
    for _ in range(steps):
        logits = train_x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = train_x.T @ (probabilities - onehot) / len(train_y) \
            + l2 * weights
        weights -= rate * gradient
    return weights


def predict(x: np.ndarray, weights: np.ndarray,
            classes: list[str]) -> list[str]:
    return [classes[index] for index in (x @ weights).argmax(axis=1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default=str(ROOT / "metrics"
                                             / "ensemble-checks"
                                             / "ensemble-1792062150.json"),
                        help="training report(s); several may be pooled with "
                             "commas. Only the *vote patterns* are used, so a "
                             "split that shares wording with the training "
                             "text is still a legitimate source of training "
                             "rows for a combiner")
    parser.add_argument("--heldout", default=str(ROOT / "metrics"
                                                 / "ensemble-checks"
                                                 / "ensemble-1792064900.json"))
    args = parser.parse_args()

    dev_paths = [Path(name.strip())
                 for name in args.dev.split(",") if name.strip()]
    dev_rows = [rows for path in dev_paths
                for rows in [json.loads(path.read_text(encoding="utf-8"))
                             ["rows"]]]
    dev = {"rows": [row for group in dev_rows for row in group]}
    held = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    candidates = sorted({row["expected"] for row in dev["rows"]})
    index = {name: position for position, name in enumerate(candidates)}

    members = sorted(dev["rows"][0]["per_lineage"])
    print(f"training rows: {len(dev['rows'])} from "
          f"{[path.name for path in dev_paths]}")
    train_x = np.array([features(row, candidates, members)
                        for row in dev["rows"]])
    train_y = np.array([index[row["expected"]] for row in dev["rows"]])
    test_x = np.array([features(row, candidates, members)
                       for row in held["rows"]])

    # Choose the regularisation on the dev split alone: tuning on the
    # held-out rows would turn the evaluation into a fitting set.
    folds = 5
    order = np.arange(len(train_y))
    best = None
    for l2 in (0.003, 0.01, 0.03, 0.1):
        scores = []
        for fold in range(folds):
            mask = order % folds == fold
            weights = softmax_regression(train_x[~mask], train_y[~mask],
                                         candidates, l2=l2)
            guesses = predict(train_x[mask], weights, candidates)
            truth = [candidates[int(value)] for value in train_y[mask]]
            scores.append(sum(1 for guess, answer in zip(guesses, truth)
                              if guess == answer) / mask.sum())
        mean = sum(scores) / folds
        print(f"  cross-validation: l2={l2:<6} accuracy {mean:.4f}")
        if best is None or mean > best[1]:
            best = (l2, mean)
    chosen_l2 = best[0]
    print(f"  chosen by {folds}-fold cross-validation on dev: "
          f"l2={chosen_l2} ({best[1]:.4f})")
    weights = softmax_regression(train_x, train_y, candidates, l2=chosen_l2)

    train_hits = sum(1 for row, guess in
                     zip(dev["rows"], predict(train_x, weights, candidates))
                     if guess == row["expected"])
    print(f"fitted on {len(dev['rows'])} dev rows "
          f"(in-sample {train_hits / len(dev['rows']):.4f}), "
          f"scored on {len(held['rows'])} held-out rows")

    guesses = predict(test_x, weights, candidates)
    learned = sum(1 for row, guess in zip(held["rows"], guesses)
                  if guess == row["expected"])

    def gate(row: dict, members: list[str]) -> tuple[str | None, str, float]:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        return result.choice, result.tier, result.mean_probability

    shipped, shipped_auto, ranked = [], [], []
    for row in held["rows"]:
        choice, tier, mean_probability = gate(row, members)
        shipped.append(choice == row["expected"])
        if tier == "auto":
            shipped_auto.append(choice == row["expected"])
        ranked.append((TIER_ORDER[tier], -mean_probability,
                       choice == row["expected"], row["expected"],
                       row["task_id"]))
    ranked.sort(key=lambda item: (item[0], item[1]))
    learned_hits = {row["task_id"]: (guess == row["expected"])
                    for row, guess in zip(held["rows"], guesses)}

    matched = len(shipped_auto)
    learned_at_matched = sum(
        1 for _, _, _, _, task_id in ranked[:matched]
        if learned_hits[task_id])
    shipped_at_matched = sum(
        1 for _, _, hit, _, _ in ranked[:matched] if hit)
    print(f"  shipped gate      full {sum(shipped) / len(shipped):.4f}  "
          f"auto {matched} at "
          f"{sum(shipped_auto) / max(1, matched):.4f}")
    print(f"  learned combiner  full {learned / len(guesses):.4f}  "
          f"matched-at-{matched} {learned_at_matched / matched:.4f}")
    print(f"  shipped at the same {matched} rows: "
          f"{shipped_at_matched / matched:.4f}")
    accepted = {task_id for _, _, _, _, task_id in ranked[:matched]}
    declined = [row["task_id"] for row in held["rows"]
                if row["task_id"] not in accepted]
    for label, group in (("the gate's own rows", accepted),
                         ("the rows the gate declines", declined)):
        ids = list(group)
        if not ids:
            continue
        learned_here = sum(1 for task_id in ids if learned_hits[task_id])
        best_member = max(
            sum(1 for task_id in ids
                if next(row for row in held["rows"]
                        if row["task_id"] == task_id)["per_lineage"][name]
                == next(row for row in held["rows"]
                        if row["task_id"] == task_id)["expected"])
            for name in members)
        print(f"  on {label} ({len(ids)}): learned "
              f"{learned_here / len(ids):.4f}, best single member "
              f"{best_member / len(ids):.4f}, oracle "
              f"{sum(1 for task_id in ids if next(row for row in held['rows'] if row['task_id'] == task_id)['expected'] in set(next(row for row in held['rows'] if row['task_id'] == task_id)['per_lineage'].values())) / len(ids):.4f}")
    print(f"  oracle (answer in the trio) "
          f"{sum(1 for row in held['rows'] if row['expected'] in set(row['per_lineage'].values())) / len(held['rows']):.4f}")

    summary = {
        "fitted_on": len(dev["rows"]), "scored_on": len(held["rows"]),
        "in_sample_accuracy": round(train_hits / len(dev["rows"]), 4),
        "learned_full": round(learned / len(guesses), 4),
        "shipped_full": round(sum(shipped) / len(shipped), 4),
        "learned_at_shipped_coverage": round(learned_at_matched / matched, 4),
        "shipped_at_shipped_coverage": round(shipped_at_matched / matched, 4),
        "coverage": matched,
    }
    out = ROOT / "metrics" / "ensemble-checks" / "learned-combiner.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
