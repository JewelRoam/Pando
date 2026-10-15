"""Would the text itself close the rest of the gap on the declined rows?

Round 139/140 left a precise remainder: on the 68 rows where the hand-written
gate declines, the best single member is 41.2%, the fitted *vote-pattern*
combiner is 54.4%, and the oracle is 67.7%. The combiner sees only how the
members voted; the question's own text is the obvious missing signal, and the
project already has a model of it (`pandosi/textmodel.py`, the cheap member of
round 126).

This measures the gain before touching the shipped feature spec: the same
vote-pattern features plus the text model's posterior over the candidates,
fitted on the dev split and scored on the wording-held-out split.

One methodological care: `assets/textmodel.json` was trained on the dev rows,
so its dev posteriors are in-sample and a combiner fitted on them would learn
to over-trust it. The text model here is trained on the *other* split's rows
instead, which leaves it out-of-sample on dev as well - the honest training
signal - and the held-out evaluation stays untouched either way.

Usage:
    python tools/combiner_text_check.py
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

from pandosi.combiner import features as vote_features  # noqa: E402
from pandosi.common import DecisionRequest, Vote, aggregate  # noqa: E402
from pandosi.textmodel import TextModel  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def load(path):
    return [json.loads(line) for line
            in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def text_features(model: TextModel, task: dict,
                  classes: list[str]) -> list[float]:
    posterior = model.posterior(task["state"], task["question"],
                                list(task["candidates"]))
    return [posterior.get(name, 0.0) for name in classes]


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
    parser.add_argument("--dev-report", default=str(
        ROOT / "metrics" / "ensemble-checks" / "ensemble-1792062150.json"))
    parser.add_argument("--dev-suite",
                        default=str(ROOT / "assets" / "big_dev.jsonl"))
    parser.add_argument("--heldout-report", default=str(
        ROOT / "metrics" / "ensemble-checks" / "ensemble-1792064900.json"))
    parser.add_argument("--heldout-suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"))
    parser.add_argument("--text-train",
                        default=str(ROOT / "assets" / "big_test.jsonl"),
                        help="rows the text model learns from (not dev)")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    dev_tasks = {task["task_id"]: task for task in load(args.dev_suite)}
    held_tasks = {task["task_id"]: task for task in load(args.heldout_suite)}
    dev = json.loads(Path(args.dev_report).read_text(encoding="utf-8"))
    held = json.loads(Path(args.heldout_report).read_text(encoding="utf-8"))
    classes = sorted({row["expected"] for row in dev["rows"]})
    lineages = sorted(dev["rows"][0]["per_lineage"])
    index = {name: position for position, name in enumerate(classes)}

    text_model = TextModel(checkpoint="text-nb-v1")
    text_model.fit(load(args.text_train))
    print(f"text model fitted on {args.text_train} (dev is out-of-sample for "
          f"it, as the held-out split is)")

    def build(rows: list[dict], tasks: dict) -> np.ndarray:
        cells = []
        for row in rows:
            votes = [(name, row["per_lineage"][name],
                      row["confidences"][name]) for name in lineages]
            cells.append(vote_features(votes, classes)
                         + text_features(text_model,
                                         tasks[row["task_id"]], classes))
        return np.array(cells)

    train_x = build(dev["rows"], dev_tasks)
    train_y = np.array([index[row["expected"]] for row in dev["rows"]])
    test_x = build(held["rows"], held_tasks)
    print(f"features: {train_x.shape[1]} "
          f"({vote_features([('a', 'refund', 0.5)], classes).__len__()} "
          f"vote-pattern + {len(classes)} text posterior)")

    order = np.arange(len(train_y))
    best = None
    for l2 in (0.003, 0.01, 0.03, 0.1):
        scores = []
        for fold in range(args.folds):
            mask = order % args.folds == fold
            weights = fit(train_x[~mask], train_y[~mask], classes, l2)
            guesses = (train_x[mask] @ weights).argmax(axis=1)
            scores.append(float((guesses == train_y[mask]).mean()))
        mean = sum(scores) / args.folds
        print(f"  {args.folds}-fold CV: l2={l2:<6} {mean:.4f}")
        if best is None or mean > best[1]:
            best = (l2, mean)
    l2 = best[0]
    weights = fit(train_x, train_y, classes, l2)
    guesses = [classes[value] for value in (test_x @ weights).argmax(axis=1)]
    hits = {row["task_id"]: (guess == row["expected"])
            for row, guess in zip(held["rows"], guesses)}

    members = lineages
    accepted, ranked = set(), []
    gate_hits = []
    for row in held["rows"]:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        gate_hits.append(result.choice == row["expected"])
        ranked.append((TIER_ORDER[result.tier], -result.mean_probability,
                       row["task_id"]))
    ranked.sort(key=lambda item: (item[0], item[1]))
    coverage = sum(1 for tier, _, _ in ranked if tier == TIER_ORDER["auto"])
    accepted = {task_id for tier, _, task_id in ranked
                if tier == TIER_ORDER["auto"]}
    declined = [row["task_id"] for row in held["rows"]
                if row["task_id"] not in accepted]
    print(f"  gate full {sum(gate_hits) / len(gate_hits):.4f}, "
          f"auto {coverage}")
    print(f"  vote-pattern + text: full "
          f"{sum(hits.values()) / len(held['rows']):.4f}")
    for label, ids in (("accepted", sorted(accepted)),
                       ("declined", declined)):
        if not ids:
            continue
        oracle = sum(1 for task_id in ids
                     if next(row for row in held["rows"]
                             if row["task_id"] == task_id)["expected"]
                     in set(next(row for row in held["rows"]
                                 if row["task_id"] == task_id)
                            ["per_lineage"].values()))
        best_member = max(
            sum(1 for task_id in ids
                if next(row for row in held["rows"]
                        if row["task_id"] == task_id)["per_lineage"][name]
                == next(row for row in held["rows"]
                        if row["task_id"] == task_id)["expected"])
            for name in members)
        print(f"  on {label} ({len(ids)}): combined "
              f"{sum(1 for task_id in ids if hits[task_id]) / len(ids):.4f}, "
              f"best member {best_member / len(ids):.4f}, "
              f"oracle {oracle / len(ids):.4f}")

    summary = {"features": int(train_x.shape[1]), "l2": l2,
               "cross_validated": round(best[1], 4),
               "full": round(sum(hits.values()) / len(held["rows"]), 4),
               "gate_full": round(sum(gate_hits) / len(gate_hits), 4),
               "declined_rows": len(declined),
               "declined_accuracy": round(
                   sum(1 for task_id in declined if hits[task_id])
                   / max(1, len(declined)), 4)}
    out = ROOT / "metrics" / "ensemble-checks" / "combiner-text.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
