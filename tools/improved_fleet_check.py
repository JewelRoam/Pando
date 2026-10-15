"""How good is the fleet with the member the project trained itself?

Round 150 trained `ft-v3` on 720 of the project's own labels - 0.9125 on an
unseen wording split against the best incumbent's 0.7375, disagreeing with its
parent on 62 of 240 decisions - and measured the gate: 0.7708 -> 0.8833 at
full coverage. What it did not measure is what the *system* answers, which is
the gate's choice where it auto-executes plus the advisory combiner's
suggestion everywhere else.

Two questions, both offline against the recorded votes:

  - does the shipped combiner - fitted on the trio, membership-agnostic since
    spec 3 - improve the four-member fleet, or does it need refitting?
  - what does a combiner fitted for the new fleet buy, measured on rows it was
    not fitted on?

The combiner's embedded text model was trained on wording set B, so it is
out-of-sample for split C: the text features are legitimate here either way.
The refit uses the first half of C and reports on the second.

Usage:
    python tools/improved_fleet_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.combiner import Combiner  # noqa: E402
from pandosi.common import Vote, aggregate  # noqa: E402
from pandosi.textmodel import TextModel  # noqa: E402

TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}
NEW = "ft-v3"


def rows_with_votes(report: dict, votes: dict, tasks: dict) -> list[dict]:
    out = []
    for record in report["rows"]:
        vote = votes[record["task_id"]]
        task = tasks[record["task_id"]]
        out.append({
            "task_id": record["task_id"],
            "expected": record["expected"],
            "candidates": task["candidates"],
            "state": task["state"],
            "question": task["question"],
            "choices": dict(record["per_lineage"], **{NEW: vote["choice"]}),
            "confidences": dict(record["confidences"],
                                **{NEW: vote["probability"]}),
        })
    return out


def decide(rows: list[dict], members: list[str],
           combiner: Combiner | None) -> dict:
    auto, auto_hits, decided, ranked = 0, 0, 0, []
    for row in rows:
        votes = [Vote(name, name, row["choices"][name],
                      row["confidences"].get(name, 0.5)) for name in members
                 if row["choices"].get(name)]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        hit = result.choice == row["expected"]
        answer = result.choice if result.tier == "auto" else None
        if result.tier == "auto":
            auto += 1
            auto_hits += hit
        if answer is None and combiner is not None:
            suggestion = combiner.suggest(
                [(name, row["choices"][name], row["confidences"][name])
                 for name in members], row["candidates"],
                state=row["state"], question=row["question"])
            if suggestion:
                answer = suggestion["choice"]
        decided += answer == row["expected"]
        ranked.append((TIER_ORDER[result.tier], -result.mean_probability, hit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return {"full": round(sum(hit for _, _, hit in ranked) / len(rows), 4),
            "auto": auto,
            "auto_accuracy": round(auto_hits / max(1, auto), 4),
            "decided": round(decided / len(rows), 4), "ranked": ranked}


def fit_combiner(train: list[dict], members: list[str],
                 text_model: TextModel | None) -> Combiner:
    classes = sorted({row["expected"] for row in train})
    index = {name: position for position, name in enumerate(classes)}
    cells = []
    for row in train:
        votes = [(name, row["choices"][name], row["confidences"][name])
                 for name in members if row["choices"].get(name)]
        posterior = None
        if text_model is not None:
            distribution = text_model.posterior(row["state"], row["question"],
                                                classes)
            posterior = [distribution.get(name, 0.0) for name in classes]
        from pandosi.combiner import features
        cells.append(features(votes, classes, posterior))
    x = np.array(cells)
    y = np.array([index[row["expected"]] for row in train])
    weights = np.zeros((x.shape[1], len(classes)))
    onehot = np.zeros((len(y), len(classes)))
    onehot[np.arange(len(y)), y] = 1.0
    for _ in range(600):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights -= 0.5 * (x.T @ (probabilities - onehot) / len(y)
                          + 0.003 * weights)
    return Combiner(classes, weights.tolist(),
                    {"fitted_on": len(train)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "metrics"
                                                / "ensemble-checks"
                                                / "ensemble-1792076803.json"))
    parser.add_argument("--votes", default=str(ROOT / "metrics"
                                               / "ensemble-checks"
                                               / "finetuned-votes.json"))
    parser.add_argument("--suite", default=str(ROOT / "assets"
                                               / "heldout2_test.jsonl"))
    parser.add_argument("--combiner", default=str(ROOT / "assets"
                                                  / "combiner.json"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    votes = {row["task_id"]: row for row
             in json.loads(Path(args.votes).read_text(encoding="utf-8"))["rows"]}
    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    rows = rows_with_votes(report, votes, tasks)
    trio = sorted(report["rows"][0]["per_lineage"])
    members = trio + [NEW]

    shipped = Combiner.load(args.combiner)
    text_model = shipped.text_model
    print(f"{len(rows)} rows; the trio and {NEW}; "
          f"the shipped combiner is spec {shipped.to_json()['spec_version']}")

    print(f"  {'fleet':28s} {'gate':>7s} {'auto':>5s} {'auto acc':>9s} "
          f"{'decided':>8s}")
    results = {}
    for label, subset in (("trio", trio), ("trio + " + NEW, members)):
        plain = decide(rows, subset, None)
        with_shipped = decide(rows, subset, shipped)
        results[label] = {"gate": plain, "shipped_combiner": with_shipped}
        print(f"  {label + ' (gate only)':28s} {plain['full']:>7.4f} "
              f"{plain['auto']:>5d} {plain['auto_accuracy']:>9.4f} "
              f"{plain['full']:>8.4f}")
        print(f"  {label + ' + shipped combiner':28s} "
              f"{with_shipped['full']:>7.4f} {with_shipped['auto']:>5d} "
              f"{with_shipped['auto_accuracy']:>9.4f} "
              f"{with_shipped['decided']:>8.4f}")

    # A combiner fitted for the four-member fleet, on rows it did not see.
    half = len(rows) // 2
    train, test = rows[:half], rows[half:]
    refit = fit_combiner(train, members, text_model)
    trio_refit = fit_combiner(train, trio, text_model)
    print(f"\n  combiners fitted on {len(train)} rows, scored on the other "
          f"{len(test)}:")
    for label, subset, model in (
            ("trio + shipped combiner", trio, shipped),
            ("trio + refitted combiner", trio, trio_refit),
            (f"trio + {NEW} + shipped combiner", members, shipped),
            (f"trio + {NEW} + refitted combiner", members, refit)):
        entry = decide(test, subset, model)
        results[f"heldout:{label}"] = {"gate": {"full": entry["full"],
                                                "auto": entry["auto"]},
                                       "decided": entry["decided"]}
        print(f"    {label:34s} gate {entry['full']:.4f}  "
              f"decided {entry['decided']:.4f}")

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key != "ranked"}
        return value
    out = ROOT / "metrics" / "ensemble-checks" / "improved-fleet.json"
    out.write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
