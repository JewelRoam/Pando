"""Is a second independent member worth a seat?

Round 126 shipped one cheap independent member; after the round-127
aggregation fix it is worth about +1.25 points to the swarm. The elastic
premise needs to know whether that is a floor or a step. If the tenth device
adds what the fourth did, scaling is a strategy; if a second independent
member adds nothing, the value was in the *first* independent voice and the
portfolio has to be designed rather than grown.

The candidates differ in what they look at, not in how hard they were tuned:
word and bigram counts (the shipped member), character trigrams, both, and a
nearest-centroid classifier that is a different algorithm on the same
features. The neural trio's votes are read from a recorded report, so no
neural model is re-run and every comparison differs only in the cheap member.

Usage:
    python tools/member_portfolio_check.py --report <ensemble-*.json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.calibrate import apply_temperature  # noqa: E402
from pandosi.common import DecisionRequest, Vote, aggregate  # noqa: E402
from pandosi.textmodel import TextModel, features  # noqa: E402

TEMPERATURES = [0.25, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.0,
                4.0, 6.0]
TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


class Centroid:
    """Nearest class centroid in token space - a different algorithm on the
    same features, included so a gain cannot be credited to naive Bayes."""

    def __init__(self, checkpoint: str = "text-centroid-v1"):
        self.checkpoint = checkpoint
        self.centroids: dict[str, Counter] = {}

    def fit(self, rows: list[dict]) -> None:
        sums: dict[str, Counter] = {}
        counts: Counter = Counter()
        for row in rows:
            label = row.get("expected")
            if label is None:
                continue
            tokens = Counter(features(row.get("state", "") + " "
                                      + row.get("question", "")))
            sums.setdefault(label, Counter()).update(tokens)
            counts[label] += 1
        self.centroids = {label: Counter({token: value / counts[label]
                                          for token, value in sums[label].items()})
                          for label in sums}

    def posterior(self, state: str, question: str,
                  candidates: list[str]) -> dict[str, float]:
        tokens = Counter(features(f"{question} {state}"))
        norm = math.sqrt(sum(value * value for value in tokens.values())) or 1.0
        scores = {}
        for choice in candidates:
            centroid = self.centroids.get(choice)
            if centroid is None:
                continue
            denominator = math.sqrt(sum(value * value
                                        for value in centroid.values())) or 1.0
            dot = sum(value * centroid.get(token, 0.0)
                      for token, value in tokens.items())
            scores[choice] = dot / (norm * denominator)
        if not scores:
            return {}
        top = max(scores.values())
        weights = {choice: math.exp((score - top) * 6.0)
                   for choice, score in scores.items()}
        total = sum(weights.values())
        return {choice: weight / total for choice, weight in weights.items()}

    def decide(self, request: DecisionRequest) -> Vote:
        probabilities = self.posterior(request.state, request.question,
                                       list(request.candidates))
        if not probabilities:
            return Vote("text", self.checkpoint, request.candidates[0], 0.0)
        choice = max(sorted(probabilities), key=lambda name: probabilities[name])
        return Vote("text", self.checkpoint, choice, probabilities[choice])


def fit_temperature(model, rows: list[dict]) -> float:
    entries = []
    for row in rows:
        vote = model.decide(DecisionRequest(state=row.get("state", ""),
                                            question=row.get("question", ""),
                                            candidates=row["candidates"]))
        entries.append((vote.probability, vote.choice == row["expected"]))
    best, temperature = None, 1.0
    for value in TEMPERATURES:
        loss = 0.0
        for probability, correct in entries:
            calibrated = min(max(apply_temperature(probability, value), 1e-7),
                             1 - 1e-7)
            loss -= math.log(calibrated if correct else 1 - calibrated)
        if best is None or loss < best:
            best, temperature = loss, value
    return temperature


def phi(rows: list[dict], left: str, right: str) -> float | None:
    left_errors = [row["per_lineage"].get(left) != row["expected"]
                   for row in rows]
    right_errors = [row["per_lineage"].get(right) != row["expected"]
                    for row in rows]
    both = sum(1 for a, b in zip(left_errors, right_errors) if a and b)
    neither = sum(1 for a, b in zip(left_errors, right_errors)
                  if not a and not b)
    left_only = sum(1 for a, b in zip(left_errors, right_errors) if a and not b)
    right_only = sum(1 for a, b in zip(left_errors, right_errors)
                     if b and not a)
    denominator = math.sqrt((neither + left_only) * (right_only + both)
                            * (neither + right_only) * (left_only + both))
    return (round((neither * both - left_only * right_only) / denominator, 4)
            if denominator else None)


def judge(rows: list[dict], members: list[str]) -> dict:
    return _judge(rows, members, None)


def weighted_decide(votes: list[Vote], weights: dict,
                    mode: str) -> tuple[str | None, str]:
    """Choose by weighted support instead of by counting heads.

    The shipped rule picks the most-voted choice, so four weak members outvote
    three strong ones - that is what the scaling curve measures. `mode="mass"`
    weights each vote by its calibrated probability (the rule round 118 found
    best on disagreements); `mode="competence"` weights each lineage by its
    accuracy on a labeled probe set, which is the same kind of quantity an
    admission decision already needs.

    The safety rule is untouched in both: the tier still requires a majority
    of *heads* behind the winner and a healthy mean probability, so a
    weighted choice can never auto-execute on support the count does not
    have.
    """
    counts = Counter(vote.choice for vote in votes)
    top = counts.most_common()
    if not top:
        return None, "escalate"
    if mode == "mass":
        ranking = {}
        for vote in votes:
            ranking[vote.choice] = (ranking.get(vote.choice, 0.0)
                                    + vote.probability)
    else:
        ranking = {}
        for vote in votes:
            ranking[vote.choice] = (ranking.get(vote.choice, 0.0)
                                    + weights.get(vote.checkpoint, 1.0))
    ordered = sorted(ranking.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None, "verify"
    winner = ordered[0][0]
    agreement = counts[winner] / len(votes)
    backers = [vote.probability for vote in votes if vote.choice == winner]
    mean_probability = sum(backers) / len(backers)
    tier = ("auto" if agreement >= 0.6 and mean_probability >= 0.6
            else "verify")
    return winner, tier


def _judge(rows: list[dict], members: list[str],
           weights: dict | None, mode: str = "competence") -> dict:
    answered, hits, auto_hits, ranked = 0, [], [], []
    auto = 0
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        if weights is None:
            result = aggregate(votes, min_quorum=min(3, len(votes)))
            choice, tier, mean_probability = (result.choice, result.tier,
                                              result.mean_probability)
        else:
            choice, tier = weighted_decide(votes, weights, mode)
            backers = [vote.probability for vote in votes
                       if vote.choice == choice]
            mean_probability = (sum(backers) / len(backers) if backers
                                else 0.0)
        hit = choice == row["expected"]
        hits.append(hit)
        answered += choice is not None
        if tier == "auto":
            auto += 1
            auto_hits.append(hit)
        ranked.append((TIER_ORDER[tier], -mean_probability, hit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return {
        "members": members,
        "full": round(sum(hits) / len(rows), 4),
        "answered": answered,
        "auto": auto,
        "accuracy_where_auto": (round(sum(auto_hits) / auto, 4) if auto
                                else None),
        "hits": hits,
        "ranked_hits": ranked,
    }


def paired_delta(left: list[bool], right: list[bool], trials: int,
                 seed: int) -> dict:
    """Bootstrap the difference between two configurations on the same rows."""
    rng = random.Random(seed)
    total = len(left)
    deltas = []
    for _ in range(trials):
        sample = [rng.randrange(total) for _ in range(total)]
        deltas.append(sum(left[index] - right[index] for index in sample)
                      / total)
    deltas.sort()
    return {
        "mean": round(sum(deltas) / trials, 4),
        "low": round(deltas[int(0.025 * trials)], 4),
        "high": round(deltas[int(0.975 * trials)], 4),
        "share_positive": round(sum(1 for value in deltas if value > 0)
                                / trials, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train", default=str(ROOT / "assets"
                                              / "big_dev.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "assets"
                                              / "heldout_test.jsonl"))
    parser.add_argument("--holdout", type=int, default=48)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    def load(path):
        return [json.loads(line) for line
                in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    train = load(args.train)
    tasks = {task["task_id"]: task for task in load(args.test)}
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    trio = sorted(report["rows"][0]["per_lineage"])
    holdout = train[-args.holdout:]
    fitted = train[:-args.holdout]

    members = {"shipped-words": TextModel.load(
        ROOT / "assets" / "textmodel.json")}
    for mode in ("chars", "mixed"):
        model = TextModel(checkpoint=f"text-{mode}-v1", feature_mode=mode)
        model.fit(fitted)
        members[mode] = model
    for name, model in members.items():
        model._temperature = fit_temperature(model, holdout)
    centroid = Centroid()
    centroid.fit(fitted)
    members["centroid"] = centroid
    members["centroid"]._temperature = fit_temperature(centroid, holdout)

    rows = []
    for record in report["rows"]:
        task = tasks[record["task_id"]]
        row = {"expected": record["expected"],
               "per_lineage": dict(record["per_lineage"]),
               "confidences": dict(record["confidences"])}
        for name, model in members.items():
            vote = model.decide(DecisionRequest(state=task["state"],
                                                question=task["question"],
                                                candidates=task["candidates"]))
            row["per_lineage"][name] = vote.choice
            row["confidences"][name] = apply_temperature(
                vote.probability, model._temperature)
        rows.append(row)

    # Competence weights from a labeled probe set the operator already has:
    # each lineage's accuracy on the dev holdout. The neural trio is scored on
    # the same rows from the dev report.
    dev_report = json.loads((ROOT / "metrics" / "ensemble-checks"
                             / "ensemble-1792062150.json").read_text(
                                 encoding="utf-8"))
    probe = dev_report["rows"][-(args.holdout):]
    weights = {}
    for name in trio:
        weights[name] = sum(1 for record in probe
                            if record["per_lineage"][name]
                            == record["expected"]) / len(probe)
    for name, model in members.items():
        hits = sum(1 for row in holdout
                   if model.decide(DecisionRequest(
                       state=row.get("state", ""),
                       question=row.get("question", ""),
                       candidates=row["candidates"])).choice
                   == row["expected"])
        weights[name] = hits / len(holdout)
    print("  competence weights from the dev holdout: "
          + ", ".join(f"{name} {value:.3f}"
                      for name, value in sorted(weights.items())))

    print(f"{len(rows)} held-out rows; trio {trio}")
    print(f"  {'member':16s} {'accuracy':>9s} {'T':>5s}  phi vs the trio")
    for name, model in members.items():
        hits = sum(1 for row in rows
                   if row["per_lineage"][name] == row["expected"])
        correlations = [phi(rows, lineage, name) for lineage in trio]
        print(f"  {name:16s} {hits / len(rows):9.4f} "
              f"{model._temperature:5.2f}  "
              + " ".join(f"{value:+.3f}" if value is not None else "  n/a"
                         for value in correlations)
              + f"   mean {sum(v for v in correlations if v is not None) / len(correlations):+.3f}")

    print(f"  {'trio + ...':16s} {'full':>7s} {'auto':>5s} "
          f"{'auto acc':>9s} {'at trio':>8s}")
    names = list(members)
    print("\n  pairwise phi over every member (the additions' resemblance to "
          "each other is what compounds or cancels):")
    every = trio + names
    print(f"    {'':18s}" + "".join(f"{name[:10]:>12s}" for name in every))
    for left in every:
        cells = "".join(
            f"{phi(rows, left, right):+12.3f}" if right != left else
            f"{'--':>12s}" for right in every)
        print(f"    {left:18s}{cells}")

    base = judge(rows, trio)
    print(f"  {'(none)':16s} {base['full']:7.4f} {base['auto']:5d} "
          f"{base['accuracy_where_auto']:9.4f} "
          f"{_at(base, base['auto']):8.4f}")
    table = {}
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(names, size):
            entry = judge(rows, trio + list(subset))
            table["+".join(subset)] = {
                key: value for key, value in entry.items()
                if key != "ranked_hits"}
            if size <= 2:
                print(f"  {'+'.join(subset):16s} {entry['full']:7.4f} "
                      f"{entry['auto']:5d} {entry['accuracy_where_auto']:9.4f} "
                      f"{_at(entry, base['auto']):8.4f}")
    ranked = sorted(table.items(), key=lambda item: -item[1]["full"])
    print("\n  best portfolios by full-coverage accuracy:")
    for name, entry in ranked[:5]:
        print(f"    {name:34s} full {entry['full']:.4f}  auto "
              f"{entry['auto']:3d} at {entry['accuracy_where_auto']:.4f}")

    print(f"\n  scaling curve, paired against the trio "
          f"({args.trials} bootstrap resamples):")
    print(f"    {'members':24s} {'full':>7s} {'95% CI':>18s} "
          f"{'vs trio':>18s} {'better':>7s}")
    scales, by_size = {}, {}
    entries = {"(none)": base, **{name: judge(rows, trio + list(name.split("+")))
                                  for name in table}}
    for name, entry in entries.items():
        size = 0 if name == "(none)" else len(name.split("+"))
        delta = ({"mean": 0.0, "low": 0.0, "high": 0.0,
                  "share_positive": 0.0} if size == 0 else
                 paired_delta(entry["hits"], base["hits"], args.trials,
                              args.seed))
        scales[name] = {"size": size, "full": entry["full"],
                        "delta_vs_trio": delta}
        by_size.setdefault(size, []).append((entry["full"], name, delta))
    for size in sorted(by_size):
        best = max(by_size[size])
        for accuracy, name, delta in sorted(by_size[size], reverse=True):
            mark = " *" if name == best[1] else ""
            print(f"    {name:24s} {accuracy:7.4f} "
                  f"[{delta['low']:+.4f},{delta['high']:+.4f}] "
                  f"{delta['mean']:+.4f}  {delta['share_positive']:6.1%}{mark}")

    summary = {"report": str(args.report), "trio": trio, "rows": len(rows),
               "baseline": {k: v for k, v in base.items()
                            if k != "ranked_hits"},
               "portfolios": {name: {k: v for k, v in entry.items()
                                     if k not in ("ranked_hits", "hits")}
                              for name, entry in table.items()},
               "scaling": scales, "weights": weights}
    weighted_table = {}
    for mode in ("mass", "competence"):
        print(f"\n  the same portfolios, choosing by {mode}-weighted support "
              f"(heads still gate auto):")
        print(f"    {'members':34s} {'heads':>7s} {mode:>10s} {'delta':>8s}")
        weighted_table[mode] = {"(none)": _judge(rows, trio, weights, mode)}
        for name in table:
            weighted_table[mode][name] = _judge(rows, trio + name.split("+"),
                                                weights, mode)
        for name in sorted(weighted_table[mode],
                           key=lambda key: -weighted_table[mode][key]["full"]):
            heads = base["full"] if name == "(none)" else table[name]["full"]
            entry = weighted_table[mode][name]
            print(f"    {name:34s} {heads:7.4f} {entry['full']:10.4f} "
                  f"{entry['full'] - heads:+8.4f}")
    summary["weighted"] = {
        mode: {name: {k: v for k, v in entry.items()
                      if k not in ("ranked_hits", "hits")}
               for name, entry in entries_.items()}
        for mode, entries_ in weighted_table.items()}
    out = ROOT / "metrics" / "ensemble-checks" / "member-portfolio.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


def _at(entry: dict, coverage: int) -> float:
    window = entry["ranked_hits"][:coverage]
    return sum(hit for _, _, hit in window) / len(window)


if __name__ == "__main__":
    raise SystemExit(main())
