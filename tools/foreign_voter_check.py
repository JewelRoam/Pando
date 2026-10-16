"""Does a voter from a different model family earn its seat?

Round 118 left the swarm's ceiling at the members: 13.5 accuracy points
between the three lineages against 1.0 point between every combiner, errors
correlated at phi up to +0.50, and no reweighting of the same votes reaching
the 91.7% the trio holds on its disagreements. Round 119 then showed that
another *checkpoint of the same family* is not a new opinion: rlcd-v7 agreed
with its parent on 240 of 240 tasks.

So the premise's remaining form is "add a member whose errors do not travel
with the others". Two voters from outside the neural family are scored here,
beside the three lineages' already-recorded votes, so no neural model is
re-run and the only thing that changes between the two ensembles is the
fourth member:

  bayes  multinomial naive Bayes over tokens and bigrams, fitted on dev
  rules  zero-shot keyword matching, nothing fitted

The pair matters because they fail differently. `bayes` learns the dev
split's phrasing, and 100 of the test split's 102 distinct first sentences
appear verbatim in dev, so its held-out score measures cue memorisation
rather than generalisation. `rules` cannot exploit that, which is why its
dev and test scores are printed side by side: a gap there is leakage, and no
gap is the honest number to reason from.

The divergence table is also the metric an admission rule would need: a
candidate that never disagrees with an admitted lineage is a copy wearing a
new name.

Usage:
    python tools/foreign_voter_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from combiner_check import MEMBER_PREFIX, choice_for, ranking  # noqa: E402
from pando.common import Vote, aggregate  # noqa: E402

ALPHA = 1.0

# Written from what a support mailbox looks like, not from the generator's
# cue list: the point of this voter is that it was never fitted on the suite,
# so a dev/test accuracy gap would expose it if it were exploiting phrasing.
RULES = {
    "refund": ["money back", "refund", "reimburse", "return my payment",
               "credit my account"],
    "chargeback": ["chargeback", "dispute", "contest the payment", "bank will",
                   "card issuer", "reverse this transaction"],
    "exchange": ["swap", "exchange", "replacement", "different unit",
                 "another size", "newer model"],
    "complaint": ["unacceptable", "terrible", "disappointed", "worst",
                  "awful", "frustrated", "a mess"],
    "praise": ["fantastic", "excellent", "great", "superb", "impressed",
               "brilliant", "well done"],
    "greeting": ["hello", "good morning", "greetings", "saying hi",
                 "how are you", "hope you are well"],
    "login_issue": ["sign in", "login", "password", "locked out",
                    "two-factor", "log in"],
    "invoice": ["invoice", "receipt", "billing", "vat", "tax document",
                "statement"],
}


def tokens(text: str) -> list[str]:
    """Lowercase words plus the adjacent pairs, which is where the cue
    phrases live ('money back', 'sign in')."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def fit(rows: list[dict]) -> dict:
    """Class-conditional token counts over labeled rows."""
    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        features = tokens(row["state"])
        counts[row["expected"]].update(features)
        totals[row["expected"]] += len(features)
        vocabulary.update(features)
    return {"counts": counts, "totals": totals,
            "vocabulary": len(vocabulary), "classes": sorted(counts)}


def posterior(model: dict, text: str, candidates: list[str]) -> dict:
    features = tokens(text)
    scores = {}
    for choice in candidates:
        counts = model["counts"].get(choice)
        if counts is None:
            continue
        total = model["totals"][choice]
        denominator = total + ALPHA * model["vocabulary"]
        score = math.log(max(counts.total(), 1))
        for feature in features:
            score += math.log((counts[feature] + ALPHA) / denominator)
        scores[choice] = score
    if not scores:
        return {}
    top = max(scores.values())
    weights = {choice: math.exp(score - top)
               for choice, score in scores.items()}
    total = sum(weights.values())
    return {choice: weight / total for choice, weight in weights.items()}


def vote_row(task: dict, model: dict) -> dict:
    """One `bayes` vote in the recorded-row shape."""
    probabilities = posterior(model, task["state"], task["candidates"])
    if not probabilities:
        return {"choice": None, "probability": 0.0}
    choice = max(sorted(probabilities), key=lambda name: probabilities[name])
    return {"choice": choice, "probability": round(probabilities[choice], 4)}


def rule_vote(task: dict) -> dict:
    """One `rules` vote: the candidate with the most keyword hits, longest
    phrase breaking ties, and a confidence in proportion to its share."""
    text = task["state"].lower()
    scores = {}
    for choice in task["candidates"]:
        phrases = RULES.get(choice, [])
        scores[choice] = sum(1 for phrase in phrases if phrase in text)
    total = sum(scores.values())
    best = max(scores.values())
    if total == 0 or best == 0:
        return {"choice": None, "probability": 0.0}
    winners = sorted(name for name, score in scores.items() if score == best)
    choice = winners[0]
    return {"choice": choice, "probability": round(best / total, 4)}


def row_for(task: dict, recorded: dict, foreign: dict, name: str) -> dict:
    per_lineage = dict(recorded["per_lineage"])
    confidences = dict(recorded["confidences"])
    per_lineage[name] = foreign["choice"]
    confidences[name] = foreign["probability"]
    votes = [Vote(name, name, choice, confidences[name])
             for name, choice in per_lineage.items() if choice is not None]
    result = aggregate(votes, min_quorum=min(3, len(votes)))
    return {"task_id": task["task_id"], "expected": task["expected"],
            "per_lineage": per_lineage, "confidences": confidences,
            "distinct": len({choice for choice in per_lineage.values()
                             if choice is not None}),
            "tier": result.tier, "agreement": result.agreement,
            "mean_probability": result.mean_probability,
            "swarm_choice": result.choice}


def errors(rows: list[dict], name: str) -> list[bool]:
    return [row["per_lineage"].get(name) != row["expected"] for row in rows]


def divergence(rows: list[dict], left: str, right: str) -> dict:
    left_errors, right_errors = errors(rows, left), errors(rows, right)
    both = sum(1 for a, b in zip(left_errors, right_errors) if a and b)
    neither = sum(1 for a, b in zip(left_errors, right_errors)
                  if not a and not b)
    left_only = sum(1 for a, b in zip(left_errors, right_errors) if a and not b)
    right_only = sum(1 for a, b in zip(left_errors, right_errors)
                     if b and not a)
    total = len(rows)
    denominator = math.sqrt((neither + left_only) * (right_only + both)
                            * (neither + right_only) * (left_only + both))
    return {"disagreement": round((left_only + right_only) / total, 4),
            "phi": round((neither * both - left_only * right_only)
                         / denominator, 4) if denominator else None}


def majority(row: dict, members: list[str]) -> tuple[str | None, int]:
    """Plain count over the named members - the only rule here that is
    immune to a member whose probabilities live on a different scale."""
    counts = Counter(row["per_lineage"].get(name) for name in members
                     if row["per_lineage"].get(name))
    if not counts:
        return None, 0
    top = counts.most_common()
    margin = top[0][1] - (top[1][1] if len(top) > 1 else 0)
    return (top[0][0] if margin > 0 else None), margin


def rank_value(row: dict, rule: str, members: list[str]) -> tuple:
    if rule == "majority":
        _, margin = majority(row, members)
        return (margin, row.get("mean_probability") or 0.0)
    return (ranking(row, rule, ""),)


def curve(rows: list[dict], rule: str, members: list[str],
          coverage: int) -> dict:
    order = sorted(rows, key=lambda row: rank_value(row, rule, members),
                   reverse=True)
    hit = [choice_for_row(row, rule, members) == row["expected"]
           for row in order]
    return {"full": round(sum(hit) / len(hit), 4),
            "at_coverage": round(sum(hit[:coverage]) / coverage, 4),
            "hits": hit}


def choice_for_row(row: dict, rule: str, members: list[str]) -> str | None:
    if rule == "majority":
        return majority(row, members)[0]
    return choice_for(row, rule, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "assets"
                                               / "big_dev.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "assets"
                                              / "big_test.jsonl"))
    parser.add_argument("--report", required=True,
                        help="the recorded neural votes for the test split")
    parser.add_argument("--coverage", type=int, default=187,
                        help="the count the shipped gate accepted")
    args = parser.parse_args()

    def load(path):
        return [json.loads(line) for line
                in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    train = load(args.train)
    test = {task["task_id"]: task for task in load(args.test)}
    recorded = json.loads(Path(args.report).read_text(encoding="utf-8"))
    neural = [name for name in recorded["rows"][0]["per_lineage"]]
    model = fit(train)

    voters = {
        "bayes": lambda task, record: vote_row(task, model),
        "rules": lambda task, record: rule_vote(task),
    }
    # The control that isolates correlation from accuracy. `shadow` is
    # typed-decisions with its misses repaired by the rules voter, so it is at
    # least as accurate as typed-decisions *and* its errors are a subset of
    # typed-decisions' errors - phi near +1 by construction, on the same
    # confidence scale. If accuracy were what the ensemble needs, shadow
    # should help at least as much as rules.
    def shadow(task: dict, record: dict) -> dict:
        base = record["per_lineage"]["typed-decisions"]
        if base == record["expected"]:
            return {"choice": base,
                    "probability": record["confidences"]["typed-decisions"]}
        fixed = rule_vote(task)["choice"]
        return {"choice": fixed or base,
                "probability": record["confidences"]["typed-decisions"]}

    voters["shadow"] = shadow
    print(f"fitted bayes on {len(train)} dev rows ({model['vocabulary']} "
          f"features); {len(recorded['rows'])} test rows")
    print("  voter accuracy, and the dev/test gap that would expose leakage")
    score = {}
    by_id = {record["task_id"]: record for record in recorded["rows"]}
    for name, voter in voters.items():
        test_rows = [test[row["task_id"]] for row in recorded["rows"]]
        test_hits = sum(1 for task in test_rows
                        if voter(task, by_id[task["task_id"]])["choice"]
                        == task["expected"])
        entry = {"test": round(test_hits / len(test_rows), 4)}
        if name in ("bayes", "rules"):
            dev_hits = sum(1 for task in train
                           if voter(task, None)["choice"] == task["expected"])
            entry["dev"] = round(dev_hits / len(train), 4)
        score[name] = entry
        dev_text = (f"dev {entry['dev']:.4f}  gap "
                    f"{entry['dev'] - entry['test']:+.4f}"
                    if "dev" in entry else "dev n/a (uses recorded row)")
        print(f"    {name:8s} test {entry['test']:.4f}  {dev_text}")

    three = recorded["rows"]
    rules = [MEMBER_PREFIX + name for name in neural] + \
            ["majority", "margin", "agreement", "gate"]
    print(f"  matched at the {args.coverage} rows the shipped gate accepts")
    print(f"    {'rule':26s} {'three':>8s} " +
          " ".join(f"{'+' + name:>9s}" for name in voters) +
          "   (full coverage)")
    curves = {}
    for rule in rules:
        row = {"three": curve(three, rule, neural, args.coverage)}
        cells = f"    {rule:26s} {row['three']['at_coverage']:8.4f} "
        fulls = f"{row['three']['full']:.4f}"
        for name, voter in voters.items():
            four = [row_for(test[record["task_id"]], record,
                            voter(test[record["task_id"]], record), name)
                    for record in three]
            row[name] = curve(four, rule, neural + [name], args.coverage)
            cells += f"{row[name]['at_coverage']:9.4f} "
            fulls += f" -> {row[name]['full']:.4f}"
        curves[rule] = row
        print(cells + f"   ({fulls})")

    divergences = {}
    for name, voter in voters.items():
        four = [row_for(test[record["task_id"]], record,
                        voter(test[record["task_id"]], record), name)
                for record in three]
        divergences[name] = {lineage: divergence(four, lineage, name)
                             for lineage in neural}
        print(f"  {name} divergence from each lineage "
              f"(phi > 0: errors together)")
        for lineage, entry in divergences[name].items():
            print(f"    {lineage:20s} disagreement "
                  f"{entry['disagreement']:.3f}  phi {entry['phi']:+.3f}")

    summary = {
        "train": Path(args.train).name,
        "test": Path(args.test).name,
        "report": str(args.report),
        "neural": neural,
        "voter_accuracy": score,
        "divergence": divergences,
        "curves": curves,
    }
    out = ROOT / "metrics" / "ensemble-checks" / "foreign-voter.json"
    out.write_text(json.dumps(
        {**summary,
         "curves": {rule: {key: {k: v for k, v in entry.items()
                                 if k != "hits"}
                           for key, entry in row.items()}
                    for rule, row in curves.items()}},
        indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
