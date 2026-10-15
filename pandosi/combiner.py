"""A fitted combiner over the quorum's votes.

Round 139 measured, on 240 wording-held-out rows, that a regularised logistic
regression over the members' vote patterns and calibrated confidences beats
every single member (80.4% against 75.4%) and - the part that matters - is
54.4% accurate on the rows the hand-written gate *declines*, where the best
member reaches 41.2% and the oracle is 67.7%.

This module is the artifact side of that measurement: the feature spec, the
model, and a loader for the file `tools/train_combiner.py` writes. A trained
combiner is advisory by construction - it answers *suggestions* for rows the
gate could not decide, and nothing here may change a tier or a choice. The
safety of the swarm stays with the hand-written thresholds; what improves is
the prior a human or a verification step starts from.

Features are addressed by **lineage**, not by node: a quorum's members depend
on which stems happen to be picked, while the admitted lineages are stable, so
an absent lineage contributes zeros.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

SPEC_VERSION = 1


def features(votes: list[tuple[str, str, float]], classes: list[str],
             lineages: list[str]) -> list[float]:
    """One row of features: per-candidate support, then per-lineage behaviour.

    `votes` is (lineage, choice, calibrated probability) for the members that
    actually answered.
    """
    counts = Counter(choice for _, choice, _ in votes)
    mass: dict[str, float] = {}
    for _, choice, probability in votes:
        mass[choice] = mass.get(choice, 0.0) + probability
    top = counts.most_common(1)
    majority = top[0][0] if top else None
    size = len(votes) or 1
    by_lineage = {lineage: (choice, probability)
                  for lineage, choice, probability in votes}
    cells = [counts.get(name, 0) / size for name in classes]
    cells += [mass.get(name, 0.0) / size for name in classes]
    cells += [(top[0][1] / size) if top else 0.0]
    cells += [0.0 if majority is None or lineage not in by_lineage
              else float(by_lineage[lineage][0] != majority)
              for lineage in lineages]
    cells += [by_lineage[lineage][1] if lineage in by_lineage else 0.0
              for lineage in lineages]
    cells += [len({choice for _, choice, _ in votes}) / size]
    cells += [1.0]
    return cells


class Combiner:
    """Weights plus the class and lineage order they were fitted with."""

    def __init__(self, classes: list[str], lineages: list[str],
                 weights: list[list[float]], meta: dict | None = None):
        self.classes = list(classes)
        self.lineages = list(lineages)
        self.weights = [list(row) for row in weights]
        self.meta = dict(meta or {})

    def suggest(self, votes: list[tuple[str, str, float]],
                candidates: list[str] | None = None) -> dict | None:
        """The fitted rule's answer for these votes, never a decision.

        With `candidates`, the answer is restricted to them: the model is
        fitted on one question schema's labels, and suggesting a label the
        question did not offer is the same contract violation a member's vote
        would be (round 140 caught exactly that in the live path).
        """
        if not votes:
            return None
        cells = features(votes, self.classes, self.lineages)
        width = len(cells)
        scores = [sum(cells[column] * self.weights[column][index]
                      for column in range(width))
                  for index in range(len(self.classes))]
        if candidates:
            allowed = set(candidates)
            if not allowed & set(self.classes):
                return None
            for index, name in enumerate(self.classes):
                if name not in allowed:
                    scores[index] = float("-inf")
        top = max(scores)
        if top == float("-inf"):
            return None
        weights = [math.exp(score - top) for score in scores]
        total = sum(weights)
        best = max(range(len(scores)), key=lambda index: scores[index])
        return {"choice": self.classes[best],
                "probability": round(weights[best] / total, 4)}

    def to_json(self) -> dict:
        return {"spec_version": SPEC_VERSION, "classes": self.classes,
                "lineages": self.lineages,
                "weights": [[round(value, 8) for value in column]
                            for column in self.weights],
                "meta": self.meta}

    @classmethod
    def from_json(cls, payload: dict) -> "Combiner":
        if payload.get("spec_version") != SPEC_VERSION:
            raise ValueError(
                f"combiner artifact spec {payload.get('spec_version')!r} is "
                f"not {SPEC_VERSION}; retrain it")
        return cls(payload["classes"], payload["lineages"],
                   payload["weights"], payload.get("meta"))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=1), "utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Combiner":
        return cls.from_json(json.loads(Path(path).read_text("utf-8")))
