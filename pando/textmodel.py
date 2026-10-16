"""A cheap second opinion whose errors do not travel with the family's.

Rounds 122-125 priced what a swarm actually buys: error independence. A
member is worth several points even at 57.5% accuracy when its mistakes do
not co-occur with the others', and nothing derived from the neural family
supplied that - a fine-tuned descendant and a reversed ballot each agreed on
100% of answers, a quantised sibling on 80%, all with positive error
correlation.

This is the other kind of member: a multinomial naive Bayes over word tokens
and adjacent pairs, standard library only, trained from labels a swarm
already accumulates and small enough to ship to a device that cannot hold a
1.6 GB graph. It is not a better model. It is a *different* one, which is
the only thing the ensemble pays for.

The emitted probability is raw - the temperature that makes it comparable to
a neural vote belongs in the same place every other lineage keeps it, in
`assets/calibration/<checkpoint>.json`, and is fitted by
`tools/train_text_model.py`.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .common import DecisionRequest, Vote

DEFAULT_CHECKPOINT = "text-nb-v1"
DEFAULT_ALPHA = 1.0


FEATURE_MODES = ("words", "chars", "mixed")


def features(text: str, mode: str = "words") -> list[str]:
    """The member's view of the text.

    `words` is words plus adjacent pairs (the cue phrases live in the pairs);
    `chars` is character trigrams over the normalized text, which sees typo'd
    and inflected forms the word view misses; `mixed` is both. Round 128 asks
    whether a second cheap member is worth a seat, and the honest version of
    that question is whether a *different view* of the same text decorrelates
    its errors - a different hyperparameter on the same view would not.
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode {mode!r}")
    words = re.findall(r"[a-z0-9']+", text.lower())
    if mode == "words":
        return words + [f"{first} {second}"
                        for first, second in zip(words, words[1:])]
    normalized = " ".join(words)
    grams = [normalized[index:index + 3]
             for index in range(len(normalized) - 2)]
    if mode == "chars":
        return grams
    return words + [f"{first} {second}"
                    for first, second in zip(words, words[1:])] + grams


class TextModel:
    """Multinomial naive Bayes over a decision request's text.

    `decide` restricted to the request's candidates, so a model trained on
    one playbook's labels can still serve a question with fewer options.
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT,
                 alpha: float = DEFAULT_ALPHA, feature_mode: str = "words"):
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature mode {feature_mode!r}")
        self.checkpoint = checkpoint
        self.alpha = float(alpha)
        self.feature_mode = feature_mode
        self._counts: dict[str, Counter] = {}
        self._totals: Counter = Counter()
        self._vocabulary = 1

    # -- training ---------------------------------------------------------
    @staticmethod
    def request_text(state: str, question: str) -> str:
        return f"{question} {state}"

    def fit(self, rows: list[dict]) -> dict:
        """Learn class-conditional counts from labeled rows."""
        counts: dict[str, Counter] = {}
        totals: Counter = Counter()
        vocabulary: set[str] = set()
        for row in rows:
            label = row.get("expected")
            if label is None:
                continue
            tokens = features(self.request_text(row.get("state", ""),
                                                row.get("question", "")),
                              self.feature_mode)
            counts.setdefault(label, Counter()).update(tokens)
            totals[label] += len(tokens)
            vocabulary.update(tokens)
        self._counts = counts
        self._totals = totals
        self._vocabulary = max(1, len(vocabulary))
        return {"labels": sorted(counts), "features": self._vocabulary,
                "tokens": sum(totals.values())}

    # -- inference --------------------------------------------------------
    def posterior(self, state: str, question: str,
                  candidates: list[str]) -> dict[str, float]:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        tokens = features(self.request_text(state, question),
                          self.feature_mode)
        logits: dict[str, float] = {}
        for choice in candidates:
            counts = self._counts.get(choice)
            if counts is None:
                continue
            denominator = self._totals[choice] + self.alpha * self._vocabulary
            score = math.log(max(self._totals[choice], 1))
            for token in tokens:
                score += math.log((counts[token] + self.alpha) / denominator)
            logits[choice] = score
        if not logits:
            return {}
        top = max(logits.values())
        weights = {choice: math.exp(score - top)
                   for choice, score in logits.items()}
        total = sum(weights.values())
        return {choice: weight / total for choice, weight in weights.items()}

    def decide(self, request: DecisionRequest) -> Vote:
        probabilities = self.posterior(request.state, request.question,
                                       list(request.candidates))
        if not probabilities:
            # Nothing learned about any candidate: an abstention, expressed
            # as a zero-confidence vote so the gate sees it as one.
            return Vote("text", self.checkpoint, request.candidates[0], 0.0)
        choice = max(sorted(probabilities), key=lambda name: probabilities[name])
        return Vote("text", self.checkpoint, choice,
                    probabilities[choice])

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "alpha": self.alpha,
            "feature_mode": self.feature_mode,
            "vocabulary": self._vocabulary,
            "totals": dict(self._totals),
            "counts": {label: dict(counter)
                       for label, counter in sorted(self._counts.items())},
        }

    def summary(self) -> dict:
        return {"checkpoint": self.checkpoint, "labels": len(self._counts),
                "features": self._vocabulary,
                "tokens": sum(self._totals.values()),
                "feature_mode": self.feature_mode}

    @classmethod
    def from_json(cls, payload: dict) -> "TextModel":
        model = cls(payload.get("checkpoint", DEFAULT_CHECKPOINT),
                    payload.get("alpha", DEFAULT_ALPHA),
                    payload.get("feature_mode", "words"))
        model._counts = {label: Counter(counter) for label, counter
                         in payload.get("counts", {}).items()}
        model._totals = Counter(payload.get("totals", {}))
        model._vocabulary = int(payload.get("vocabulary", 1))
        return model

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), ensure_ascii=False),
                          encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TextModel":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
