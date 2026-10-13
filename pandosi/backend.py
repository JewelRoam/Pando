"""Decision backends. MockBackend enables the full protocol without model weights."""

from __future__ import annotations

import hashlib

from .common import DecisionRequest, Vote


class MockBackend:
    """Deterministic pseudo-decision backend for protocol testing.

    The choice is derived from a stable hash of (state, question). When
    ``correct_choice`` is set, the backend answers it with ``agree_rate``
    probability; otherwise it answers the hashed choice. This lets tests
    control agreement across replicas.
    """

    def __init__(self, node_id: str, checkpoint: str = "mock",
                 correct_choice: str | None = None, agree_rate: float = 1.0):
        self.node_id = node_id
        self.checkpoint = checkpoint
        self.correct_choice = correct_choice
        self.agree_rate = agree_rate

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        digest = hashlib.sha256(f"{req.state}|{req.question}".encode()).digest()
        hashed_choice = req.candidates[digest[0] % len(req.candidates)]

        choice = hashed_choice
        probability = 0.55 + (digest[1] % 40) / 100.0  # 0.55 - 0.94
        if self.correct_choice is not None:
            agrees = (digest[2] % 100) / 100.0 < self.agree_rate
            choice = self.correct_choice if agrees else hashed_choice
            probability = 0.9 if agrees else 0.4
        return Vote(self.node_id, self.checkpoint, choice, round(probability, 4))


class LayaBackend:
    """Laya decision-model backend (convaiinnovations/laya, Apache-2.0).

    Requires ``transformers`` and a local checkpoint. Not auto-imported so
    the protocol stack runs dependency-free in mock mode.
    """

    def __init__(self, node_id: str, checkpoint: str = "english"):
        try:
            from laya import load  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pip install laya (or transformers), then retry"
            ) from exc
        self.node_id = node_id
        self.checkpoint = checkpoint
        self._model = load(f"convaiinnovations/laya-{checkpoint}")

    def decide(self, req: DecisionRequest) -> Vote:
        result = self._model.decide(req.state, req.question, req.candidates)
        return Vote(self.node_id, self.checkpoint,
                    result.choice, float(result.probability))
