"""Shared message schemas and helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class DecisionRequest:
    """A single typed question over a state."""

    state: str
    question: str
    candidates: list[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "DecisionRequest":
        data = json.loads(raw)
        return cls(
            state=data["state"],
            question=data["question"],
            candidates=data.get("candidates", []),
            request_id=data.get("request_id", ""),
        )


@dataclass
class Vote:
    """One replica's answer to a DecisionRequest."""

    node_id: str
    checkpoint: str
    choice: str
    probability: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class DecisionResult:
    """Aggregated quorum output with confidence gating."""

    choice: str | None
    agreement: float
    mean_probability: float
    tier: str  # auto | verify | escalate
    votes: list[dict] = field(default_factory=list)


def aggregate(votes: list[Vote], min_quorum: int = 3) -> DecisionResult:
    """Aggregate votes into a gated decision.

    Tiers:
      auto     - strict majority and healthy mean probability
      verify   - majority but weak confidence, or tie
      escalate - quorum not met
    """
    if len(votes) < min_quorum:
        return DecisionResult(None, 0.0, 0.0, "escalate", [v.to_json() for v in votes])

    counts: dict[str, int] = {}
    probs: dict[str, list[float]] = {}
    for v in votes:
        counts[v.choice] = counts.get(v.choice, 0) + 1
        probs.setdefault(v.choice, []).append(v.probability)

    best_choice = max(counts, key=counts.get)
    top = counts[best_choice]
    agreement = top / len(votes)
    mean_probability = sum(probs[best_choice]) / len(probs[best_choice])

    # Tie between the two most-voted choices is not a majority.
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) > 1 and sorted_counts[0] == sorted_counts[1]:
        return DecisionResult(None, agreement, mean_probability, "verify",
                              [v.to_json() for v in votes])

    if agreement >= 0.6 and mean_probability >= 0.6:
        tier = "auto"
    else:
        tier = "verify"
    return DecisionResult(best_choice, agreement, mean_probability, tier,
                          [v.to_json() for v in votes])


def bypass_proxy_for_lan() -> None:
    """All PandoSI traffic is LAN-local; a system/env proxy must never route
    it. urllib on Windows honours registry proxies, which black-hole
    loopback/LAN addresses (observed: register POSTs retrying forever)."""
    import os
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except OSError:
        lan_ip = "127.0.0.1"
    bypass = {"localhost", "127.0.0.1", lan_ip}
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        merged = ",".join(sorted(set(filter(None, current.split(","))) | bypass))
        os.environ[var] = merged
