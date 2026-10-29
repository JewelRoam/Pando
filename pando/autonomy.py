"""Observe -> decide -> act: the half of the loop PandoRSI did not have.

Round 98 measured Laya's built-in act head on eight states through both
ecosystems: it returns 1.0 for everything, so the model cannot decide
*whether* to engage. What it can do is answer under a gate, and the swarm
already computes that gate - the `tier` field (`auto` / `verify` /
`escalate`). This module drives the loop: offer observations to the swarm,
act only when the swarm says it is competent, and record every outcome.

Acting is deliberately pluggable. The loop owns the safety property (never
act outside the allowed tier) and nothing else.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .gate import TierPolicy
from .net import open_url


@dataclass
class Observation:
    """Something the system noticed and might have to decide about."""

    observation_id: str
    state: str
    question: str
    candidates: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """What the loop did with one observation."""

    observation_id: str
    tier: str
    acted: bool
    choice: str | None
    agreement: float
    mean_probability: float
    responded: int
    quorum_size: int
    reason: str


class RootDecider:
    """Ask a live root for a decision; the same call the CLI makes."""

    def __init__(self, root_url: str, pairing_token: str | None = None,
                 timeout: float = 120):
        self.root_url = root_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if pairing_token:
            self._headers["Authorization"] = f"Bearer {pairing_token}"
        self.timeout = timeout

    def __call__(self, observation: Observation) -> dict:
        body = json.dumps({"state": observation.state,
                           "question": observation.question,
                           "candidates": observation.candidates,
                           "request_id": observation.observation_id})
        request = urllib.request.Request(
            f"{self.root_url}/decide", data=body.encode("utf-8"),
            headers=self._headers, method="POST")
        with open_url(request, timeout=self.timeout) as resp:
            return json.loads(resp.read())


class Journal:
    """Append-only record of every outcome, acted or held."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, outcome: Outcome) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(outcome), sort_keys=True) + "\n")


class AutonomyLoop:
    """Turn a decision stream into actions, gated by the swarm's tier.

    ``allowed_tiers`` is the safety policy: a decision is executed only when
    the swarm's own verdict is in the allowed set. Everything else is
    recorded as held, with the reason, so an operator can review what the
    system declined to do on its own.

    ``min_responders`` is the second half of that policy, and it is not
    optional in practice: the coordinator sets its quorum threshold to
    ``min(3, len(selected))``, so a lone surviving voter still reports tier
    ``auto`` with agreement 1.0. Gating on the tier alone would let an
    autonomous loop act on a single machine (measured in round 98).
    """

    def __init__(self, decide, actuator=None, allowed_tiers=("auto",),
                 min_responders: int = 1, require_calibration: bool = True,
                 journal: Journal | None = None):
        self.decide = decide
        self.actuator = actuator
        # Kept as attributes for callers that read them back; the rule itself
        # lives in pando/gate.py so this loop and pando/genesis.py cannot drift.
        self.policy = TierPolicy(allowed_tiers=allowed_tiers,
                                 min_responders=min_responders,
                                 require_calibration=require_calibration)
        self.allowed_tiers = self.policy.allowed_tiers
        self.min_responders = self.policy.min_responders
        self.require_calibration = self.policy.require_calibration
        self.journal = journal
        self.outcomes: list[Outcome] = []

    def step(self, observation: Observation) -> Outcome:
        result = self.decide(observation)
        verdict = self.policy.evaluate(result)
        acted = verdict.acted
        if acted and self.actuator is not None:
            self.actuator(observation, verdict.choice, result)
        outcome = Outcome(
            observation_id=observation.observation_id,
            tier=verdict.tier,
            acted=acted,
            choice=verdict.choice,
            agreement=verdict.agreement,
            mean_probability=verdict.mean_probability,
            responded=verdict.responded,
            quorum_size=verdict.quorum_size,
            reason=verdict.reason,
        )
        self.outcomes.append(outcome)
        if self.journal is not None:
            self.journal.append(outcome)
        return outcome

    def run(self, observations) -> list[Outcome]:
        self._started = time.time()
        return [self.step(observation) for observation in observations]

    def summary(self) -> dict:
        acted = [outcome for outcome in self.outcomes if outcome.acted]
        return {
            "observations": len(self.outcomes),
            "acted": len(acted),
            "held": len(self.outcomes) - len(acted),
            "tiers": {tier: sum(1 for outcome in self.outcomes
                                if outcome.tier == tier)
                      for tier in {o.tier for o in self.outcomes}},
            "elapsed_s": round(time.time() - self._started, 3)
            if getattr(self, "_started", None) else None,
        }
