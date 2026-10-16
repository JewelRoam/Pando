"""One gate, two loops: apply the swarm's own verdict, identically.

``Coordinator.decide`` returns four things that matter to anyone about to act:
a ``tier`` (``auto`` / ``verify`` / ``escalate``), a ``choice``, a ``responded``
count, and the checkpoints whose confidence scale is unknown
(``uncalibrated_checkpoints``). Two loops consume that verdict --
``pando/autonomy.py`` acts on observations and ``pando/genesis.py`` adopts
proposals -- and they must not drift, because "what the swarm allowed" is a
single safety property, not a per-module preference.

The rule has three parts, and all three are needed:

1. ``tier in allowed_tiers`` -- the swarm's own competence verdict.
2. ``responded >= min_responders`` -- the coordinator sets its quorum to
   ``min(3, len(selected))``, so a lone surviving voter still reports ``auto``
   with agreement 1.0. Gating on the tier alone would let one machine act for
   all (measured, round 98).
3. ``not uncalibrated_checkpoints`` -- the auto thresholds assume a known
   confidence scale. A vote with no temperature table is on an unknown scale,
   so the rule that let it through cannot be trusted (measured, round 100).

This module is the one place that decision is made.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateVerdict:
    """Whether the swarm's verdict authorises acting, and otherwise why not."""

    acted: bool
    tier: str
    choice: str | None
    responded: int
    quorum_size: int
    agreement: float
    mean_probability: float
    uncalibrated: list[str]
    reason: str


class TierPolicy:
    """Turn a ``Coordinator.decide`` result into act / hold + a reason.

    ``allowed_tiers`` is the safety policy: a decision is carried out only when
    the swarm's own verdict is in the allowed set. Everything else is recorded
    as held, with the reason, so an operator can review what the system
    declined to do on its own.
    """

    def __init__(self, allowed_tiers=("auto",), min_responders: int = 1,
                 require_calibration: bool = True):
        self.allowed_tiers = tuple(allowed_tiers)
        self.min_responders = int(min_responders)
        self.require_calibration = bool(require_calibration)

    def evaluate(self, result: dict) -> GateVerdict:
        tier = str(result.get("tier", "escalate"))
        choice = result.get("choice")
        responded = int(result.get("responded", 0))
        uncalibrated = list(result.get("uncalibrated_checkpoints") or [])

        acted = (tier in self.allowed_tiers and choice is not None
                 and responded >= self.min_responders)
        if acted and self.require_calibration and uncalibrated:
            acted = False

        if acted:
            reason = (f"tier {tier!r} allowed with {responded} responders "
                      f">= {self.min_responders}")
        elif self.require_calibration and uncalibrated:
            reason = "uncalibrated checkpoint(s): " + ", ".join(uncalibrated)
        elif tier not in self.allowed_tiers:
            reason = f"tier {tier!r} not in {list(self.allowed_tiers)}"
        elif choice is None:
            reason = "no choice was returned"
        else:
            reason = (f"only {responded} responders; policy requires "
                      f"{self.min_responders}")

        return GateVerdict(
            acted=acted, tier=tier, choice=choice, responded=responded,
            quorum_size=int(result.get("quorum_size", 0)),
            agreement=float(result.get("agreement", 0.0)),
            mean_probability=float(result.get("mean_probability", 0.0)),
            uncalibrated=uncalibrated, reason=reason)
