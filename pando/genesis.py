"""Genesis: the group authors its own agenda.

Round 98/99 measured the swarm honestly: it can choose among options handed to
it (51.8% overall, 92.9% on what it accepts) and it can refuse, but it cannot
find its own decisions — "the agenda must come from outside". ``pando/
autonomy.py`` makes that literal: every ``Observation`` arrives with its
``candidates`` already filled in. The group is an agenda *follower*.

Genesis is the layer that removes the outside hand. Members propose their own
candidate actions from the group's own history, the group scores them against
what worked before, adopts the best, executes it, and writes the outcome back
into the memory that shapes the next round. The options become a function of
the group's own past, not of an operator's menu.

The design bet is not "bigger models". It is that many weak proposers + a
shared memory + cheap recombination + outcome-conditioned selection add up to a
search over strategies — a genetic algorithm over *what to do*, not a bigger
brain. See docs/DESIGN-genesis.md.

This module owns the mechanism and nothing about model quality: ``StubProposer``
(deterministic, no model) proves the loop closes and that its behaviour is a
function of recorded outcomes. Swap in a real proposer only after that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .identity import verify as _verify_signature


# The group's action space: the primitives the swarm actually controls.
# A proposal is a composition and parameterisation of these. If an action is
# not here it is unexecutable words, not a plan (docs/DESIGN-genesis.md §5).
# The set is small and enumerable today; that is the honest boundary.
ACTION_SPACE: dict[str, str] = {
    "seed.export": "write a seed bundle for this member (identity + grants + state)",
    "seed.replicate": "carry a seed bundle to another host",
    "recruit.invite": "issue an invitation to a candidate member",
    "reproduce.spawn": "spawn a replica of an existing member (spore/graft)",
    "policy.set": "change a voting rule, gate threshold, or role grant",
    "culture.refine": "refine/retrain a member, distribute a playbook update",
    "resource.rebalance": "move a member to a different host",
    "meta.propose": "propose a new proposal operator (the seed of Tier 2)",
}


@dataclass
class Proposal:
    """One member's claim about what the group should do next."""

    proposal_id: str
    author: str
    goal: str
    action: str
    expected_effect: str = ""
    cost: float = 0.0
    cites: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        """The content a signature covers (everything but the proof)."""
        return {
            "proposal_id": self.proposal_id,
            "author": self.author,
            "goal": self.goal,
            "action": self.action,
            "expected_effect": self.expected_effect,
            "cost": self.cost,
            "cites": list(self.cites),
        }


@dataclass
class ProposalOutcome:
    """What happened to one adopted proposal. ``success`` is None when not run."""

    proposal_id: str
    action: str
    adopted: bool
    executed: bool
    success: bool | None
    detail: str = ""


class ScoreBoard:
    """Expected instrumental gain, from what similar actions did before.

    This is the gradient, and it is instrumental — not the axiom. Survival is a
    telos, not a measurement (docs/DESIGN-genesis.md §6); what a loop can learn
    from is the record: did this kind of action work last time. A Beta-style
    smoothing keeps an unseen action at the prior (0.5), so with no history
    every action ties and the tie-break decides.
    """

    def __init__(self, prior: float = 0.5, strength: float = 2.0):
        self.prior = float(prior)
        self.strength = float(strength)
        self.successes: dict[str, float] = {}
        self.trials: dict[str, float] = {}

    def observe(self, outcome: ProposalOutcome) -> None:
        if not outcome.executed or outcome.success is None:
            return
        self.trials[outcome.action] = self.trials.get(outcome.action, 0.0) + 1.0
        if outcome.success:
            self.successes[outcome.action] = (
                self.successes.get(outcome.action, 0.0) + 1.0)

    def score(self, action: str) -> float:
        wins = self.successes.get(action, 0.0)
        tries = self.trials.get(action, 0.0)
        return (wins + self.prior * self.strength) / (tries + self.strength)

    def table(self) -> dict[str, float]:
        actions = set(self.successes) | set(self.trials)
        return {action: round(self.score(action), 4)
                for action in sorted(actions)}


class NullScoreBoard(ScoreBoard):
    """Amnesiac control: every action scores the prior, always.

    Used to prove that any behavioural difference between two otherwise
    identical loops comes from the memory and not from the machinery.
    """

    def observe(self, outcome: ProposalOutcome) -> None:
        return

    def score(self, action: str) -> float:
        return self.prior


class StubProposer:
    """Deterministic recombiner: same candidate action set every round.

    It is deliberately dumb. It emits a fixed set of actions and only varies
    its ``cites`` (references to the most recent adopted action), which is the
    recombination hook. The point of holding the candidate set constant is that
    then the *only* thing that can change the loop's behaviour between rounds
    is the recorded outcome — which is exactly what the drill must isolate.
    """

    def __init__(self, actions: list[str], author: str = "stub"):
        self.actions = list(actions)
        self.author = author

    def propose(self, round_no: int, history: dict) -> list[Proposal]:
        last = history["outcomes"][-1]["action"] if history["outcomes"] else None
        cites = [f"r{round_no - 1}:{last}"] if last else []
        return [
            Proposal(
                proposal_id=f"p{round_no}:{action}",
                author=self.author,
                goal=f"keep the group alive: {action}",
                action=action,
                expected_effect=ACTION_SPACE.get(action, "?"),
                cost=0.0,
                cites=list(cites),
            )
            for action in self.actions
        ]


class Journal:
    """Append-only record of every proposed outcome, acted or not."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, outcome: ProposalOutcome) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(outcome), sort_keys=True) + "\n")


@dataclass
class Round:
    """One propose -> score -> adopt -> execute -> remember cycle."""

    round_no: int
    proposals: list[Proposal]
    feasible: list[Proposal]
    adopted: Proposal | None
    outcome: ProposalOutcome


class GenesisLoop:
    """Make the agenda endogenous: the group proposes, scores, adopts, remembers.

    ``proposer`` is any callable ``(round_no, history) -> list[Proposal]``.
    ``actuator`` is a callable ``(Proposal) -> bool`` that returns whether the
    action succeeded; it is the only place the loop touches the world, and it is
    pluggable on purpose.

    ``scoreboard`` is the memory used to rank proposals. Pass a
    ``NullScoreBoard`` to get the amnesiac negative control.

    The loop does not own safety. In the real system the top proposal is still
    gated by the swarm's own tier (``auto`` / ``verify`` / ``escalate``) from
    ``Coordinator.decide`` — exactly as ``pando/autonomy.py`` gates its actions.
    On a single box with no swarm, this loop is the mechanism only.
    """

    def __init__(self, proposer, actuator, scoreboard: ScoreBoard | None = None,
                 action_space: dict[str, str] | None = None,
                 journal: Journal | None = None):
        self.proposer = proposer
        self.actuator = actuator
        self.scoreboard = scoreboard if scoreboard is not None else ScoreBoard()
        self.action_space = dict(action_space or ACTION_SPACE)
        self.journal = journal
        self.proposals: list[Proposal] = []
        self.outcomes: list[ProposalOutcome] = []
        self.rounds: list[Round] = []

    def history(self) -> dict:
        return {
            "rounds": len(self.rounds),
            "proposals": [asdict(p) for p in self.proposals],
            "outcomes": [asdict(o) for o in self.outcomes],
        }

    def step(self) -> Round:
        round_no = len(self.rounds) + 1
        raw = list(self.proposer.propose(round_no, self.history()))
        feasible = [p for p in raw if p.action in self.action_space]
        # Rank by learned instrumental gain, then by action name so the
        # tie-break is deterministic and reproducible.
        ranked = sorted(feasible,
                        key=lambda p: (-self.scoreboard.score(p.action), p.action))
        adopted = ranked[0] if ranked else None

        success: bool | None = None
        detail = "no feasible proposal"
        if adopted is not None:
            success = bool(self.actuator(adopted))
            detail = "actuator reported " + ("success" if success else "failure")
        outcome = ProposalOutcome(
            proposal_id=adopted.proposal_id if adopted else "",
            action=adopted.action if adopted else "",
            adopted=adopted is not None,
            executed=adopted is not None,
            success=success,
            detail=detail,
        )

        self.proposals.extend(raw)
        self.outcomes.append(outcome)
        self.scoreboard.observe(outcome)
        if self.journal is not None:
            self.journal.append(outcome)
        round_result = Round(round_no=round_no, proposals=raw, feasible=feasible,
                             adopted=adopted, outcome=outcome)
        self.rounds.append(round_result)
        return round_result

    def run(self, rounds: int) -> list[Round]:
        return [self.step() for _ in range(rounds)]

    def summary(self) -> dict:
        return {
            "rounds": len(self.rounds),
            "adopted": [r.outcome.action for r in self.rounds],
            "successes": sum(1 for r in self.rounds if r.outcome.success),
            "scoreboard": self.scoreboard.table(),
        }


# --------------------------------------------------------------------------
# Transport: a proposal as a signed, self-describing record.
#
# GenesisLoop works on in-process Proposal objects. For the agenda to be the
# *group's*, a proposal must travel between members and survive without a
# trusted middleman, by the same rule as registration: the public key is the
# claim, the node name is only a label (docs/IDENTITY-SPEC.md). These helpers
# turn a Proposal into a record any transport can carry -- the mailbox today,
# the S2 group log when it lands -- and back, with the signature checked on the
# way in.
# --------------------------------------------------------------------------

PROPOSAL_KIND = "proposal"


def sign_proposal(proposal: Proposal, identity) -> dict:
    """A transferable, signed proposal record: payload + author's proof."""
    payload = proposal.payload()
    return {"kind": PROPOSAL_KIND, "version": 1, "payload": payload,
            **identity.proof(payload)}


def verify_proposal(record, expected_pubkey: str | None = None) -> Proposal | None:
    """Return the Proposal if *record* is a well-formed, signed proposal.

    Returns None (never raises) on anything malformed, unsigned, or tampered
    with, so a caller can filter a mixed log without try/except per entry.

    ``expected_pubkey`` is the pin: when given, the record must carry exactly
    that key. Signature validity alone only proves *someone* signed it; it does
    not prove the author is a member, because anyone can sign under their own
    key. Pinning the member's key is what makes the author name meaningful.
    """
    if not isinstance(record, dict) or record.get("kind") != PROPOSAL_KIND:
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    pubkey = str(record.get("pubkey", ""))
    if expected_pubkey is not None and pubkey != expected_pubkey:
        return None
    if not _verify_signature(pubkey, payload, str(record.get("sig", ""))):
        return None
    try:
        return Proposal(
            proposal_id=str(payload["proposal_id"]),
            author=str(payload["author"]),
            goal=str(payload["goal"]),
            action=str(payload["action"]),
            expected_effect=str(payload.get("expected_effect", "")),
            cost=float(payload.get("cost", 0.0)),
            cites=list(payload.get("cites", [])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def proposals_from_log(entries, pinned: dict[str, str] | None = None
                       ) -> list[Proposal]:
    """Every verifiable proposal in a mixed log, in order; invalid ones dropped.

    ``pinned`` maps node_id -> public key. When given, a proposal whose author
    is a known member must carry that member's key, so a different key cannot
    speak under a member's name.
    """
    found: list[Proposal] = []
    for entry in entries:
        expected = None
        if pinned and isinstance(entry, dict):
            payload = entry.get("payload")
            author = payload.get("author") if isinstance(payload, dict) else None
            if isinstance(author, str):
                expected = pinned.get(author)
        proposal = verify_proposal(entry, expected_pubkey=expected)
        if proposal is not None:
            found.append(proposal)
    return found
