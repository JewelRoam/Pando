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
brain. See docs/design/specs/genesis.md.

This module owns the mechanism and nothing about model quality: ``StubProposer``
(deterministic, no model) proves the loop closes and that its behaviour is a
function of recorded outcomes. Swap in a real proposer only after that.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .gate import TierPolicy
from .identity import verify as _verify_signature


# The group's action space: the primitives the swarm actually controls.
# A proposal is a composition and parameterisation of these. If an action is
# not here it is unexecutable words, not a plan (docs/design/specs/genesis.md §5).
# The set is small and enumerable today; that is the honest boundary.
ACTION_SPACE: dict[str, str] = {
    "seed.export": "write a seed bundle for this member (identity + grants + state)",
    "seed.replicate": "carry a seed bundle to another host",
    "recruit.invite": "issue an invitation to a candidate member",
    "reproduce.spawn": "spawn a replica of an existing member (spore/graft)",
    "policy.set": "change a bounded voting rule or gate threshold (min_adopt, veto)",
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
    # The key that signed this record. Empty on objects built in-process.
    # It is not part of payload(); the signature covers payload only.
    pubkey: str = ""
    # Host name for seed.replicate. Empty stays out of the signed payload so
    # older signatures, which never had the key, still verify.
    destination: str = ""
    # Machine-readable arguments for the action (``policy.set`` names its knobs
    # here and nowhere else). Signed with everything else, and likewise left out
    # of the payload while empty so proposals signed before it existed verify.
    params: dict = field(default_factory=dict)

    def payload(self) -> dict:
        """The content a signature covers (everything but the proof)."""
        body = {
            "proposal_id": self.proposal_id,
            "author": self.author,
            "goal": self.goal,
            "action": self.action,
            "expected_effect": self.expected_effect,
            "cost": self.cost,
            "cites": list(self.cites),
        }
        if self.destination:
            body["destination"] = self.destination
        if self.params:
            body["params"] = dict(self.params)
        return body


@dataclass
class ProposalOutcome:
    """What happened to one adopted proposal. ``success`` is None when not run.

    ``gate_tier`` is the swarm's own verdict for the round that produced this
    outcome (empty when no ``decide`` was wired and the loop chose alone).
    ``detail`` carries the policy's reason whenever the loop held instead of
    acted, so a withheld action is auditable and not merely absent.
    """

    proposal_id: str
    action: str
    adopted: bool
    executed: bool
    success: bool | None
    detail: str = ""
    gate_tier: str = ""
    held_reason: str = ""


class ScoreBoard:
    """Expected instrumental gain, from what similar actions did before.

    This is the gradient, and it is instrumental — not the axiom. Survival is a
    telos, not a measurement (docs/design/specs/genesis.md §6); what a loop can learn
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


class LogProposer:
    """Propose from the group log: candidates are other members' proposals.

    This is the difference between "the loop is clever" and "the agenda is the
    group's". The candidates are not invented by the loop; they are the signed
    proposals members put on the log, verified and pinned on the way in (see
    ``proposals_from_log``). ``source`` is a list of records or a zero-arg
    callable returning one, so a live log that grows between rounds is just
    ``lambda: fetch_recent()``.
    """

    def __init__(self, source, pinned: dict[str, str] | None = None):
        self._source = source
        self.pinned = dict(pinned) if pinned else None

    def propose(self, round_no: int, history: dict) -> list[Proposal]:
        records = self._source() if callable(self._source) else self._source
        return proposals_from_log(records, pinned=self.pinned)


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
    """One propose -> score -> offer -> adopt -> execute -> remember cycle."""

    round_no: int
    proposals: list[Proposal]
    feasible: list[Proposal]
    adopted: Proposal | None
    outcome: ProposalOutcome
    offered: list[Proposal] = field(default_factory=list)
    verdict: dict | None = None


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
                 journal: Journal | None = None, decide=None,
                 policy: TierPolicy | None = None, max_candidates: int | None = None):
        self.proposer = proposer
        self.actuator = actuator
        self.scoreboard = scoreboard if scoreboard is not None else ScoreBoard()
        self.action_space = dict(action_space or ACTION_SPACE)
        self.journal = journal
        # ``decide`` is the swarm, in the shape ``pando/autonomy.py`` already
        # uses: a callable returning Coordinator.decide's dict. ``None`` keeps
        # the mechanism-only mode the drills run (adopt unilaterally), which is
        # fine to measure the gradient and not fine for a live member.
        self.decide = decide
        self.policy = policy if policy is not None else TierPolicy()
        self.max_candidates = max_candidates
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

        offered: list[Proposal] = []
        verdict: dict | None = None
        adopted: Proposal | None = ranked[0] if ranked else None
        held_reason = ""

        if self.decide is not None and ranked:
            # The menu handed to the swarm is the group's own ranked proposals;
            # the swarm names one by proposal_id, and the policy decides whether
            # that verdict is enough to act on. A hold is a recorded outcome,
            # not an absence of one.
            offered = (ranked[:self.max_candidates]
                       if self.max_candidates else ranked)
            verdict = self.decide(round_no, offered, self.history())
            decision = self.policy.evaluate(verdict)
            chosen_id = str(verdict.get("choice") or "")
            picked = next((p for p in offered
                           if p.proposal_id == chosen_id), None)
            if decision.acted and picked is not None:
                adopted = picked
            else:
                adopted = None
                if picked is not None:
                    held_reason = decision.reason
                elif not chosen_id:
                    # No answer at all (e.g. the swarm timed out, responded=0).
                    # Say that, rather than quoting an empty choice back.
                    held_reason = "the swarm returned no choice"
                else:
                    held_reason = (f"swarm chose {chosen_id!r}, "
                                   f"not on the menu")

        success: bool | None = None
        detail = "no feasible proposal"
        if adopted is not None:
            success = bool(self.actuator(adopted))
            detail = "actuator reported " + ("success" if success else "failure")
        elif held_reason:
            detail = "held: " + held_reason
        outcome = ProposalOutcome(
            proposal_id=adopted.proposal_id if adopted else "",
            action=adopted.action if adopted else "",
            adopted=adopted is not None,
            executed=adopted is not None,
            success=success,
            detail=detail,
            gate_tier=str((verdict or {}).get("tier", "")),
            held_reason=held_reason,
        )

        self.proposals.extend(raw)
        self.outcomes.append(outcome)
        self.scoreboard.observe(outcome)
        if self.journal is not None:
            self.journal.append(outcome)
        round_result = Round(round_no=round_no, proposals=raw, feasible=feasible,
                             adopted=adopted, outcome=outcome,
                             offered=offered, verdict=verdict)
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
# claim, the node name is only a label (docs/design/IDENTITY.md). These helpers
# turn a Proposal into a record any transport can carry -- the mailbox today,
# the S2 group log when it lands -- and back, with the signature checked on the
# way in.
# --------------------------------------------------------------------------

PROPOSAL_KIND = "proposal"
REACTION_KIND = "reaction"
# What a decision model may put under a proposal. A stance is a label, like a
# reaction on a message, not another proposal.
REACTION_STANCES = ("赞同", "搁置", "反对")


def sign_reaction(target_id: str, node_id: str, checkpoint: str, stance: str,
                  identity, probability: float | None = None) -> dict:
    """A signed note that one decision model answered one proposal."""
    payload = {
        "target_id": str(target_id),
        "node_id": str(node_id),
        "checkpoint": str(checkpoint),
        "stance": str(stance),
    }
    if probability is not None:
        payload["probability"] = float(probability)
    return {"kind": REACTION_KIND, "version": 1, "payload": payload,
            **identity.proof(payload)}


def verify_reaction(record, expected_pubkey: str | None = None) -> dict | None:
    """Return the reaction payload when the signature holds, else None."""
    if not isinstance(record, dict) or record.get("kind") != REACTION_KIND:
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    stance = str(payload.get("stance") or "")
    target = str(payload.get("target_id") or "")
    node = str(payload.get("node_id") or "")
    if not target or not node or not stance.strip():
        return None
    pubkey = str(record.get("pubkey", ""))
    if expected_pubkey is not None and pubkey != expected_pubkey:
        return None
    if not _verify_signature(pubkey, payload, str(record.get("sig", ""))):
        return None
    return {
        "target_id": target,
        "node_id": node,
        "checkpoint": str(payload.get("checkpoint") or ""),
        "stance": stance,
        "signed": True,
    }




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
            pubkey=pubkey,
            destination=str(payload.get("destination") or ""),
            params=dict(payload.get("params") or {}),
        )
    except (KeyError, TypeError, ValueError):
        return None


def proposals_from_log(entries, pinned: dict[str, str] | None = None
                       ) -> list[Proposal]:
    """Every verifiable proposal in a mixed log, in order; the rest dropped.

    Two shapes are accepted per entry, and the difference between them is the
    trust boundary, not a convenience:

    - a **signed record** (a dict): the untrusted transport form. Its signature
      is checked here, and when ``pinned`` maps its author to a key the record
      must carry exactly that key -- a different key cannot speak under a
      member's name.
    - a ``Proposal``: already past the verification boundary. It was either
      built in this process by a trusted proposer or returned by
      ``pando.proposal_mail.fetch_proposals``, which enforced the pin on the
      way in. A ``Proposal`` may carry the pubkey copied at verification. ``pinned`` still cannot be re-checked from the object alone
      and does not -- which is exactly why the pin has to bite at the fetch.

    Passing ``Proposal`` objects through is what lets the two halves compose:
    ``LogProposer(source=lambda: fetch_proposals(url, pinned=pin))`` is the
    live wiring, and re-verifying an object that has no signature to check
    would silently empty the menu.
    """
    found: list[Proposal] = []
    for entry in entries:
        if isinstance(entry, Proposal):
            found.append(entry)
            continue
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


class SwarmProposalDecider:
    """Ask a live root which of the group's own proposals to adopt.

    The menu handed to the models is the loop's ranked proposals. A stem
    answers with one token, so the question has to carry the meaning: a bare
    proposal_id would be unanswerable, and a numbered menu would put the loop's
    own ordering back in charge. So every candidate is ``proposal_id`` and the
    question names, for each, what the action actually is and who authored it.

    The reply is the coordinator's verdict dict, unchanged -- ``tier``,
    ``choice``, ``responded``, ``agreement``, ``uncalibrated_checkpoints`` --
    so the same ``TierPolicy`` that gates ``pando/autonomy.py`` gates this.
    """

    def __init__(self, root_url: str, pairing_token: str | None = None,
                 timeout: float = 120, max_menu: int = 8):
        from .common import bypass_proxy_for_lan

        bypass_proxy_for_lan()
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({})))
        self.root_url = root_url.rstrip("/")
        self.timeout = timeout
        self.max_menu = max(1, int(max_menu))
        self._headers = {"Content-Type": "application/json"}
        if pairing_token:
            self._headers["Authorization"] = f"Bearer {pairing_token}"

    def render_menu(self, proposals: list[Proposal]) -> tuple[list[str], str]:
        menu = list(proposals)[:self.max_menu]
        lines = [f"{p.proposal_id} = {p.action} (by {p.author})"
                 for p in menu]
        question = ("Which of the group's own proposals should be adopted? "
                    "Answer with one proposal id. Menu: " + "; ".join(lines))
        return [p.proposal_id for p in menu], question

    def __call__(self, round_no: int, proposals: list[Proposal],
                 history: dict) -> dict:
        candidates, question = self.render_menu(proposals)
        body = json.dumps({
            "state": f"genesis-round-{round_no}",
            "question": question,
            "candidates": candidates,
            "request_id": f"genesis-r{round_no}",
        })
        request = urllib.request.Request(
            f"{self.root_url}/decide", data=body.encode("utf-8"),
            headers=self._headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            return json.loads(resp.read())
