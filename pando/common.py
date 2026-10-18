"""Shared message schemas and helpers."""

from __future__ import annotations

# What a member is for. A device usually holds both kinds: a decision model
# that answers proposals, and an LLM that writes them. They are not the same
# job, so one member never does both: the LLM does not also hold a vote, and
# a decision model does not write proposals.
DECIDING_ROLE = "decision"
PROPOSING_ROLE = "proposal"
ROLES = (DECIDING_ROLE, PROPOSING_ROLE)

import json
import hashlib
import uuid
from dataclasses import asdict, dataclass, field


def canonical_question(question: str,
                       candidates: list[str]) -> tuple[str, list[str]]:
    """Stable id and option order for the same logical decision schema.

    The prompt name alone is not a schema: two requests named ``intent`` can
    carry different option sets. The digest prevents RLCD training from
    merging incompatible label spaces.
    """
    base = (question or "decision").strip() or "decision"
    options = sorted({str(candidate) for candidate in candidates})
    if not options:
        return base, options
    digest = hashlib.sha256("\x1f".join(options).encode("utf-8")).hexdigest()
    return f"{base}.{digest[:8]}", options


@dataclass
class DecisionRequest:
    """A single typed question over a state."""

    state: str
    question: str
    candidates: list[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    family: str | None = None

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
            family=data.get("family"),
        )


@dataclass
class Vote:
    """One replica's answer to a DecisionRequest.

    ``neutral`` marks a choice-only vote that carries no confidence signal.
    Its choice still counts toward agreement and quorum, but its compatibility
    probability is excluded from confidence aggregation and calibration.
    """

    node_id: str
    checkpoint: str
    choice: str
    probability: float
    neutral: bool = False

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


def aggregate(votes: list[Vote], min_quorum: int = 3,
              auto_min_agreement: float = 0.6,
              auto_min_probability: float = 0.6,
              count_lineages: bool = False,
              blocs: dict | None = None) -> DecisionResult:
    """Aggregate votes into a gated decision.

    Tiers:
      auto     - strict majority and healthy mean probability
      verify   - majority but weak confidence, or tie
      escalate - quorum not met

    With count_lineages the quorum is counted over *lineages* rather than
    stems: same-lineage members are reduced to one vote first (round 119
    measured a same-lineage pair agreeing on 240 of 240 tasks, so a
    replicated lineage would otherwise outvote a distinct one 2:1 on
    evidence it does not have). min_quorum then applies to lineages too.

    `blocs` generalises that from a name to a group: {checkpoint: group id}.
    Round 128 measured four cheap members - each decorrelated from the neural
    trio, strongly correlated with each other (phi up to +0.898) - overriding
    a unanimous trio on 37 rows and being wrong on 32 of them, and neither
    reweighting the choice nor counting heads fixed it. Members whose errors
    move together are one opinion however they are named.
    """
    if blocs is not None or count_lineages:
        votes = collapse_lineages(votes, blocs)
    if not votes:
        return DecisionResult(None, 0.0, 0.0, "escalate", [])
    if len(votes) < min_quorum:
        return DecisionResult(None, 0.0, 0.0, "escalate", [v.to_json() for v in votes])

    counts: dict[str, int] = {}
    probs: dict[str, list[float]] = {}
    for v in votes:
        counts[v.choice] = counts.get(v.choice, 0) + 1
        if not v.neutral:
            probs.setdefault(v.choice, []).append(v.probability)

    best_choice = max(counts, key=counts.get)
    top = counts[best_choice]
    agreement = top / len(votes)
    backing_probs = probs.get(best_choice, [])
    # A choice backed only by neutral votes has no reported confidence. Treat
    # that as unsafe for auto rather than inventing a value for the gate.
    mean_probability = (sum(backing_probs) / len(backing_probs)
                        if backing_probs else 0.0)

    # Tie between the two most-voted choices is not a majority, and a tie is
    # never a decision - but it is not necessarily silence either. Three
    # members cannot tie; four can, and round 126 measured what refusing to
    # answer costs when a weak independent member turns disagreements into
    # 2-2 splits: accepted coverage fell and the information that member
    # brought was discarded rather than weighed. When the confidences
    # disagree, the weighted choice is what the count threw away, so it is
    # returned at the verify tier - a quorum that cannot agree still cannot
    # act alone.
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) > 1 and sorted_counts[0] == sorted_counts[1]:
        mass: dict[str, float] = {}
        for vote in votes:
            if vote.neutral:
                continue
            mass[vote.choice] = mass.get(vote.choice, 0.0) + vote.probability
        ranked_mass = sorted(mass.items(), key=lambda item: (-item[1],
                                                             item[0]))
        if len(ranked_mass) > 1 and ranked_mass[0][1] > ranked_mass[1][1]:
            return DecisionResult(ranked_mass[0][0], agreement,
                                  mean_probability, "verify",
                                  [v.to_json() for v in votes])
        return DecisionResult(None, agreement, mean_probability, "verify",
                              [v.to_json() for v in votes])

    if (agreement >= auto_min_agreement
            and mean_probability >= auto_min_probability):
        tier = "auto"
    else:
        tier = "verify"
    return DecisionResult(best_choice, agreement, mean_probability, tier,
                          [v.to_json() for v in votes])


def collapse_lineages(votes: list[Vote],
                      blocs: dict | None = None) -> list[Vote]:
    """Reduce a quorum to one vote per lineage, or per bloc of lineages.

    The lineage's vote is the choice carrying the most probability mass among
    its members, at the mean probability of the members that backed it. A
    tie between equally heavy choices resolves by choice name, so the result
    never depends on arrival order. With `blocs`, lineages that share a group
    id are collapsed together - the operator declares which lineages are one
    opinion, the same way the playbook declares admissions and artifacts.
    """
    groups: dict[str, list[Vote]] = {}
    for vote in votes:
        key = (blocs or {}).get(vote.checkpoint, vote.checkpoint)
        groups.setdefault(key, []).append(vote)

    collapsed: list[Vote] = []
    for key, members in sorted(groups.items()):
        confident = [vote for vote in members if not vote.neutral]
        if not confident:
            counts: dict[str, int] = {}
            for vote in members:
                counts[vote.choice] = counts.get(vote.choice, 0) + 1
            choice = max(sorted(counts), key=lambda name: counts[name])
            collapsed.append(Vote(
                members[0].node_id, key, choice, 0.5, neutral=True))
            continue

        mass: dict[str, float] = {}
        for vote in confident:
            mass[vote.choice] = mass.get(vote.choice, 0.0) + vote.probability
        choice = max(sorted(mass), key=lambda name: mass[name])
        backers = [vote for vote in confident if vote.choice == choice]
        collapsed.append(Vote(
            backers[0].node_id, key, choice,
            sum(vote.probability for vote in backers) / len(backers)))
    return collapsed


def mcnemar_test(a_right_b_wrong: int, b_right_a_wrong: int) -> dict:
    """Paired two-arm comparison over the *discordant* rows (McNemar).

    `a_right_b_wrong` is b (arm A right, arm B wrong), `b_right_a_wrong` is c.
    The difference in counts is (c - b). Under H0 every discordant row is a
    coin flip, so var(b - c) = b + c and the standard error is sqrt(b + c).

    Do NOT "simplify" that to sqrt(b + c) / 2. The halved SE doubles every z,
    and it shipped: equal_budget_check.py and vote_ablation.py both carried it
    until 2026-10-19, which is what turned 002 (b=21, c=15) from a p=0.41 tie
    into the published "z = -2.00, beyond noise". One definition, one place.

    Returns delta, se, z and the exact two-sided McNemar p (no scipy): the
    exact test conditions on the discordant count, so it needs no normal
    approximation and is the honest companion for small b + c.
    """
    import math

    b, c = int(a_right_b_wrong), int(b_right_a_wrong)
    discordant = b + c
    delta = c - b
    if discordant == 0:
        return {"b": b, "c": c, "discordant": 0, "delta": 0,
                "se": 0.0, "z": None, "p_exact": None}
    se = math.sqrt(discordant)
    lo = min(b, c)
    tail = sum(math.comb(discordant, i) for i in range(lo + 1)) / 2 ** discordant
    return {"b": b, "c": c, "discordant": discordant, "delta": delta,
            "se": se, "z": delta / se, "p_exact": min(1.0, 2.0 * tail)}


def read_secret_file(path: str | Path) -> str:
    """Read a small newline-terminated secret from a user-owned file."""
    from pathlib import Path as SecretPath

    secret_path = SecretPath(path).expanduser()
    try:
        value = secret_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read secret file {secret_path}: {exc}") from exc
    if not value:
        raise ValueError(f"secret file is empty: {secret_path}")
    if len(value) > 4096 or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"secret file has invalid content: {secret_path}")
    return value


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
