"""Shared message schemas and helpers."""

from __future__ import annotations

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
        probs.setdefault(v.choice, []).append(v.probability)

    best_choice = max(counts, key=counts.get)
    top = counts[best_choice]
    agreement = top / len(votes)
    mean_probability = sum(probs[best_choice]) / len(probs[best_choice])

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
        mass: dict[str, float] = {}
        for vote in members:
            mass[vote.choice] = mass.get(vote.choice, 0.0) + vote.probability
        choice = max(sorted(mass), key=lambda name: mass[name])
        backers = [vote for vote in members if vote.choice == choice]
        collapsed.append(Vote(
            backers[0].node_id, key, choice,
            sum(vote.probability for vote in backers) / len(backers)))
    return collapsed


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
