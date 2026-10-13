"""Playbook: the signed strategy layer (blueprint section 1).

A playbook is a TOML file authored offline (human or System-2 LLM). It defines
the question schemas the swarm answers, confidence gating thresholds, and
action bindings. It contains no code and never executes - the swarm only
answers its declared questions; actions are the harness's responsibility.

Verification model:
- canonical_hash(): sha256 over canonical JSON of the parsed playbook.
- The spore manifest signs playbook hashes; stems refuse unsigned/unknown
  playbooks (pinned in a follow-up to the stem's manifest).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from pathlib import Path

REQUIRED_META = {"name", "version"}
VALID_QTYPES = {"choice", "score", "noul"}


@dataclasses.dataclass
class Playbook:
    name: str
    version: int
    meta: dict
    gating: dict
    questions: list[dict]
    actions: list[dict]
    source_path: str | None = None

    def canonical_json(self) -> str:
        data = dataclasses.asdict(self)
        data.pop("source_path", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def load_playbook(path: Path) -> Playbook:
    """Load and validate a playbook TOML file. Raises ValueError on any
    structural problem - a malformed playbook must never reach the swarm."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    meta = raw.get("meta") or {}
    missing = REQUIRED_META - set(meta)
    if missing:
        raise ValueError(f"playbook meta missing keys: {sorted(missing)}")
    if not isinstance(meta.get("version"), int):
        raise ValueError("playbook meta.version must be an integer")

    questions = raw.get("questions") or []
    # A gating-only playbook (no questions) is valid: it overrides thresholds
    # for a question set defined elsewhere.
    seen_ids = set()
    for q in questions:
        qid = q.get("id")
        if not qid:
            raise ValueError("question missing id")
        if qid in seen_ids:
            raise ValueError(f"duplicate question id {qid!r}")
        seen_ids.add(qid)
        if q.get("type") not in VALID_QTYPES:
            raise ValueError(
                f"question {qid!r}: type must be one of {sorted(VALID_QTYPES)}")
        if q["type"] in ("choice", "score") and not q.get("criteria"):
            raise ValueError(f"question {qid!r}: {q['type']} needs criteria")

    actions = raw.get("actions") or []
    for a in actions:
        if not a.get("when") or not a.get("do"):
            raise ValueError("action missing 'when' or 'do'")

    return Playbook(
        name=meta["name"],
        version=meta["version"],
        meta=meta,
        gating=raw.get("gating") or {},
        questions=questions,
        actions=actions,
        source_path=str(path),
    )


def question_specs(pb: Playbook) -> dict[str, dict]:
    """Convert playbook questions into Laya predict() question specs."""
    specs = {}
    for q in pb.questions:
        spec: dict = {"type": q["type"], "instructions": q["instructions"]}
        if q["type"] in ("choice", "score"):
            spec["criteria"] = q["criteria"]
        specs[q["id"]] = spec
    return specs
