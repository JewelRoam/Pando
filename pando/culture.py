"""Stage a signed playbook candidate without changing the running culture.

``culture.refine`` covers a broad set of ideas, and most of them are not safe
to hot-apply: retraining a member, replacing the playbook, or admitting a
lineage can all change how every decision is made. This module implements the
smallest useful actuator: an adopted proposal may write a *candidate* culture
artifact to a local outbox, signed by the proposing member. Nothing loads the
candidate, starts a process, or opens a socket.

The typed intent is deliberately narrow and shared with the proposer::

    {"playbook": {<complete playbook mapping>}}

The mapping is validated by the same parser used for a live playbook. The
candidate is stored as canonical JSON plus a detached Ed25519 attestation, so a
human or a later supervisor can review and apply it without trusting prose.
Staging is an action; applying a culture update is not an actuator in this
module.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .identity import Identity
from .playbook import parse_playbook

DEFAULT_OUTBOX = "~/.pandosi/culture-outbox"
ALLOWED_TOP_LEVEL = {
    "meta",
    "gating",
    "questions",
    "actions",
    "lineages",
    "combiner",
}


class CultureError(ValueError):
    """A culture.refine intent was refused. The message is the audit reason."""


def validate(params) -> tuple[dict | None, str]:
    """Validate the typed ``culture.refine`` intent. Pure.

    The playbook is a complete candidate rather than a patch. A patch language
    would need its own equivalence rules and would make it too easy for the
    proposer and executor to disagree about what a signed value means. The
    candidate is not applied here, so the conservative choice is to make the
    full object explicit and validate all of it.
    """
    if not isinstance(params, dict) or set(params) != {"playbook"}:
        return None, 'params must be {"playbook": {<complete playbook>}}'
    raw = params["playbook"]
    if not isinstance(raw, dict):
        return None, "params.playbook must be a mapping"
    unknown = set(raw) - ALLOWED_TOP_LEVEL
    if unknown:
        return None, ("unknown playbook fields: "
                      + ", ".join(sorted(str(name) for name in unknown)))
    try:
        # JSON round-trip catches values that cannot survive the signed wire
        # format before the proposer signs them.
        candidate = copy.deepcopy(json.loads(json.dumps(raw, ensure_ascii=False)))
        parse_playbook(candidate, source_path="<culture.refine proposal>")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid playbook: {exc}"
    return candidate, ""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _safe_component(value: object) -> str:
    name = str(value or "")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise CultureError("proposal id is not a safe path component")
    return name


def stage(proposal, *, identity: Identity,
          outbox: str | Path = DEFAULT_OUTBOX) -> dict:
    """Write one signed, inactive playbook candidate into ``outbox``.

    Returns the candidate hash and the directory written. No existing
    playbook, policy store, root, or model is touched.
    """
    candidate, why = validate(getattr(proposal, "params", None))
    if candidate is None:
        raise CultureError(why)
    proposal_id = _safe_component(getattr(proposal, "proposal_id", ""))
    root = Path(outbox).expanduser() / proposal_id
    payload = {
        "action": "culture.refine",
        "proposal_id": proposal_id,
        "author": str(getattr(proposal, "author", "")),
        "playbook": candidate,
    }
    playbook_hash = parse_playbook(
        candidate, source_path="<culture.refine proposal>").hash()
    candidate_text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    attestation = {
        "algorithm": "ed25519",
        "playbook_hash": playbook_hash,
        "proof": identity.proof(payload),
        "status": "staged-not-applied",
    }
    manifest = {
        "action": "culture.refine",
        "applied": False,
        "author": payload["author"],
        "playbook_hash": playbook_hash,
        "proposal_id": proposal_id,
        "status": "staged-not-applied",
    }
    _write_atomic(root / "candidate.json", candidate_text)
    _write_atomic(root / "attestation.json",
                  json.dumps(attestation, sort_keys=True, indent=2) + "\n")
    _write_atomic(root / "manifest.json",
                  json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"directory": str(root), "playbook_hash": playbook_hash}
