"""Stage a signed invitation without admitting a member.

``recruit.invite`` is the smallest recruiting action: a member issues a fresh
pairing token, signs the invitation that carries it, and writes the artifact
into a local outbox for review. Nothing here registers a stem, installs the
token into a running root, starts a process, or opens a socket. Turning a
staged invitation into a member stays a deployment step — write the token into
the root's ``--pairing-token-file`` and restart it — the same way a staged
``culture.refine`` candidate stays inactive until a human applies it.

The invitation is a *bearer* token: ``pando/coordinator.py`` matches a secret, not a
person. A ``candidate`` label may travel with it, but the running root does not
enforce that label, and the manifest records the label without pretending
otherwise.

Because the token is random, staging is idempotent by construction: once
``manifest.json`` exists the directory is never rewritten. Regenerating the
token on a retry would silently invalidate an invitation that had already been
copied somewhere, which is exactly the kind of second-order failure a retry
loop must not introduce.

The optional typed intent is narrow and deliberately a superset of what the
current proposer emits (today ``recruit.invite`` proposals carry no params)::

    {"candidate": "<label>", "expires_in_hours": <int>}

Both keys are optional. The token is the only secret, and it exists on disk
only inside the outbox: never in git, never in the mailbox, never in the
ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from .identity import Identity, canonical
from .identity import verify as verify_signature

DEFAULT_OUTBOX = "~/.pando/invites"
DEFAULT_TTL_HOURS = 24 * 7
MAX_TTL_HOURS = 24 * 365
ALLOWED_PARAMS = {"candidate", "expires_in_hours"}


class InviteError(ValueError):
    """A recruit.invite intent or artifact was refused. Message is the reason."""


def validate(params) -> tuple[dict | None, str]:
    """Validate the typed ``recruit.invite`` intent. Pure.

    An absent or empty params block is the normal case today. A narrow optional
    block lets a later proposer name a candidate label or a shorter lifetime
    without inventing a second wire format; anything else is refused rather
    than ignored, so a value the executor did not understand can never ride
    along inside a signature it does not describe.
    """
    if params is None or params == {} or params == []:
        return {}, ""
    if not isinstance(params, dict):
        return None, ('params must be {} or '
                      '{"candidate": "<label>", "expires_in_hours": <int>}')
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        return None, ("unknown recruit.invite params: "
                      + ", ".join(sorted(str(name) for name in unknown)))
    checked: dict = {}
    candidate = params.get("candidate")
    if candidate is not None:
        if not isinstance(candidate, str) or not candidate.strip():
            return None, "params.candidate must be a non-empty string"
        if len(candidate.strip()) > 128:
            return None, "params.candidate must be at most 128 characters"
        checked["candidate"] = candidate.strip()
    ttl = params.get("expires_in_hours")
    if ttl is not None:
        if isinstance(ttl, bool) or not isinstance(ttl, int):
            return None, "params.expires_in_hours must be an integer"
        if not 1 <= ttl <= MAX_TTL_HOURS:
            return None, (f"params.expires_in_hours must be in 1..{MAX_TTL_HOURS}")
        checked["expires_in_hours"] = ttl
    return checked, ""


def _write_atomic(path: Path, text: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(text, encoding="utf-8")
    os.chmod(staging, mode)
    os.replace(staging, path)


def _safe_component(value: object) -> str:
    name = str(value or "")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise InviteError("proposal id is not a safe path component")
    return name


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InviteError(f"unreadable invitation artifact {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise InviteError(f"invitation artifact {path.name} is not an object")
    return data


def stage(proposal, *, identity: Identity,
          outbox: str | Path = DEFAULT_OUTBOX,
          now: float | None = None,
          ttl_hours: int = DEFAULT_TTL_HOURS) -> dict:
    """Write one signed, inactive invitation into ``outbox``.

    Returns the invite hash, its expiry and the directory written. Nothing
    live is touched: no root, no stem, no model, no member table. Re-staging an
    existing proposal returns the artifact already on disk instead of minting a
    second token, so a retry cannot invalidate an invitation already copied.
    """
    checked, why = validate(getattr(proposal, "params", None))
    if checked is None:
        raise InviteError(why)
    proposal_id = _safe_component(getattr(proposal, "proposal_id", ""))
    root = Path(outbox).expanduser() / proposal_id
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        return {
            "directory": str(root),
            "expires_at": int(existing.get("expires_at") or 0),
            "invite_hash": str(existing.get("invite_hash") or ""),
            "staged": False,
        }
    if root.exists():
        raise InviteError(
            "invitation directory exists without a manifest; refusing to overwrite")
    issued_at = int(time.time() if now is None else now)
    ttl = int(checked.get("expires_in_hours", ttl_hours))
    payload = {
        "action": "recruit.invite",
        "author": str(getattr(proposal, "author", "")),
        "expires_at": issued_at + ttl * 3600,
        "goal": str(getattr(proposal, "goal", "")),
        "issued_at": issued_at,
        "proposal_id": proposal_id,
        "token": secrets.token_urlsafe(18),
    }
    if checked.get("candidate"):
        payload["candidate"] = checked["candidate"]
    invite_hash = hashlib.sha256(canonical(payload)).hexdigest()
    attestation = {
        "algorithm": "ed25519",
        "invite_hash": invite_hash,
        "proof": identity.proof(payload),
        "status": "staged-not-applied",
    }
    manifest = {
        "action": "recruit.invite",
        "applied": False,
        "author": payload["author"],
        "candidate": str(payload.get("candidate") or ""),
        "expires_at": payload["expires_at"],
        "invite_hash": invite_hash,
        "proposal_id": proposal_id,
        "status": "staged-not-applied",
    }
    _write_atomic(root / "invite.json",
                  json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             indent=2) + "\n", 0o600)
    _write_atomic(root / "attestation.json",
                  json.dumps(attestation, sort_keys=True, indent=2) + "\n", 0o600)
    _write_atomic(root / "manifest.json",
                  json.dumps(manifest, sort_keys=True, indent=2) + "\n", 0o644)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return {
        "directory": str(root),
        "expires_at": payload["expires_at"],
        "invite_hash": invite_hash,
        "staged": True,
    }


def verify(directory: str | Path) -> bool:
    """True when a staged invitation is intact and signed by its issuer.

    A reviewer can run this without trusting the manifest: the hash is
    recomputed from ``invite.json`` and the Ed25519 proof is checked against the
    payload that hash covers.
    """
    root = Path(directory).expanduser()
    try:
        payload = _read_json(root / "invite.json")
        attestation = _read_json(root / "attestation.json")
        manifest = _read_json(root / "manifest.json")
    except InviteError:
        return False
    invite_hash = hashlib.sha256(canonical(payload)).hexdigest()
    if invite_hash != str(manifest.get("invite_hash") or ""):
        return False
    if invite_hash != str(attestation.get("invite_hash") or ""):
        return False
    proof = attestation.get("proof")
    if not isinstance(proof, dict):
        return False
    return verify_signature(str(proof.get("pubkey") or ""), payload,
                            str(proof.get("sig") or ""))
