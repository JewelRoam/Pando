"""Signed delegation of one standing deployment grant between PandoRSI hosts.

The human authorizes the *deployment action* once, locally.  Other nodes may
then receive an expiring, narrowly scoped copy of that decision without a
person reviewing every proposal.  A delegated grant is not permission to send
code: the executor still accepts only a fixed catalogue profile name.

The wire object is an Ed25519-signed JSON envelope.  Receivers pin the Mac's
node public key, so a Windows host cannot mint a grant for itself even when it
controls its local ledger.  Revocation is deletion on the issuer; a receiver
removes only grants whose local ``origin`` says they came from this delegation.
Network failure never erases a still-valid local grant.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from .grants import (audit_status, default_path as default_grants_path,
                     grant_fingerprint, grant_status, install_grant_fields,
                     load_grants, parse_timestamp,
                     remove_grant_if_origin)
from .identity import Identity, canonical
from .identity import verify as verify_signature
from .net import open_url

ACTION = "deploy_operator"
RESOURCE = "profile-catalogue/v1"
SCOPE = "operator-host"
AUDIENCE = "operator-host"
ORIGIN = "mac-delegation/v1"
MAX_TTL = dt.timedelta(days=7)
CLOCK_SKEW = dt.timedelta(minutes=5)

#: The Mac node identity that is allowed to delegate this action.  This is a
#: pin, not a value learned from the HTTP response: otherwise a compromised
#: responder could include its own public key and sign its own authorization.
MAC_OPERATOR_PUBKEY = "NFqRlNyH6eOKjhOCqcazwQBWTKZ2_HR-dreInL4djq8="

ENVELOPE_FIELDS = frozenset({
    "version", "audience", "action", "resource", "scope", "approved_by",
    "approved_at", "expires_at", "evidence", "signature",
})
GRANT_FIELDS = ("resource", "scope", "approved_by", "approved_at",
                "expires_at", "evidence")


class DelegationError(ValueError):
    """A delegation envelope is malformed, unauthorized or no longer valid."""


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _require_text(payload: dict, key: str, limit: int = 2000) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise DelegationError(f"{key} must be a non-empty string of at most {limit} bytes")
    return value.strip()


def verify_envelope(envelope: dict, *, expected_pubkey: str = MAC_OPERATOR_PUBKEY,
                    now: dt.datetime | None = None) -> dict:
    """Validate the exact wire shape, lifetime and pinned signature."""
    if not isinstance(envelope, dict):
        raise DelegationError("delegation envelope must be a JSON object")
    if set(envelope) != ENVELOPE_FIELDS:
        missing = sorted(ENVELOPE_FIELDS - set(envelope))
        extra = sorted(set(envelope) - ENVELOPE_FIELDS)
        raise DelegationError(f"delegation fields mismatch: missing={missing}, extra={extra}")
    if envelope.get("version") != 1 or isinstance(envelope.get("version"), bool):
        raise DelegationError("delegation version must be 1")
    expected = {"audience": AUDIENCE, "action": ACTION,
                "resource": RESOURCE, "scope": SCOPE}
    for key, value in expected.items():
        if envelope.get(key) != value:
            raise DelegationError(f"delegation {key} must be {value!r}")
    _require_text(envelope, "approved_by", 256)
    _require_text(envelope, "evidence", 2000)
    signature = _require_text(envelope, "signature", 512)
    try:
        approved = parse_timestamp(_require_text(envelope, "approved_at", 80))
        expires = parse_timestamp(_require_text(envelope, "expires_at", 80))
    except (ValueError, TypeError) as exc:
        raise DelegationError(f"invalid delegation timestamp: {exc}") from exc
    current = now or _utc_now()
    if approved > current + CLOCK_SKEW:
        raise DelegationError("delegation approved_at is in the future")
    if expires <= approved:
        raise DelegationError("delegation expires_at must be after approved_at")
    if expires - approved > MAX_TTL:
        raise DelegationError("delegation lifetime exceeds 168 hours")
    if current >= expires:
        raise DelegationError("delegation has expired")
    payload = {key: envelope[key] for key in ENVELOPE_FIELDS if key != "signature"}
    if not verify_signature(expected_pubkey, payload, signature):
        raise DelegationError("delegation signature is not from the pinned operator key")
    return copy.deepcopy(payload)


def issue_envelope(grant_path: Path | str | None = None,
                   identity: Identity | None = None, *,
                   identity_path: Path | str | None = None,
                   now: dt.datetime | None = None) -> dict:
    """Turn one active local deploy grant into a signed wire envelope."""
    path = Path(grant_path) if grant_path else default_grants_path()
    status = grant_status(ACTION, path, now=now)
    if status.get("status") != "active":
        raise DelegationError(f"deploy_operator grant is {status.get('status')}: "
                              f"{status.get('reason')}")
    if status.get("audit", {}).get("status") != "valid":
        raise DelegationError("deploy_operator grant audit is not valid")
    grant = status.get("grant")
    if not isinstance(grant, dict):
        raise DelegationError("deploy_operator grant is not an object")
    if grant.get("resource") != RESOURCE or grant.get("scope") != SCOPE:
        raise DelegationError("deploy_operator grant does not match the fixed catalogue scope")
    if grant.get("origin"):
        raise DelegationError("a delegated grant cannot issue another delegation")
    signer = identity
    if signer is None:
        if identity_path is None:
            from .identity import default_path as default_identity_path
            identity_path = default_identity_path()
        key_path = Path(identity_path).expanduser()
        try:
            signer = Identity.from_json(json.loads(key_path.read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DelegationError(f"operator identity is unavailable: {exc}") from exc
    if signer.public_key != MAC_OPERATOR_PUBKEY:
        raise DelegationError("operator identity is not the pinned Mac signer")
    payload = {"version": 1, "audience": AUDIENCE, "action": ACTION,
               "resource": RESOURCE, "scope": SCOPE}
    for key in ("approved_by", "approved_at", "expires_at", "evidence"):
        try:
            payload[key] = _require_text(grant, key)
        except DelegationError as exc:
            raise DelegationError(f"local deploy_operator grant is invalid: {exc}") from exc
    envelope = dict(payload, signature=signer.sign(payload))
    # Signing must fail closed too.  The receiver pins the Mac key, so accepting
    # a differently-keyed local signer here would only produce an envelope that
    # every node correctly rejects later -- with a much less obvious failure.
    verify_envelope(envelope, expected_pubkey=MAC_OPERATOR_PUBKEY, now=now)
    return envelope


def _delegated_fields(payload: dict) -> dict:
    fields = {key: str(payload[key]) for key in GRANT_FIELDS}
    fields["origin"] = ORIGIN
    fields["delegation_id"] = hashlib.sha256(canonical(payload)).hexdigest()
    return fields


def install_envelope(envelope: dict, grants_path: Path | str | None = None, *,
                     expected_pubkey: str = MAC_OPERATOR_PUBKEY,
                     now: dt.datetime | None = None) -> dict:
    """Verify and install a delegation, preserving a human-authored grant."""
    payload = verify_envelope(envelope, expected_pubkey=expected_pubkey, now=now)
    path = Path(grants_path) if grants_path else default_grants_path()
    current = load_grants(path).get(ACTION)
    if isinstance(current, dict) and current.get("origin") != ORIGIN:
        return {"status": "manual-preserved", "changed": False,
                "grants_path": str(path),
                "reason": "a locally installed grant already authorizes this action"}
    fields = _delegated_fields(payload)
    target, changed = install_grant_fields(
        ACTION, fields, path,
        decision="delegated-installed" if not current else "delegated-renewed",
        reason=f"signed delegation {fields['delegation_id'][:16]} installed")
    return {"status": "installed" if changed else "unchanged",
            "changed": changed, "grants_path": str(target),
            "delegation_id": fields["delegation_id"],
            "grant_fingerprint": grant_fingerprint(fields),
            "expires_at": fields["expires_at"]}


def operator_status(grant_path: Path | str | None = None,
                    identity_path: Path | str | None = None, *,
                    now: dt.datetime | None = None) -> dict:
    """Non-secret status for remote verification and operational debugging."""
    path = Path(grant_path) if grant_path else default_grants_path()
    try:
        status = grant_status(ACTION, path, now=now)
    except PermissionError as exc:
        return {"authorized": False, "grant_status": "unreadable",
                "grants_path": str(path), "reason": str(exc),
                "audit": audit_status(path.with_name(path.name + ".audit.jsonl"))}
    grant = status.get("grant") if isinstance(status.get("grant"), dict) else {}
    audit = status.get("audit") or audit_status(path.with_name(path.name + ".audit.jsonl"))
    authorized = bool(status.get("status") == "active"
                      and audit.get("status") == "valid"
                      and grant.get("resource") == RESOURCE
                      and grant.get("scope") == SCOPE)
    result = {
        "authorized": authorized,
        "grant_status": status.get("status"),
        "reason": status.get("reason"),
        "grants_path": str(path),
        "resource": grant.get("resource"),
        "scope": grant.get("scope"),
        "approved_by": grant.get("approved_by"),
        "approved_at": grant.get("approved_at"),
        "expires_at": grant.get("expires_at"),
        "origin": grant.get("origin"),
        "delegation_id": grant.get("delegation_id"),
        "grant_fingerprint": grant_fingerprint(grant) if grant else None,
        "audit": audit,
    }
    if identity_path:
        key_path = Path(identity_path).expanduser()
        try:
            signer = Identity.from_json(json.loads(key_path.read_text(encoding="utf-8")))
            result["signer_pubkey"] = signer.public_key
            result["signer_fingerprint"] = signer.node_fingerprint
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            result["signer_pubkey"] = None
    return result


def refresh_and_install(urls: list[str] | tuple[str, ...], *,
                        token_file: Path | str, grants_path: Path | str | None = None,
                        expected_pubkey: str = MAC_OPERATOR_PUBKEY,
                        timeout: float = 8.0,
                        now: dt.datetime | None = None) -> dict:
    """Fetch from the first answering grant shelf and update local state safely.

    A transport error or a bad signature preserves the existing verified grant.
    Only an explicit 404 answer on every configured shelf is treated as
    revocation, and even then only a grant carrying this module's origin is
    deleted.
    """
    ordered = list(dict.fromkeys(str(url).rstrip("/") for url in urls
                                 if str(url or "").strip()))
    if not ordered:
        return {"status": "disabled", "changed": False}
    try:
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {"status": "error", "changed": False,
                "reason": f"mailbox token unavailable: {exc}"}
    attempts: list[dict] = []
    for url in ordered:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}",
                          "Accept": "application/json"}, method="GET")
        attempt = {"url": url}
        try:
            with open_url(request, timeout) as response:
                raw = response.read(256 * 1024 + 1)
            attempt["http"] = 200
            if len(raw) > 256 * 1024:
                raise DelegationError("delegation response is too large")
            envelope = json.loads(raw)
            installed = install_envelope(
                envelope, grants_path, expected_pubkey=expected_pubkey, now=now)
            return {"status": installed["status"], "changed": installed["changed"],
                    "source": url, "grant": installed,
                    "attempts": attempts + [{"url": url, "http": 200}]}
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            try:
                detail = exc.read(240).decode("utf-8", "replace")
            finally:
                exc.close()
            attempts.append({"url": url, "http": code, "detail": detail})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
    answered = [row for row in attempts if "http" in row]
    all_absent = bool(answered) and len(answered) == len(attempts) \
        and all(row["http"] == 404 for row in answered)
    if all_absent:
        removed = remove_grant_if_origin(ACTION, ORIGIN, grants_path)
        return {"status": "revoked" if removed else "absent",
                "changed": removed, "attempts": attempts}
    if any(row.get("http") == 200 for row in attempts):
        return {"status": "error", "changed": False,
                "reason": "grant response could not be verified",
                "attempts": attempts}
    return {"status": "unreachable" if not answered else "error",
            "changed": False, "reason": "no usable signed grant response",
            "attempts": attempts}
