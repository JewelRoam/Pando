"""Operator-owned, expiring capability grants.

Grants are deliberately not code permissions. They record that a human
explicitly approved a sensitive capability, for how long, and with what local
evidence. Missing or expired grants fail closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

GRANT_ACTIONS = ("lan_listen", "lab_replication", "service_autostart",
                 "reproduce", "retire")


def default_path() -> Path:
    configured = os.environ.get("PANDOSI_GRANTS")
    return Path(configured) if configured else \
        Path.home() / ".pandosi" / "grants.toml"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def format_timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_grants(path: Path | str | None = None) -> dict[str, dict]:
    grant_path = Path(path) if path else default_path()
    if not grant_path.exists():
        return {}
    try:
        data = tomllib.loads(grant_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PermissionError(
            f"refusing startup with unreadable grants file {grant_path}: {exc}"
        ) from exc
    grants = data.get("grants", {})
    return grants if isinstance(grants, dict) else {}


def grant_status(action: str, path: Path | str | None = None,
                 now: dt.datetime | None = None,
                 include_audit: bool = True) -> dict:
    grant_path = Path(path) if path else default_path()
    grants = load_grants(grant_path)
    grant = grants.get(action)
    if not isinstance(grant, dict):
        result = {"action": action, "status": "missing", "grant": None,
                  "expires_at": None,
                  "reason": f"no {action} grant"}
    else:
        try:
            expires = parse_timestamp(str(grant["expires_at"]))
        except (KeyError, ValueError, TypeError) as exc:
            result = {"action": action, "status": "invalid", "grant": grant,
                      "expires_at": None,
                      "reason": f"invalid expires_at: {exc}"}
        else:
            current = now or _now()
            expires_text = format_timestamp(expires)
            if current >= expires:
                result = {"action": action, "status": "expired",
                          "grant": grant, "expires_at": expires_text,
                          "reason": f"expired at {expires_text}"}
            else:
                result = {"action": action, "status": "active",
                          "grant": grant, "expires_at": expires_text,
                          "reason": f"expires at {expires_text}"}
    if isinstance(grant, dict) and action == "lab_replication":
        resource = str(grant.get("resource", "sandbox/lab"))
    elif isinstance(grant, dict) and action == "reproduce":
        resource = str(grant.get("resource", "localhost"))
    elif isinstance(grant, dict) and action == "retire":
        resource = str(grant.get("resource", "any-member"))
    elif action == "lan_listen":
        resource = "0.0.0.0:8080"
    elif action == "service_autostart":
        resource = "user-agent/pandosi"
    else:
        resource = "sandbox/lab"
    result["grants_path"] = str(grant_path)
    result["next_command"] = (
        f"python -m pandosi.grants grant {action} --resource {resource} "
        "--hours 24 --yes")
    if include_audit:
        result["audit"] = audit_status(default_audit_path(grant_path))
    return result


def require_grant(action: str, path: Path | str | None = None,
                  now: dt.datetime | None = None, audit: bool = True,
                  audit_path: Path | str | None = None) -> dict:
    if action not in GRANT_ACTIONS:
        raise ValueError(f"unknown grant action: {action}")
    status = grant_status(action, path, now, include_audit=False)
    target_audit = Path(audit_path) if audit_path else \
        default_audit_path(path)
    if status["status"] == "active":
        if audit:
            try:
                append_audit_entry(action, "allowed", status["reason"], path,
                                   target_audit, status["grant"])
            except (OSError, PermissionError) as exc:
                raise PermissionError(
                    f"{action} refused: grant audit unavailable ({exc})") from exc
        return status["grant"]
    example = (f"python -m pandosi.grants grant {action} "
               "--resource 0.0.0.0:8080 --hours 24 --yes")
    if action == "lab_replication":
        example = (f"python -m pandosi.grants grant {action} "
                   "--resource sandbox/lab --hours 4 --yes")
    elif action == "reproduce":
        example = (f"python -m pandosi.grants grant {action} "
                   "--resource <target-host-or-device> --hours 1 --yes")
    elif action == "retire":
        example = (f"python -m pandosi.grants grant {action} "
                   "--resource <member-to-remove> --hours 1 --yes")
    message = (f"{action} refused: {status['status']} ({status['reason']}). "
               f"Approve explicitly with: {example}")
    if audit:
        try:
            append_audit_entry(action, "denied", status["reason"], path,
                               target_audit, status["grant"])
        except (OSError, PermissionError) as exc:
            message += f" (grant audit unavailable: {exc})"
    raise PermissionError(message)


def default_audit_path(grants_path: Path | str | None = None) -> Path:
    grant_path = Path(grants_path) if grants_path else default_path()
    return grant_path.with_name(grant_path.name + ".audit.jsonl")


def _canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _grant_fingerprint(grant: dict | None) -> str:
    if not grant:
        return "sha256:" + ("0" * 64)
    digest = hashlib.sha256(_canonical_json(grant).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def append_audit_entry(action: str, decision: str, reason: str,
                       grants_path: Path | str | None = None,
                       audit_path: Path | str | None = None,
                       grant: dict | None = None) -> dict:
    """Append a hash-chained local audit event and return that event."""
    audit_file = Path(audit_path) if audit_path else \
        default_audit_path(grants_path)
    valid, previous, problems = verify_audit(audit_file)
    if audit_file.exists() and not valid:
        raise PermissionError(
            f"grant audit {audit_file} failed verification: {problems}; "
            "refusing to authorize while evidence is tampered")
    prior_hash = previous.get("last_hash", "0" * 64) if valid \
        else "0" * 64

    event = {
        "timestamp": format_timestamp(_now()),
        "action": action,
        "decision": decision,
        "reason": reason,
        "grant": _grant_fingerprint(grant),
        "previous_hash": prior_hash,
    }
    event["entry_hash"] = hashlib.sha256(
        _canonical_json(event).encode("utf-8")).hexdigest()
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with audit_file.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def verify_audit(path: Path | str | None = None) -> tuple[bool, dict, list[str]]:
    """Verify the local hash chain without trusting its last summary line."""
    audit_file = Path(path) if path else default_audit_path()
    if not audit_file.exists():
        return True, {"entries": 0, "last_hash": "0" * 64}, []
    previous_hash = "0" * 64
    entries = 0
    problems: list[str] = []
    try:
        lines = audit_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return False, {"entries": entries, "last_hash": previous_hash}, \
            [f"unreadable: {exc}"]
    for line_no, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
            stored_hash = event.pop("entry_hash")
            expected = hashlib.sha256(
                _canonical_json(event).encode("utf-8")).hexdigest()
            if stored_hash != expected:
                problems.append(f"line {line_no}: entry hash mismatch")
            if event.get("previous_hash") != previous_hash:
                problems.append(f"line {line_no}: chain parent mismatch")
            previous_hash = stored_hash
            entries += 1
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            problems.append(f"line {line_no}: invalid JSON entry")
    if problems:
        return False, {"entries": entries, "last_hash": previous_hash}, problems
    return True, {"entries": entries, "last_hash": previous_hash}, []


def audit_status(audit_path: Path | str | None = None) -> dict:
    audit_file = Path(audit_path) if audit_path else default_audit_path()
    valid, summary, problems = verify_audit(audit_file)
    status = "absent" if not audit_file.exists() else \
        ("valid" if valid else "invalid")
    return {"path": str(audit_file), "status": status,
            "entries": summary["entries"], "problems": problems}


def _toml_string(value: str) -> str:
    # JSON string escaping is a valid TOML basic-string subset for scalar
    # metadata this module writes (no control characters are accepted here).
    return json.dumps(value, ensure_ascii=False)


def save_grants(grants: dict[str, dict], path: Path | str | None = None) -> Path:
    grant_path = Path(path) if path else default_path()
    grant_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# PandoSI authorization ledger.",
             "# Each capability must expire; revocation is deletion.", ""]
    for action in sorted(grants):
        lines.append(f"[grants.{action}]")
        fields = grants[action]
        for key in ("resource", "scope", "approved_by", "approved_at",
                    "expires_at", "evidence"):
            if key in fields:
                lines.append(f"{key} = {_toml_string(str(fields[key]))}")
        lines.append("")
    grant_path.write_text("\n".join(lines), encoding="utf-8")
    return grant_path


def _preflight_audit(path: Path | str | None = None) -> None:
    target = default_audit_path(path)
    valid, _, problems = verify_audit(target)
    if not valid:
        raise PermissionError(
            f"grant audit {target} is invalid: {problems}; refusing to "
            "modify the authorization ledger")


def _mutate_grant_with_audit(action: str, decision: str, reason: str,
                             path: Path | str | None, mutate) -> Path:
    target = Path(path) if path else default_path()
    old_bytes = target.read_bytes() if target.exists() else None
    try:
        output = mutate()
        grant = load_grants(target).get(action)
        append_audit_entry(action, decision, reason, target,
                           default_audit_path(target), grant)
        return output
    except Exception:
        if old_bytes is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(old_bytes)
        raise


def save_grant(action: str, resource: str, scope: str, approved_by: str,
               expires_at: str, evidence: str | None = None,
               path: Path | str | None = None) -> Path:
    if action not in GRANT_ACTIONS:
        raise ValueError(f"unknown grant action: {action}")
    parse_timestamp(expires_at)  # fail before mutating an existing ledger
    _preflight_audit(path)
    grants = load_grants(path)
    approved_at = format_timestamp(_now())
    if evidence is None:
        digest = hashlib.sha256(
            f"{action}\n{resource}\n{scope}\n{approved_by}\n{approved_at}"
            .encode("utf-8")).hexdigest()
        evidence = f"sha256:{digest}"
    grants[action] = {
        "resource": resource,
        "scope": scope,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "expires_at": expires_at,
        "evidence": evidence,
    }

    def write():
        return save_grants(grants, path)

    return _mutate_grant_with_audit(
        action, "created", f"approved until {expires_at}", path, write)


def revoke_grant(action: str, path: Path | str | None = None) -> bool:
    if action not in GRANT_ACTIONS:
        raise ValueError(f"unknown grant action: {action}")
    _preflight_audit(path)
    grants = load_grants(path)
    if action not in grants:
        return False
    del grants[action]

    def write():
        return save_grants(grants, path)

    _mutate_grant_with_audit(action, "revoked", "operator deleted grant",
                             path, write)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None,
                        help="grants TOML path (default: ~/.pandosi/grants.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    grant = sub.add_parser("grant", help="record one explicit grant")
    grant.add_argument("action", choices=GRANT_ACTIONS)
    grant.add_argument("--resource", required=True)
    grant.add_argument("--scope", default="operator-host")
    grant.add_argument("--approved-by", default="interactive")
    group = grant.add_mutually_exclusive_group(required=True)
    group.add_argument("--hours", type=float)
    group.add_argument("--expires-at")
    grant.add_argument("--evidence")
    grant.add_argument("--yes", action="store_true",
                       help="confirm this is an interactive human approval")

    revoke = sub.add_parser("revoke", help="delete one grant")
    revoke.add_argument("action", choices=GRANT_ACTIONS)

    sub.add_parser("show", help="print grant statuses as JSON")
    sub.add_parser("audit", help="verify the local grant audit hash chain")
    args = parser.parse_args()
    path = Path(args.path) if args.path else None

    if args.command == "grant" and not args.yes:
        parser.error("sensitive grants require --yes after the user approves")

    if args.command == "grant":
        if args.expires_at:
            expires = format_timestamp(parse_timestamp(args.expires_at))
        else:
            expires = format_timestamp(_now() + dt.timedelta(hours=args.hours))
        out = save_grant(args.action, args.resource, args.scope,
                         args.approved_by, expires, args.evidence, path)
        print(f"granted {args.action} until {expires}: {out}")
        return

    if args.command == "revoke":
        removed = revoke_grant(args.action, path)
        print(f"revoked {args.action}" if removed else
              f"{args.action} was not granted")
        return

    if args.command == "audit":
        target = default_audit_path(path)
        status = audit_status(target)
        print(json.dumps(status, indent=2))
        raise SystemExit(0 if status["status"] != "invalid" else 2)

    statuses = {action: grant_status(action, path)
                for action in GRANT_ACTIONS}
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
