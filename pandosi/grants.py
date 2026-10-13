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

GRANT_ACTIONS = ("lan_listen", "lab_replication")


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
                 now: dt.datetime | None = None) -> dict:
    grants = load_grants(path)
    grant = grants.get(action)
    if not isinstance(grant, dict):
        return {"status": "missing", "grant": None,
                "reason": f"no {action} grant"}
    try:
        expires = parse_timestamp(str(grant["expires_at"]))
    except (KeyError, ValueError, TypeError) as exc:
        return {"status": "invalid", "grant": grant,
                "reason": f"invalid expires_at: {exc}"}
    current = now or _now()
    if current >= expires:
        return {"status": "expired", "grant": grant,
                "reason": f"expired at {format_timestamp(expires)}"}
    return {"status": "active", "grant": grant,
            "reason": f"expires at {format_timestamp(expires)}"}


def require_grant(action: str, path: Path | str | None = None,
                  now: dt.datetime | None = None) -> dict:
    if action not in GRANT_ACTIONS:
        raise ValueError(f"unknown grant action: {action}")
    status = grant_status(action, path, now)
    if status["status"] == "active":
        return status["grant"]
    example = (f"python -m pandosi.grants grant {action} "
               "--resource 0.0.0.0:8080 --hours 24 --yes")
    if action == "lab_replication":
        example = (f"python -m pandosi.grants grant {action} "
                   "--resource sandbox/lab --hours 4 --yes")
    raise PermissionError(
        f"{action} refused: {status['status']} ({status['reason']}). "
        f"Approve explicitly with: {example}")


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


def save_grant(action: str, resource: str, scope: str, approved_by: str,
               expires_at: str, evidence: str | None = None,
               path: Path | str | None = None) -> Path:
    if action not in GRANT_ACTIONS:
        raise ValueError(f"unknown grant action: {action}")
    parse_timestamp(expires_at)  # fail before mutating an existing ledger
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
    return save_grants(grants, path)


def revoke_grant(action: str, path: Path | str | None = None) -> bool:
    grants = load_grants(path)
    if action not in grants:
        return False
    del grants[action]
    save_grants(grants, path)
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

    statuses = {action: grant_status(action, path)
                for action in GRANT_ACTIONS}
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
