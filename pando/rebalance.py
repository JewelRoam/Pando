"""Draft a signed member-move plan without moving anything.

``resource.rebalance`` asks the group to run an existing member somewhere else:
drain the host that is out of memory or swap, and let a boxier machine carry the
member instead. It is the one action whose *point* is to touch two machines at
once, so the smallest useful actuator drafts the move and stops there: a plan in
a local outbox naming the member, the target host, and every reason the parent
still refuses to move it.

Nothing here stops a member, starts a member, copies a directory, opens a
socket, or writes a grant. The move itself stays a deployment step — the same
boundary ``culture.refine`` and ``recruit.invite`` sit behind.

The typed intent is labels only::

    {"node_id": "<member label>", "target_host": "<host label>"}

A ``path``, ``command``, ``url`` or any other key is refused rather than
ignored. The member's state directory and the stop/start order are exactly what
the proposer must not be able to choose, because a proposal that can name a path
can name ``/`` and a proposal that can name an order can name "start the second
copy first" — two members answering to one identity, which is worse than an
outage.

A plan holds no secret, so re-staging rewrites it instead of freezing it: a
draft that keeps reporting a refusal which has since been answered is a stale
accusation, not a safety property.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .grants import grant_status
from .identity import Identity, canonical
from .identity import verify as verify_signature

DEFAULT_OUTBOX = "~/.pando/rebalance-outbox"
ALLOWED_PARAMS = {"node_id", "target_host"}
MAX_LABEL = 128

NO_STATE = ("no state directory: params carry labels only, so the gate names "
            "the member's state path")

#: What a human (or a later supervisor) must supply before this plan can run.
#: Written into every artifact so the hold is legible without the code.
REQUIRES = (
    "a reproduce grant for the target host (python -m pando.grants grant "
    "reproduce --resource <target-host> --hours 1 --yes) - moving a member "
    "copies its identity, so it is a reproduce-class operation",
    "a service_autostart grant on the destination, or the member arrives "
    "unable to run",
    "the destination to confirm a sandbox (container/VM)",
    "a gate-supplied state directory and stop-before-start order - the old "
    "copy must be stopped first, or two members answer to one identity",
    "the source member to be idle: a member draining mid-decision leaves the "
    "quorum one vote short at exactly the wrong moment",
)


class RebalanceError(ValueError):
    """A resource.rebalance intent was refused. The message is the audit reason."""


def validate(params) -> tuple[dict | None, str]:
    """Validate the typed ``resource.rebalance`` intent. Pure.

    Only labels are accepted. The whole point of this actuator is that the
    proposal cannot choose a path or an argv, so an unknown key is refused
    rather than ignored.
    """
    if params is None or params == {} or params == []:
        return {}, ""
    if not isinstance(params, dict):
        return None, ('params must be {} or '
                      '{"node_id": "<label>", "target_host": "<label>"}')
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        return None, ("unknown resource.rebalance params: "
                      + ", ".join(sorted(str(name) for name in unknown))
                      + " (this action accepts labels only; the state path and "
                        "the stop/start order are supplied by the gate)")
    checked: dict = {}
    for key in sorted(ALLOWED_PARAMS):
        value = params.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            return None, f"params.{key} must be a non-empty string"
        if len(value.strip()) > MAX_LABEL:
            return None, f"params.{key} must be at most {MAX_LABEL} characters"
        checked[key] = value.strip()
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
        raise RebalanceError("proposal id is not a safe path component")
    return name


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebalanceError(f"unreadable rebalance artifact {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise RebalanceError(f"rebalance artifact {path.name} is not an object")
    return data


def stage(proposal, *, identity: Identity,
          outbox: str | Path = DEFAULT_OUTBOX,
          grants_path: Path | str | None = None,
          now: float | None = None) -> dict:
    """Write one signed, inactive move plan into ``outbox``.

    Returns the plan hash, the refusal list and the directory written. No
    member is stopped, no member is started, no state moves.
    """
    checked, why = validate(getattr(proposal, "params", None))
    if checked is None:
        raise RebalanceError(why)
    proposal_id = _safe_component(getattr(proposal, "proposal_id", ""))
    node_id = str(checked.get("node_id") or "")
    target_host = str(checked.get("target_host") or "")

    refusal: list[str] = [NO_STATE]
    if not target_host:
        refusal.append("no target host: the proposal names a member but not "
                       "where it should run")

    # Read-only: a staging round must not append to the grant audit trail.
    grant = grant_status("reproduce", path=grants_path, include_audit=False)
    autostart = grant_status("service_autostart", path=grants_path,
                             include_audit=False)
    issued_at = int(time.time() if now is None else now)
    plan = {
        "action": "resource.rebalance",
        "approved": False,
        "author": str(getattr(proposal, "author", "")),
        "goal": str(getattr(proposal, "goal", "")),
        "grants": {
            "reproduce": {
                "action": "reproduce",
                "reason": str(grant.get("reason") or ""),
                "status": str(grant.get("status") or ""),
            },
            "service_autostart": {
                "action": "service_autostart",
                "reason": str(autostart.get("reason") or ""),
                "status": str(autostart.get("status") or ""),
            },
        },
        "issued_at": issued_at,
        "moved": False,
        "proposal_id": proposal_id,
        "refusals": refusal,
        "requested": {"node_id": node_id, "target_host": target_host},
        "requires": list(REQUIRES),
    }
    plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
    attestation = {
        "algorithm": "ed25519",
        "plan_hash": plan_hash,
        "proof": identity.proof(plan),
        "status": "staged-not-applied",
    }
    manifest = {
        "action": "resource.rebalance",
        "applied": False,
        "approved": False,
        "author": plan["author"],
        "issued_at": issued_at,
        "moved": False,
        "plan_hash": plan_hash,
        "proposal_id": proposal_id,
        "refusals": refusal,
        "status": "staged-not-applied",
        "target_host": target_host,
    }
    root = Path(outbox).expanduser() / proposal_id
    _write_atomic(root / "plan.json",
                  json.dumps(plan, ensure_ascii=False, sort_keys=True,
                             indent=2) + "\n", 0o600)
    _write_atomic(root / "attestation.json",
                  json.dumps(attestation, sort_keys=True, indent=2) + "\n",
                  0o600)
    _write_atomic(root / "manifest.json",
                  json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                  0o644)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return {
        "directory": str(root),
        "plan_hash": plan_hash,
        "refusals": refusal,
        "staged": True,
    }


def verify(directory: str | Path) -> bool:
    """True when a staged move plan is intact, unsigned-for-nothing and inert.

    A reviewer can run this without trusting the manifest: the hash is
    recomputed from ``plan.json`` and the Ed25519 proof is checked against the
    payload that hash covers. A plan that claims to be approved or moved fails
    verification, so a hand-edited artifact cannot borrow the signature of the
    one that was reviewed.
    """
    root = Path(directory).expanduser()
    try:
        plan = _read_json(root / "plan.json")
        attestation = _read_json(root / "attestation.json")
        manifest = _read_json(root / "manifest.json")
    except RebalanceError:
        return False
    if plan.get("approved") is not False or plan.get("moved") is not False:
        return False
    if manifest.get("applied") is not False or manifest.get("moved") is not False:
        return False
    plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
    if plan_hash != str(manifest.get("plan_hash") or ""):
        return False
    if plan_hash != str(attestation.get("plan_hash") or ""):
        return False
    proof = attestation.get("proof")
    if not isinstance(proof, dict):
        return False
    return verify_signature(str(proof.get("pubkey") or ""), plan,
                            str(proof.get("sig") or ""))
