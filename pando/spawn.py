"""Stage a signed reproduction plan without spawning anything.

``reproduce.spawn`` is the heaviest action in the space: it copies a payload to
another device, starts a process there, and grows the quorum. This module is
the smallest useful actuator for it — an adopted proposal writes a *plan* into
a local outbox saying what would be copied, where, under which lineage, and
every reason the parent side still refuses to do it.

``pando/reproduce.py`` already knows how to decide and how to launch. This
module only calls its planner (``plan_reproduction``); it never calls
``reproduce()``. Nothing here transfers a byte, starts a process, opens a
socket, or writes a grant.

The staged plan is inert by construction:

- the proposal cannot name a path or an argv. Its params carry labels only
  (``node_id``, ``lineage``), so a model can never put a command into an
  artifact a human might later run. The payload and the target argv are what
  the gate supplies, and the artifact lists them as requirements.
- ``sandbox_confirmed`` is always ``False`` here, which is the planner's own
  rule for "the receiving device has not confirmed a sandbox". So every
  artifact this module writes has ``approved: false`` and ``spawned: false``.
  Staging a plan cannot be mistaken for approving one.
- the grant state is *read*, never requested. ``grant_status`` is used rather
  than ``require_grant`` on purpose: a staging round must not append an audit
  entry, and it must not be the thing that decides a human approved anything.

Re-staging the same proposal rewrites its plan instead of freezing the first
one, so the artifact cannot keep reporting a refusal that has since been
answered (a grant added, a payload staged). A plan holds no secret, so there
is nothing for a rewrite to invalidate — unlike ``recruit.invite``, whose
token must never be re-minted.
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
from .reproduce import Target, plan_reproduction

DEFAULT_OUTBOX = "~/.pandosi/reproduce-outbox"
ALLOWED_PARAMS = {"lineage", "node_id"}
MAX_LABEL = 128

NO_PAYLOAD = ("no payload: params carry labels only, so the gate names the "
              "artifact to replicate")

#: What a human (or a later supervisor) must supply before this plan can run.
#: Written into every artifact so the hold is legible without the code.
REQUIRES = (
    "a reproduce grant (python -m pando.grants grant reproduce "
    "--resource <target-host-or-device> --hours 1 --yes)",
    "the receiving device to confirm a sandbox (container/VM)",
    "a payload artifact this host can serve",
    "a target argv and host, supplied by the gate - never by the proposal",
)


class SpawnError(ValueError):
    """A reproduce.spawn intent was refused. The message is the audit reason."""


def validate(params) -> tuple[dict | None, str]:
    """Validate the typed ``reproduce.spawn`` intent. Pure.

    Only labels are accepted. A ``command`` or ``payload`` key is refused
    rather than ignored: the whole point of this actuator is that the
    proposal cannot carry an executable surface into the outbox.
    """
    if params is None or params == {} or params == []:
        return {}, ""
    if not isinstance(params, dict):
        return None, ('params must be {} or '
                      '{"node_id": "<label>", "lineage": "<label>"}')
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        return None, ("unknown reproduce.spawn params: "
                      + ", ".join(sorted(str(name) for name in unknown))
                      + " (this action accepts labels only; the payload and "
                        "the target argv are supplied by the gate)")
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
        raise SpawnError("proposal id is not a safe path component")
    return name


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpawnError(f"unreadable repro artifact {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpawnError(f"repro artifact {path.name} is not an object")
    return data


def stage(proposal, *, identity: Identity,
          outbox: str | Path = DEFAULT_OUTBOX,
          spore: str | Path | None = None,
          declared_lineages: list[str] | None = None,
          declared_artifacts: dict | None = None,
          grants_path: Path | str | None = None,
          now: float | None = None) -> dict:
    """Write one signed, inactive reproduction plan into ``outbox``.

    Returns the plan hash, the refusal list and the directory written. No
    payload moves, no process starts, no grant is requested.
    """
    checked, why = validate(getattr(proposal, "params", None))
    if checked is None:
        raise SpawnError(why)
    proposal_id = _safe_component(getattr(proposal, "proposal_id", ""))
    lineage = str(checked.get("lineage") or "")
    node_id = str(checked.get("node_id") or "")

    refusal: list[str] = []
    digest: str | None = None
    payload_path = ""
    if spore is None:
        refusal.append(NO_PAYLOAD)
    else:
        payload = Path(spore).expanduser()
        planned = plan_reproduction(
            payload, lineage, Target(node_id=node_id, command=[]),
            declared_lineages=declared_lineages,
            declared_artifacts=declared_artifacts,
            grants_path=grants_path,
            sandbox_confirmed=False)
        refusal = list(planned.refusals)
        digest = planned.artifact_sha256
        payload_path = str(payload)

    # Read-only: a staging round must not append to the grant audit trail.
    grant = grant_status("reproduce", path=grants_path, include_audit=False)
    issued_at = int(time.time() if now is None else now)
    plan = {
        "action": "reproduce.spawn",
        "approved": False,
        "author": str(getattr(proposal, "author", "")),
        "goal": str(getattr(proposal, "goal", "")),
        "grant": {
            "action": "reproduce",
            "reason": str(grant.get("reason") or ""),
            "status": str(grant.get("status") or ""),
        },
        "issued_at": issued_at,
        "payload": {"path": payload_path, "sha256": digest},
        "proposal_id": proposal_id,
        "refusals": refusal,
        "requested": {"lineage": lineage, "node_id": node_id},
        "requires": list(REQUIRES),
        "spawned": False,
    }
    plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
    attestation = {
        "algorithm": "ed25519",
        "plan_hash": plan_hash,
        "proof": identity.proof(plan),
        "status": "staged-not-applied",
    }
    manifest = {
        "action": "reproduce.spawn",
        "applied": False,
        "approved": False,
        "author": plan["author"],
        "issued_at": issued_at,
        "plan_hash": plan_hash,
        "proposal_id": proposal_id,
        "refusals": refusal,
        "status": "staged-not-applied",
    }
    root = Path(outbox).expanduser() / proposal_id
    _write_atomic(root / "plan.json",
                  json.dumps(plan, ensure_ascii=False, sort_keys=True,
                             indent=2) + "\n", 0o600)
    _write_atomic(root / "attestation.json",
                  json.dumps(attestation, sort_keys=True, indent=2) + "\n",
                  0o600)
    _write_atomic(root / "manifest.json",
                  json.dumps(manifest, sort_keys=True, indent=2) + "\n", 0o644)
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
    """True when a staged plan is intact, unsigned-for-nothing and inert.

    A reviewer can run this without trusting the manifest: the hash is
    recomputed from ``plan.json`` and the Ed25519 proof is checked against the
    payload that hash covers. A plan that claims to be approved or spawned
    fails verification, so a hand-edited artifact cannot borrow the signature
    of the one that was reviewed.
    """
    root = Path(directory).expanduser()
    try:
        plan = _read_json(root / "plan.json")
        attestation = _read_json(root / "attestation.json")
        manifest = _read_json(root / "manifest.json")
    except SpawnError:
        return False
    if plan.get("approved") is not False or plan.get("spawned") is not False:
        return False
    if manifest.get("applied") is not False:
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
