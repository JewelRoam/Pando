"""Draft a signed proposal-operator spec without installing an operator.

``meta.propose`` is the seed of Tier 2: the group asking for a *new way to
propose*, rather than for one more proposal. Today's operator set is fixed in
``pando/genesis.py`` — a human wrote every one of them, and the gate can only
run what is compiled into it. This module implements the smallest useful
actuator for that request: an adopted proposal writes a *spec* into a local
outbox naming the operator it wants, and lists every reason the parent still
refuses to install one.

Nothing here defines a class, imports a module, evaluates a string, writes a
source file, or opens a socket. A new operator is code. Code that arrives from
a proposal is the one thing this whole action space must never do silently, so
the artifact carries labels and prose and nothing else: the implementation is
supplied by the gate, by a human, and reviewed before the gate can load it.

The typed intent is deliberately narrow::

    {"operator": "<label>", "rationale": "<prose>"}

A ``code``, ``source``, ``module``, ``path`` or ``command`` key is refused
rather than ignored — not because the draft would run it, but so that a
proposal that tries is *visible in the ledger* instead of quietly reduced to a
label. A refused intent keeps its reason and no ``skipped`` marker, so it is
retried and re-refused next round rather than looking handled.

A spec holds no secret, so re-staging rewrites it: a draft that keeps listing a
requirement which has since been met is a stale accusation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .identity import Identity, canonical
from .identity import verify as verify_signature

DEFAULT_OUTBOX = "~/.pandosi/operator-outbox"
ALLOWED_PARAMS = {"operator", "rationale"}
MAX_LABEL = 128
MAX_RATIONALE = 500

#: Keys that would carry an executable surface. Named explicitly so the
#: refusal message can say *what* was seen, not just "unknown key".
CODE_KEYS = ("code", "source", "module", "import", "path", "command", "cwd",
             "argv", "payload", "url", "endpoint", "api_key", "token")

#: What a human (or a later supervisor) must supply before this spec can be
#: installed. Written into every artifact so the hold is legible without code.
REQUIRES = (
    "an operator implementation, supplied by the gate - a human writes the "
    "class; the proposal never carries code",
    "a gate version that can load operators at all: today pando/genesis.py "
    "hard-codes its operator set, so Tier 2 needs a loader first",
    "human review of the operator's action space before it may emit anything "
    "- an operator that can propose outside ACTION_SPACE widens the gate "
    "without a vote",
    "no new paid or network dependency: an operator that calls a cloud API is "
    "a paid resource plus an external dependency and must be asked about "
    "first, never switched on quietly",
    "a rollback: the previous operator set stays runnable, because an operator "
    "that can only be replaced by editing the running gate is not reversible",
)


class OperatorError(ValueError):
    """A meta.propose intent was refused. The message is the audit reason."""


def validate(params) -> tuple[dict | None, str]:
    """Validate the typed ``meta.propose`` intent. Pure.

    Labels and prose only. An executable-surface key is refused rather than
    ignored so the attempt reaches the ledger instead of vanishing.
    """
    if params is None or params == {} or params == []:
        return {}, ""
    if not isinstance(params, dict):
        return None, ('params must be {} or '
                      '{"operator": "<label>", "rationale": "<prose>"}')
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        return None, ("unknown meta.propose params: "
                      + ", ".join(sorted(str(name) for name in unknown))
                      + " (this action accepts a label and prose only; the "
                        "operator implementation is supplied by the gate, "
                        "never by the proposal)")
    checked: dict = {}
    label = params.get("operator")
    if label is not None:
        if not isinstance(label, str) or not label.strip():
            return None, "params.operator must be a non-empty string"
        if len(label.strip()) > MAX_LABEL:
            return None, f"params.operator must be at most {MAX_LABEL} characters"
        checked["operator"] = label.strip()
    rationale = params.get("rationale")
    if rationale is not None:
        if not isinstance(rationale, str):
            return None, "params.rationale must be a string"
        text = rationale.strip()
        if len(text) > MAX_RATIONALE:
            return None, (f"params.rationale must be at most {MAX_RATIONALE} "
                          "characters")
        if text:
            checked["rationale"] = text
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
        raise OperatorError("proposal id is not a safe path component")
    return name


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorError(f"unreadable operator spec {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise OperatorError(f"operator spec {path.name} is not an object")
    return data


def stage(proposal, *, identity: Identity,
          outbox: str | Path = DEFAULT_OUTBOX,
          now: float | None = None) -> dict:
    """Write one signed, inactive operator spec into ``outbox``.

    Returns the spec hash, the refusal list and the directory written. No
    module is imported, no class is defined, no gate is touched.
    """
    checked, why = validate(getattr(proposal, "params", None))
    if checked is None:
        raise OperatorError(why)
    proposal_id = _safe_component(getattr(proposal, "proposal_id", ""))
    operator = str(checked.get("operator") or "")

    refusal: list[str] = []
    if not operator:
        refusal.append("no operator named: the proposal asks for a new "
                       "operator without saying which one")
    refusal.append("no implementation: the spec names an operator, and a "
                   "human-supplied class is what would make it real")

    issued_at = int(time.time() if now is None else now)
    spec = {
        "action": "meta.propose",
        "approved": False,
        "author": str(getattr(proposal, "author", "")),
        "goal": str(getattr(proposal, "goal", "")),
        "installed": False,
        "issued_at": issued_at,
        "proposal_id": proposal_id,
        "rationale": str(checked.get("rationale") or ""),
        "refusals": refusal,
        "requested": {"operator": operator},
        "requires": list(REQUIRES),
    }
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    attestation = {
        "algorithm": "ed25519",
        "proof": identity.proof(spec),
        "spec_hash": spec_hash,
        "status": "staged-not-applied",
    }
    manifest = {
        "action": "meta.propose",
        "applied": False,
        "approved": False,
        "author": spec["author"],
        "installed": False,
        "issued_at": issued_at,
        "operator": operator,
        "proposal_id": proposal_id,
        "refusals": refusal,
        "spec_hash": spec_hash,
        "status": "staged-not-applied",
    }
    root = Path(outbox).expanduser() / proposal_id
    _write_atomic(root / "spec.json",
                  json.dumps(spec, ensure_ascii=False, sort_keys=True,
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
        "refusals": refusal,
        "spec_hash": spec_hash,
        "staged": True,
    }


def verify(directory: str | Path) -> bool:
    """True when a staged operator spec is intact, unsigned-for-nothing, inert.

    A reviewer can run this without trusting the manifest: the hash is
    recomputed from ``spec.json`` and the Ed25519 proof is checked against the
    payload that hash covers. A spec that claims to be approved or installed
    fails verification, so a hand-edited artifact cannot borrow the signature of
    the one that was reviewed.
    """
    root = Path(directory).expanduser()
    try:
        spec = _read_json(root / "spec.json")
        attestation = _read_json(root / "attestation.json")
        manifest = _read_json(root / "manifest.json")
    except OperatorError:
        return False
    if spec.get("approved") is not False or spec.get("installed") is not False:
        return False
    if manifest.get("applied") is not False or manifest.get("installed") is not False:
        return False
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if spec_hash != str(manifest.get("spec_hash") or ""):
        return False
    if spec_hash != str(attestation.get("spec_hash") or ""):
        return False
    proof = attestation.get("proof")
    if not isinstance(proof, dict):
        return False
    return verify_signature(str(proof.get("pubkey") or ""), spec,
                            str(proof.get("sig") or ""))
