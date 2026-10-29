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
from dataclasses import replace
from pathlib import Path

from .identity import Identity, canonical
from .identity import verify as verify_signature

DEFAULT_OUTBOX = "~/.pando/operator-outbox"
#: Where a reviewed operator lands once the gate has agreed to run it. One file
#: per verb, mode 0600, *beside* the spec it came from and never inside it: the
#: outbox is what was asked for, this directory is what the gate will load, and
#: a draft that is also an install would be executable without a review.
DEFAULT_INSTALLED = "~/.pando/operators"

#: Append-only record of installs and retirements. The rollback requirement in
#: ``REQUIRES`` is only met if a retirement leaves a trace, so the file moves to
#: ``retired/`` and this ledger keeps the reason.
DEFAULT_INSTALL_LEDGER = "~/.pando/operator-install-ledger.jsonl"

#: One row per automatic catalogue deployment attempt. Kept separate from the
#: install ledger: the latter records the host's operator lifecycle, while this
#: one records why the standing deployment grant did or did not release a
#: particular staged spec.
DEFAULT_DEPLOY_LEDGER = "~/.pando/operator-deploy-ledger.jsonl"

#: The keys a reviewer may supply as the operator's *implementation*. Both
#: describe how an existing actuator is reused; neither carries code, and a key
#: that would is refused by name rather than ignored.
IMPLEMENTATION_KEYS = {"binds_to", "params"}

#: The actuators an installed verb may bind to: exactly the ones
#: ``pando/execute.py`` stages into an outbox by name. The three that are absent
#: are absent for reasons rather than by omission -- ``seed.export`` and
#: ``seed.replicate`` carry a key and a destination, and ``policy.set`` writes
#: the live rule store instead of an inbox. A binding outside this set would be
#: a verb the gate accepts and never carries out, which is the empty shell this
#: module exists to stop, so it is refused at install time rather than
#: discovered as a silent no-op later.
#:
#: A binding to *any* of these still cannot widen the gate: the reviewer fixed
#: the params, so the most a proposal can do is ask for a step someone already
#: reviewed. ``tests/test_smoke.py`` holds this set equal to the executor's
#: table so the two cannot drift apart.
BINDABLE = frozenset({
    "culture.refine",
    "recruit.invite",
    "reproduce.spawn",
    "resource.rebalance",
    "meta.propose",
})
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
    "an operator implementation, supplied by the gate - a reviewer writes the "
    "*binding*, not the proposal: which actuator the new verb reuses, with "
    "which fixed params, under whose name. Nothing a proposal carries can "
    "reach that: a draft supplies a label and prose",
    "a gate that loads the installed set: ``installed`` reads the records, "
    "``merge_action_space`` adds them to the built-ins, and ``pando/execute.py`` "
    "resolves a verb to the actuator it binds to. What is still hard-coded is "
    "the built-in set in ``pando/genesis.py``",
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
    refusal.append("no implementation: the spec names an operator, and the "
                   "gate has to bind it to an actuator before it can run")

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


def verify(directory: str | Path,
           expected_pubkey: str | None = None) -> bool:
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
    pubkey = str(proof.get("pubkey") or "")
    if expected_pubkey is not None and pubkey != expected_pubkey:
        return False
    return verify_signature(pubkey, spec, str(proof.get("sig") or ""))
# ---------------------------------------------------------------------------
# The gate's half: turning a reviewed draft into a verb the gate will run.
#
# Everything above stops at a signed draft and says so. That hold is the point
# -- a proposal may not widen the gate -- but a hold with no way to clear it is
# not caution, it is a wall, and the group spent 33 of 40 rounds asking a wall.
# The functions below are the clearing mechanism, and they are written to make
# the *proposal* unable to use them: a spec supplies a name and prose, and
# every executable fact (which actuator the verb reuses, with what fixed
# params, under whose review) arrives from the caller.
# ---------------------------------------------------------------------------


def _emit_ledger(ledger: str | Path, entry: dict) -> None:
    path = Path(ledger).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def installed(directory: str | Path = DEFAULT_INSTALLED) -> dict[str, dict]:
    """Every operator this gate has agreed to run, keyed by name.

    Fail-closed on a broken file rather than skipping it: one unreadable
    record means the operator set on this host is not what the ledgers say it
    is, and a gate that quietly runs the smaller set is the failure this whole
    module exists to make impossible. A missing directory is not a problem --
    it is the ordinary state of a host that has installed nothing.
    """
    root = Path(directory).expanduser()
    found: dict[str, dict] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.json")):
        record = _read_json(path)
        name = str(record.get("operator") or "")
        binds = str(record.get("binds_to") or "")
        if not name or not binds:
            raise OperatorError(
                f"installed operator {path.name} names no operator or binds "
                "to nothing; the gate will not guess what it meant")
        if name in found:
            raise OperatorError(
                f"installed operator {name} is defined twice; the gate will "
                "not pick one")
        found[name] = record
    return found


def binding(record: dict) -> tuple[str, dict]:
    """``(actuator, fixed params)`` for one installed record, fail-closed.

    Read by the executor on every round, so a record that lost its binding or
    grew params that are not an object stops the round with a reason instead of
    running something the reviewer did not write.
    """
    actuator = str(record.get("binds_to") or "")
    if not actuator:
        raise OperatorError(
            f"installed operator {record.get('operator') or '?'} binds to "
            "nothing; the gate will not guess what it meant")
    params = record.get("params")
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise OperatorError(
            f"installed operator {record.get('operator') or '?'} has params "
            "that are not an object")
    return actuator, dict(params)


def rebound(proposal, record: dict):
    """The same proposal aimed at the actuator the record binds to.

    An installed verb is a fixed parameterisation of an actuator the gate
    already runs, so the executor rewrites the proposal rather than growing a
    second dispatch path. The proposal id, author, goal and signature are
    untouched -- only ``action`` and ``params`` move, and they move to values a
    reviewer chose before the proposal was written. What the group voted on is
    still what the ledger names.
    """
    actuator, params = binding(record)
    return replace(proposal, action=actuator, params=params)


def merge_action_space(core: dict, extra: dict) -> dict[str, str]:
    """The action space the gate actually runs: built-ins plus installed verbs.

    A name that collides with a built-in is refused rather than shadowed.
    Redefining a verb the gate already has is precisely the move that would
    need a vote, and ``dict.update`` is not one.
    """
    merged = dict(core)
    for name, record in extra.items():
        if name in merged:
            raise OperatorError(
                f"installed operator {name} collides with an action this gate "
                "already runs; it must be renamed, not shadowed")
        merged[name] = str(record.get("rationale")
                           or f"binds to {record.get('binds_to')}")
    return merged


def install(spec_dir: str | Path, *, identity: Identity, action_space: dict,
            implementation: dict, reviewer: str = "",
            validate_params=None,
            installed_dir: str | Path = DEFAULT_INSTALLED,
            ledger: str | Path = DEFAULT_INSTALL_LEDGER,
            now: float | None = None) -> dict:
    """Turn one reviewed draft into an operator this gate will load.

    Three things must hold, and a proposal can supply none of them: the draft
    must still verify (same spec, same signature, still inert), a reviewer must
    supply the *implementation*, and the name must not already mean something.
    The draft names the verb; it never decides what the verb does.

    The implementation is deliberately not code. It is a binding to an actuator
    the gate already runs, plus the params that actuator will see -- so a new
    verb can name a rule the group wants without any path by which a proposal
    reaches an interpreter. A binding to an action this gate cannot run is
    refused rather than stored: a verb with no actuator is the empty shell that
    made ``meta.propose`` unreachable for 33 rounds.
    """
    spec_path = Path(spec_dir).expanduser()
    if not verify(spec_path):
        raise OperatorError(
            "spec is not installable: it does not verify (tampered with, or it "
            "already claims to be approved or installed)")
    spec = _read_json(spec_path / "spec.json")
    manifest = _read_json(spec_path / "manifest.json")
    requested = spec.get("requested")
    name = str((requested or {}).get("operator") or "").strip() \
        if isinstance(requested, dict) else ""
    if not name:
        raise OperatorError("spec names no operator: there is nothing to install")
    if name in action_space:
        raise OperatorError(f"{name} is already an action this gate runs; a new "
                            "operator has to be a new verb")
    if not isinstance(implementation, dict):
        raise OperatorError('implementation must be {"binds_to": "<action>", '
                            '"params": {...}}')
    code_keys = sorted(set(implementation) & set(CODE_KEYS))
    if code_keys:
        raise OperatorError(
            "implementation carries executable surface: " + ", ".join(code_keys)
            + " (an implementation is a binding to an actuator the gate already "
              "runs, never code)")
    unknown = sorted(set(implementation) - IMPLEMENTATION_KEYS - set(CODE_KEYS))
    if unknown:
        raise OperatorError("unknown implementation keys: " + ", ".join(unknown))
    binds_to = str(implementation.get("binds_to") or "")
    if not binds_to:
        raise OperatorError(
            "implementation names no binds_to: a new verb with no actuator is "
            "the empty shell this action exists to stop")
    if binds_to not in action_space:
        raise OperatorError(f"implementation binds to {binds_to}, which this "
                            "gate cannot run")
    if binds_to not in BINDABLE:
        raise OperatorError(
            f"implementation binds to {binds_to}, which this executor stages "
            "into an outbox nowhere: a new verb may only reuse one of "
            + ", ".join(sorted(BINDABLE))
            + " (the other three carry a key, a destination or the live rule "
              "store, and are not outbox stages)")
    params = implementation.get("params")
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise OperatorError("implementation.params must be an object")
    if validate_params is not None and params:
        # The bound actuator has the last word on its own params. Without this
        # a reviewer could install a verb the actuator refuses on every round
        # -- the empty shell with extra steps -- and the refusal would only
        # show up in the ledger as a failure nobody asked for.
        checked_params, why = validate_params(params)
        if checked_params is None:
            raise OperatorError(
                f"implementation.params are not ones {binds_to} accepts: {why}")

    target = Path(installed_dir).expanduser()
    dest = target / f"{_safe_component(name)}.json"
    if dest.exists():
        raise OperatorError(f"{name} is already installed; retire it first")

    installed_at = int(time.time() if now is None else now)
    payload = {
        "author": str(spec.get("author") or ""),
        "binds_to": binds_to,
        "installed_at": installed_at,
        "operator": name,
        "params": params,
        "proposal_id": str(spec.get("proposal_id") or ""),
        "rationale": str(spec.get("rationale") or ""),
        "reviewer": str(reviewer or ""),
        "source_spec_hash": str(manifest.get("spec_hash") or ""),
        "status": "installed",
    }
    record = dict(payload)
    record["attestation"] = {
        "algorithm": "ed25519",
        "proof": identity.proof(payload),
    }
    _write_atomic(dest, json.dumps(record, ensure_ascii=False, sort_keys=True,
                                   indent=2) + "\n", 0o600)
    _emit_ledger(ledger, payload)
    return record


def uninstall(name: str, *, reason: str = "",
              installed_dir: str | Path = DEFAULT_INSTALLED,
              ledger: str | Path = DEFAULT_INSTALL_LEDGER,
              now: float | None = None) -> dict:
    """Retire an operator without erasing that it ran.

    Rollback is one of the spec's own requirements, not a nicety: the file
    moves to ``retired/``, the ledger keeps the reason, and the gate stops
    loading it on the next read. Nothing else on the host is touched, and the
    spec that asked for it keeps its own hash.
    """
    target = Path(installed_dir).expanduser()
    source = target / f"{_safe_component(name)}.json"
    if not source.is_file():
        raise OperatorError(f"{name} is not installed")
    record = _read_json(source)
    retired_at = int(time.time() if now is None else now)
    retired_dir = target / "retired"
    retired_dir.mkdir(parents=True, exist_ok=True)
    dest = retired_dir / f"{name}.{retired_at}.json"
    os.replace(source, dest)
    entry = {
        "operator": name,
        "reason": str(reason or ""),
        "retired_at": retired_at,
        "status": "retired",
    }
    _emit_ledger(ledger, entry)
    return {
        **entry,
        "restored_from_proposal": str(record.get("proposal_id") or ""),
        "retired_to": str(dest),
    }
