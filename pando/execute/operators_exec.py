"""Operator lifecycle on the executor side: deploy-attempt ledger + auto-deploy.

``auto_deploy_profiles`` installs at most one exact catalogue profile from an
adopted local spec, only when a standing human grant permits the deployment
action itself.  The implementation and params come from
``pando.operator_profiles``; this module only proves the signed proposal was
adopted, its spec is intact and local, and the grant matches.
"""
import datetime as dt
import json
from pathlib import Path

from ..genesis import ACTION_SPACE, Proposal
from ..grants import grant_fingerprint, require_matching_grant
from ..operator_profiles import OperatorProfileError
from ..operator_profiles import VALIDATORS as OPERATOR_VALIDATORS
from ..operator_profiles import resolve as resolve_profile
from ..operators import BINDABLE
from ..operators import OperatorError
from ..operators import install as install_operator
from ..operators import verify as verify_operator

from .judge import adopted

def addressed_replications(proposals: list[Proposal], reactions: list[dict],
                           done: set[str], signer: str,
                           knobs: dict | None = None) -> list[tuple[str, str]]:
    """Adopted seed.replicate ids signed here that name a host.

    A pull host gets an offer in the outbox. For every other named host there
    is no push transport; naming it is not permission to open a socket.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if proposal.pubkey != signer or not proposal.destination:
            continue
        if adopted(proposal, reactions, knobs):
            found.append((proposal.proposal_id, proposal.destination))
    return found


def next_installed(proposals: list[Proposal], reactions: list[dict],
                   done: set[str], signer: str | None,
                   installed_operators: dict,
                   knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted proposal naming an installed verb this host authored.

    One per round, like every other actuator here: a round that staged every
    installed verb at once would put several unrelated artifacts in flight on
    the strength of one vote each, and the ledger would stop being a
    description of what happened in order.
    """
    for proposal in proposals:
        if proposal.proposal_id in done:
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if proposal.action not in installed_operators:
            continue
        if adopted(proposal, reactions, knobs, installed_operators):
            return proposal
    return None


def _deployment_attempt_key(row: dict) -> tuple[str, str, str, str]:
    return (str(row.get("proposal_id") or ""),
            str(row.get("spec_hash") or ""),
            str(row.get("status") or ""),
            str(row.get("reason") or ""))


def _deployment_attempt_keys(path: Path) -> set[tuple[str, str, str, str]]:
    found: set[tuple[str, str, str, str]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            found.add(_deployment_attempt_key(row))
    return found


def _append_deployment_attempt(path: Path, row: dict,
                               seen: set[tuple[str, str, str, str]]) -> bool:
    """Append one distinct attempt and return whether it was new.

    Refusals are retried by the executor, so the same missing grant or invalid
    spec must not add an identical row every minute. ``at`` is deliberately not
    part of the key: time passing is not new evidence.
    """
    key = _deployment_attempt_key(row)
    if key in seen:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    seen.add(key)
    return True


def _read_operator_spec(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorError(f"unreadable operator spec {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"operator spec {path.name} is not an object")
    return value
def auto_deploy_profiles(proposals: list[Proposal], reactions: list[dict], *,
                         identity, signer: str, knobs: dict | None,
                         outbox: str | Path, installed_dir: str | Path,
                         install_ledger: str | Path,
                         deploy_ledger: str | Path,
                         grants_path: str | Path | None,
                         host_config: dict | None = None,
                         installed_operators: dict | None = None,
                         installed_error: str = "",
                         now: dt.datetime | None = None) -> list[dict]:
    """Install at most one exact catalogue profile from adopted local specs.

    A proposal can name a profile, never supply one. The implementation and
    params come from ``pando.operator_profiles``; this function only proves
    that the signed proposal was adopted, its spec is intact and local, and a
    standing human grant permits the deployment action itself. It then calls
    the same ``operators.install`` path used by the manual tool.
    """
    out_root = Path(outbox).expanduser()
    ledger_path = Path(deploy_ledger).expanduser()
    seen = _deployment_attempt_keys(ledger_path)
    installed = dict(installed_operators or {})
    attempts: list[dict] = []

    def note(*, proposal_id: str, spec_hash: str, status: str, reason: str,
             operator: str = "", profile: dict | None = None,
             grant: dict | None = None, installed_at: int | None = None,
             directory: Path | None = None) -> dict:
        row = {
            "action": "deploy_operator",
            "proposal_id": proposal_id,
            "spec_hash": spec_hash,
            "status": status,
            "reason": reason,
        }
        if operator:
            row["operator"] = operator
        if profile:
            row["profile"] = "auto-profile-catalogue/v1"
            row["binds_to"] = profile["binds_to"]
            row["params"] = profile["params"]
        if grant:
            row["grant_fingerprint"] = grant_fingerprint(grant)
            row["grant_evidence"] = str(grant.get("evidence") or "")
        if installed_at is not None:
            row["installed_at"] = installed_at
        if directory is not None:
            row["path"] = str(directory)
        try:
            # Resolve through the package so a test that patches
            # ``pando.execute._append_deployment_attempt`` is honoured.
            from . import _append_deployment_attempt
            _append_deployment_attempt(ledger_path, row, seen)
        except OSError as exc:
            # The deployment ledger is part of the authorization boundary.
            # Callers decide whether a failed write must stop before or after
            # installation; this helper must never turn a disk error into an
            # unhandled exception or a second, recursive write attempt.
            row["ledger_error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(row)
        return row

    for proposal in proposals:
        if proposal.action != "meta.propose":
            continue
        if not adopted(proposal, reactions, knobs):
            continue
        proposal_id = str(proposal.proposal_id or "")
        if (not proposal_id or proposal_id in {".", ".."}
                or "/" in proposal_id or "\\" in proposal_id):
            note(proposal_id=proposal_id, spec_hash="", status="refused",
                 reason="proposal id is not a safe path component")
            continue
        spec_dir = out_root / proposal_id
        if not spec_dir.is_dir():
            continue

        if proposal.pubkey != signer:
            note(proposal_id=proposal_id, spec_hash="", status="refused",
                 reason="proposal was not signed by this host identity")
            continue
        if not verify_operator(spec_dir, expected_pubkey=signer):
            note(proposal_id=proposal_id, spec_hash="", status="refused",
                 reason="operator spec signature or hash verification failed")
            continue
        try:
            spec = _read_operator_spec(spec_dir / "spec.json")
            manifest = _read_operator_spec(spec_dir / "manifest.json")
        except OperatorError as exc:
            note(proposal_id=proposal_id, spec_hash="", status="refused",
                 reason=str(exc))
            continue
        spec_hash = str(manifest.get("spec_hash") or "")
        requested = spec.get("requested")
        if (spec.get("action") != "meta.propose"
                or spec.get("proposal_id") != proposal_id
                or manifest.get("action") != "meta.propose"
                or manifest.get("proposal_id") != proposal_id):
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 reason="operator spec does not identify this meta.propose")
            continue
        if not isinstance(requested, dict) or set(requested) != {"operator"}:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 reason="requested must contain exactly one operator key")
            continue
        name = requested.get("operator")
        proposal_params = proposal.params if isinstance(proposal.params, dict) else {}
        if not isinstance(name, str) or proposal_params.get("operator") != name:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 reason="operator spec does not match the signed proposal")
            continue
        if manifest.get("operator") != name:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name,
                 reason="manifest operator does not match the signed spec")
            continue
        try:
            from .. import hostconfig
            profile = resolve_profile(
                name, fleet=hostconfig.profile_params(host_config or {}, name))
        except OperatorProfileError as exc:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, reason=str(exc))
            continue
        if profile["binds_to"] not in BINDABLE:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile,
                 reason="catalogue binding is outside the executor boundary")
            continue
        checked, why = OPERATOR_VALIDATORS[profile["binds_to"]](profile["params"])
        if checked is None:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile,
                 reason=f"catalogue params failed validation: {why}")
            continue

        existing = installed.get(name)
        if existing is not None:
            same_spec = str(existing.get("source_spec_hash") or "") == spec_hash
            same_binding = (existing.get("binds_to") == profile["binds_to"]
                            and existing.get("params") == profile["params"])
            if same_spec and same_binding:
                note(proposal_id=proposal_id, spec_hash=spec_hash,
                     status="already-installed", operator=name,
                     profile=profile, installed_at=existing.get("installed_at"),
                     directory=Path(installed_dir).expanduser() / f"{name}.json",
                     reason="this exact spec and catalogue binding are installed")
                continue
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile,
                 reason="a different operator with this name is installed; retire it first")
            continue
        if installed_error:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile,
                 reason=f"installed operator state is unreadable: {installed_error}")
            return attempts

        try:
            grant = require_matching_grant(
                "deploy_operator", resource="profile-catalogue/v1",
                scope="operator-host", path=grants_path, now=now)
        except (PermissionError, ValueError, OSError) as exc:
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile,
                 reason=f"deployment grant refused: {exc}")
            return attempts

        # Record the released grant before mutating the installed set. If the
        # process dies during install, the deploy ledger still answers why the
        # action was attempted rather than showing an unexplained operator.
        # A ledger that cannot be written is a pre-install refusal: without
        # that durable evidence the executor must not mutate the installed set.
        released = note(
            proposal_id=proposal_id, spec_hash=spec_hash, status="granted",
            operator=name, profile=profile, grant=grant,
            reason="deployment grant matched; installing fixed profile")
        if "ledger_error" in released:
            released["status"] = "refused"
            released["reason"] = (
                "deployment ledger unavailable; operator not installed: "
                + released["ledger_error"])
            return attempts
        try:
            record = install_operator(
                spec_dir, identity=identity, action_space=ACTION_SPACE,
                implementation={"binds_to": profile["binds_to"],
                                "params": profile["params"]},
                reviewer=profile["reviewer"],
                validate_params=OPERATOR_VALIDATORS[profile["binds_to"]],
                installed_dir=installed_dir, ledger=install_ledger)
        except Exception as exc:  # noqa: BLE001 - one action must not kill the loop
            note(proposal_id=proposal_id, spec_hash=spec_hash, status="refused",
                 operator=name, profile=profile, grant=grant,
                 reason=f"install refused: {type(exc).__name__}: {exc}")
            return attempts
        # Installation has already changed the host. A failure to append the
        # final row must be reported as ledger_error on an installed result,
        # never rewritten as a refusal: a retry could otherwise try to install
        # the same operator again.
        note(proposal_id=proposal_id, spec_hash=spec_hash, status="installed",
             operator=name, profile=profile, grant=grant,
             installed_at=record.get("installed_at"),
             directory=Path(installed_dir).expanduser() / f"{name}.json",
             reason="installed fixed profile from the authorized catalogue")
        return attempts
    return attempts
