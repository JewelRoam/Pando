"""Vote tallying, the adoption judge, and the retry budget.

The judge is a pure function of its arguments: policy knobs are passed in,
not read from the disk, so a round judges every proposal under the same
rules and tests can fix a policy without touching the filesystem.
"""
import json
from collections import Counter
from pathlib import Path

from ..genesis import ACTION_SPACE, Proposal
from ..operators import OperatorError
from ..operators import binding
from ..operators import rebound

from .consts import ADOPT, DECLINE, HOLD

MAX_ATTEMPTS = 3
def failure_counts(ledger: Path) -> Counter:
    """Rounds in which each proposal raised, keyed by proposal id."""
    found: Counter = Counter()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("failed") and not row.get("executed") and row.get("proposal_id"):
            found[str(row["proposal_id"])] += 1
    return found


def deferred_reasons(ledger: Path) -> set[tuple[str, str]]:
    """``(proposal_id, reason)`` pairs the ledger has already explained.

    A deferral is neither a failure nor a settlement: the proposal stays
    unsettled and is re-attempted next round, because the reason may be a
    precondition a human can satisfy (a passphrase) rather than a defect in
    the proposal. Writing the row every round would grow the ledger without
    adding information, so the pair is written once and the per-round status
    file carries it afterwards.
    """
    found: set[tuple[str, str]] = set()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("deferred") and row.get("proposal_id"):
            found.add((str(row["proposal_id"]), str(row["deferred"])))
    return found


def settle_after_failure(record: dict, failures: Counter) -> dict:
    """Turn a repeatedly-failing record into a settled refusal, in place.

    The row keeps its ``failed`` reason and gains ``skipped`` plus ``attempts``.
    ``skipped`` is what ``left_ids`` reads, so the proposal stops being picked
    while the reason it was dropped stays in the ledger. Same three states as
    everywhere else here: ``executed`` true means done, ``skipped`` means
    settled, neither means retry.
    """
    if record.get("executed") or record.get("skipped"):
        return record
    pid = str(record.get("proposal_id") or "")
    attempts = failures.get(pid, 0) + 1
    if pid and attempts >= MAX_ATTEMPTS:
        record["skipped"] = "gave-up"
        record["attempts"] = attempts
    return record

def tally(proposal: Proposal, reactions: list[dict]) -> dict[str, list[str]]:
    """One vote per decision model. The author does not count."""
    buckets = {ADOPT: [], HOLD: [], DECLINE: []}
    seen: set[str] = set()
    for reaction in reactions:
        if str(reaction.get("target_id") or "") != proposal.proposal_id:
            continue
        node = str(reaction.get("node_id") or "").strip()
        stance = str(reaction.get("stance") or "").strip()
        if not node or node == proposal.author or node in seen:
            continue
        if stance not in buckets:
            continue
        seen.add(node)
        buckets[stance].append(node)
    return buckets


def judge(proposal: Proposal, reactions: list[dict], *,
          min_adopt: int = 2, veto: bool = True,
          actuators: dict | None = None,
          installed_operators: dict | None = None) -> dict:
    """Return what should happen. Does not perform the action by itself.

    ``actuators`` maps an action name to ``(Proposal) -> bool``. Missing
    means the colony may have adopted the proposal and still must not act.

    ``installed_operators`` is what ``pando.operators.installed`` read: the
    verbs this host has agreed to run, keyed by name. A verb in there is a
    fixed parameterisation of an actuator the gate already runs, so it is
    judged like any other action and then resolved to its binding. The map
    defaults to empty, which is the fail-closed direction: without the records
    an installed verb is not in the action space, and a proposal naming one is
    held rather than guessed at.
    """
    votes = tally(proposal, reactions)
    result = {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action,
        "author": proposal.author,
        "adopt": list(votes[ADOPT]),
        "hold": list(votes[HOLD]),
        "decline": list(votes[DECLINE]),
        "adopted": False,
        "executed": False,
        "success": None,
        "detail": "",
    }
    record = installed_operators.get(proposal.action) \
        if installed_operators else None
    if proposal.action not in ACTION_SPACE and record is None:
        result["detail"] = "held: action is not in the action space"
        return result
    if veto and votes[DECLINE]:
        result["detail"] = "held: a decision model declined"
        return result
    if len(votes[ADOPT]) < min_adopt:
        result["detail"] = (f"held: {len(votes[ADOPT])} adopt "
                            f"< {min_adopt}")
        return result
    result["adopted"] = True
    target = proposal
    actor = proposal.action
    if record is not None:
        # A verb nobody wrote code for. The record says which actuator it
        # reuses and with what params; a record that cannot say stops here
        # rather than reaching for the proposal's own params, which are the
        # proposer's and not the reviewer's.
        try:
            actor, _fixed = binding(record)
        except OperatorError as exc:
            result["detail"] = f"adopted, not executed: {exc}"
            return result
        result["binds_to"] = actor
        target = rebound(proposal, record)
    actuator = (actuators or {}).get(actor)
    if actuator is None:
        result["detail"] = "adopted, not executed: no actuator"
        return result
    try:
        ok = bool(actuator(target))
    except Exception as exc:
        result["detail"] = f"actuator failed before completion: {exc}"
        return result
    result["executed"] = True
    result["success"] = ok
    result["detail"] = "actuator reported " + ("success" if ok else "failure")
    return result




def adopted(proposal: Proposal, reactions: list[dict],
            knobs: dict | None = None,
            installed_operators: dict | None = None) -> bool:
    """Does the gate adopt this proposal, under the current policy?

    ``knobs`` is what ``pando/policy.py`` loaded. The knob names are the
    keyword arguments of ``judge`` on purpose, so the store is passed straight
    through. It is read here and not inside ``judge`` so that the judge stays a
    pure function of its arguments and the tests can fix a policy without
    touching the disk.
    """
    return bool(judge(proposal, reactions, actuators={},
                      installed_operators=installed_operators,
                      **dict(knobs or {}))["adopted"])
