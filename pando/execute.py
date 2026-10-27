"""Decide whether a proposal may be executed, and carry out the few it owns.

The colony can already propose and vote. Handing every adopted proposal
straight to an actuator would run almost every one of them: the decision
models currently answer 赞同 far more often than they refuse. So an action
runs only when three things are true — the action is in the action space,
enough other decision models adopted it under the policy in force, and a
tested actuator for that exact action exists.

``ACTUATORS`` is that boundary, and it is short on purpose. As of today:

- ``seed.export`` — seal this host's identity into a bundle. Wired.
- ``seed.replicate`` — stage that bundle for a host that pulls it. The offer
  waits in the local outbox; there is no push transport, and naming a host is
  not permission to open a socket to it.
- ``policy.set`` — move a knob in ``pando/policy.py``. Wired, bounded, and the
  only actuator that changes what the gate does next round.
- ``culture.refine`` — stage a signed, validated playbook *candidate* into a
  local outbox. Wired up to a review queue, not to a hot swap: nothing loads
  the candidate, so adopting a culture change can never rewrite the rules in
  the same breath as proposing it. Applying stays a human/gate step.
- ``recruit.invite`` — mint one signed, *inactive* pairing invitation into a
  local outbox. Wired up to a review queue, not to an admission: nothing writes
  the token into a running root, and the manifest is ``applied: false``. A
  staged invitation is a credential, so the artifact is written 0600 and never
  published to the mailbox or the ledger.
- ``reproduce.spawn`` — stage a signed reproduction *plan* into a local
  outbox: what would be copied, where, and every reason the parent still
  refuses. Same review-queue boundary as culture.refine, and wider: the
  proposal may carry labels only, ``sandbox_confirmed`` is always false, and
  nothing here calls ``pando.reproduce.reproduce``. A plan cannot be an
  approval.
- ``resource.rebalance`` — draft a signed *move* plan into a local outbox:
  which member, onto which host, and every precondition that is still unmet.
  Labels only (``node_id``, ``target_host``); a ``path`` or ``command`` is
  refused, so a proposal cannot name the state directory or the stop/start
  order. Nothing stops a member, starts one, or copies a byte.
- ``meta.propose`` — draft a signed *operator spec* into a local outbox: the
  name of a new proposal operator and the prose asking for it. This is the one
  action whose payload is code in the end, so the artifact carries labels only
  and ``code``/``source``/``module``/``path``/``command`` are refused rather
  than ignored. Nothing imports a module or defines a class.

That closes the action space: all eight declared actions now have an actuator,
and six of the eight stop at a signed draft in a local outbox. The two that
touch the world on their own are ``seed.export`` (writes a sealed bundle under
this host's own ``executions/``) and ``policy.set`` (moves one bounded knob).
Neither is reachable from a proposal alone.

Nothing here starts itself, and nothing here opens a socket. A dry run reads
the mailbox and prints counts. A ``--once`` round seals at most one seed,
applies at most one rule change, stages at most one culture candidate, mints at
most one invitation, and drafts at most one plan for each of reproduction,
member move, and a new operator — each recorded in an append-only ledger.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from .culture import DEFAULT_OUTBOX as DEFAULT_CULTURE_OUTBOX
from .culture import CultureError
from .culture import stage as stage_culture
from .genesis import ACTION_SPACE, Proposal
from .grants import grant_fingerprint, require_matching_grant
from .invite import DEFAULT_OUTBOX as DEFAULT_INVITE_OUTBOX
from .invite import InviteError
from .invite import stage as stage_invite
from .operator_profiles import OperatorProfileError
from .operator_profiles import VALIDATORS as OPERATOR_VALIDATORS
from .operator_profiles import names as profile_names
from .operator_profiles import resolve as resolve_profile
from .operators import BINDABLE
from .operators import DEFAULT_DEPLOY_LEDGER
from .operators import DEFAULT_INSTALLED
from .operators import DEFAULT_INSTALL_LEDGER
from .operators import DEFAULT_OUTBOX as DEFAULT_OPERATOR_OUTBOX
from .operators import OperatorError
from .operators import binding
from .operators import installed as load_installed
from .operators import install as install_operator
from .operators import rebound
from .operators import stage as stage_operator
from .operators import verify as verify_operator
from .rebalance import DEFAULT_OUTBOX as DEFAULT_REBALANCE_OUTBOX
from .rebalance import RebalanceError
from .rebalance import stage as stage_rebalance
from .spawn import DEFAULT_OUTBOX as DEFAULT_SPAWN_OUTBOX
from .spawn import SpawnError
from .spawn import stage as stage_spawn
from .policy import DEFAULT_PATH as DEFAULT_POLICY_PATH
from .policy import PolicyError
from .policy import apply as apply_policy
from .policy import load as load_policy
from .proposal_mail import (fetch_proposals, is_reaction_task,
                            mailbox_chain)

ADOPT = "赞同"
HOLD = "搁置"
DECLINE = "反对"
STANCES = (ADOPT, HOLD, DECLINE)

#: The actions this program can actually carry out. Everything else in
#: ``ACTION_SPACE`` is a declared intent: adopted, never executed. Kept as
#: one list so the boundary is readable from outside and cannot drift
#: between the judge, the executor and the ledger.
ACTUATORS: tuple[str, ...] = (
    "seed.export",
    "seed.replicate",
    "policy.set",
    "culture.refine",
    "recruit.invite",
    "reproduce.spawn",
    "resource.rebalance",
    "meta.propose",
)



#: How an installed verb is staged: the actuator it binds to, the outbox that
#: actuator writes, and the flag that relocates it. A new verb has no stager of
#: its own -- reusing the one it binds to is the whole point -- so this table is
#: the only place that knows how to carry out a verb nobody wrote code for.
#: Kept equal to ``pando.operators.BINDABLE`` by a test, so a verb can never be
#: installable and unstaged at once.
INSTALLED_STAGERS: dict[str, tuple] = {
    "culture.refine": (stage_culture, DEFAULT_CULTURE_OUTBOX, "culture_outbox"),
    "recruit.invite": (stage_invite, DEFAULT_INVITE_OUTBOX, "invite_outbox"),
    "reproduce.spawn": (stage_spawn, DEFAULT_SPAWN_OUTBOX, "spawn_outbox"),
    "resource.rebalance": (stage_rebalance, DEFAULT_REBALANCE_OUTBOX,
                           "rebalance_outbox"),
    "meta.propose": (stage_operator, DEFAULT_OPERATOR_OUTBOX,
                     "operator_outbox"),
}


def stage_bound(proposal, args, identity) -> dict:
    """Stage an installed verb through the actuator its record binds to."""
    entry = INSTALLED_STAGERS.get(proposal.action)
    if entry is None:
        raise OperatorError(
            f"no executor stages {proposal.action}; an installed verb may only "
            "bind to one of " + ", ".join(sorted(INSTALLED_STAGERS)))
    stager, default_outbox, arg_name = entry
    return stager(proposal, identity=identity,
                  outbox=getattr(args, arg_name, default_outbox))


def _titled(title: str, prefix: str) -> tuple[str, str] | None:
    """Second and third tokens of a ``[proposal]`` or ``[reaction]`` title.

    The mailbox listing has titles, not bodies. The id is the second token
    and the action or stance is the third, followed by ``by``.
    """
    parts = str(title).split()
    if len(parts) < 5 or parts[0] != prefix or parts[3] != "by":
        return None
    if not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


#: How deep the executor looks for its own un-acted work, per actuator. Every
#: actuator runs at most once per round, so this is only a bound on the cost
#: of a round, never a statement about which proposals are allowed to run.
DEFAULT_PER_ACTION = 40

#: Where a host keeps its executor state. The Mac has used ``~/.pandosi`` since
#: the first round; a node brought up under the shorter ``~/.pando`` should pass
#: ``--state-dir`` once rather than spell out ten paths -- spelling them out per
#: unit is how one host ends up with its ledger in one tree and its outboxes in
#: another, and how a seal writes a rollout home nobody reads.
DEFAULT_STATE_DIR = "~/.pandosi"

#: Path arguments that live inside the state dir, as argument name -> (leaf,
#: what the leaf is for). ``--identity`` is deliberately absent: both hosts
#: keep the identity in ``~/.pando``, which predates this flag and must not be
#: silently relocated by it.
STATE_PATHS: dict[str, tuple[str, str]] = {
    "token_file": ("mailbox.token", "the mailbox token this host reads"),
    "passphrase_file": ("seed.passphrase", "the passphrase that seals a seed"),
    "out_root": ("executions", "where a sealed seed is written"),
    "ledger": ("execute-ledger.jsonl", "what this host has already acted on"),
    "policy": ("policy.json", "the bounded self-modification store"),
    "culture_outbox": ("culture-outbox", "staged culture.refine candidates"),
    "invite_outbox": ("invites", "staged recruit.invite tokens"),
    "spawn_outbox": ("reproduce-outbox", "drafted reproduce.spawn plans"),
    "rebalance_outbox": ("rebalance-outbox", "drafted resource.rebalance plans"),
    "operator_outbox": ("operator-outbox", "drafted meta.propose specs"),
    # Not an outbox: this is the set the gate will *load* and run, so it sits
    # beside the state root rather than inside the draft tree. Defaulting it
    # here as well as in ``pando.operators`` keeps the two spellings of the
    # same directory from drifting apart on a host that relocated its state.
    "operators_dir": ("operators", "installed operator bindings this host runs"),
    "grants": ("grants.toml", "expiring capability grants"),
    "operator_install_ledger": (
        "operator-install-ledger.jsonl",
        "installed operator lifecycle ledger"),
    "operator_deploy_ledger": (
        "operator-deploy-ledger.jsonl",
        "automatic profile deployment ledger"),
}


def _author_of(title: str, prefix: str = "[proposal]") -> str:
    """The author token of a ``[proposal]`` title, or ``""``.

    ``publish_proposal`` writes ``"[proposal] <id> <action> by <author>"`` and
    an author is a single token, so the fifth token is the whole name. It is
    read without fetching the body, which is what lets retention tell this
    host's drafts from a peer's before paying for a GET.
    """
    parts = str(title).split()
    if len(parts) < 5 or parts[0] != prefix or parts[3] != "by":
        return ""
    return parts[4]


def rows_for_execution(tasks: list, limit: int,
                       per_action: int = DEFAULT_PER_ACTION,
                       settled: frozenset | set = frozenset(),
                       authors: set | None = None) -> list:
    """The rows one round may act on, in listing order.

    Three layers:

    1. the last ``limit`` rows -- the live window the dry run and the ledger
       describe;
    2. every unsettled ``seed.export`` and ``seed.replicate``. Those two are
       the only actions with a second host on the other end, so the send/receive
       bookkeeping has to see a peer's offer as well as this host's: an
       addressed replication pushed out of the window would be offered twice.
       A proposal already named by ``settled`` needs neither its own body nor
       its votes again;
    3. with ``authors`` set, the oldest ``per_action`` proposals of each
       actuator action that this host signs under and that no ledger row has
       settled yet (``settled``).

    Layer 3 is the one that is easy to leave out and hard to notice missing.
    Without it an adopted ``culture.refine`` / ``recruit.invite`` /
    ``reproduce.spawn`` / ``resource.rebalance`` / ``meta.propose`` /
    ``policy.set`` older than ``limit`` rows is invisible, so the only actions
    that ever reach an actuator are the two seeds. Oldest-first on purpose: a
    backlog is drained from the front, and newest-first would strand every
    proposal behind a window that keeps sliding away from it. It keys on the
    author name in the title rather than the pubkey because the body has not
    been fetched yet; ``next_*`` re-checks the pubkey, so a name collision
    buys a wasted fetch, never a wrong execution.

    Reactions naming any kept proposal ride along. A proposal judged on half
    its votes reads as "not adopted", which is indistinguishable in the ledger
    from a real refusal.
    """
    rows = [row for row in tasks if isinstance(row, dict)]
    window = rows[-limit:] if limit > 0 else []
    keep = {row.get("id") for row in window}
    names = {str(name) for name in (authors or ()) if name}
    seed_ids = set()
    act_ids = set()
    taken: Counter = Counter()
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[proposal]")
        if parsed is None:
            continue
        proposal_id, action = parsed
        if action in ("seed.export", "seed.replicate"):
            if proposal_id in settled:
                continue
            seed_ids.add(proposal_id)
            keep.add(row.get("id"))
            continue
        if not names or per_action <= 0 or action not in ACTUATORS:
            continue
        if taken[action] >= per_action or proposal_id in settled:
            continue
        if _author_of(str(row.get("title") or "")) not in names:
            continue
        taken[action] += 1
        act_ids.add(proposal_id)
        keep.add(row.get("id"))
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[reaction]")
        if parsed and (parsed[0] in seed_ids or parsed[0] in act_ids):
            keep.add(row.get("id"))
    return [row for row in rows if row.get("id") in keep]


def _hosts_table(path: str = "~/.pandosi/hosts.json") -> dict:
    """This host's ``hosts.json``, or ``{}`` when it is missing or corrupt.

    The executor already reads the same file for its mailbox and pull
    destinations; reading the author name from it too keeps one source of
    truth for "who is this host".
    """
    try:
        return json.loads(
            Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


#: The localhost conventions, in the order the connectivity doctrine lists
#: them: the Mac's mailbox port first, the win node's second. They are used
#: only when neither the command line nor this host's table names a mailbox,
#: so a node with no config still looks at its own door instead of dialling a
#: peer's address that this checkout is not allowed to know.
DEFAULT_MAILBOXES = ("http://127.0.0.1:8776", "http://127.0.0.1:8778")


def host_mailboxes(table: dict | None) -> list[str]:
    """The mailbox URLs a ``hosts.json`` names, in order. Blank -> ``[]``."""
    urls = table.get("mailboxes") if isinstance(table, dict) else None
    if not isinstance(urls, list):
        return []
    return [str(url).strip() for url in urls if str(url or "").strip()]


def executor_mailboxes(mailbox: str, table: dict | None, fallbacks=()) -> list[str]:
    """The ordered shelf list one round may read.

    Command line first, then *every* mailbox this host's table names, then
    ``--mailbox-fallback``. The table goes in whole rather than as
    ``table[1:]``: when the command line names a primary, the table's own
    first entry is still a shelf this host is meant to read, and dropping it
    is how a node polls a peer's replica while its own mailbox fills up
    unread. Duplicates collapse in ``mailbox_chain``, so the caller may pass
    the same address in all three places -- which is exactly what the pi5 and
    Windows units do, so that a moved address is reused everywhere a person
    was already writing it.
    """
    named = host_mailboxes(table)
    primary = str(mailbox or "").strip() or (named[0] if named else "")
    chain = mailbox_chain(primary, named + list(fallbacks or []))
    return chain or list(DEFAULT_MAILBOXES)


def local_authors(table: dict | None, extra=()) -> set[str]:
    """The author names this host may drain in layer 3 of the row selection.

    ``hosts.json`` already carries this host's ``proposer_id``. ``extra`` is
    for a second local name (a host can sign under more than one). An empty
    result disables layer 3 instead of guessing: a wrong name here fails
    silently -- the host would drain nothing beyond the seed pair -- while an
    empty set is merely the behaviour that preceded this function.
    """
    names = {str(name) for name in (extra or ()) if name}
    if isinstance(table, dict) and table.get("proposer_id"):
        names.add(str(table["proposer_id"]))
    return names


#: How many rounds one proposal may raise before the executor stops retrying
#: it. Retrying is the right default -- a locked file, a model that is still
#: loading, a passphrase that has not been typed yet are all transient -- but a
#: proposal whose params can never satisfy the actuator's contract raises the
#: same error for ever. Every actuator takes the *oldest* unsettled proposal, so
#: without a budget one such proposal blocks that action's whole backlog while
#: the ledger grows a row every round.
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


def next_policy_set(proposals: list[Proposal], reactions: list[dict],
                    done: set[str], signer: str | None = None,
                    knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``policy.set`` this host authored and not yet applied.

    One per round, like the seed export, and for the same two reasons: a round
    should not rewrite every knob at once, and a rule change applied halfway
    through a round would leave the ledger describing a rule that never held.
    A change applied here is in force from the next round.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "policy.set":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def next_culture_refine(proposals: list[Proposal], reactions: list[dict],
                        done: set[str], signer: str | None = None,
                        knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``culture.refine`` this host authored and not yet staged.

    Staging is one-way: once a candidate is written to the outbox the ledger
    row marks it done, so a later round will not restage the same proposal.
    Only the signer's own proposals are staged — a foreign culture change is
    recorded, never written locally, for the same reason a foreign seed is not
    sealed here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "culture.refine":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_culture_refines(proposals: list[Proposal], reactions: list[dict],
                            done: set[str], signer: str,
                            knobs: dict | None = None) -> list[str]:
    """Adopted ``culture.refine`` ids another host authored.

    Recorded rather than dropped: a foreign culture candidate that is neither
    staged here nor named here would leave the ledger silent about a culture
    change the group adopted. It is not staged locally — the authoring host
    owns its own outbox.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "culture.refine":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_recruit_invite(proposals: list[Proposal], reactions: list[dict],
                        done: set[str], signer: str | None = None,
                        knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``recruit.invite`` this host authored and not yet staged.

    Only the signer's own invitation is minted: an invitation is issued with
    the issuer's key and into the issuer's outbox, so a foreign invitation is
    named, never written here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "recruit.invite":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_recruit_invites(proposals: list[Proposal], reactions: list[dict],
                            done: set[str], signer: str,
                            knobs: dict | None = None) -> list[str]:
    """Adopted ``recruit.invite`` ids another host authored.

    Recorded rather than dropped: an invitation the group adopted is a real
    outcome even though this host must not mint it.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "recruit.invite":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_reproduce_spawn(proposals: list[Proposal], reactions: list[dict],
                         done: set[str], signer: str | None = None,
                         knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``reproduce.spawn`` this host authored and not yet drafted.

    Only the signer's own proposal is drafted here. The plan is written into the
    authoring host's outbox, so a foreign reproduction stays foreign: it is
    named in the ledger and never written locally.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "reproduce.spawn":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_reproduce_spawns(proposals: list[Proposal], reactions: list[dict],
                             done: set[str], signer: str,
                             knobs: dict | None = None) -> list[str]:
    """Adopted ``reproduce.spawn`` ids another host authored.

    Recorded rather than dropped: a reproduction the group adopted is a real
    outcome even though this host must not draft it. The authoring host owns
    its own outbox.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "reproduce.spawn":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_rebalance(proposals: list[Proposal], reactions: list[dict],
                   done: set[str], signer: str | None = None,
                   knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``resource.rebalance`` this host authored, not yet drafted.

    Only the signer's own proposal is drafted: the plan is written into the
    authoring host's outbox, and a foreign move is named, never re-hosted here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "resource.rebalance":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_rebalances(proposals: list[Proposal], reactions: list[dict],
                       done: set[str], signer: str,
                       knobs: dict | None = None) -> list[str]:
    """Adopted ``resource.rebalance`` ids another host authored."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "resource.rebalance":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_meta_propose(proposals: list[Proposal], reactions: list[dict],
                      done: set[str], signer: str | None = None,
                      knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``meta.propose`` this host authored, not yet drafted."""
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "meta.propose":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_meta_proposes(proposals: list[Proposal], reactions: list[dict],
                          done: set[str], signer: str,
                          knobs: dict | None = None) -> list[str]:
    """Adopted ``meta.propose`` ids another host authored."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "meta.propose":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def export_seed(proposal: Proposal, *, home, identity, rollout_dir, out_root,
                 passphrase_file=None) -> bool:
    """Write one seed bundle for this proposal, under ``out_root``.

    Uses the existing seed tool. The destination is the proposal id, not the
    live seed path, and a bundle is never written inside a git worktree.
    ``passphrase_file`` seals the identity. Without it the bundle holds a
    bare private key, which is only acceptable inside a test directory.
    """
    import argparse

    module = _seed_module()
    out = Path(out_root).expanduser() / proposal.proposal_id
    rc = module.cmd_export(argparse.Namespace(
        out=str(out),
        home=str(Path(home).expanduser()),
        identity=str(Path(identity).expanduser()),
        rollout_dir=str(Path(rollout_dir).expanduser()),
        force=False,
        passphrase_file=(str(passphrase_file) if passphrase_file else None),
        proposal_id=proposal.proposal_id,
        author=proposal.author,
    ))
    return rc == 0 and (out / "manifest.json").is_file()


def summarize(results: list[dict]) -> dict:
    counts = Counter(row["detail"].split(":")[0] if row["detail"] else "?"
                     for row in results)
    # Stable buckets a person can read without parsing every detail line.
    return {
        "considered": len(results),
        "held_not_in_space": sum(row["detail"].startswith("held: action")
                                 for row in results),
        "held_veto": sum(row["detail"].startswith("held: a decision")
                         for row in results),
        "held_quorum": sum(row["detail"].startswith("held: ")
                           and "adopt <" in row["detail"]
                           for row in results),
        "adopted_not_executed": sum(
            row["adopted"] and not row["executed"] for row in results),
        "executed": sum(row["executed"] for row in results),
        "details": dict(counts),
    }



def done_ids(ledger: Path) -> set[str]:
    found: set[str] = set()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("executed") and row.get("proposal_id"):
            found.add(str(row["proposal_id"]))
    return found


def left_ids(ledger: Path) -> set[str]:
    """Seed exports already recorded as another host's job."""
    found: set[str] = set()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("skipped") and row.get("proposal_id") and not row.get("executed"):
            found.add(str(row["proposal_id"]))
    return found


def next_seed_export(proposals: list[Proposal], reactions: list[dict],
                     done: set[str], signer: str | None = None,
                     knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted seed.export this host is allowed to seal.

    The bundle is this machine's identity. A proposal signed by another
    host must be sealed there. ``signer`` is this machine's public key;
    when it is set, a proposal with any other key is left alone.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.export":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def write_status(path: Path, payload: dict) -> None:
    """One file, replaced each run. The ledger stays append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def foreign_adopted_seeds(proposals: list[Proposal], reactions: list[dict],
                          done: set[str], signer: str,
                          knobs: dict | None = None) -> list[str]:
    """Adopted seed.export ids this host must not seal."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.export":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def foreign_policy_sets(proposals: list[Proposal], reactions: list[dict],
                        done: set[str], signer: str,
                        knobs: dict | None = None) -> list[str]:
    """Adopted policy.set ids another host authored. It applies them itself.

    Recorded rather than dropped: now that ``policy.set`` has an actuator, a
    foreign rule change that is neither applied here nor named here would leave
    the ledger silent about work the group adopted.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "policy.set":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def foreign_replications(proposals: list[Proposal], reactions: list[dict],
                         done: set[str], signer: str,
                         knobs: dict | None = None) -> list[str]:
    """Adopted seed.replicate ids signed by someone else. Not ours to send."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if not proposal.pubkey or proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def local_replications(proposals: list[Proposal], reactions: list[dict],
                       done: set[str], signer: str,
                       knobs: dict | None = None) -> list[str]:
    """Adopted seed.replicate ids signed here that name no destination."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if proposal.pubkey != signer or proposal.destination:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def known_hosts(data: dict | None) -> dict[str, str] | None:
    """Host name to root URL. None when there is no table to check against."""
    if not isinstance(data, dict):
        return None
    found: dict[str, str] = {}
    for row in data.get("roots") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        url = str(row.get("url") or "").strip()
        if name and url:
            found[name] = url
    return found


def pull_destinations(data: dict | None) -> set[str]:
    """Hosts that pull a sealed bundle. A pull host needs no root URL."""
    if not isinstance(data, dict):
        return set()
    found = set()
    for row in data.get("roots") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name and str(row.get("seed") or "").strip() == "pull":
            found.add(name)
    return found


def replication_skip(destination: str, hosts: dict[str, str] | None) -> str:
    """Why this named copy is not sent. This function does not open the URL.

    A pull host is handled by ``offer_replication``, which copies a sealed
    bundle into a local outbox. Neither function opens a socket.
    """
    if hosts is not None and destination not in hosts:
        return "unknown-host"
    return "no-transport"


def _seed_module():
    import importlib.util
    tool = Path(__file__).resolve().parents[1] / "tools" / "seed.py"
    spec = importlib.util.spec_from_file_location("pandosi_seed", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("seed tool is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_sealed(out_root) -> Path | None:
    """Newest bundle under ``out_root`` that is still sealed and signed."""
    root = Path(out_root).expanduser()
    if not root.is_dir():
        return None
    seed = _seed_module()
    found = [child for child in root.iterdir()
             if child.is_dir() and seed.judge_bundle(child).get("accepted")]
    if not found:
        return None
    return max(found, key=lambda child: child.stat().st_mtime)


def offer_replication(proposal_id: str, destination: str, *, bundle, outbox) -> dict:
    """Stage one sealed bundle for a pull host. Does not open a socket."""
    row = {"proposal_id": proposal_id, "action": "seed.replicate",
           "destination": destination, "executed": False}
    if bundle is None:
        row["skipped"] = "no-bundle"
        return row
    seed = _seed_module()
    kept = seed.receive_bundle(bundle, outbox, name=proposal_id)
    if not kept.get("copied"):
        if kept.get("reason") == "already-received":
            row["skipped"] = "offered"
            row["bundle"] = Path(bundle).name
            return row
        row["skipped"] = str(kept.get("reason") or "no-bundle")
        return row
    row["skipped"] = "offered"
    row["bundle"] = Path(bundle).name
    return row


def received_acks(ledger: Path, outbox) -> list[dict]:
    """Offers whose peer has acked, and which are not yet marked received."""
    offered: list[tuple[str, str]] = []
    done: set[str] = set()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        proposal_id = str(row.get("proposal_id") or "")
        if not proposal_id:
            continue
        if row.get("received"):
            done.add(proposal_id)
        elif row.get("skipped") == "offered":
            offered.append((proposal_id, str(row.get("destination") or "")))
    outbox = Path(outbox).expanduser()
    found = []
    for proposal_id, destination in offered:
        if proposal_id in done:
            continue
        ack = outbox / f"{proposal_id}.ack.json"
        if not ack.is_file():
            continue
        found.append({"proposal_id": proposal_id, "action": "seed.replicate",
                      "destination": destination, "executed": True,
                      "received": True})
    return found


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
            profile = resolve_profile(name)
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


def local_unacted(proposals: list[Proposal], reactions: list[dict],
                  done: set[str], signer: str,
                  knobs: dict | None = None,
                  installed_operators: dict | None = None) -> list[tuple[str, str]]:
    """Adopted proposals this host signed that have no actuator.

    Anything in ``ACTUATORS`` is left out: those are recorded by their own
    path (seed.export seals, seed.replicate wants a destination), so calling
    them unactuated here would double-count them as failures. An installed verb
    is left out for the same reason once its host has the record: it has a
    stager, and ``next_installed`` is what reports it.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.pubkey != signer:
            continue
        if proposal.action in ACTUATORS:
            continue
        if installed_operators and proposal.action in installed_operators:
            continue
        if adopted(proposal, reactions, knobs, installed_operators):
            found.append((proposal.proposal_id, proposal.action))
    return found


def dry_run(proposals: list[Proposal], reactions: list[dict], **policy) -> dict:
    """Judge every proposal. No actuator, no write."""
    results = [judge(proposal, reactions, actuators={}, **policy)
               for proposal in proposals]
    for result in results:
        # ``judge`` cannot distinguish "this host has no implementation" from
        # "the dry run deliberately supplied none". Saying the former here
        # made an adopted action look like a missing actuator even when the
        # real executor had one, which sent operators chasing a false fault.
        if result["detail"] == "adopted, not executed: no actuator":
            result["detail"] = "adopted, not executed: dry run (actuator not invoked)"
    return {"summary": summarize(results), "results": results}


def _load_reactions(url: str, token: str | None, limit: int,
                    rows: list | None = None) -> list[dict]:
    from .proposal_mail import _request, fetch_bodies
    from .genesis import verify_reaction
    if rows is None:
        listing = _request(url, "/tasks?status=all", token, "GET")
        tasks = listing.get("tasks") or []
        tasks = tasks[-limit:]
    else:
        tasks = rows
    wanted = [task.get("id") for task in tasks
              if isinstance(task, dict)
              and is_reaction_task(str(task.get("title") or ""))]
    # One batched read instead of one GET per reaction row. A round can carry
    # 1600 of them, and serial HTTP was the whole cost of the pass.
    bodies = fetch_bodies(url, token, wanted)
    found = []
    for task_id in wanted:
        body = (bodies.get(str(task_id)) or {}).get("body")
        if not isinstance(body, str):
            continue
        try:
            record = json.loads(body)
        except json.JSONDecodeError:
            continue
        view = verify_reaction(record)
        if view is not None:
            found.append(view)
    return found



def run_once(proposals: list[Proposal], reactions: list[dict], args) -> int:
    """Carry out at most one seed.export and at most one policy.set.

    The home directory is empty on purpose: the bundle must not copy the
    mailbox token or task files. Only a seed.export signed by this host is
    sealed. A seed.replicate with no destination is recorded and not sent.

    The policy store is read once at the top, so every judgment in a round uses
    the same rules. A rule change applied here is in force from the next round,
    which keeps the ledger from describing a rule that never held. A culture
    candidate is only staged, never loaded, so it cannot change this round.
    """
    ledger = Path(args.ledger).expanduser()
    from .identity import Identity
    identity = Identity.from_json(json.loads(
        Path(args.identity).expanduser().read_text(encoding="utf-8")))
    signer = identity.public_key
    failures = failure_counts(ledger)
    deferred = deferred_reasons(ledger)

    def record_row(record: dict) -> None:
        """One ledger append, with the retry budget applied first."""
        settle_after_failure(record, failures)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    knobs = load_policy(getattr(args, "policy_path", DEFAULT_POLICY_PATH))
    # The installed verbs, read once per round for the same reason the policy
    # store is: every judgment in a round has to use the same set. A record
    # that will not parse leaves the set empty rather than guessing, so the
    # built-ins keep running and only the installed verbs stop -- and the
    # reason travels in the status file, because a gate that quietly runs the
    # smaller set is the failure this whole module exists to make impossible.
    installed_operators: dict = {}
    installed_error = ""
    try:
        installed_operators = load_installed(
            getattr(args, "operators_dir", DEFAULT_INSTALLED))
    except OperatorError as exc:
        installed_error = str(exc)
    deployment_attempts: list[dict] = []
    chosen = next_seed_export(proposals, reactions, done_ids(ledger), signer,
                              knobs)
    # A missing passphrase is a property of this host, not a defect in the
    # proposal, so it must not end the round: only the seal needs a key, and
    # none of the six stage-only actuators do. The proposal is left unsettled
    # -- a passphrase typed tomorrow has to release it -- and the reason is
    # written once per pair, so a host that cannot seal does not grow a ledger
    # row every round. It travels in the status file either way.
    phrase = Path(args.passphrase_file).expanduser()
    deferred_export = chosen is not None and not phrase.is_file()
    sealed = None
    if deferred_export:
        print("no passphrase file; refusing to write a plaintext key",
              flush=True)
        if (chosen.proposal_id, "no-passphrase") not in deferred:
            record_row({"proposal_id": chosen.proposal_id,
                        "action": "seed.export",
                        "executed": False,
                        "deferred": "no-passphrase"})
    elif chosen is not None:
        home = Path(getattr(args, "seed_home",
                                   DEFAULT_STATE_DIR + "/seed-home-empty")
                   ).expanduser()
        home.mkdir(parents=True, exist_ok=True)
        try:
            sealed = export_seed(
                chosen, home=home, identity=args.identity,
                rollout_dir=home / "rollouts", out_root=args.out_root,
                passphrase_file=phrase)
            failure = ""
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            sealed = None
            failure = f"{type(exc).__name__}: {exc}"
        record = {
            "proposal_id": chosen.proposal_id,
            "action": "seed.export",
            "executed": bool(sealed),
            "sealed": bool(sealed),
        }
        if not sealed:
            # No ``skipped`` yet, so the seal is retried -- until
            # ``MAX_ATTEMPTS``. The reason still travels, or a broken
            # actuator would be indistinguishable from an idle one.
            record["failed"] = failure or "export wrote no manifest"
        record_row(record)
    # A rule change is applied after the export and before the replicate
    # bookkeeping, so this round's own ledger row is already in ``done_ids``
    # when the later passes look for unactuated work.
    change = next_policy_set(proposals, reactions, done_ids(ledger), signer,
                             knobs)
    if change is not None:
        adopters = len(tally(change, reactions)[ADOPT])
        record = {"proposal_id": change.proposal_id, "action": "policy.set"}
        try:
            outcome = apply_policy(
                change, getattr(args, "policy_path", DEFAULT_POLICY_PATH),
                adopters=adopters)
        except PolicyError as exc:
            # No ``skipped``: a refused change is retried next round and
            # refused again, and the reason travels, so a change that can never
            # be applied stays visible instead of looking handled.
            record["executed"] = False
            record["failed"] = f"PolicyError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["applied"] = outcome["applied"]
        record_row(record)
    # A culture change is staged after the export and rule change. Staging is
    # one-way and inert: it writes a signed, validated candidate into the
    # outbox for later review and never loads it, so it cannot rewrite the
    # rules this round.
    refine = next_culture_refine(proposals, reactions, done_ids(ledger),
                                 signer, knobs)
    if refine is not None:
        record = {"proposal_id": refine.proposal_id, "action": "culture.refine"}
        try:
            staged = stage_culture(
                refine, identity=identity,
                outbox=getattr(args, "culture_outbox", DEFAULT_CULTURE_OUTBOX))
        except CultureError as exc:
            # No ``skipped``: a refused candidate is retried next round and
            # refused again, and the reason travels, so a candidate that can
            # never be staged stays visible instead of looking handled.
            record["executed"] = False
            record["failed"] = f"CultureError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["playbook_hash"] = staged["playbook_hash"]
            record["directory"] = staged["directory"]
        record_row(record)
    # An invitation is minted last and is inert: the token is written 0600 into
    # a local outbox with ``applied: false``. Nothing installs it into a root,
    # so adopting a recruit can never admit a member in the same breath.
    invite = next_recruit_invite(proposals, reactions, done_ids(ledger),
                                 signer, knobs)
    if invite is not None:
        record = {"proposal_id": invite.proposal_id,
                  "action": "recruit.invite"}
        try:
            staged = stage_invite(
                invite, identity=identity,
                outbox=getattr(args, "invite_outbox", DEFAULT_INVITE_OUTBOX))
        except InviteError as exc:
            # No ``skipped``: a refused invitation is retried next round and
            # refused again, so one that can never be staged stays visible.
            record["executed"] = False
            record["failed"] = f"InviteError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["invite_hash"] = staged["invite_hash"]
            record["expires_at"] = staged["expires_at"]
            record["directory"] = staged["directory"]
        record_row(record)
    # A reproduction is drafted last and is inert by construction: the plan
    # names what would be copied and where, and lists every reason the parent
    # still refuses. Nothing is copied, no process starts, and this module
    # never calls ``pando.reproduce.reproduce``.
    spawn = next_reproduce_spawn(proposals, reactions, done_ids(ledger),
                                 signer, knobs)
    if spawn is not None:
        record = {"proposal_id": spawn.proposal_id, "action": "reproduce.spawn"}
        try:
            staged = stage_spawn(
                spawn, identity=identity,
                outbox=getattr(args, "spawn_outbox", DEFAULT_SPAWN_OUTBOX))
        except SpawnError as exc:
            # No ``skipped``: a refused draft is retried next round and refused
            # again, so a plan that can never be staged stays visible.
            record["executed"] = False
            record["failed"] = f"SpawnError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["plan_hash"] = staged["plan_hash"]
            record["refusals"] = staged["refusals"]
            record["directory"] = staged["directory"]
        record_row(record)
    # A member move is drafted next, and it is inert: the plan names the
    # member and the target host and lists every precondition. No member is
    # stopped, none is started, no state directory is touched.
    move = next_rebalance(proposals, reactions, done_ids(ledger), signer, knobs)
    if move is not None:
        record = {"proposal_id": move.proposal_id,
                  "action": "resource.rebalance"}
        try:
            staged = stage_rebalance(
                move, identity=identity,
                outbox=getattr(args, "rebalance_outbox",
                               DEFAULT_REBALANCE_OUTBOX))
        except RebalanceError as exc:
            # No ``skipped``: a refused draft is retried next round and refused
            # again, so a move that can never be staged stays visible.
            record["executed"] = False
            record["failed"] = f"RebalanceError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["plan_hash"] = staged["plan_hash"]
            record["refusals"] = staged["refusals"]
            record["directory"] = staged["directory"]
        record_row(record)
    # A proposal for a *new operator* is drafted last of all. It is the one
    # artifact in this file that must never become code: the spec carries a
    # label and prose, and the implementation is what the gate supplies.
    meta = next_meta_propose(proposals, reactions, done_ids(ledger), signer,
                             knobs)
    if meta is not None:
        record = {"proposal_id": meta.proposal_id, "action": "meta.propose"}
        try:
            staged = stage_operator(
                meta, identity=identity,
                outbox=getattr(args, "operator_outbox",
                               DEFAULT_OPERATOR_OUTBOX))
        except OperatorError as exc:
            # No ``skipped``: a refused draft is retried next round and refused
            # again, so a spec that can never be staged stays visible.
            record["executed"] = False
            record["failed"] = f"OperatorError: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            record["spec_hash"] = staged["spec_hash"]
            record["refusals"] = staged["refusals"]
            record["directory"] = staged["directory"]
        record_row(record)
    # Once the spec is staged, the fixed catalogue may deploy it without a
    # per-proposal human review. The standing grant authorizes the deployment
    # action, not arbitrary code: the profile, binding and params are compiled
    # into pando.operator_profiles and the proposal can only name one exactly.
    try:
        deployment_attempts = auto_deploy_profiles(
            proposals, reactions, identity=identity, signer=signer, knobs=knobs,
            outbox=getattr(args, "operator_outbox", DEFAULT_OPERATOR_OUTBOX),
            installed_dir=getattr(args, "operators_dir", DEFAULT_INSTALLED),
            install_ledger=getattr(args, "operator_install_ledger",
                                   DEFAULT_INSTALL_LEDGER),
            deploy_ledger=getattr(args, "operator_deploy_ledger",
                                  DEFAULT_DEPLOY_LEDGER),
            grants_path=getattr(args, "grants", None),
            installed_operators=installed_operators,
            installed_error=installed_error)
    except Exception as exc:  # noqa: BLE001 - one audit failure must not kill the loop
        deployment_attempts = [{
            "action": "deploy_operator",
            "status": "refused",
            "reason": f"automatic deployment failed closed: {type(exc).__name__}: {exc}",
        }]
    # A verb the group installed is staged through the actuator it binds to,
    # last of all and one per round like everything else. The params are the
    # record's, not the proposal's: a proposal names the verb, a reviewer
    # decided what it does, and that is the only reason a name nobody wrote
    # code for is allowed to reach an outbox at all.
    if installed_operators:
        bound = next_installed(proposals, reactions, done_ids(ledger), signer,
                               installed_operators, knobs)
        if bound is not None:
            record = {"proposal_id": bound.proposal_id, "action": bound.action}
            try:
                record["binds_to"] = binding(
                    installed_operators[bound.action])[0]
                staged = stage_bound(
                    rebound(bound, installed_operators[bound.action]),
                    args, identity)
            except OperatorError as exc:
                # No ``skipped``: a verb that cannot be staged is retried next
                # round and refused again, so it stays visible.
                record["executed"] = False
                record["failed"] = f"OperatorError: {exc}"
            except Exception as exc:  # noqa: BLE001 - one action must not end the loop
                record["executed"] = False
                record["failed"] = f"{type(exc).__name__}: {exc}"
            else:
                record["executed"] = True
                # Whatever the bound stager reported, under the verb's own
                # name: the ledger says what the group asked for and which
                # actuator carried it out.
                for key, value in staged.items():
                    record[key] = value
            record_row(record)
    # An export must not crowd out a replicate. Both are recorded.
    if True:
        seen = done_ids(ledger) | left_ids(ledger)
        foreign = foreign_adopted_seeds(proposals, reactions, seen, signer,
                                        knobs)
        replicates = local_replications(proposals, reactions, seen, signer,
                                        knobs)
        addressed = addressed_replications(proposals, reactions, seen, signer,
                                           knobs)
        hosts = getattr(args, "hosts", None)
        foreign_repl = foreign_replications(proposals, reactions, seen, signer,
                                            knobs)
        foreign_policy = foreign_policy_sets(proposals, reactions, seen, signer,
                                             knobs)
        foreign_culture = foreign_culture_refines(proposals, reactions, seen,
                                                  signer, knobs)
        foreign_invites = foreign_recruit_invites(proposals, reactions, seen,
                                                  signer, knobs)
        foreign_spawns = foreign_reproduce_spawns(proposals, reactions, seen,
                                                  signer, knobs)
        foreign_moves = foreign_rebalances(proposals, reactions, seen, signer,
                                           knobs)
        foreign_meta = foreign_meta_proposes(proposals, reactions, seen, signer,
                                             knobs)
        unacted = local_unacted(proposals, reactions, seen, signer, knobs,
                                installed_operators)
        pulls = getattr(args, "pull", None) or set()
        outbox = getattr(args, "outbox", "~/.pandosi/outbox")
        acks = received_acks(ledger, outbox)
        if (foreign or replicates or addressed or unacted or foreign_repl
                or foreign_policy or foreign_culture or foreign_invites
                or foreign_spawns or foreign_moves or foreign_meta
                or acks):
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as handle:
                for proposal_id in foreign:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "seed.export",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in replicates:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "seed.replicate",
                        "executed": False,
                        "skipped": "no-destination",
                    }, sort_keys=True) + "\n")
                bundle = latest_sealed(args.out_root) if pulls else None
                for proposal_id, destination in addressed:
                    # A pull host is recorded whether or not the offer landed:
                    # ``offer_replication`` returns the reason it did not
                    # (``no-bundle`` when nothing is sealed yet). Dropping that
                    # row made a failed copy read exactly like one that was
                    # never attempted -- which is how win sat at zero received
                    # seeds for 35 rounds without leaving a trace.
                    if destination in pulls:
                        record = offer_replication(
                            proposal_id, destination, bundle=bundle, outbox=outbox)
                    else:
                        record = {
                            "proposal_id": proposal_id,
                            "action": "seed.replicate",
                            "destination": destination,
                            "executed": False,
                            "skipped": replication_skip(destination, hosts),
                        }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                for record in received_acks(ledger, outbox):
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                for proposal_id in foreign_repl:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "seed.replicate",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_policy:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "policy.set",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_culture:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "culture.refine",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_invites:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "recruit.invite",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_spawns:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "reproduce.spawn",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_moves:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "resource.rebalance",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id in foreign_meta:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": "meta.propose",
                        "executed": False,
                        "skipped": "other-host",
                    }, sort_keys=True) + "\n")
                for proposal_id, action in unacted:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": action,
                        "executed": False,
                        "skipped": "no-actuator",
                    }, sort_keys=True) + "\n")
        payload = {"did": "idle", "read_from": getattr(args, "read_from", ""),
                   "mailbox_tried": list(getattr(args, "mailbox_chain", []) or []),
                   "foreign_seed_exports": foreign,
                   "seed_export_deferred": ("no-passphrase"
                                            if deferred_export else None),
                   "replication_without_destination": replicates,
                   "replication_not_sent": [item[0] for item in addressed],
                   "foreign_replications": foreign_repl,
                   "foreign_policy_sets": foreign_policy,
                   "foreign_culture_refines": foreign_culture,
                   "foreign_recruit_invites": foreign_invites,
                   "foreign_reproduce_spawns": foreign_spawns,
                   "foreign_rebalances": foreign_moves,
                   "foreign_meta_proposes": foreign_meta,
                   "installed_operators": sorted(installed_operators),
                   "installed_operators_error": installed_error,
                   "operator_deployments": deployment_attempts,
                   "no_actuator": [item[0] for item in unacted]}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        write_status(ledger.with_name("execute-status.json"), payload)
        if sealed is None:
            return 0
        offered = [item[0] for item in addressed
                   if item[1] in (getattr(args, "pull", None) or set())]
        done = {"did": "executed", "proposal_id": chosen.proposal_id,
                "executed": bool(sealed), "offered": offered,
                "read_from": getattr(args, "read_from", ""),
                "mailbox_tried": list(getattr(args, "mailbox_chain", []) or []),
                "seed_export_deferred": ("no-passphrase"
                                         if deferred_export else None),
                "operator_deployments": deployment_attempts}
        print(json.dumps(done, sort_keys=True), flush=True)
        write_status(ledger.with_name("execute-status.json"), done)
        return 0 if sealed else 1


def build_parser() -> argparse.ArgumentParser:
    """The executor's flags, with state paths left blank for ``resolve_state``.

    A blank default is not "no ledger": it means "wherever ``--state-dir``
    points". Keeping the leaf names here and the root in one place is what lets
    a non-Mac host relocate its whole state with a single flag.
    """
    parser = argparse.ArgumentParser(
        description="judge proposals; does not execute them")
    parser.add_argument("--mailbox", default=None,
                        help="the mailbox to read first; default: this host's "
                             "hosts.json mailboxes[0], then the localhost "
                             "conventions (127.0.0.1:8776, 127.0.0.1:8778)")
    parser.add_argument("--mailbox-fallback", action="append", default=[],
                        help="read this mailbox if the ones before it do not "
                             "answer (repeatable)")
    parser.add_argument("--hosts", default="~/.pandosi/hosts.json",
                        help="this host's table: proposer_id, mailboxes, roots")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                        help="the directory every blank path below is relative "
                             f"to (default: {DEFAULT_STATE_DIR})")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--author", action="append", default=[],
                        help="an author name this host signs under; repeatable. "
                             "Added to hosts.json proposer_id. Only these "
                             "authors are drained past the row window.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="one round: at most one adopted seed.export and "
                             "at most one adopted policy.set")
    parser.add_argument("--identity", default="~/.pando/identity.json")
    for name, (leaf, purpose) in STATE_PATHS.items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, default="",
                            help=f"{purpose} (default: <state-dir>/{leaf})")
    return parser


def resolve_state(args: argparse.Namespace) -> argparse.Namespace:
    """Fill every blank state path from ``--state-dir``. An explicit one wins.

    Called on the parsed namespace rather than inside ``main`` so the rule can
    be tested without a mailbox: a host that quietly resolves its ledger to a
    different tree than its outboxes is indistinguishable from an idle one.
    """
    base = Path(str(getattr(args, "state_dir", "") or DEFAULT_STATE_DIR)
                ).expanduser()
    for name, (leaf, _) in STATE_PATHS.items():
        if not getattr(args, name, ""):
            setattr(args, name, str(base / leaf))
    args.state_dir = str(base)
    args.seed_home = str(base / "seed-home-empty")
    args.policy_path = args.policy
    return args


def main() -> int:
    args = resolve_state(build_parser().parse_args())
    if args.dry_run == args.once:
        print("pass exactly one of --dry-run or --once", flush=True)
        return 2
    token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    from .proposal_mail import listing_any
    table = _hosts_table(args.hosts)
    mailboxes = executor_mailboxes(args.mailbox, table, args.mailbox_fallback)
    listing, source, read_error = listing_any(mailboxes, token, status="all")
    ledger = Path(args.ledger).expanduser()
    if source is None:
        # Nobody answered. That is a property of the network, not of any
        # proposal, so it is recorded where a person looks -- the status file
        # -- and not as a ledger row keyed by a proposal id that does not
        # exist. A dead address used to reach this same point through the
        # empty-listing path and print an idle round, which is how an
        # unreachable executor looked healthy for a day.
        payload = {"did": "unreachable", "error": read_error,
                   "mailbox_tried": list(mailboxes)}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        write_status(ledger.with_name("execute-status.json"), payload)
        return 1
    args.read_from = source
    args.mailbox_chain = list(mailboxes)
    rows = rows_for_execution(
        listing.get("tasks") or [], args.limit,
        authors=local_authors(table, args.author),
        settled=done_ids(ledger) | left_ids(ledger))
    # One shelf per round: the rows, the bodies those rows point at, and the
    # reactions all come from the mailbox the listing came from. Mixing them
    # is not a union -- it is a round that reads one host's row ids and
    # another host's bodies.
    proposals = fetch_proposals(source, token, status="all", rows=rows)
    reactions = _load_reactions(source, token, args.limit, rows=rows)
    if args.once:
        if table:
            args.hosts = known_hosts(table)
            args.pull = pull_destinations(table)
        else:
            args.hosts = None
            args.pull = set()
        return run_once(proposals, reactions, args)
    report = dry_run(proposals, reactions)
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    shown = 0
    for row in report["results"]:
        if not row["adopted"]:
            continue
        print(f"{row['proposal_id']} {row['action']} "
              f"adopt={len(row['adopt'])} {row['detail']}", flush=True)
        shown += 1
        if shown >= 5:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
