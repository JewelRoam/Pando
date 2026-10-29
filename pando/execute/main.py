"""One execution round: judge, act at most once per action, record everything.

The round order is deliberate: seal the seed, apply at most one policy
change, stage at most one culture candidate, mint at most one invitation,
draft at most one reproduction / move / operator spec, deploy at most one
catalogue profile, then reconcile the seed pair and every foreign outcome.
Each step writes at most one append-only ledger row, so the ledger is always
a description of what happened, in order.

Nothing here starts itself and nothing opens a socket beyond the mailbox
read the round began with.
"""
import argparse
import json
from pathlib import Path
from typing import Callable, NamedTuple

from ..culture import DEFAULT_OUTBOX as DEFAULT_CULTURE_OUTBOX
from ..culture import CultureError
from ..culture import stage as stage_culture
from ..genesis import Proposal
from ..identity import Identity
from ..invite import DEFAULT_OUTBOX as DEFAULT_INVITE_OUTBOX
from ..invite import InviteError
from ..invite import stage as stage_invite
from ..operators import (DEFAULT_DEPLOY_LEDGER, DEFAULT_INSTALLED,
                         DEFAULT_INSTALL_LEDGER,
                         DEFAULT_OUTBOX as DEFAULT_OPERATOR_OUTBOX,
                         OperatorError, binding, installed as load_installed,
                         rebound, stage as stage_operator)
from ..policy import DEFAULT_PATH as DEFAULT_POLICY_PATH
from ..policy import PolicyError
from ..policy import apply as apply_policy
from ..policy import load as load_policy
from ..proposal_mail import fetch_proposals, is_reaction_task, query_tasks_any
from ..rebalance import DEFAULT_OUTBOX as DEFAULT_REBALANCE_OUTBOX
from ..rebalance import RebalanceError
from ..rebalance import stage as stage_rebalance
from ..spawn import DEFAULT_OUTBOX as DEFAULT_SPAWN_OUTBOX
from ..spawn import SpawnError
from ..spawn import stage as stage_spawn

from .actuators import stage_bound
from .consts import ACTUATORS, ADOPT
from .selectors import (DEFAULT_STATE_DIR, STATE_PATHS,
                        _hosts_table)
from .judge import (deferred_reasons, failure_counts, judge, settle_after_failure,
                    tally)
from .next import (foreign_culture_refines, foreign_meta_proposes,
                   foreign_rebalances, foreign_recruit_invites,
                   foreign_reproduce_spawns, next_culture_refine,
                   next_installed, next_meta_propose, next_policy_set,
                   next_rebalance, next_recruit_invite, next_reproduce_spawn)
from .operators_exec import auto_deploy_profiles
from .reconcile import known_hosts, pull_destinations
from .next import next_seed_export
from . import seed_export as _seed_export_mod
from .seed_export import (addressed_replications, done_ids,
                          foreign_adopted_seeds, foreign_policy_sets,
                          foreign_replications, latest_sealed, left_ids,
                          local_replications, offer_replication, received_acks,
                          replication_skip, write_status)
from .selectors import (execution_selectors, executor_mailboxes, local_authors,
                        rows_for_execution)


class _Stager(NamedTuple):
    """One one-per-round actuator: pick at most one proposal, stage it, record.

    A refused action has no ``skipped`` state: it is retried next round and
    refused again, and the reason travels in the ledger row, so an action
    that can never be staged stays visible instead of looking handled.
    """

    action: str
    pick: Callable    # (proposals, reactions, done_ids, signer, knobs)
    stage: Callable     # (picked, args, identity, reactions) -> dict
    error: type         # domain refusal; anything else is recorded generically
    carried: tuple      # staged keys copied into the ledger row


def _stage_policy_set(picked, args, identity, reactions) -> dict:
    # Applied after the export and before the replicate bookkeeping, so this
    # round's own ledger row is already in ``done_ids`` when the later passes
    # look for unactuated work.
    adopters = len(tally(picked, reactions)[ADOPT])
    return apply_policy(
        picked, getattr(args, "policy_path", DEFAULT_POLICY_PATH),
        adopters=adopters)


def _stage_culture_refine(picked, args, identity, reactions) -> dict:
    # Staging is one-way and inert: it writes a signed, validated candidate
    # into the outbox for later review and never loads it, so it cannot
    # rewrite this round's rules.
    return stage_culture(
        picked, identity=identity,
        outbox=getattr(args, "culture_outbox", DEFAULT_CULTURE_OUTBOX))


def _stage_recruit_invite(picked, args, identity, reactions) -> dict:
    # Inert: the token is written 0600 into a local outbox with
    # ``applied: false``. Nothing installs it into a root, so adopting a
    # recruit can never admit a member in the same breath.
    return stage_invite(
        picked, identity=identity,
        outbox=getattr(args, "invite_outbox", DEFAULT_INVITE_OUTBOX))


def _stage_reproduce_spawn(picked, args, identity, reactions) -> dict:
    # Inert by construction: the plan names what would be copied and where,
    # and lists every reason the parent still refuses. Nothing is copied, no
    # process starts, and this module never calls ``pando.reproduce.reproduce``.
    return stage_spawn(
        picked, identity=identity,
        outbox=getattr(args, "spawn_outbox", DEFAULT_SPAWN_OUTBOX))


def _stage_resource_rebalance(picked, args, identity, reactions) -> dict:
    # Inert: the plan names the member and the target host and lists every
    # precondition. No member is stopped, none is started, no state directory
    # is touched.
    return stage_rebalance(
        picked, identity=identity,
        outbox=getattr(args, "rebalance_outbox", DEFAULT_REBALANCE_OUTBOX))


def _stage_meta_propose(picked, args, identity, reactions) -> dict:
    # The one artifact in this file that must never become code: the spec
    # carries a label and prose, and the implementation is what the gate
    # supplies.
    return stage_operator(
        picked, identity=identity,
        outbox=getattr(args, "operator_outbox", DEFAULT_OPERATOR_OUTBOX))


#: The one-per-round actuators, in round order. Adding one is a row here plus
#: its selector/actuator modules; the loop in ``run_once`` does not change.
_STAGERS: tuple = (
    _Stager("policy.set", next_policy_set, _stage_policy_set, PolicyError,
            ("applied",)),
    _Stager("culture.refine", next_culture_refine, _stage_culture_refine,
            CultureError, ("playbook_hash", "directory")),
    _Stager("recruit.invite", next_recruit_invite, _stage_recruit_invite,
            InviteError, ("invite_hash", "expires_at", "directory")),
    _Stager("reproduce.spawn", next_reproduce_spawn, _stage_reproduce_spawn,
            SpawnError, ("plan_hash", "refusals", "directory")),
    _Stager("resource.rebalance", next_rebalance, _stage_resource_rebalance,
            RebalanceError, ("plan_hash", "refusals", "directory")),
    _Stager("meta.propose", next_meta_propose, _stage_meta_propose,
            OperatorError, ("spec_hash", "refusals", "directory")),
)

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
    from .seed_export import summarize
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
    from ..proposal_mail import _request, fetch_bodies
    from ..genesis import verify_reaction
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
    from ..identity import Identity
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
            # Resolve ``export_seed`` through the package at call time so a
            # test that patches ``pando.execute.export_seed`` is honoured;
            # a plain ``from .seed_export import`` here would freeze the
            # pre-patch function object into this module.
            from . import export_seed as _export_seed
            sealed = _export_seed(
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
    # The one-per-round actuators run in table order, after the export and
    # before the replicate bookkeeping, so each round's own ledger row is
    # already in ``done_ids`` when the later passes look for unactuated work.
    for spec in _STAGERS:
        picked = spec.pick(proposals, reactions, done_ids(ledger), signer,
                           knobs)
        if picked is None:
            continue
        record = {"proposal_id": picked.proposal_id, "action": spec.action}
        try:
            staged = spec.stage(picked, args, identity, reactions)
        except spec.error as exc:
            record["executed"] = False
            record["failed"] = f"{spec.error.__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - one action must not end the loop
            record["executed"] = False
            record["failed"] = f"{type(exc).__name__}: {exc}"
        else:
            record["executed"] = True
            for key in spec.carried:
                record[key] = staged[key]
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
            # Foreign adoptions of every other action: seen here, acted on
            # elsewhere, recorded with the same "other-host" skip.
            for ids, skip_action in (
                    (foreign_repl, "seed.replicate"),
                    (foreign_policy, "policy.set"),
                    (foreign_culture, "culture.refine"),
                    (foreign_invites, "recruit.invite"),
                    (foreign_spawns, "reproduce.spawn"),
                    (foreign_moves, "resource.rebalance"),
                    (foreign_meta, "meta.propose")):
                for proposal_id in ids:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": skip_action,
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
    table = _hosts_table(args.hosts)
    mailboxes = executor_mailboxes(args.mailbox, table, args.mailbox_fallback)
    authors = local_authors(table, args.author)
    ledger = Path(args.ledger).expanduser()
    settled = done_ids(ledger) | left_ids(ledger)
    # New mailboxes return only the latest window plus seed offers, this
    # host's actuator backlog, and their reactions. Old mailboxes ignore the
    # route and fall back to the old full listing, so mixed-version peers lose
    # no work. The full listing must not be requested with ``limit`` here:
    # the focus rows may all be older than the newest 40.
    listing, source, read_error = query_tasks_any(
        mailboxes, token, selectors=execution_selectors(authors),
        limit=args.limit, exclude_proposal_ids=settled, status="all")
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
        authors=authors, settled=settled)
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
