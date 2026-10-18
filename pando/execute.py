"""Decide whether a proposal may be executed. Do not touch the world.

The colony can already propose and vote. It cannot yet do the thing it voted
for, and wiring that straight through would run almost every proposal: the
decision models currently answer 赞同 far more often than they refuse.

So this module only judges. An action runs only when three things are true:
the action is in the action space, enough other decision models adopted it,
and a tested actuator for that exact action was passed in. The default
actuator set is empty. A judgment of "adopted, not executed" is a hold on
the world, recorded as such, not a success.

Nothing here starts itself. A dry run reads the mailbox and prints counts.
It does not write an outcome, restart a process, or copy a seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .genesis import ACTION_SPACE, Proposal
from .proposal_mail import fetch_proposals, is_reaction_task

ADOPT = "赞同"
HOLD = "搁置"
DECLINE = "反对"
STANCES = (ADOPT, HOLD, DECLINE)

#: The actions this program can actually carry out. Everything else in
#: ``ACTION_SPACE`` is a declared intent: adopted, never executed. Kept as
#: one list so the boundary is readable from outside and cannot drift
#: between the judge, the executor and the ledger.
ACTUATORS: tuple[str, ...] = ("seed.export", "seed.replicate")



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


def rows_for_execution(tasks: list, limit: int) -> list:
    """Recent rows, plus every seed.export and seed.replicate, and the reactions that name them.

    The recent window is the last ``limit`` rows. An adopted seed.export
    does not stop being adopted because newer mail pushed it out of that
    window, so those proposals, and the reactions whose titles name them,
    are kept as well. Order stays the listing order.
    """
    rows = [row for row in tasks if isinstance(row, dict)]
    window = rows[-limit:] if limit > 0 else []
    keep = {row.get("id") for row in window}
    seed_ids = set()
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[proposal]")
        if parsed and parsed[1] in ("seed.export", "seed.replicate"):
            seed_ids.add(parsed[0])
            keep.add(row.get("id"))
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[reaction]")
        if parsed and parsed[0] in seed_ids:
            keep.add(row.get("id"))
    return [row for row in rows if row.get("id") in keep]

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
          actuators: dict | None = None) -> dict:
    """Return what should happen. Does not perform the action by itself.

    ``actuators`` maps an action name to ``(Proposal) -> bool``. Missing
    means the colony may have adopted the proposal and still must not act.
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
    if proposal.action not in ACTION_SPACE:
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
    actuator = (actuators or {}).get(proposal.action)
    if actuator is None:
        result["detail"] = "adopted, not executed: no actuator"
        return result
    try:
        ok = bool(actuator(proposal))
    except Exception as exc:
        result["detail"] = f"actuator failed before completion: {exc}"
        return result
    result["executed"] = True
    result["success"] = ok
    result["detail"] = "actuator reported " + ("success" if ok else "failure")
    return result




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
                     done: set[str], signer: str | None = None) -> Proposal | None:
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
        result = judge(proposal, reactions, actuators={})
        if result["adopted"]:
            return proposal
    return None


def write_status(path: Path, payload: dict) -> None:
    """One file, replaced each run. The ledger stays append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def foreign_adopted_seeds(proposals: list[Proposal], reactions: list[dict],
                          done: set[str], signer: str) -> list[str]:
    """Adopted seed.export ids this host must not seal."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.export":
            continue
        if proposal.pubkey == signer:
            continue
        if judge(proposal, reactions, actuators={})["adopted"]:
            found.append(proposal.proposal_id)
    return found


def foreign_replications(proposals: list[Proposal], reactions: list[dict],
                         done: set[str], signer: str) -> list[str]:
    """Adopted seed.replicate ids signed by someone else. Not ours to send."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if not proposal.pubkey or proposal.pubkey == signer:
            continue
        if judge(proposal, reactions, actuators={})["adopted"]:
            found.append(proposal.proposal_id)
    return found


def local_replications(proposals: list[Proposal], reactions: list[dict],
                       done: set[str], signer: str) -> list[str]:
    """Adopted seed.replicate ids signed here that name no destination."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if proposal.pubkey != signer or proposal.destination:
            continue
        if judge(proposal, reactions, actuators={})["adopted"]:
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
                           done: set[str], signer: str) -> list[tuple[str, str]]:
    """Adopted seed.replicate ids signed here that name a host.

    Naming a host is not permission to send. There is still no transport.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.replicate":
            continue
        if proposal.pubkey != signer or not proposal.destination:
            continue
        if judge(proposal, reactions, actuators={})["adopted"]:
            found.append((proposal.proposal_id, proposal.destination))
    return found


def local_unacted(proposals: list[Proposal], reactions: list[dict],
                  done: set[str], signer: str) -> list[tuple[str, str]]:
    """Adopted proposals this host signed that have no actuator.

    Anything in ``ACTUATORS`` is left out: those are recorded by their own
    path (seed.export seals, seed.replicate wants a destination), so calling
    them unactuated here would double-count them as failures.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.pubkey != signer:
            continue
        if proposal.action in ACTUATORS:
            continue
        if judge(proposal, reactions, actuators={})["adopted"]:
            found.append((proposal.proposal_id, proposal.action))
    return found


def dry_run(proposals: list[Proposal], reactions: list[dict], **policy) -> dict:
    """Judge every proposal. No actuator, no write."""
    results = [judge(proposal, reactions, actuators={}, **policy)
               for proposal in proposals]
    return {"summary": summarize(results), "results": results}


def _load_reactions(url: str, token: str | None, limit: int,
                    rows: list | None = None) -> list[dict]:
    from .proposal_mail import _request
    from .genesis import verify_reaction
    if rows is None:
        listing = _request(url, "/tasks?status=all", token, "GET")
        tasks = listing.get("tasks") or []
        tasks = tasks[-limit:]
    else:
        tasks = rows
    found = []
    for task in tasks:
        if not isinstance(task, dict) or not is_reaction_task(str(task.get("title") or "")):
            continue
        full = _request(url, f"/task/{task.get('id')}", token, "GET")
        body = (full.get("task") or {}).get("body")
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
    """Export one sealed seed, or do nothing if none is waiting.

    The home directory is empty on purpose: the bundle must not copy the
    mailbox token or task files. Only a seed.export signed by this host is sealed. A seed.replicate
    with no destination is recorded and not sent.
    """
    phrase = Path(args.passphrase_file).expanduser()
    if not phrase.is_file():
        print("no passphrase file; refusing to write a plaintext key", flush=True)
        return 2
    ledger = Path(args.ledger).expanduser()
    from .identity import Identity
    signer = Identity.from_json(json.loads(
        Path(args.identity).expanduser().read_text(encoding="utf-8"))).public_key
    chosen = next_seed_export(proposals, reactions, done_ids(ledger), signer)
    sealed = None
    if chosen is not None:
        home = Path("~/.pandosi/seed-home-empty").expanduser()
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
        ledger.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "proposal_id": chosen.proposal_id,
            "action": "seed.export",
            "executed": bool(sealed),
            "sealed": bool(sealed),
        }
        if not sealed:
            # No ``skipped``: a failed seal is retried next round, so the
            # adopted proposal is not dropped. The reason still travels, or
            # a broken actuator is indistinguishable from an idle one.
            record["failed"] = failure or "export wrote no manifest"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    # An export must not crowd out a replicate. Both are recorded.
    if True:
        seen = done_ids(ledger) | left_ids(ledger)
        foreign = foreign_adopted_seeds(proposals, reactions, seen, signer)
        replicates = local_replications(proposals, reactions, seen, signer)
        addressed = addressed_replications(proposals, reactions, seen, signer)
        hosts = getattr(args, "hosts", None)
        foreign_repl = foreign_replications(proposals, reactions, seen, signer)
        unacted = local_unacted(proposals, reactions, seen, signer)
        pulls = getattr(args, "pull", None) or set()
        outbox = getattr(args, "outbox", "~/.pandosi/outbox")
        acks = received_acks(ledger, outbox)
        if foreign or replicates or addressed or unacted or foreign_repl or acks:
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
                    if destination in pulls:
                        record = offer_replication(
                            proposal_id, destination, bundle=bundle, outbox=outbox)
                        if record.get("skipped") != "offered":
                            continue
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
                for proposal_id, action in unacted:
                    handle.write(json.dumps({
                        "proposal_id": proposal_id,
                        "action": action,
                        "executed": False,
                        "skipped": "no-actuator",
                    }, sort_keys=True) + "\n")
        payload = {"did": "idle", "foreign_seed_exports": foreign,
                   "replication_without_destination": replicates,
                   "replication_not_sent": [item[0] for item in addressed],
                   "foreign_replications": foreign_repl,
                   "no_actuator": [item[0] for item in unacted]}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        write_status(ledger.with_name("execute-status.json"), payload)
        if sealed is None:
            return 0
        offered = [item[0] for item in addressed
                   if item[1] in (getattr(args, "pull", None) or set())]
        done = {"did": "executed", "proposal_id": chosen.proposal_id,
                "executed": bool(sealed), "offered": offered}
        print(json.dumps(done, sort_keys=True), flush=True)
        write_status(ledger.with_name("execute-status.json"), done)
        return 0 if sealed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="judge proposals; does not execute them")
    parser.add_argument("--mailbox", default="http://127.0.0.1:8776")
    parser.add_argument("--token-file", default="~/.pandosi/mailbox.token")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="export one adopted seed that is not already done")
    parser.add_argument("--passphrase-file", default="~/.pandosi/seed.passphrase")
    parser.add_argument("--identity", default="~/.pando/identity.json")
    parser.add_argument("--out-root", default="~/.pandosi/executions")
    parser.add_argument("--ledger", default="~/.pandosi/execute-ledger.jsonl")
    args = parser.parse_args()
    if args.dry_run == args.once:
        print("pass exactly one of --dry-run or --once", flush=True)
        return 2
    token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    from .proposal_mail import _request
    listing = _request(args.mailbox, "/tasks?status=all", token, "GET")
    rows = rows_for_execution(listing.get("tasks") or [], args.limit)
    proposals = fetch_proposals(args.mailbox, token, status="all", rows=rows)
    reactions = _load_reactions(args.mailbox, token, args.limit, rows=rows)
    if args.once:
        path = Path("~/.pandosi/hosts.json").expanduser()
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
            args.hosts = known_hosts(table)
            args.pull = pull_destinations(table)
        except (OSError, json.JSONDecodeError):
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
