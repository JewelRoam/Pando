"""Sealing seeds into bundles and offering them to pull hosts.

The only actuator pair that touches durable state on this host's own disk:
``export_seed`` seals this host's identity into ``out_root/<proposal_id>``;
``offer_replication`` copies a sealed bundle into a local outbox for a host
that pulls.  Neither opens a socket.

The send/receive bookkeeping (done/left ids, foreign and local replication
views, ack tracking) lives here because those rows are what the ledger
appends when a round reconciles the seed pair.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ..genesis import Proposal

from . import reconcile
from .judge import adopted
from .reconcile import _seed_module

# Re-exported for callers that imported them here historically.
from .reconcile import known_hosts, pull_destinations, replication_skip

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
