"""Ask this machine's own decision models what they think of a proposal.

A proposal is a message. A reaction is the short answer a decision model
puts under it: agree, hold, or decline. It sits on that proposal the way a
chat reaction sits on a message, and it is not another item in the feed.

The reactor only calls stems registered on this machine. A root that still
lists another device's members is not a reason to ask them. Each end reacts
for itself.

One proposal per pass. Walking the whole history through the resident model
would spend the machine on display, not on thought. The first pass answers
the newest proposal and marks the older ones seen.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import DecisionRequest
from .genesis import REACTION_STANCES, sign_reaction
from .identity import load_or_create
from .net import local_addresses, open_url
from .proposal_mail import fetch_proposals, publish_reaction


def select_local_stems(nodes: dict, addresses: set[str]) -> list[dict]:
    """Stems whose host is this machine. Remote members are left alone."""
    chosen = []
    for node_id, info in (nodes or {}).items():
        if not isinstance(info, dict):
            continue
        host = str(info.get("host") or "")
        url = str(info.get("url") or "")
        if host not in addresses or not url:
            continue
        chosen.append({
            "node_id": str(node_id),
            "url": url.rstrip("/"),
            "checkpoint": str(info.get("checkpoint") or ""),
            "host": host,
        })
    chosen.sort(key=lambda row: row["node_id"])
    return chosen


def load_seen(path: Path) -> set[str]:
    """Proposal ids already answered or deliberately skipped."""
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return seen
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = str(row.get("target_id") or "")
        if target and (row.get("skip") or row.get("done")):
            seen.add(target)
    return seen


def append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_token(path: str | None) -> str | None:
    if not path:
        return None
    try:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def fetch_nodes(root_url: str, token: str | None, timeout: float = 6) -> dict:
    headers = {}
    if token:
        headers["X-Pairing-Token"] = token
    request = urllib.request.Request(
        f"{root_url.rstrip('/')}/nodes", headers=headers, method="GET")
    with open_url(request, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    return nodes if isinstance(nodes, dict) else {}


def ask_stem(stem: dict, proposal, timeout: float) -> dict:
    """One stem, one proposal, one of the three stances."""
    goal = (proposal.goal or "").replace("\n", " ")[:160]
    question = (f"是否采纳这条提案？提案者 {proposal.author}，"
                f"动作 {proposal.action}，目标 {goal}")
    req = DecisionRequest(
        state=str(proposal.proposal_id)[:180],
        question=question,
        candidates=list(REACTION_STANCES),
        request_id=f"react-{proposal.proposal_id}"[:80],
    )
    request = urllib.request.Request(
        f"{stem['url']}/decide", data=req.to_json(),
        headers={"Content-Type": "application/json"}, method="POST")
    with open_url(request, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    vote = payload.get("vote") if isinstance(payload, dict) else None
    if not isinstance(vote, dict):
        raise RuntimeError(f"no vote from {stem['node_id']}")
    stance = str(vote.get("choice") or "")
    if stance not in REACTION_STANCES:
        raise RuntimeError(f"{stem['node_id']} answered {stance!r}")
    probability = vote.get("probability")
    return {"stance": stance,
            "probability": None if vote.get("neutral") else probability}


def react_once(proposals, stems: list[dict], identity, mailbox: str,
               token: str | None, ledger: Path, timeout: float) -> dict:
    """Answer one not-yet-answered proposal.

    On the first run, every proposal but the newest is marked skipped, so a
    restart does not walk history through the models.
    """
    seen = load_seen(ledger)
    ordered = [item for item in proposals if getattr(item, "proposal_id", "")]
    if not ordered:
        return {"did": "idle", "reason": "no proposals"}
    if not seen:
        for old in ordered[:-1]:
            append_ledger(ledger, {"target_id": old.proposal_id, "skip": True,
                                   "at": time.time()})
            seen.add(old.proposal_id)
    pending = [item for item in ordered if item.proposal_id not in seen]
    if not pending:
        return {"did": "idle", "reason": "caught up", "seen": len(seen)}
    if not stems:
        return {"did": "idle", "reason": "no local stems"}
    proposal = pending[0]
    votes = []
    errors = []
    for stem in stems:
        if stem["node_id"] == proposal.author:
            continue
        try:
            vote = ask_stem(stem, proposal, timeout)
        except (urllib.error.URLError, TimeoutError, RuntimeError,
                json.JSONDecodeError, OSError) as exc:
            errors.append(f"{stem['node_id']}: {exc}")
            continue
        record = sign_reaction(
            proposal.proposal_id, stem["node_id"], stem["checkpoint"],
            vote["stance"], identity, probability=vote["probability"])
        posted = publish_reaction(record, stem["node_id"], mailbox, token,
                                  timeout=timeout)
        votes.append({"node_id": stem["node_id"], "stance": vote["stance"],
                      "claimed": posted.get("claimed")})
    if errors and not votes:
        return {"did": "error", "target": proposal.proposal_id, "errors": errors}
    append_ledger(ledger, {
        "target_id": proposal.proposal_id, "done": True, "at": time.time(),
        "votes": votes, "errors": errors,
    })
    return {"did": "reacted", "target": proposal.proposal_id,
            "votes": votes, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="local decision-model reactions")
    parser.add_argument("--root", default="http://127.0.0.1:8765")
    parser.add_argument("--pairing-token-file", default="~/.pandosi/pairing.token")
    parser.add_argument("--mailbox", default="http://127.0.0.1:8776")
    parser.add_argument("--token-file", default="~/.pandosi/mailbox.token")
    parser.add_argument("--ledger", default="~/.pandosi/reaction-ledger.jsonl")
    parser.add_argument("--interval", type=float, default=40)
    parser.add_argument("--timeout", type=float, default=50)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    identity = load_or_create()
    pairing = _read_token(args.pairing_token_file)
    token = _read_token(args.token_file)
    ledger = Path(args.ledger).expanduser()
    addresses = local_addresses()
    while True:
        try:
            nodes = fetch_nodes(args.root, pairing)
            stems = select_local_stems(nodes, addresses)
            proposals = fetch_proposals(args.mailbox, token, status="all",
                                        max_items=30, timeout=args.timeout)
            report = react_once(proposals, stems, identity, args.mailbox,
                                token, ledger, args.timeout)
            print(time.strftime("%H:%M:%S"),
                  json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001 - a pass must not kill the loop
            print(time.strftime("%H:%M:%S"), f"pass failed: {exc}", flush=True)
            report = {"did": "error"}
        if args.once:
            return 0 if report.get("did") != "error" else 1
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
