"""Ask this machine's own decision models what they think of a proposal.

A proposal is a message. A reaction is the short answer a decision model
puts under it: agree, hold, or decline. It sits on that proposal the way a
chat reaction sits on a message, and it is not another item in the feed.

Only decision models answer. An LLM writes proposals and does not hold a
vote, so asking one to react would put a model in a job it was never given.
Whether a member decides travels in its own ``role`` field; anything that
does not say ``decision`` is left out.

The reactor only calls stems registered on this machine. A root that still
lists another device's members is not a reason to ask them. Each end reacts
for itself.

One proposal per pass. Walking the whole history through the resident model
would spend the machine on display, not on thought. The first pass answers
the newest proposal and marks the older ones seen.

A reactor that only walks forward from the oldest unseen proposal wedges as
soon as it falls behind: every pass spends a model forward-pass on a proposal
that was already decided hours ago, and the feed keeps producing new ones. So
a proposal older than ``--max-age`` is marked skipped instead of answered.
Age travels in the id the proposer stamped, so the sweep needs the listing
only, never a signed body.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import DECIDING_ROLE, DecisionRequest
from .genesis import REACTION_STANCES, sign_reaction
from .identity import load_or_create
from .net import local_addresses, open_url
from .proposal_mail import (delivery_failed, fetch_proposals_any,
                            is_proposal_task, mailbox_chain,
                            proposal_age_seconds, proposal_id_from_title,
                            publish_reaction, _request)


def select_local_stems(nodes: dict, addresses: set[str]) -> list[dict]:
    """Decision models on this machine. Everything else is left alone.

    A member with no ``role`` is treated as a decision model, so a root
    that predates the field keeps answering the way it did before.
    """
    chosen = []
    for node_id, info in (nodes or {}).items():
        if not isinstance(info, dict):
            continue
        host = str(info.get("host") or "")
        url = str(info.get("url") or "")
        if host not in addresses or not url:
            continue
        role = str(info.get("role") or DECIDING_ROLE)
        if role != DECIDING_ROLE:
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


#: Most stale proposals one pass will retire. A backlog is bounded by how
#: long the reactor was down, and one pass should not write an unbounded
#: ledger row count; the next pass sweeps whatever is left.
STALE_SWEEP_LIMIT = 2000

#: The append-only feed has tens of thousands of rows. Both the stale-title
#: scan and the signed-body read only need a recent tail; fetching the whole
#: history every 40 seconds made display dominate reaction work.
STALE_SWEEP_FEED_LIMIT = 2000
REACT_FEED_LIMIT = 500

#: A proposal older than this is history, not an open question. An hour is
#: long enough for a proposal to cross three mailboxes and for every host to
#: poll, and short enough that a vote still lands while the proposal matters.
DEFAULT_MAX_AGE = 3600.0


def stale_ids(proposal_ids, seen, max_age: float, now: float | None = None):
    """Unanswered proposal ids older than *max_age*, oldest first.

    An id with no stamp (a genesis seed) has no age and is never swept: it is
    not stale, it is unattributable, and guessing would drop the seed's own
    proposals.
    """
    moment = time.time() if now is None else now
    found = []
    for pid in proposal_ids:
        pid = str(pid)
        if not pid or pid in seen:
            continue
        age = proposal_age_seconds(pid, moment)
        if age is not None and age > max_age:
            found.append((age, pid))
    found.sort(reverse=True)
    return [pid for _age, pid in found]


def sweep_stale(mailboxes, token: str | None, seen: set[str],
                max_age: float, ledger: Path, timeout: float,
                now: float | None = None,
                limit: int | None = STALE_SWEEP_FEED_LIMIT) -> int:
    """Retire every proposal in the chain that is too old to answer.

    Reads titles only. Returns how many ids were newly marked skipped.
    """
    if not max_age or max_age <= 0:
        return 0
    rows = []
    for url in mailboxes:
        path = "/tasks?status=all"
        if limit is not None:
            path += f"&limit={int(limit)}"
        listing = _request(url, path, token, "GET",
                           timeout=timeout)
        if isinstance(listing, dict) and listing.get("error"):
            continue
        rows = listing.get("tasks") if isinstance(listing, dict) else None
        if rows:
            break
    if not rows:
        return 0
    stamped = [proposal_id_from_title(str(row.get("title") or ""))
               for row in rows
               if isinstance(row, dict)
               and is_proposal_task(str(row.get("title") or ""))]
    moment = time.time() if now is None else now
    swept = 0
    for pid in stale_ids(stamped, seen, max_age, moment):
        if swept >= STALE_SWEEP_LIMIT:
            break
        append_ledger(ledger, {"target_id": pid, "skip": True,
                               "stale": True, "at": moment})
        seen.add(pid)
        swept += 1
    return swept


def append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_delivered(path: Path) -> dict[str, set[str]]:
    """Which (proposal, member) reactions the mailbox has already accepted.

    A proposal is only ``done`` after every local decision model's reaction
    was accepted. Remembering successful posts separately means a partial
    mailbox failure can be retried without re-asking models that already got
    through. The old done-row format is read too, so upgrades keep their
    existing history.
    """
    delivered: dict[str, set[str]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return delivered
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        target = str(row.get("target_id") or "")
        if not target:
            continue
        if row.get("delivered"):
            node = str(row.get("node_id") or "")
            if node:
                delivered.setdefault(target, set()).add(node)
        if row.get("done"):
            for vote in row.get("votes") or []:
                if isinstance(vote, dict) and vote.get("node_id"):
                    delivered.setdefault(target, set()).add(str(vote["node_id"]))
    return delivered


def _mailboxes(urls) -> list[str]:
    if isinstance(urls, str):
        return mailbox_chain(urls)
    urls = list(urls or [])
    return mailbox_chain(urls[0] if urls else "", urls[1:])


def publish_reaction_any(record: dict, node_id: str, urls,
                         token: str | None, timeout: float) -> dict:
    """Publish a reaction to the first mailbox that hears it.

    A dead mailbox and a 5xx go to the next URL; a 4xx is a final refusal
    and is returned without asking another host.
    """
    last = {"error": "unreachable: no mailbox configured"}
    for url in _mailboxes(urls):
        result = publish_reaction(record, node_id, url, token, timeout)
        if not delivery_failed(result):
            if isinstance(result, dict):
                result.setdefault("mailbox", url)
            return result
        last = result
    return last


def _accepted(result) -> bool:
    if not isinstance(result, dict) or result.get("error"):
        return False
    return (result.get("task") is not None
            or result.get("claimed") is not None
            or result.get("claim_warning") is not None)


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


def react_once(proposals, stems: list[dict], identity, mailbox,
               token: str | None, ledger: Path, timeout: float,
               mailbox_fallbacks=None, max_age: float | None = None) -> dict:
    """Answer one not-yet-answered proposal.

    On the first run, every proposal but the newest is marked skipped, so a
    restart does not walk history through the models. A reaction is only
    counted as done after a mailbox accepted it; a failed post leaves the
    proposal retryable instead of silently treating a lost vote as a vote.

    ``max_age`` retires proposals that are already history. Without it a
    reactor that fell behind answers the oldest unseen proposal for ever and
    never reaches a current one.
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
    retired: list[str] = []
    if max_age and max_age > 0:
        retired = stale_ids([item.proposal_id for item in pending], seen,
                            max_age)
        for pid in retired:
            append_ledger(ledger, {"target_id": pid, "skip": True,
                                   "stale": True, "at": time.time()})
            seen.add(pid)
        if retired:
            expired = set(retired)
            pending = [item for item in pending
                       if item.proposal_id not in expired]
    if not pending:
        return {"did": "idle", "reason": "caught up", "seen": len(seen),
                "stale": len(retired)}
    if not stems:
        return {"did": "idle", "reason": "no local stems"}
    proposal = pending[0]
    delivered = load_delivered(ledger).get(proposal.proposal_id, set())
    votes = []
    errors = []
    for stem in stems:
        if stem["node_id"] == proposal.author or stem["node_id"] in delivered:
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
        posted = publish_reaction_any(record, stem["node_id"], mailbox, token,
                                      timeout)
        if not _accepted(posted):
            detail = posted.get("error") if isinstance(posted, dict) else posted
            errors.append(f"{stem['node_id']}: publish: {detail}")
            continue
        append_ledger(ledger, {
            "target_id": proposal.proposal_id,
            "node_id": stem["node_id"],
            "delivered": True,
            "stance": vote["stance"],
            "mailbox": posted.get("mailbox"),
            "at": time.time(),
        })
        delivered.add(stem["node_id"])
        votes.append({"node_id": stem["node_id"], "stance": vote["stance"],
                      "claimed": posted.get("claimed")})
    expected = {stem["node_id"] for stem in stems
                if stem["node_id"] != proposal.author}
    complete = expected <= delivered
    append_ledger(ledger, {
        "target_id": proposal.proposal_id, "done": complete, "at": time.time(),
        "votes": votes, "errors": errors,
    })
    if errors or not complete:
        return {"did": "error", "target": proposal.proposal_id,
                "votes": votes, "errors": errors, "complete": complete}
    return {"did": "reacted", "target": proposal.proposal_id,
            "votes": votes, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="local decision-model reactions")
    parser.add_argument("--root", default="http://127.0.0.1:8765")
    parser.add_argument("--pairing-token-file", default="~/.pando/pairing.token")
    parser.add_argument("--mailbox", default="http://127.0.0.1:8776")
    parser.add_argument("--mailbox-fallback", action="append", default=[],
                        help="mailbox to try after the primary (repeatable)")
    parser.add_argument("--token-file", default="~/.pando/mailbox.token")
    parser.add_argument("--ledger", default="~/.pando/reaction-ledger.jsonl")
    parser.add_argument("--interval", type=float, default=40)
    parser.add_argument("--timeout", type=float, default=50)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE,
                        help="retire proposals older than this many seconds "
                             "instead of answering them (0 = no limit)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    identity = load_or_create()
    pairing = _read_token(args.pairing_token_file)
    token = _read_token(args.token_file)
    ledger = Path(args.ledger).expanduser()
    addresses = local_addresses()
    mailboxes = mailbox_chain(args.mailbox, args.mailbox_fallback)
    max_age = args.max_age if args.max_age and args.max_age > 0 else None
    while True:
        try:
            nodes = fetch_nodes(args.root, pairing)
            stems = select_local_stems(nodes, addresses)
            seen = load_seen(ledger)
            swept = sweep_stale(mailboxes, token, seen, max_age, ledger,
                                args.timeout)
            proposals, _source, read_error = fetch_proposals_any(
                mailboxes, token, status="all", timeout=args.timeout,
                max_items=REACT_FEED_LIMIT, skip_ids=seen)
            if read_error and not proposals:
                raise RuntimeError(read_error)
            report = react_once(proposals, stems, identity, mailboxes,
                                token, ledger, args.timeout,
                                max_age=max_age)
            if swept:
                report["swept"] = swept
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
