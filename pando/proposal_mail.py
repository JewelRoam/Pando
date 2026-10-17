"""Carry a signed proposal over the mailbox: the group's log, today.

``docs/DESIGN-genesis.md`` §7 left one hole open by name: the signed proposal
record is transport-agnostic, but *nothing carried it*. A live member could
sign a proposal and no other member could receive it, so the agenda was the
group's only inside one process.

This module is the smallest wiring that closes it, over the transport the group
already runs rather than a new one. The mailbox at :8776 is already the place
where members leave each other messages, and it already has the two properties
this needs: an append-only task list, and a body that can hold arbitrary text.

The convention, by the same rule as ``[chat]``:

- Title ``[proposal] ...`` is a **message class**, not work. Automations must
  not claim it, exactly as they must not claim ``[chat]``.
- The body is the JSON of the signed record from ``pando.genesis``.
- On the way in, every record is re-verified and dropped if it is unsigned,
  tampered with, or carrying a key other than the pinned member's. The mailbox
  is a courier, not a source of truth: nothing here trusts what it read.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .genesis import (Proposal, proposals_from_log, sign_proposal,
                      sign_reaction)
from .net import open_url

TITLE_PREFIX = "[proposal]"
PROPOSAL_PREFIX = TITLE_PREFIX


def is_proposal_task(title: str) -> bool:
    """True when a mailbox task title marks a message rather than work."""
    return str(title).lstrip().startswith(TITLE_PREFIX)


REACTION_PREFIX = "[reaction]"


def is_reaction_task(title: str) -> bool:
    """True when a mailbox task is a decision model's answer to a proposal.

    Same rule as ``[chat]`` and ``[proposal]``: it is a message, not work.
    """
    return str(title).lstrip().startswith(REACTION_PREFIX)


def _request(url: str, path: str, token: str | None, method: str,
             payload: dict | None = None, timeout: float = 15) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{url.rstrip('/')}{path}", data=body,
                                     headers=headers, method=method)
    try:
        with open_url(request, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        return {"error": f"HTTP {exc.code}", "detail": detail}
    except urllib.error.URLError as exc:
        return {"error": f"unreachable: {exc}"}


def publish_proposal(proposal: Proposal, identity, url: str,
                     token: str | None = None, timeout: float = 15) -> dict:
    """Sign *proposal* with *identity* and leave it on the mailbox.

    Returns the mailbox's response. The title carries the author and action so
    a human reading the feed can tell what was proposed without opening the
    body; the body is the record itself.
    """
    record = sign_proposal(proposal, identity)
    title = (f"{TITLE_PREFIX} {proposal.proposal_id} "
             f"{proposal.action} by {proposal.author}")
    posted = _request(url, "/task", token, "POST",
                      {"from": proposal.author, "title": title,
                       "body": json.dumps(record, sort_keys=True)},
                      timeout=timeout)
    # A proposal is a message class, not work. Left open it would sit in the
    # same queue a claim-handling automation drains, and the convention "do not
    # claim [proposal]" would be the only thing standing between the agenda and
    # being executed as a task. So claim it as its author, the way
    # tools/agent_chat.py claims a chat message, and say so loudly if the claim
    # does not take -- a silent failure here is a proposal that looks like work.
    task = posted.get("task") if isinstance(posted, dict) else None
    task_id = task.get("id") if isinstance(task, dict) else None
    if task_id:
        claimed = _request(url, "/claim", token, "POST",
                           {"id": task_id, "by": proposal.author},
                           timeout=timeout)
        took = isinstance(claimed, dict) and "task" in claimed
        posted["claimed"] = took
        if not took:
            posted["claim_warning"] = (
                f"sent {task_id} but the claim failed ({claimed}); it may be "
                f"read as a task - claim or close it manually")
    return posted



def publish_reaction(record: dict, node_id: str, url: str,
                     token: str | None = None, timeout: float = 15) -> dict:
    """Leave a signed reaction on the mailbox and claim it so it is not work.

    ``record`` is the output of ``sign_reaction``. The title names the proposal
    and the stance so a reader can see who answered without opening the body.
    """
    payload = record.get("payload") or {}
    title = (f"{REACTION_PREFIX} {payload.get('target_id')} "
             f"{payload.get('stance')} by {node_id}")
    posted = _request(url, "/task", token, "POST",
                      {"from": node_id, "title": title,
                       "body": json.dumps(record, sort_keys=True)},
                      timeout=timeout)
    task = posted.get("task") if isinstance(posted, dict) else None
    task_id = task.get("id") if isinstance(task, dict) else None
    if task_id:
        claimed = _request(url, "/claim", token, "POST",
                           {"id": task_id, "by": node_id}, timeout=timeout)
        took = isinstance(claimed, dict) and "task" in claimed
        posted["claimed"] = took
        if not took:
            posted["claim_warning"] = (
                f"sent {task_id} but the claim failed ({claimed}); it may be "
                f"read as a task - claim or close it manually")
    return posted


def fetch_proposals(url: str, token: str | None = None,
                    pinned: dict[str, str] | None = None,
                    status: str = "all", timeout: float = 15,
                    max_items: int | None = None,
                    rows: list | None = None) -> list[Proposal]:
    """Every verifiable proposal on the mailbox, in the order it was left.

    Unverifiable ones are dropped rather than raising, so a mixed feed (chat,
    work, forgeries) still yields the genuine proposals -- the same contract as
    ``proposals_from_log``. ``pinned`` maps node_id -> public key; a proposal
    whose author is a known member must carry that member's key.

    ``rows`` is a listing already chosen by the caller. When it is given,
    this does not list the mailbox again and does not apply ``max_items``.
    """
    if rows is None:
        listing = _request(url, f"/tasks?status={status}", token, "GET",
                           timeout=timeout)
        tasks = listing.get("tasks") or []
        if max_items is not None:
            tasks = tasks[-max_items:]
    else:
        tasks = rows
    records = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if not is_proposal_task(str(task.get("title", ""))):
            continue
        full = _request(url, f"/task/{task.get('id')}", token, "GET",
                        timeout=timeout)
        body = (full.get("task") or {}).get("body")
        if not isinstance(body, str):
            continue
        try:
            records.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return proposals_from_log(records, pinned=pinned)
