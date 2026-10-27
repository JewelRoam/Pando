"""Chat is a field on a member, not a name in a program.

A member can speak in the group only when its record has ``chat.model``.
Decision-only members (a text model, an ONNX voter) do not have that field,
so they are not given a chat identity by a list somewhere else. The stem
serves ``POST /chat`` for its own model. Each machine's chatd answers only
for members whose host is itself.
"""

from __future__ import annotations

import re

PEER_CAP = 2
_MODEL_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
_ADDRESSEE = re.compile(r"^@([A-Za-z0-9][A-Za-z0-9_.-]{0,63})$")


def normalize_chat(value) -> dict | None:
    """The only chat record we keep: ``{"model": "<id>"}``.

    Anything else (a URL, a role, a display name) is dropped. The endpoint
    is the member's own stem URL; it is not copied into this field.
    """
    if not isinstance(value, dict):
        return None
    model = value.get("model")
    if not isinstance(model, str) or not _MODEL_OK.fullmatch(model.strip()):
        return None
    return {"model": model.strip()}


def title_addressee(title: str) -> str | None:
    """The single @token in a ``[chat]`` title, or None.

    ``@@name`` is not a mention. The body is ignored: a reply is always
    addressed back to its sender, and reading the body would make two
    members answer each other forever.
    """
    text = str(title or "")
    if not text.startswith("[chat]"):
        return None
    rest = text[len("[chat]"):].strip()
    head, sep, _ = rest.partition(":")
    if not sep:
        head = rest
    match = _ADDRESSEE.fullmatch(head.strip())
    return match.group(1) if match else None


def message_text(task: dict) -> str:
    body = str(task.get("body") or "").strip()
    if body:
        return body
    title = str(task.get("title") or "")
    rest = title[len("[chat]"):].strip() if title.startswith("[chat]") else title
    _, sep, preview = rest.partition(":")
    return (preview if sep else rest).strip()


def trailing_run(senders: list[str], sender: str) -> int:
    run = 0
    for item in reversed(senders):
        if item != sender:
            break
        run += 1
    return run


def should_answer(member_id: str, sender: str, addressee: str | None,
                  chat_ids: set[str], peer_run: int,
                  peer_cap: int = PEER_CAP) -> bool:
    """Whether this member answers this one message.

    A person (anyone who is not themselves a chat member) reaches a member
    with ``@member`` or ``@all``. Another chat member reaches it only by
    naming it, and only ``peer_cap`` times in a row, so two models cannot
    ping-pong on ``@all``.
    """
    if not member_id or sender == member_id or not addressee:
        return False
    if addressee not in (member_id, "all"):
        return False
    if sender in chat_ids:
        if addressee != member_id:
            return False
        if peer_run >= peer_cap:
            return False
    return True


def next_reply(members: list[dict], chat_ids: set[str], tasks: list[dict],
               answered: set[tuple[str, str]],
               peer_senders: dict[str, list[str]]) -> dict | None:
    """Oldest still-unanswered chat message that some local member should take."""
    known = set(chat_ids) | {str(item.get("node_id") or "") for item in members}
    ordered = sorted(tasks, key=lambda task: str(task.get("created_at") or ""))
    for task in ordered:
        source = str(task.get("id") or "")
        if not source or not str(task.get("title") or "").startswith("[chat]"):
            continue
        addressee = title_addressee(str(task.get("title") or ""))
        sender = str(task.get("from") or "")
        for member in members:
            node_id = str(member.get("node_id") or "")
            if not node_id or (node_id, source) in answered:
                continue
            run = trailing_run(peer_senders.get(node_id) or [], sender)
            if should_answer(node_id, sender, addressee, known, run):
                return {"member": member, "task": task}
    return None


OPEN_COOLDOWN = 3600.0


def chat_peer_ids(members: list[dict], known: set[str],
                  opened: dict[tuple[str, str], float] | None = None) -> set[str]:
    """Remote chat peers currently discovered or previously addressed.

    A root can be dark while the mailbox still carries a usable replica. An
    ``open`` ledger row is evidence that a local chat member previously chose
    this peer from a live chat-member record, so forgetting that peer just
    because ``/nodes`` timed out would turn a partition into permanent silence.
    """
    local = {str(item.get("node_id") or "") for item in members}
    local.discard("")
    peers = {str(node) for node in (known or set()) if node}
    for sender, peer in (opened or {}):
        sender = str(sender or "")
        peer = str(peer or "")
        if sender in local and peer:
            peers.add(peer)
    peers.difference_update(local)
    return peers


def next_opening(members: list[dict], known: set[str],
                 opened: dict[tuple[str, str], float], now: float,
                 cooldown: float = OPEN_COOLDOWN) -> dict | None:
    """A conversation this machine should start, or None.

    Every ``[chat]`` row this colony has ever written is an answer:
    2026-10-19, 159 chat tasks, zero of them member-to-member.
    ``should_answer`` already lets one member reach another by naming it --
    nothing was ever going to use that, because nothing initiates. A colony
    of members who can only answer is a panel of respondents, not a group.

    Peers are the chat members that are *not* on this host. A member talking
    to a sibling it shares a chatd with is one process talking to itself, and
    it would satisfy the letter of "member-to-member" while leaving the
    colony exactly as silent as before.

    The pick is a rotation, not a latch: the local member that opened least
    recently speaks, to the peer it has opened to least recently, ties broken
    by node id so the choice is stable across restarts. *cooldown* is per
    host, not per pair -- a per-pair throttle would scale its total allowed
    chatter with the square of the colony, which is the direction the wrong
    one grows.
    """
    local = [str(item.get("node_id") or "") for item in members]
    local = [node_id for node_id in local if node_id]
    if not local:
        return None
    peers = sorted(chat_peer_ids(members, known, opened))
    if not peers:
        return None
    if opened and now - max(opened.values()) < cooldown:
        return None

    def last_from(sender: str) -> float:
        return max((opened.get((sender, peer), float("-inf")) for peer in peers),
                   default=float("-inf"))

    sender = min(local, key=lambda node_id: (last_from(node_id), node_id))
    peer = min(peers, key=lambda node_id: (opened.get((sender, node_id),
                                                      float("-inf")), node_id))
    member = next(item for item in members
                  if str(item.get("node_id") or "") == sender)
    return {"member": member, "to": peer}
