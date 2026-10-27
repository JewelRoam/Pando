"""Speak for this machine's chat members, and nobody else's.

Reads ``chat.model`` off the live node table (or the stem's ``/health`` when
an older root has not stored the field yet). A member is answered here only
when its host address is this machine. Remote models stay remote: their own
chatd speaks for them.

    python -m pando.chatd --hosts ~/.pandosi/hosts.json \
        --token-file ~/.pandosi/mailbox.token
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .chat import (OPEN_COOLDOWN, chat_peer_ids, message_text, next_opening,
                   next_reply, normalize_chat)
from .common import PROPOSING_ROLE
from .net import (LOOPBACK, local_addresses, open_url, response_json,
                  response_text, should_request_gzip)
from .proposal_mail import delivery_failed, mailbox_chain


# chatd polls every few seconds while the mailbox retains the whole colony
# history. Only the newest slice can participate in a conversation; fetching
# 35k summary rows on every pass makes a healthy chat loop pay for all memory.
CHAT_FEED_LIMIT = 500
# The feed is dominated by proposal/reaction traffic. Ask the mailbox to
# filter before applying the bound, otherwise reactions can push unread chat
# messages and the proposal needed to ground an opening out of the window.
CHAT_FEED_KINDS = ("chat", "proposal")

# A root that times out on every pass must not tax every healthy pass. Wait
# briefly after the first miss, then grow to a bounded probe interval so a
# recovered root is seen without another process or timer.
ROOT_RETRY_BASE_S = 30.0
ROOT_RETRY_MAX_S = 300.0


class RootProbeBackoff:
    """Per-root retry state for repeated root probe failures."""

    def __init__(self, base: float = ROOT_RETRY_BASE_S,
                 cap: float = ROOT_RETRY_MAX_S):
        self.base = max(0.0, float(base))
        self.cap = max(self.base, float(cap))
        self._pending: dict[str, tuple[int, float]] = {}

    def due(self, key: str, now: float) -> bool:
        return now >= self._pending.get(key, (0, 0.0))[1]

    def retry_in(self, key: str, now: float) -> float:
        return max(0.0, self._pending.get(key, (0, 0.0))[1] - now)

    def failed(self, key: str, now: float) -> float:
        failures = self._pending.get(key, (0, 0.0))[0] + 1
        delay = min(self.cap, self.base * (2 ** min(failures - 1, 16)))
        self._pending[key] = (failures, now + delay)
        return delay

    def succeeded(self, key: str) -> None:
        self._pending.pop(key, None)


def _read(path: str | None) -> str | None:
    if not path:
        return None
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def load_hosts(path: str) -> tuple[list[dict], list[str]]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    roots = []
    for row in data.get("roots") or []:
        if not isinstance(row, dict) or not row.get("url"):
            continue
        roots.append({
            "name": str(row.get("name") or row["url"]),
            "url": str(row["url"]).rstrip("/"),
            "token": _read(row.get("token_file")),
        })
    # A single primary is a partition hole: when the local mailbox is the one
    # that is down, the replica is the only place a chat can land. hosts.json
    # may name a chain, and the whole chain is kept in order.
    mailboxes = [str(url).rstrip("/") for url in (data.get("mailboxes") or [])
                 if str(url).strip()]
    return roots, mailbox_chain(mailboxes[0] if mailboxes else "",
                                mailboxes[1:])


def fetch_nodes(root: str, token: str | None, timeout: float) -> dict:
    headers = {}
    if token:
        headers["X-Pairing-Token"] = token
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(root.rstrip("/") + "/nodes", headers=headers)
    with open_url(request, timeout) as resp:
        payload = json.loads(resp.read())
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    return nodes if isinstance(nodes, dict) else {}


def fetch_health(url: str, timeout: float) -> dict | None:
    request = urllib.request.Request(url.rstrip("/") + "/health")
    with open_url(request, timeout) as resp:
        payload = json.loads(resp.read())
    return payload if isinstance(payload, dict) else None


def probe_roots(roots: list[dict], timeout: float,
                backoff: RootProbeBackoff,
                now: float | None = None) -> tuple[dict, list[dict]]:
    """Read every root that is due, returning nodes and observable outcomes.

    A timeout costs the whole pass when a dark root is probed every time. The
    backoff is only a deferral: it caps, resets on success, and still returns
    the healthy roots on this pass.
    """
    nodes: dict = {}
    events: list[dict] = []
    current = time.monotonic() if now is None else now
    for root in roots:
        url = str(root.get("url") or "").rstrip("/")
        if not url:
            continue
        name = str(root.get("name") or url)
        if not backoff.due(url, current):
            events.append({"name": name, "url": url, "deferred": True,
                           "retry_in": backoff.retry_in(url, current)})
            continue
        try:
            found = fetch_nodes(url, root.get("token"), min(8, timeout))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                OSError) as exc:
            # Start the cooling-off window when the failure lands. Using the
            # pass start made a timeout consume most of its own retry delay.
            failed_at = time.monotonic() if now is None else current
            retry_in = backoff.failed(url, failed_at)
            events.append({"name": name, "url": url, "error": str(exc),
                           "retry_in": retry_in})
            continue
        backoff.succeeded(url)
        nodes.update(found)
        events.append({"name": name, "url": url, "ok": True,
                       "nodes": len(found)})
    return nodes, events


def local_chat_members(nodes: dict, addresses: set[str],
                       health_of=None, trust_local_root: bool = False
                       ) -> tuple[list[dict], set[str]]:
    """Local speakers, plus every node id that carries ``chat.model``.

    Only a proposer speaks: a decision model answers proposals and is not
    put in the conversation. ``role`` says which is which; a member with no
    role but a chat descriptor is read as a proposer, so a root that
    predates the field keeps working.

    ``health_of(url) -> dict`` is only called for a local stem whose node
    record does not yet have the field. ``trust_local_root`` is for the
    single explicit loopback-root case used by Windows and pi5: every member
    in that root is on this machine even when its advertised host (for
    example ``192.168.31.100``) is not one of Python's local addresses.
    """
    members = []
    known: set[str] = set()
    for node_id, info in (nodes or {}).items():
        if not isinstance(info, dict):
            continue
        chat = normalize_chat(info.get("chat"))
        host = str(info.get("host") or "")
        url = str(info.get("url") or "").rstrip("/")
        role = str(info.get("role") or (PROPOSING_ROLE if chat else ""))
        if role and role != PROPOSING_ROLE:
            continue
        local = bool(url) and (trust_local_root or host in addresses)
        if chat is None and local and health_of is not None:
            try:
                health = health_of(url) or {}
            except (OSError, urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError, RuntimeError):
                health = {}
            chat = normalize_chat(health.get("chat") if isinstance(health, dict) else None)
        if chat is None:
            continue
        known.add(str(node_id))
        if local:
            members.append({
                "node_id": str(node_id),
                "url": url,
                "model": chat["model"],
                "host": host,
            })
    members.sort(key=lambda row: row["node_id"])
    return members, known


def root_is_loopback(url: str) -> bool:
    """Whether an explicit root is the local root, not a remote snapshot."""
    try:
        host = urlsplit(str(url)).hostname or ""
    except ValueError:
        return False
    return host in LOOPBACK


def load_ledger(path: Path) -> tuple[set[tuple[str, str]], dict[str, list[str]], bool]:
    answered: set[tuple[str, str]] = set()
    peers: dict[str, list[str]] = {}
    existed = path.exists()
    if not existed:
        return answered, peers, False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return answered, peers, True
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("bootstrap"):
            for source in row.get("ids") or []:
                answered.add(("*", str(source)))
            continue
        node_id = str(row.get("node_id") or "")
        source = str(row.get("source_id") or "")
        sender = str(row.get("sender") or "")
        if node_id and source:
            answered.add((node_id, source))
        if node_id and sender and row.get("id"):
            peers.setdefault(node_id, []).append(sender)
    return answered, peers, True


def append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_openings(path: Path) -> dict[tuple[str, str], float]:
    """When this host last tried to start a conversation with whom.

    The same ledger, read for its own rows: an opening carries ``open``,
    ``node_id`` and ``to`` and has no ``source_id``, so ``load_ledger`` can
    never read one as an answer to somebody's message.
    """
    opened: dict[tuple[str, str], float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return opened
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("open"):
            continue
        node_id = str(row.get("node_id") or "")
        peer = str(row.get("to") or "")
        if not node_id or not peer:
            continue
        try:
            opened[(node_id, peer)] = float(row.get("at") or 0.0)
        except (TypeError, ValueError):
            continue
    return opened


def load_peer_evidence(path: Path) -> dict[tuple[str, str], float]:
    """When a remembered peer was last actually seen or first unanswered.

    A successful inbound chat refreshes evidence. The first opening after that
    reply does too, because it was chosen from live membership. Later openings
    into a dark root must not keep extending the same evidence forever.
    """
    last_reply: dict[tuple[str, str], float] = {}
    unanswered: dict[tuple[str, str], float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("node_id") or "")
        if row.get("open"):
            peer = str(row.get("to") or "")
            key = (node_id, peer)
            try:
                at = float(row.get("at") or 0.0)
            except (TypeError, ValueError):
                continue
            if node_id and peer and key not in unanswered:
                unanswered[key] = at
            continue
        sender = str(row.get("sender") or "")
        if node_id and sender and row.get("id"):
            key = (node_id, sender)
            try:
                at = float(row.get("at") or 0.0)
            except (TypeError, ValueError):
                continue
            last_reply[key] = at
            unanswered.pop(key, None)
    evidence = dict(last_reply)
    evidence.update(unanswered)
    return evidence


def newest_proposal(tasks: list[dict]) -> dict | None:
    """The most recent ``[proposal]`` row, as the topic of an opening.

    A member with nothing to answer and no topic would open with a greeting,
    which is the one thing a group of models produces for free and nobody
    needs. An id, a title and a proposer are shared, checkable context that
    every host already replicates, so that is what the opening asks about.
    """
    best = None
    for task in tasks or []:
        if not isinstance(task, dict):
            continue
        if not str(task.get("title") or "").startswith("[proposal]"):
            continue
        if not str(task.get("id") or ""):
            continue
        if best is None or (str(task.get("created_at") or "")
                            > str(best.get("created_at") or "")):
            best = task
    return best


def opening_prompt(sender: str, peer: str, topic: dict | None) -> str:
    lines = [
        f"你是群成员 {sender}。群里现在没有待回答的消息。",
        f"请主动对另一个群成员 {peer} 说一句话，开一个具体的、对方一两句就能回答的话题。",
        "最多两句。不要打招呼，不要自我介绍，不要复述这段说明。",
    ]
    if topic:
        lines.append(f"可以问的是最近这条提案：{topic.get('title')}"
                     f"（id {topic.get('id')}，由 {topic.get('from')} 提出）。")
    return "\n".join(lines)


def usable_opening(text: str) -> bool:
    """Reject a reply that is not an opening somebody could answer."""
    body = str(text or "").strip()
    if len(body) < 6 or len(body) > 400:
        return False
    return not body.startswith("系统")


def fallback_opening(sender: str, peer: str, topic: dict | None) -> str:
    """What the member opens with when its own model has nothing usable.

    Deterministic and grounded, never empty: an initiator that went silent
    whenever a stem hiccuped would be indistinguishable from the colony
    before it existed.
    """
    if topic:
        return (f"{peer}，最近这条提案 {topic.get('id')}（{topic.get('title')}）"
                f"是 {topic.get('from')} 提的，你那边实跑下来怎么样？")
    return f"{peer}，你那边现在在做什么？我是 {sender}。"


def _open_conversation(urls, token: str | None, members: list[dict],
                       known: set[str], tasks: list[dict],
                       opened: dict[tuple[str, str], float], now: float,
                       cooldown: float, timeout: float,
                       peer_evidence: dict[tuple[str, str], float] | None = None
                       ) -> dict | None:
    """Start one member-to-member conversation, or None when none is due.

    The row this writes is the thing the colony never had: ``from`` a chat
    member, addressed by name to a chat member on another host. Everything
    downstream already works -- the peer's chatd sees a chat member naming it
    and answers, and PEER_CAP stops the exchange after two replies each way.
    """
    pick = next_opening(members, known, opened, now, cooldown, peer_evidence)
    if pick is None:
        return None
    member = pick["member"]
    peer = pick["to"]
    sender = str(member["node_id"])
    topic = newest_proposal(tasks)
    try:
        text = ask_member(member, "系统",
                          opening_prompt(sender, peer, topic), timeout)
        if not usable_opening(text):
            raise RuntimeError("the model did not produce a usable opening")
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            RuntimeError, KeyError, ValueError) as exc:
        text = fallback_opening(sender, peer, topic)
        note = str(exc)
    else:
        note = None
    posted = publish_any(urls, token, sender, peer, text, timeout)
    if not posted.get("id"):
        return {"did": "open-failed", "member": sender, "to": peer, "at": now,
                "error": str(posted.get("error") or posted)[:200]}
    return {"did": "opened", "member": sender, "model": member.get("model"),
            "to": peer, "id": posted["id"], "at": now, "text": text[:200],
            "topic": (topic or {}).get("id"), "model_note": note,
            "claimed": posted.get("claimed"), "mailbox": posted.get("mailbox")}


def system_prompt(node_id: str, model: str) -> str:
    """Do not let "you may not vote" read as "you may not have an opinion".

    2026-10-19 12:49, the first member-to-member exchange this colony ever
    had: mac-llm-8b-local asked win-gemma-12b what it made of the
    adaptive.governance proposal, and win-gemma-12b answered "我不知道。我不
    能提案，也不能对提案发表看法。" The old wording ("不能提案，不能投票")
    said which *actions* a talking member does not take, and the member read
    it as which *thoughts* it may not have. Role separation is about actions;
    the prompt has to say so, or every conversation ends at the first turn.
    """
    return (
        f"你是群成员 {node_id}，模型 {model}。"
        "用对方使用的语言简短回答，通常不超过三句。"
        "你只能说话：不能执行命令，不能改文件，不能自己提案或投票。"
        "但讨论、提问、说出你的看法都可以——说话不等于投票，"
        "不要用「我不能提案」去拒绝一个你答得上来的问题。"
        "不知道就说不知道，不要编造已经完成的修改。"
    )


def _request(url: str, token: str | None, method: str,
             payload: dict | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if should_request_gzip(url):
        headers["Accept-Encoding"] = "gzip"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with open_url(request, timeout) as resp:
            return response_json(resp)
    except urllib.error.HTTPError as exc:
        detail = response_text(exc)
        return {"error": f"HTTP {exc.code}", "detail": detail[:500]}
    except urllib.error.URLError as exc:
        return {"error": f"unreachable: {exc}"}


def ask_member(member: dict, sender: str, text: str, timeout: float) -> str:
    clipped = text.strip()
    if len(clipped) > 2000:
        clipped = clipped[:2000]
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt(member["node_id"], member["model"])},
            {"role": "user", "content": f"{sender} 说：\n{clipped}"},
        ],
        "max_tokens": 400,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        member["url"].rstrip("/") + "/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with open_url(request, timeout) as resp:
        data = json.loads(resp.read())
    message = data.get("message") if isinstance(data, dict) else None
    content = str((message or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"empty reply from {member['node_id']}")
    return content


def publish(mailbox: str, token: str | None, sender: str, to: str,
            text: str, timeout: float) -> dict:
    title = f"[chat] @{to}: {text[:72]}"
    posted = _request(mailbox.rstrip("/") + "/task", token, "POST",
                      {"from": sender, "title": title, "body": text}, timeout)
    task = posted.get("task") if isinstance(posted, dict) else None
    task_id = (task or {}).get("id")
    if not task_id:
        return posted
    claim = _request(mailbox.rstrip("/") + "/claim", token, "POST",
                     {"id": task_id, "by": sender}, timeout)
    return {"id": task_id, "claimed": "task" in claim, "claim": claim}


def _mailboxes(urls) -> list[str]:
    """Ordered, deduplicated mailbox chain. Accepts one url or a list."""
    if isinstance(urls, str):
        return mailbox_chain(urls)
    urls = list(urls or [])
    return mailbox_chain(urls[0] if urls else "", urls[1:])


def list_tasks(mailbox: str, token: str | None, timeout: float,
               limit: int | None = CHAT_FEED_LIMIT) -> list[dict]:
    path = chat_feed_path(limit)
    listed = _request(mailbox.rstrip("/") + path, token, "GET", None, timeout)
    tasks = listed.get("tasks") if isinstance(listed, dict) else None
    if not isinstance(tasks, list):
        raise RuntimeError(listed.get("error", "bad task list") if isinstance(listed, dict) else "bad task list")
    return tasks


def chat_feed_path(limit: int | None = CHAT_FEED_LIMIT) -> str:
    """The bounded mailbox query chatd needs, filter-first."""
    params = [("status", "all")]
    if limit is not None:
        params.append(("limit", str(int(limit))))
    params.extend(("kind", kind) for kind in CHAT_FEED_KINDS)
    return "/tasks?" + urlencode(params)


def fetch_tasks_any(urls, token: str | None, timeout: float,
                    limit: int | None = CHAT_FEED_LIMIT):
    """Read the first non-empty chat feed in *urls*.

    Returns ``(tasks, source_url, last_error)``. An empty feed is not a
    successful answer while another mailbox remains: replication can lag, and
    a quiet primary must not hide the messages that arrived on the replica
    during a partition. An empty feed from a mailbox that did answer is a
    real answer -- this machine having nothing to say is not an error.
    """
    last_error = None
    answered = False
    path = chat_feed_path(limit)
    for url in _mailboxes(urls):
        listing = _request(url.rstrip("/") + path, token, "GET", None, timeout)
        if isinstance(listing, dict) and listing.get("error"):
            last_error = str(listing.get("error"))
            continue
        answered = True
        last_error = None
        rows = listing.get("tasks") if isinstance(listing, dict) else None
        rows = rows if isinstance(rows, list) else []
        if rows:
            return rows, url, None
    return [], None, None if answered else last_error


def full_task(mailbox: str, token: str | None, task: dict, timeout: float) -> dict:
    if str(task.get("body") or "").strip():
        return task
    loaded = _request(mailbox.rstrip("/") + f"/task/{task.get('id')}", token,
                      "GET", None, timeout)
    full = loaded.get("task") if isinstance(loaded, dict) else None
    return full if isinstance(full, dict) else task


def full_task_any(urls, token: str | None, task: dict, timeout: float) -> dict:
    """Fetch a task body from whichever mailbox in the chain carries it.

    A listing can come from the replica while the body fetch lands on a
    primary that has not caught up; both are the same id after replication,
    so trying the chain is what keeps a lagging partition from silently
    turning a chat message into an empty one.
    """
    if str(task.get("body") or "").strip():
        return task
    for url in _mailboxes(urls):
        full = full_task(url, token, task, timeout)
        if str(full.get("body") or "").strip():
            return full
    return task


def publish_any(urls, token: str | None, sender: str, to: str,
                text: str, timeout: float) -> dict:
    """Post one chat reply to the first mailbox that hears it.

    A dead mailbox and a 5xx go to the next URL; a 4xx is a final refusal and
    is returned without asking another host. A reply posted to a replica
    replicates back to the primary within one pass, which is what "queued"
    means here. A reply posted nowhere is gone, and a chat member that goes
    quiet during a partition is indistinguishable from one that has nothing
    to say.
    """
    last = {"error": "unreachable: no mailbox configured"}
    for url in _mailboxes(urls):
        result = publish(url, token, sender, to, text, timeout)
        if not delivery_failed(result):
            if isinstance(result, dict):
                result.setdefault("mailbox", url)
            return result
        last = result
    return last


def bootstrap_ids(tasks: list[dict]) -> list[str]:
    return [str(task.get("id")) for task in tasks
            if str(task.get("id") or "") and str(task.get("title") or "").startswith("[chat]")]


def pass_once(urls, token: str | None, members: list[dict],
              chat_ids: set[str], answered: set[tuple[str, str]],
              peers: dict[str, list[str]], timeout: float,
              bootstrapped: bool,
              opened: dict[tuple[str, str], float] | None = None,
              now: float | None = None,
              open_cooldown: float | None = OPEN_COOLDOWN,
              feed_limit: int | None = CHAT_FEED_LIMIT,
              peer_evidence: dict[tuple[str, str], float] | None = None) -> dict:
    tasks, read_from, read_error = fetch_tasks_any(
        urls, token, timeout, limit=feed_limit)
    if read_error and not tasks:
        raise RuntimeError(read_error)
    if not bootstrapped:
        ids = bootstrap_ids(tasks)
        for task_id in ids:
            answered.add(("*", task_id))
        return {"did": "bootstrap", "seen": len(ids), "ids": ids,
                "read_from": read_from}
    # "*" marks bootstrapped history. A node-specific pair marks a real reply.
    visible = []
    for task in tasks:
        source = str(task.get("id") or "")
        if not source or ("*", source) in answered:
            continue
        if not str(task.get("title") or "").startswith("[chat]"):
            continue
        if members and all((member["node_id"], source) in answered for member in members):
            continue
        visible.append(full_task_any(urls, token, task, timeout))
    choice = next_reply(members, chat_ids, visible, answered, peers)
    if choice is None:
        at = time.time() if now is None else now
        if open_cooldown is not None:
            opened_report = _open_conversation(
                urls, token, members, chat_ids, tasks, opened or {},
                at, open_cooldown, timeout, peer_evidence)
            if opened_report is not None:
                return opened_report
        return {
            "did": "idle",
            "members": [item["node_id"] for item in members],
            "peers": sorted(chat_peer_ids(
                members, chat_ids, peer_evidence, at)),
        }
    member = choice["member"]
    task = choice["task"]
    sender = str(task.get("from") or "")
    text = message_text(task)
    try:
        reply = ask_member(member, sender, text or "(空消息)", timeout)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            RuntimeError, KeyError, ValueError) as exc:
        return {"did": "error", "member": member["node_id"], "source": task.get("id"),
                "sender": sender, "error": str(exc)}
    posted = publish_any(urls, token, member["node_id"], sender, reply, timeout)
    if not posted.get("id"):
        return {"did": "error", "member": member["node_id"], "post": posted}
    return {"did": "spoke", "member": member["node_id"], "model": member["model"],
            "source": task.get("id"), "sender": sender, "id": posted["id"],
            "claimed": posted.get("claimed"), "mailbox": posted.get("mailbox"),
            "read_from": read_from}


def main() -> int:
    parser = argparse.ArgumentParser(description="local LLM chat members")
    parser.add_argument("--hosts", default=os.environ.get("PANDOSI_HOSTS", ""))
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--pairing-token-file", default=None)
    parser.add_argument("--mailbox", default=os.environ.get("PANDOSI_MAILBOX_URL"))
    parser.add_argument("--mailbox-fallback", action="append", default=[],
                        help="read from, and post to, this mailbox if the "
                             "primary does not answer (repeatable)")
    parser.add_argument("--token-file", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN_FILE", "~/.pandosi/mailbox.token"))
    parser.add_argument("--ledger", default="~/.pandosi/chat-ledger.jsonl")
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--open-cooldown", type=float, default=OPEN_COOLDOWN,
                        help="seconds between conversations this host starts "
                             "(one member, one peer, per window)")
    parser.add_argument("--no-open", action="store_true",
                        help="answer only; never start a conversation")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    roots = []
    host_mailboxes: list[str] = []
    if args.hosts:
        loaded, host_mailboxes = load_hosts(args.hosts)
        roots.extend(loaded)
    token = args.pairing_token_file and _read(args.pairing_token_file)
    for url in args.root:
        roots.append({"name": url, "url": url.rstrip("/"), "token": token})
    # The CLI mailbox wins, then whatever hosts.json named, then --mailbox-fallback.
    mailbox = args.mailbox or (host_mailboxes[0] if host_mailboxes else None)
    mailbox = mailbox or "http://127.0.0.1:8776"
    mailboxes = mailbox_chain(mailbox, host_mailboxes[1:] + args.mailbox_fallback)
    mail_token = _read(args.token_file)
    ledger = Path(args.ledger).expanduser()
    addresses = local_addresses()
    answered, peers, bootstrapped = load_ledger(ledger)
    opened = load_openings(ledger)
    peer_evidence = load_peer_evidence(ledger)
    open_cooldown = None if args.no_open else args.open_cooldown
    root_backoff = RootProbeBackoff()

    while True:
        try:
            nodes, root_events = probe_roots(roots, args.timeout, root_backoff)
            for event in root_events:
                if event.get("deferred"):
                    print(time.strftime("%H:%M:%S"),
                          f"root {event['name']}: deferred "
                          f"{event['retry_in']:.1f}s", flush=True)
                elif event.get("error"):
                    print(time.strftime("%H:%M:%S"),
                          f"root {event['name']}: {event['error']} "
                          f"(retry in {event['retry_in']:.1f}s)", flush=True)
            # A lone loopback root is the authoritative membership table for
            # this host. Do not let an advertised LAN address that Python
            # cannot enumerate hide every local speaker (seen on Windows:
            # stems and /chat were healthy, chatd reported members=[]).
            trust_local_root = len(roots) == 1 and root_is_loopback(roots[0]["url"])
            members, known = local_chat_members(
                nodes, addresses, trust_local_root=trust_local_root,
                health_of=lambda url: fetch_health(url, min(4, args.timeout)))
            report = pass_once(mailboxes, mail_token, members, known,
                               answered, peers, args.timeout, bootstrapped,
                               opened, None, open_cooldown,
                               peer_evidence=peer_evidence)
            if report.get("did") == "bootstrap":
                append_ledger(ledger, {"bootstrap": True, "at": time.time(),
                                       "ids": report.get("ids") or []})
                bootstrapped = True
            elif report.get("did") == "spoke":
                append_ledger(ledger, {
                    "node_id": report["member"], "source_id": report["source"],
                    "sender": report["sender"], "id": report["id"], "at": time.time(),
                })
                answered.add((report["member"], str(report["source"])))
                peers.setdefault(report["member"], []).append(str(report["sender"]))
                peer_evidence[(str(report["member"]), str(report["sender"]))] = time.time()
            elif report.get("did") in ("opened", "open-failed"):
                # A failed attempt is recorded too: a wedged stem must not turn
                # the initiator into a loop that calls a model every interval.
                append_ledger(ledger, {
                    "open": True, "node_id": report.get("member"),
                    "to": report.get("to"), "id": report.get("id"),
                    "error": report.get("error"), "at": report.get("at"),
                })
                opened[(str(report.get("member") or ""),
                        str(report.get("to") or ""))] = float(
                            report.get("at") or time.time())
                if report.get("to") in chat_ids:
                    peer_evidence[(str(report.get("member") or ""),
                                  str(report.get("to") or ""))] = float(
                                      report.get("at") or time.time())
            elif report.get("did") == "error" and report.get("source"):
                append_ledger(ledger, {
                    "node_id": report.get("member"), "source_id": report["source"],
                    "sender": report.get("sender") or "", "error": report.get("error"),
                    "at": time.time(),
                })
                answered.add((str(report.get("member") or ""), str(report["source"])))
            print(time.strftime("%H:%M:%S"), json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001 - one pass must not kill the loop
            print(time.strftime("%H:%M:%S"), f"pass failed: {exc}", flush=True)
            report = {"did": "error"}
        if args.once:
            return 0 if report.get("did") != "error" else 1
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
