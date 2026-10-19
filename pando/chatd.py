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

from .chat import (OPEN_COOLDOWN, message_text, next_opening, next_reply,
                   normalize_chat)
from .common import PROPOSING_ROLE
from .net import local_addresses, open_url
from .proposal_mail import delivery_failed, mailbox_chain


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


def local_chat_members(nodes: dict, addresses: set[str],
                       health_of=None) -> tuple[list[dict], set[str]]:
    """Local speakers, plus every node id that carries ``chat.model``.

    Only a proposer speaks: a decision model answers proposals and is not
    put in the conversation. ``role`` says which is which; a member with no
    role but a chat descriptor is read as a proposer, so a root that
    predates the field keeps working.

    ``health_of(url) -> dict`` is only called for a local stem whose node
    record does not yet have the field.
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
        local = bool(url) and host in addresses
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
                       cooldown: float, timeout: float) -> dict | None:
    """Start one member-to-member conversation, or None when none is due.

    The row this writes is the thing the colony never had: ``from`` a chat
    member, addressed by name to a chat member on another host. Everything
    downstream already works -- the peer's chatd sees a chat member naming it
    and answers, and PEER_CAP stops the exchange after two replies each way.
    """
    pick = next_opening(members, known, opened, now, cooldown)
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
    return (
        f"你是群成员 {node_id}，模型 {model}。"
        "用对方使用的语言简短回答，通常不超过三句。"
        "你只能说话：不能执行命令，不能改文件，不能提案，不能投票。"
        "不知道就说不知道，不要编造已经完成的修改。"
    )


def _request(url: str, token: str | None, method: str,
             payload: dict | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with open_url(request, timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
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


def list_tasks(mailbox: str, token: str | None, timeout: float) -> list[dict]:
    listed = _request(mailbox.rstrip("/") + "/tasks?status=all", token,
                      "GET", None, timeout)
    tasks = listed.get("tasks") if isinstance(listed, dict) else None
    if not isinstance(tasks, list):
        raise RuntimeError(listed.get("error", "bad task list") if isinstance(listed, dict) else "bad task list")
    return tasks


def fetch_tasks_any(urls, token: str | None, timeout: float):
    """Read the first non-empty chat feed in *urls*.

    Returns ``(tasks, source_url, last_error)``. An empty feed is not a
    successful answer while another mailbox remains: replication can lag, and
    a quiet primary must not hide the messages that arrived on the replica
    during a partition. An empty feed from a mailbox that did answer is a
    real answer -- this machine having nothing to say is not an error.
    """
    last_error = None
    answered = False
    for url in _mailboxes(urls):
        listing = _request(url.rstrip("/") + "/tasks?status=all", token,
                           "GET", None, timeout)
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
              open_cooldown: float | None = OPEN_COOLDOWN) -> dict:
    tasks, read_from, read_error = fetch_tasks_any(urls, token, timeout)
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
        if open_cooldown is not None:
            opened_report = _open_conversation(
                urls, token, members, chat_ids, tasks, opened or {},
                time.time() if now is None else now, open_cooldown, timeout)
            if opened_report is not None:
                return opened_report
        return {"did": "idle", "members": [item["node_id"] for item in members]}
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
    open_cooldown = None if args.no_open else args.open_cooldown

    while True:
        try:
            nodes: dict = {}
            for root in roots:
                try:
                    found = fetch_nodes(root["url"], root.get("token"), min(8, args.timeout))
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                    print(time.strftime("%H:%M:%S"), f"root {root['name']}: {exc}", flush=True)
                    continue
                nodes.update(found)
            members, known = local_chat_members(
                nodes, addresses,
                health_of=lambda url: fetch_health(url, min(4, args.timeout)))
            report = pass_once(mailboxes, mail_token, members, known,
                               answered, peers, args.timeout, bootstrapped,
                               opened, None, open_cooldown)
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
