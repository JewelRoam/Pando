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

from .chat import message_text, next_reply, normalize_chat
from .net import local_addresses, open_url


def _read(path: str | None) -> str | None:
    if not path:
        return None
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def load_hosts(path: str) -> tuple[list[dict], str | None]:
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
    mailboxes = data.get("mailboxes") or []
    mailbox = str(mailboxes[0]).rstrip("/") if mailboxes else None
    return roots, mailbox


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


def list_tasks(mailbox: str, token: str | None, timeout: float) -> list[dict]:
    listed = _request(mailbox.rstrip("/") + "/tasks?status=all", token,
                      "GET", None, timeout)
    tasks = listed.get("tasks") if isinstance(listed, dict) else None
    if not isinstance(tasks, list):
        raise RuntimeError(listed.get("error", "bad task list") if isinstance(listed, dict) else "bad task list")
    return tasks


def full_task(mailbox: str, token: str | None, task: dict, timeout: float) -> dict:
    if str(task.get("body") or "").strip():
        return task
    loaded = _request(mailbox.rstrip("/") + f"/task/{task.get('id')}", token,
                      "GET", None, timeout)
    full = loaded.get("task") if isinstance(loaded, dict) else None
    return full if isinstance(full, dict) else task


def bootstrap_ids(tasks: list[dict]) -> list[str]:
    return [str(task.get("id")) for task in tasks
            if str(task.get("id") or "") and str(task.get("title") or "").startswith("[chat]")]


def pass_once(mailbox: str, token: str | None, members: list[dict],
              chat_ids: set[str], answered: set[tuple[str, str]],
              peers: dict[str, list[str]], timeout: float,
              bootstrapped: bool) -> dict:
    tasks = list_tasks(mailbox, token, timeout)
    if not bootstrapped:
        ids = bootstrap_ids(tasks)
        for source in ids:
            answered.add(("*", source))
        return {"did": "bootstrap", "seen": len(ids), "ids": ids}
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
        visible.append(full_task(mailbox, token, task, timeout))
    choice = next_reply(members, chat_ids, visible, answered, peers)
    if choice is None:
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
    posted = publish(mailbox, token, member["node_id"], sender, reply, timeout)
    if not posted.get("id"):
        return {"did": "error", "member": member["node_id"], "post": posted}
    return {"did": "spoke", "member": member["node_id"], "model": member["model"],
            "source": task.get("id"), "sender": sender, "id": posted["id"],
            "claimed": posted.get("claimed")}


def main() -> int:
    parser = argparse.ArgumentParser(description="local LLM chat members")
    parser.add_argument("--hosts", default=os.environ.get("PANDOSI_HOSTS", ""))
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--pairing-token-file", default=None)
    parser.add_argument("--mailbox", default=os.environ.get("PANDOSI_MAILBOX_URL"))
    parser.add_argument("--token-file", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN_FILE", "~/.pandosi/mailbox.token"))
    parser.add_argument("--ledger", default="~/.pandosi/chat-ledger.jsonl")
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    roots = []
    mailbox = args.mailbox
    if args.hosts:
        loaded, host_mailbox = load_hosts(args.hosts)
        roots.extend(loaded)
        mailbox = mailbox or host_mailbox
    token = args.pairing_token_file and _read(args.pairing_token_file)
    for url in args.root:
        roots.append({"name": url, "url": url.rstrip("/"), "token": token})
    mailbox = mailbox or "http://127.0.0.1:8776"
    mail_token = _read(args.token_file)
    ledger = Path(args.ledger).expanduser()
    addresses = local_addresses()
    answered, peers, bootstrapped = load_ledger(ledger)

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
            report = pass_once(mailbox, mail_token, members, known, answered,
                               peers, args.timeout, bootstrapped)
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
