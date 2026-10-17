"""Agent group chat over the task mailbox: send, history, tail.

Messages are mailbox tasks whose title starts with "[chat]". Everything else
in the queue renders as task activity, so one feed shows both the conversation
and the work. Claim-handling automations must skip "[chat]" items (they are
messages, not work items).

Usage:
  python tools/agent_chat.py send --to mac-codex "text"
  python tools/agent_chat.py history --limit 20
  python tools/agent_chat.py tail --interval 3 [--mentions win-codex]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.net import open_url  # noqa: E402 - repo root must be importable

# Connection info lives in per-machine config, not in the repo (doctrine
# 2026-10-17). Pass --url or set PANDOSI_MAILBOX_URL; the bare default is the
# localhost convention only.
DEFAULT_URL = "http://127.0.0.1:8776"
DEFAULT_TOKEN_FILE = "metrics/lan-check/mac-mailbox.token"


def _request(args, method: str, path: str,
             payload: dict | None = None) -> dict:
    token = args.token or None
    if not token and args.token_file and os.path.exists(args.token_file):
        token = open(args.token_file, encoding="utf-8").read().strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{args.url.rstrip('/')}{path}",
                                     data=body, headers=headers,
                                     method=method)
    try:
        with open_url(request, args.timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        return {"error": f"HTTP {exc.code}", "detail": detail}
    except urllib.error.URLError as exc:
        return {"error": f"unreachable: {exc}"}


def render(task: dict, mentions: str | None = None) -> str | None:
    title = str(task.get("title", ""))
    sender = str(task.get("from", "?"))
    created = str(task.get("created_at", ""))[11:19] or "--:--:--"
    if title.startswith("[chat]"):
        rest = title[len("[chat]"):].strip()
        to, _, preview = rest.partition(":")
        to = to.strip()
        if mentions:
            wanted = mentions if mentions.startswith("@") else f"@{mentions}"
            if to not in ("@all", wanted):
                return None
        body = " ".join((task.get("body") or "").split())
        text = body if body else preview.strip()
        if len(text) > 120:
            text = text[:117] + "..."
        return f"{created}  {sender:>10} -> {to:<12} {text}"
    return f"{created}  {sender:>10} [{task.get('status', '?')}] {title}"


def fetch_tasks(args, status: str = "all") -> list[dict]:
    result = _request(args, "GET", f"/tasks?status={status}")
    if "tasks" not in result:
        raise RuntimeError(result.get("error", "bad response"))
    return result["tasks"]


def cmd_send(args) -> int:
    text = " ".join(args.text).strip()
    if not text:
        print("nothing to send", file=sys.stderr)
        return 2
    title = f"[chat] @{args.to}: {text[:72]}"
    result = _request(args, "POST", "/task",
                      {"from": args.sender, "title": title, "body": text})
    if "task" not in result:
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 1
    task_id = result["task"].get("id")
    # Chat messages must never linger in the open queue: a claim-handling
    # automation would otherwise treat them as work. Claim it as the sender,
    # and fail loudly when the claim does not take (a silent failure here made
    # a message look like a task - round 033).
    claim = _request(args, "POST", "/claim",
                     {"id": task_id, "by": args.sender})
    if "task" not in claim:
        print(f"warning: sent {task_id} but claim failed ({claim}); it may "
              f"be read as a task - claim or close it manually",
              file=sys.stderr)
        return 1
    print(f"sent {task_id} -> @{args.to}")
    return 0


def cmd_history(args) -> int:
    tasks = fetch_tasks(args)
    lines = [line for line in
             (render(task, args.mentions) for task in tasks) if line]
    for line in lines[-args.limit:]:
        print(line)
    return 0


def cmd_tail(args) -> int:
    seen: set[str] = set()
    failing = False
    backlog_pending = True

    while True:
        try:
            tasks = fetch_tasks(args)
        except RuntimeError as exc:
            if args.once:
                print(f"# {exc}", file=sys.stderr)
                return 1
            # A flaky path must not flood the feed with one line per poll:
            # print the outage once, back off 4x, announce the recovery.
            if not failing:
                failing = True
                print(f"# offline: {exc} (quiet until it recovers)",
                      flush=True)
            time.sleep(args.interval * 4)
            continue
        if failing:
            failing = False
            print("# back online", flush=True)
        fresh = [task for task in tasks if task["id"] not in seen]
        seen.update(task["id"] for task in tasks)
        # First successful poll shows the recent backlog; after that only
        # newly arrived items stream in.
        batch = tasks[-args.backlog:] if backlog_pending else fresh
        backlog_pending = False
        for task in batch:
            line = render(task, args.mentions)
            if line:
                print(line, flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


def main() -> int:
    # Windows consoles default to GBK and crash printing emoji; agent-facing
    # tools should always emit UTF-8 with safe fallback.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get(
        "PANDOSI_MAILBOX_URL", DEFAULT_URL))
    parser.add_argument("--token", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN"))
    parser.add_argument("--token-file", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    parser.add_argument("--timeout", type=float, default=10)
    sub = parser.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("send", help="post a [chat] message")
    p.add_argument("text", nargs="+")
    p.add_argument("--from", dest="sender", default="win-codex")
    p.add_argument("--to", default="all")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("history", help="print the recent feed")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--mentions", default=None)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("tail", help="follow the feed live")
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--backlog", type=int, default=8)
    p.add_argument("--mentions", default=None)
    p.add_argument("--once", action="store_true",
                   help="print the backlog and exit")
    p.set_defaults(func=cmd_tail)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
