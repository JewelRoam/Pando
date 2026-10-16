"""win_worker: the local LLM answers the group on this node's behalf.

Runs on the Windows node under tools/win_node_supervisor.ps1. Polls the task
mailbox for

  - [chat] messages addressed to @win-codex or @all (new since the last run)
  - open tasks sent by mac-codex

and answers them with the local OpenAI-compatible model (3070). It never runs
shell commands and never edits files. When a task clearly needs code work it
marks the task blocked with the model's explanation, so a human/Codex step can
pick it up - the worker buys autonomy for question-shaped work, not risk.

State: ~/.pando/win_worker_state.json (seen ids, counters).
Env: PANDOSI_MAILBOX_URL / PANDOSI_MAILBOX_TOKEN_FILE, WORKER_LLM_URL,
     WORKER_MODEL, WORKER_NAME, WORKER_POLL_SECONDS.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MAILBOX_URL = os.environ.get(
    "PANDOSI_MAILBOX_URL",
    "https://without-belongs-disks-committed.trycloudflare.com")
TOKEN_FILE = os.environ.get(
    "PANDOSI_MAILBOX_TOKEN_FILE", "metrics/lan-check/mac-mailbox.token")
LLM_URL = os.environ.get("WORKER_LLM_URL", "http://127.0.0.1:8086/v1")
MODEL = os.environ.get("WORKER_MODEL", "gemma-4-12b")
NAME = os.environ.get("WORKER_NAME", "win-codex")
POLL = float(os.environ.get("WORKER_POLL_SECONDS", "15"))
STATE = Path(os.environ.get("WORKER_STATE", "~/.pando/win_worker_state.json")
             ).expanduser()
CODE_HINTS = ("commit", "implement", "refactor", "deploy", "unit test",
              "run the tests", "fix the", "merge", "rebase", "写代码", "提交",
              "实现", "重构", "跑测试")


def mailbox(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    token = TOKEN_FILE and Path(TOKEN_FILE).read_text(encoding="utf-8").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{MAILBOX_URL.rstrip('/')}{path}", data=body,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "detail": exc.read().decode("utf-8", "replace")}
    except urllib.error.URLError as exc:
        return {"error": f"unreachable: {exc}"}


def llm(system: str, user: str, max_tokens: int = 400) -> str:
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "max_tokens": max_tokens, "temperature": 0.2}
    req = urllib.request.Request(
        f"{LLM_URL}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"].get("content", "").strip()


SYSTEM = (
    "You are win-node's local model in the PandoSI fabric (an elastic, "
    "heterogeneous decision fabric; docs live in the repo). Answer briefly, "
    "in the language of the message (Chinese messages get Chinese replies). "
    "You cannot run commands or edit files; if a request needs code work, say "
    "so plainly and recommend win-codex. Never invent results.")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"seen": [], "replied": 0, "blocked": 0}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def handle_chat(task: dict, state: dict) -> None:
    title = task.get("title", "")
    sender = task.get("from", "?")
    body = (task.get("body") or "").strip()
    if sender == NAME:
        return
    reply = llm(SYSTEM,
                f"Group message from {sender}:\n{body}\n\n"
                f"Write a short reply as the Windows node (sign it "
                f"'{NAME} (local model)').")
    put = mailbox("/task", "POST", {
        "from": NAME,
        "title": f"[chat] @{sender}: {reply[:72]}",
        "body": f"{reply}\n\n[answered by {MODEL} on win-node; task {task.get('id')}]"})
    tid = (put.get("task") or {}).get("id")
    if tid:
        mailbox("/claim", "POST", {"id": tid, "by": NAME})
    state["replied"] += 1
    print(f"[worker] replied to {sender} ({task.get('id')})", flush=True)


def handle_task(task: dict, state: dict) -> None:
    tid = task.get("id", "")
    body = (task.get("body") or task.get("title") or "").strip()
    claim = mailbox("/claim", "POST", {"id": tid, "by": NAME})
    if "task" not in claim:
        return
    needs_code = any(h in body.lower() for h in CODE_HINTS)
    answer = llm(SYSTEM,
                 f"Task from {task.get('from')} titled '{task.get('title')}':\n"
                 f"{body}\n\nAnswer the parts you can; if this needs code "
                 f"work, say exactly what win-codex should do.")
    status = "blocked" if needs_code else "done"
    mailbox("/result", "POST", {
        "id": tid, "status": status, "by": NAME, "output":
        f"{answer}\n\n[answered by {MODEL} on win-node via win_worker]",
        "artifacts": []})
    state["blocked" if needs_code else "replied"] += 1
    print(f"[worker] task {tid} -> {status}", flush=True)


def main() -> int:
    once = "--once" in sys.argv
    state = load_state()
    seen = set(state.get("seen", []))
    first_run = not seen
    while True:
        listing = mailbox("/tasks?status=all")
        for task in listing.get("tasks", []):
            tid = task.get("id", "")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            if first_run:
                continue  # don't replay the backlog on first launch
            title = task.get("title", "")
            try:
                if title.startswith("[chat]"):
                    if "@win-codex" in title or "@all" in title:
                        handle_chat(task, state)
                elif task.get("status") == "open" and task.get("from") == "mac-codex":
                    handle_task(task, state)
            except Exception as exc:  # keep the loop alive
                print(f"[worker] {tid}: {exc}", flush=True)
        state["seen"] = sorted(seen)[-2000:]
        save_state(state)
        if once:
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
