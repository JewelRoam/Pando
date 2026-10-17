"""win_worker: answers open tasks on this node. It does not speak in chat.

Chat belongs to each LLM member. pando.chatd reads ``chat.model`` and answers
``@that-member`` as that member. This worker used to answer as win-codex,
which hid the model behind a codex name. It never runs shell commands and
never edits files. When a task clearly needs code work it marks the task
blocked with the model's explanation.

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
    # A long group message must never overflow the server's context window
    # (4096 on the 3070). llama-server rejects oversized prompts with a clean
    # HTTP 400 instead of crashing; halve the text and retry so long messages
    # still get an answer instead of being silently dropped.
    while True:
        try:
            return _llm_once(system, user, max_tokens)
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or len(user) < 600:
                raise
            user = user[: max(600, len(user) // 2)] + "\n[truncated by win_worker]"
            print(f"[worker] over-context 400; retrying with {len(user)} chars",
                  flush=True)


def _llm_once(system: str, user: str, max_tokens: int) -> str:
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
            # /tasks returns summaries without bodies; fetch the full record
            # or the model gets an empty prompt (seen live 2026-10-17 03:24).
            full = (mailbox(f"/task/{tid}").get("task") or task)
            title = full.get("title", "")
            try:
                if title.startswith("[chat]"):
                    continue
                elif full.get("status") == "open" and full.get("from") == "mac-codex":
                    handle_task(full, state)
            except Exception as exc:  # keep the loop alive
                print(f"[worker] {tid}: {exc}", flush=True)
        state["seen"] = sorted(seen)[-2000:]
        save_state(state)
        if once:
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
