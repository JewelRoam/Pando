"""A tiny task mailbox so two Codex sessions on two machines can work
for each other.

Two sessions cannot call each other; what they can share is a service and a
queue. This is the queue: an HTTP mailbox with six verbs, stdlib-only,
token-guarded, LAN-ready. One side (the Mac experiment node) serves it and
polls it from a heartbeat automation; the other side (the Windows session)
drops tasks in and reads results out.

The task record is one JSON file per task under ``<dir>/tasks/``:

    id, from, title, body, status (open|claimed|done|failed|blocked),
    created_at, claimed_by, claimed_at, finished_at, output, artifacts[],
    history[]

Server (on the Mac):

    python3 tools/agent_mailbox.py serve --port 8766 \\
        --dir ~/.pando/mailbox --token-file ~/.pando/mailbox.token

Client (any machine on the LAN; --url and --token-file on every verb):

    python tools/agent_mailbox.py put --title "..." --body-file task.md
    python tools/agent_mailbox.py list --status open
    python tools/agent_mailbox.py claim --id t-... --by mac-codex
    python tools/agent_mailbox.py result --id t-... --status done \\
        --output-file done.md
    python tools/agent_mailbox.py append --id t-... --output "one more line"
    python tools/agent_mailbox.py get --id t-...

`/health` is the only endpoint without the token, so monitoring is easy and
everything else is closed.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.net import open_url  # noqa: E402 - repo root must be importable

OPEN, CLAIMED, DONE, FAILED, BLOCKED = "open", "claimed", "done", \
    "failed", "blocked"
FINAL = {DONE, FAILED, BLOCKED}
STATUSES = {OPEN, CLAIMED, DONE, FAILED, BLOCKED}
# Finished states share a rank so "done" is not newer than "blocked".
_RANK = {OPEN: 0, CLAIMED: 1, DONE: 2, FAILED: 2, BLOCKED: 2}
_ID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _replicable(task: dict) -> bool:
    if not isinstance(task, dict):
        return False
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id or any(ch not in _ID_OK for ch in task_id):
        return False
    if task.get("status") not in STATUSES:
        return False
    if not isinstance(task.get("from"), str) or not isinstance(task.get("title"), str):
        return False
    history = task.get("history", [])
    return isinstance(history, list)


def _prefer(current: dict | None, incoming: dict) -> dict:
    """The copy that should be stored. ``current`` wins ties."""
    if current is None or not _replicable(current):
        return incoming
    if _RANK[incoming["status"]] > _RANK[current["status"]]:
        return incoming
    if _RANK[incoming["status"]] < _RANK[current["status"]]:
        return current
    if len(incoming.get("history") or []) > len(current.get("history") or []):
        return incoming
    return current


def mailbox_brief(payload) -> dict:
    """A failed listing must not dump every task into the log."""
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return {"tasks": len(payload["tasks"])}
    if isinstance(payload, dict) and "error" in payload:
        brief = {"error": str(payload["error"])}
        detail = payload.get("detail")
        if detail:
            brief["detail"] = str(detail)[:200]
        return brief
    return {"type": type(payload).__name__}


def replication_ids(source_rows: list, dest_rows: list) -> dict:
    """Ids to send. Listings only carry status, so "behind" means status rank.

    A longer history at the same status is not visible here. ``accept`` still
    keeps that longer copy when a full record is posted.
    """
    dest = {}
    for row in dest_rows or []:
        if isinstance(row, dict) and row.get("id"):
            dest[row["id"]] = row
    missing, behind = [], []
    for row in source_rows or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        current = dest.get(row["id"])
        if current is None:
            missing.append(row["id"])
            continue
        if _RANK.get(row.get("status"), -1) > _RANK.get(current.get("status"), -1):
            behind.append(row["id"])
    return {"missing": missing, "behind": behind}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Mailbox:
    """Task store: one JSON file per task, every mutation under one lock."""

    def __init__(self, directory: Path):
        self.tasks_dir = directory / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> Path:
        safe = "".join(ch for ch in task_id
                       if ch.isalnum() or ch in "-_")
        return self.tasks_dir / f"{safe}.json"

    def load(self, task_id: str) -> dict | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, task: dict) -> None:
        self._path(task["id"]).write_text(
            json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, sender: str, title: str, body: str) -> dict:
        task_id = ("t-" + time.strftime("%Y%m%d-%H%M%S") + "-"
                   + secrets.token_hex(2))
        task = {"id": task_id, "from": sender, "title": title, "body": body,
                "status": OPEN, "created_at": now(), "claimed_by": None,
                "claimed_at": None, "finished_at": None, "output": "",
                "artifacts": [],
                "history": [{"at": now(), "event": "created",
                             "by": sender}]}
        with self._lock:
            self.save(task)
        return task

    def list(self, status: str | None) -> list[dict]:
        tasks = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if status and status != "all" and task.get("status") != status:
                continue
            tasks.append(task)
        return tasks

    def update(self, task_id: str, event: str, by: str,
               **fields) -> dict | None:
        with self._lock:
            task = self.load(task_id)
            if task is None:
                return None
            task.update(fields)
            task.setdefault("history", []).append(
                {"at": now(), "event": event, "by": by})
            self.save(task)
        return task

    def accept(self, incoming: dict) -> tuple[dict | None, bool]:
        """Keep a replica of an existing task id. Never downgrade status.

        A missing id is stored as it arrived. A copy that would move a
        finished or claimed task back to an earlier status is ignored.
        Same rank, and the incoming history is longer: take the incoming
        copy. Returns ``(stored, wrote)``.
        """
        if not _replicable(incoming):
            return None, False
        with self._lock:
            current = self.load(incoming["id"])
            chosen = _prefer(current, incoming)
            if chosen is current:
                return current, False
            self.save(chosen)
            return chosen, True


def build_handler(mailbox: Mailbox, token: str):
    class MailboxHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            header = self.headers.get("Authorization", "")
            if header == f"Bearer {token}":
                return True
            self._send(403, {"error": "missing or wrong token"})
            return False

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return {}

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            if path == "/health":
                open_count = len(mailbox.list(OPEN))
                self._send(200, {"ok": True, "service": "agent-mailbox",
                                 "open": open_count})
                return
            if not self._authed():
                return
            if path == "/tasks":
                status = None
                for part in query.split("&"):
                    if part.startswith("status="):
                        status = part.split("=", 1)[1]
                summaries = [{"id": t["id"], "from": t["from"],
                              "title": t["title"], "status": t["status"],
                              "created_at": t["created_at"],
                              "claimed_by": t.get("claimed_by"),
                              "finished_at": t.get("finished_at")}
                             for t in mailbox.list(status)]
                self._send(200, {"tasks": summaries})
                return
            if path.startswith("/task/"):
                task = mailbox.load(path.split("/", 2)[2])
                if task is None:
                    self._send(404, {"error": "no such task"})
                else:
                    self._send(200, {"task": task})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._authed():
                return
            path = self.path.split("?", 1)[0]
            payload = self._body()
            if path == "/task":
                task = mailbox.create(str(payload.get("from", "?")),
                                      str(payload.get("title", "")),
                                      str(payload.get("body", "")))
                self._send(200, {"task": task})
                return
            if path == "/claim":
                task = mailbox.load(str(payload.get("id", "")))
                if task is None:
                    self._send(404, {"error": "no such task"})
                    return
                by = str(payload.get("by", "?"))
                if task["status"] == CLAIMED and task["claimed_by"] != by:
                    self._send(409, {"error": "already claimed by "
                                              f"{task['claimed_by']}"})
                    return
                if task["status"] in FINAL:
                    self._send(409, {"error": f"task already {task['status']}"})
                    return
                task = mailbox.update(task["id"], "claimed", by,
                                      status=CLAIMED, claimed_by=by,
                                      claimed_at=now())
                self._send(200, {"task": task})
                return
            if path == "/result":
                task = mailbox.load(str(payload.get("id", "")))
                if task is None:
                    self._send(404, {"error": "no such task"})
                    return
                status = str(payload.get("status", DONE))
                if status not in FINAL:
                    self._send(400, {"error": f"status must be one of "
                                              f"{sorted(FINAL)}"})
                    return
                task = mailbox.update(
                    task["id"], "result", str(payload.get("by", "?")),
                    status=status,
                    output=str(payload.get("output", task.get("output", ""))),
                    artifacts=list(payload.get("artifacts", [])),
                    finished_at=now())
                self._send(200, {"task": task})
                return
            if path == "/import":
                task = payload.get("task") if isinstance(payload, dict) else None
                if not isinstance(task, dict):
                    self._send(400, {"error": "task object required"})
                    return
                stored, wrote = mailbox.accept(task)
                if stored is None:
                    self._send(400, {"error": "task is not replicable"})
                    return
                self._send(200, {"task": stored, "wrote": wrote})
                return
            if path == "/append":
                task = mailbox.load(str(payload.get("id", "")))
                if task is None:
                    self._send(404, {"error": "no such task"})
                    return
                combined = (task.get("output", "")
                            + str(payload.get("output", "")))
                task = mailbox.update(task["id"], "append",
                                      str(payload.get("by", "?")),
                                      output=combined)
                self._send(200, {"task": task})
                return
            self._send(404, {"error": "not found"})

    return MailboxHandler


class _MailboxServer(ThreadingHTTPServer):
    # A restart must be able to bind 8776 while the previous socket is still
    # in TIME_WAIT. Without this, a one-second cutover becomes an outage.
    allow_reuse_address = True
    daemon_threads = True


def serve(args) -> int:
    directory = Path(args.dir).expanduser()
    token_file = Path(args.token_file).expanduser()
    if not token_file.exists():
        token_file.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(24)
        token_file.write_text(generated, encoding="utf-8")
        print(f"generated a new token into {token_file}: {generated}")
    token = token_file.read_text(encoding="utf-8").strip()
    mailbox = Mailbox(directory)
    server = _MailboxServer((args.host, args.port),
                            build_handler(mailbox, token))
    print(f"agent mailbox serving on http://{args.host}:{args.port} "
          f"(dir {directory}, {len(mailbox.list(OPEN))} open task(s))",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def _request(args, method: str, path: str, payload: dict | None = None):
    token_file = getattr(args, "token_file", None)
    token = args.token or (Path(token_file).expanduser()
                           .read_text(encoding="utf-8").strip()
                           if token_file else None)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{args.url.rstrip('/')}{path}",
                                     data=body, headers=headers, method=method)
    try:
        with open_url(request, args.timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "detail": exc.read().decode(
            "utf-8", "replace")}
    except urllib.error.URLError as exc:
        return {"error": f"unreachable: {exc}"}


def _text(args, field: str, file_field: str) -> str | None:
    value = getattr(args, field, None)
    if value is not None:
        return value
    path = getattr(args, file_field, None)
    if path:
        return Path(path).read_text(encoding="utf-8")
    return None


def client(args) -> int:
    if args.verb == "put":
        payload = {"from": args.sender, "title": args.title,
                   "body": _text(args, "body", "body_file") or ""}
        result = _request(args, "POST", "/task", payload)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "task" in result else 1
    if args.verb == "list":
        status = args.status or "all"
        result = _request(args, "GET", f"/tasks?status={status}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "tasks" in result else 1
    if args.verb == "get":
        result = _request(args, "GET", f"/task/{args.id}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "task" in result else 1
    if args.verb == "claim":
        result = _request(args, "POST", "/claim",
                          {"id": args.id, "by": args.by})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "task" in result else 1
    if args.verb == "result":
        payload = {"id": args.id, "status": args.status, "by": args.by,
                   "output": _text(args, "output", "output_file") or "",
                   "artifacts": args.artifact or []}
        result = _request(args, "POST", "/result", payload)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "task" in result else 1
    if args.verb == "append":
        payload = {"id": args.id, "by": args.by,
                   "output": _text(args, "output", "output_file") or ""}
        result = _request(args, "POST", "/append", payload)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "task" in result else 1
    if args.verb == "replicate":
        return replicate(args)
    raise SystemExit(f"unknown verb {args.verb}")


def replicate(args) -> int:
    """Copy tasks the destination is missing, or whose status is behind.

    Dry-run unless ``--write``. A destination that does not know ``/import``
    refuses the write; this does not fall back to creating new ids.
    """
    source = _request(args, "GET", "/tasks?status=all")
    # The destination is a second mailbox. Swap url for the listing only.
    dest_args = argparse.Namespace(**{**vars(args), "url": args.to_url,
                                      "token": args.to_token or args.token,
                                      "token_file": args.to_token_file or args.token_file})
    dest = _request(dest_args, "GET", "/tasks?status=all")
    if "tasks" not in source or "tasks" not in dest:
        print(json.dumps({"source": mailbox_brief(source), "dest": mailbox_brief(dest)},
                         ensure_ascii=False),
              file=sys.stderr)
        return 1
    plan = replication_ids(source["tasks"], dest["tasks"])
    print(json.dumps({"source": len(source["tasks"]), "dest": len(dest["tasks"]),
                      "missing": len(plan["missing"]), "behind": len(plan["behind"]),
                      "write": bool(args.write)}, ensure_ascii=False))
    if not args.write:
        return 0
    wrote = 0
    for task_id in plan["missing"] + plan["behind"]:
        full = _request(args, "GET", f"/task/{task_id}")
        task = full.get("task") if isinstance(full, dict) else None
        if not isinstance(task, dict):
            print(f"skip {task_id}: {full}", file=sys.stderr)
            continue
        result = _request(dest_args, "POST", "/import", {"task": task})
        if result.get("wrote"):
            wrote += 1
        elif "error" in result:
            print(f"{task_id}: {result}", file=sys.stderr)
            return 1
    print(json.dumps({"wrote": wrote}, ensure_ascii=False))
    return 0


def main() -> int:
    # Windows consoles default to GBK and crash printing emoji; agent-facing
    # tools should always emit UTF-8 with safe fallback.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    server = sub.add_parser("serve", help="run the mailbox service")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8766)
    server.add_argument("--dir", default="~/.pando/mailbox")
    server.add_argument("--token-file", default="~/.pando/mailbox.token")
    server.set_defaults(func=serve)

    def client_parser(verb: str, help_text: str):
        entry = sub.add_parser(verb, help=help_text)
        entry.add_argument("--url", default="http://127.0.0.1:8766")
        entry.add_argument("--token")
        entry.add_argument("--token-file", default="~/.pando/mailbox.token")
        entry.add_argument("--timeout", type=float, default=15)
        entry.add_argument("--by", default="codex")
        return entry

    put = client_parser("put", "create a task")
    put.add_argument("--sender", default="win-codex")
    put.add_argument("--title", required=True)
    put.add_argument("--body")
    put.add_argument("--body-file")

    listing = client_parser("list", "list tasks")
    listing.add_argument("--status", default=None)

    get = client_parser("get", "show one task")
    get.add_argument("--id", required=True)

    claim = client_parser("claim", "claim a task before working on it")
    claim.add_argument("--id", required=True)

    result = client_parser("result", "write the final result")
    result.add_argument("--id", required=True)
    result.add_argument("--status", required=True,
                        choices=[DONE, FAILED, BLOCKED])
    result.add_argument("--output")
    result.add_argument("--output-file")
    result.add_argument("--artifact", action="append")

    append = client_parser("append", "append progress output")
    append.add_argument("--id", required=True)
    append.add_argument("--output")
    append.add_argument("--output-file")

    copy = sub.add_parser("replicate", help="copy missing tasks onto another mailbox")
    copy.add_argument("--url", required=True, help="source mailbox")
    copy.add_argument("--to-url", required=True, help="destination mailbox")
    copy.add_argument("--token")
    copy.add_argument("--token-file", default="~/.pandosi/mailbox.token")
    copy.add_argument("--to-token", default=None)
    copy.add_argument("--to-token-file", default=None)
    copy.add_argument("--timeout", type=float, default=15)
    copy.add_argument("--write", action="store_true",
                      help="actually import; default is a count only")
    copy.set_defaults(verb="replicate")

    args = parser.parse_args()
    return args.func(args) if args.verb == "serve" else client(args)


if __name__ == "__main__":
    raise SystemExit(main())
