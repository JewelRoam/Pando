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
import bisect
import gzip
import hmac
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.net import (open_url, response_json,  # noqa: E402
                       response_text, should_request_gzip)
from pando.grant_delegation import (  # noqa: E402 - repo root must be importable
    DelegationError, issue_envelope, operator_status,
)
from pando.grants import (  # noqa: E402 - repo root must be importable
    default_path as default_grants_path, grant_status,
)
from pando.identity import default_path as default_identity_path  # noqa: E402
from pando.telemetry import (  # noqa: E402 - repo root must be importable
    PI5_PUBKEY, TelemetryError, TelemetryReceiver,
    TelemetrySignatureError,
)
from tools.seed_outbox import code_bundle  # noqa: E402

OPEN, CLAIMED, DONE, FAILED, BLOCKED = "open", "claimed", "done", \
    "failed", "blocked"
FINAL = {DONE, FAILED, BLOCKED}
STATUSES = {OPEN, CLAIMED, DONE, FAILED, BLOCKED}
#: How many task records one ``/tasks/bodies`` call may return. A caller that
#: names more gets the first ``MAX_BODIES`` back and asks again for the rest.
MAX_BODIES = 2000
#: The observatory only renders a recent chat tail and a compact work queue.
#: Keep its purpose-built read bounded even if a caller sends huge limits.
MAX_FEED_WINDOW = 2000
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


def _proposal_parts(title) -> tuple[str, str, str] | None:
    """``(proposal_id, action, author)`` from an executor title.

    The mailbox has stored these titles since the first proposal.  This is a
    read-side convention only: no task is rewritten and older rows remain
    valid.  A malformed title is simply not selectable.
    """
    parts = str(title or "").split()
    if (len(parts) < 5 or parts[0] != "[proposal]"
            or parts[3] != "by"):
        return None
    return parts[1], parts[2], parts[4]


def _reaction_target(title) -> str:
    parts = str(title or "").split()
    if (len(parts) < 5 or parts[0] != "[reaction]"
            or parts[3] != "by"):
        return ""
    return parts[1]


def _task_kind(title) -> str:
    """The leading ``[kind]`` marker, lower-cased, or an empty string."""
    token = str(title or "").strip().split(None, 1)[0]
    if len(token) < 3 or not token.startswith("[") or not token.endswith("]"):
        return ""
    return token[1:-1].strip().lower()


def _clean_kinds(kinds) -> list[str]:
    """Normalise a caller's kind filter without changing its order."""
    if isinstance(kinds, str):
        kinds = [kinds]
    clean = []
    for value in kinds or ():
        kind = str(value).strip().strip("[]").lower()
        if kind and kind not in clean:
            clean.append(kind)
    return clean


def _summary_order(row: dict) -> tuple[str, str]:
    """Stable mailbox order: creation time, then id as the tie-breaker."""
    return str(row.get("created_at", "")), str(row.get("id", ""))


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
        # Listings ask for a small amount of metadata from every task.  At
        # tens of thousands of files, re-reading every JSON body for /health
        # and every tail poll makes the mailbox look offline.  Keep one
        # metadata copy in memory and refresh it on our own writes; full
        # bodies are still read on demand by load()/list().
        self._summaries: dict[str, dict] = {}
        # Derived, incrementally maintained indexes behind the summaries. A
        # focused executor poll used to copy and sort every summary on every
        # round, even though the mailbox itself changes one task at a time.
        self._ordered: list[dict] = []
        self._by_status: dict[str, list[dict]] = {}
        self._by_kind: dict[str, list[dict]] = {}
        self._by_status_kind: dict[tuple[str, str], list[dict]] = {}
        self._kinded: list[dict] = []
        self._work: list[dict] = []
        self._proposal_by_task: dict[str, tuple[str, str, str]] = {}
        self._reactions_by_target: dict[str, list[str]] = {}
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(task, dict) and task.get("id"):
                self._summaries[task["id"]] = self._summary(task)
        self._build_indexes()

    @staticmethod
    def _insert_ordered(rows: list[dict], row: dict) -> None:
        bisect.insort(rows, row, key=_summary_order)

    @staticmethod
    def _remove_ordered(rows: list[dict], row: dict) -> None:
        key = _summary_order(row)
        index = bisect.bisect_left(rows, key, key=_summary_order)
        while index < len(rows) and _summary_order(rows[index]) == key:
            if rows[index].get("id") == row.get("id"):
                rows.pop(index)
                return
            index += 1

    @staticmethod
    def _status_rows(by_status: dict[str, list[dict]],
                     row: dict) -> list[dict]:
        return by_status.setdefault(str(row.get("status", "")), [])

    def _index_summary(self, row: dict) -> None:
        self._insert_ordered(self._ordered, row)
        self._insert_ordered(self._status_rows(self._by_status, row), row)
        kind = _task_kind(row.get("title"))
        if kind:
            self._insert_ordered(self._kinded, row)
            self._insert_ordered(
                self._by_kind.setdefault(kind, []), row)
            status_kind = (str(row.get("status", "")), kind)
            self._insert_ordered(
                self._by_status_kind.setdefault(status_kind, []), row)
        else:
            self._insert_ordered(self._work, row)
        task_id = str(row.get("id", ""))
        parsed = _proposal_parts(row.get("title"))
        if parsed:
            self._proposal_by_task[task_id] = parsed
        target = _reaction_target(row.get("title"))
        if target:
            self._reactions_by_target.setdefault(target, []).append(task_id)

    def _unindex_summary(self, row: dict) -> None:
        self._remove_ordered(self._ordered, row)
        self._remove_ordered(self._status_rows(self._by_status, row), row)
        kind = _task_kind(row.get("title"))
        if kind:
            self._remove_ordered(self._kinded, row)
            rows = self._by_kind.get(kind)
            if rows is not None:
                self._remove_ordered(rows, row)
                if not rows:
                    self._by_kind.pop(kind, None)
            status_kind = (str(row.get("status", "")), kind)
            rows = self._by_status_kind.get(status_kind)
            if rows is not None:
                self._remove_ordered(rows, row)
                if not rows:
                    self._by_status_kind.pop(status_kind, None)
        else:
            self._remove_ordered(self._work, row)
        task_id = str(row.get("id", ""))
        self._proposal_by_task.pop(task_id, None)
        target = _reaction_target(row.get("title"))
        if target:
            task_ids = self._reactions_by_target.get(target)
            if task_ids is not None and task_id in task_ids:
                task_ids.remove(task_id)
                if not task_ids:
                    self._reactions_by_target.pop(target, None)

    def _build_indexes(self) -> None:
        self._ordered = sorted(self._summaries.values(), key=_summary_order)
        self._by_status = {}
        self._by_kind = {}
        self._by_status_kind = {}
        self._kinded = []
        self._work = []
        self._proposal_by_task = {}
        self._reactions_by_target = {}
        for row in self._ordered:
            self._status_rows(self._by_status, row).append(row)
            kind = _task_kind(row.get("title"))
            if kind:
                self._kinded.append(row)
                self._by_kind.setdefault(kind, []).append(row)
                self._by_status_kind.setdefault(
                    (str(row.get("status", "")), kind), []).append(row)
            else:
                self._work.append(row)
            task_id = str(row.get("id", ""))
            parsed = _proposal_parts(row.get("title"))
            if parsed:
                self._proposal_by_task[task_id] = parsed
            target = _reaction_target(row.get("title"))
            if target:
                self._reactions_by_target.setdefault(target, []).append(task_id)

    @staticmethod
    def _summary(task: dict) -> dict:
        return {"id": task.get("id"), "from": task.get("from"),
                "title": task.get("title"), "status": task.get("status"),
                "created_at": task.get("created_at", ""),
                "claimed_by": task.get("claimed_by"),
                "finished_at": task.get("finished_at")}

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
        previous = self._summaries.get(task["id"])
        if previous is not None:
            self._unindex_summary(previous)
        summary = self._summary(task)
        self._summaries[task["id"]] = summary
        self._index_summary(summary)

    def summaries(self, status: str | None = None,
                  limit: int | None = None,
                  kinds=None) -> list[dict]:
        """Return cached listing rows, filtered by kind before limiting.

        Filtering after ``limit`` lets a noisy kind displace the rows a
        consumer actually needs. ``kinds`` uses the leading title marker, so
        ``["chat", "proposal"]`` selects those two classes without walking
        the entire history.
        """
        kinds = _clean_kinds(kinds)
        with self._lock:
            if not kinds:
                if status is None or status == "all":
                    rows = self._ordered
                else:
                    rows = self._by_status.get(status, [])
                if limit is not None:
                    rows = rows[-limit:] if limit else []
                return [dict(row) for row in rows]

            selected = []
            for kind in kinds:
                if status is None or status == "all":
                    rows = self._by_kind.get(kind, [])
                else:
                    rows = self._by_status_kind.get((status, kind), [])
                # Each list is already in mailbox order. Taking its tail is
                # enough to find the newest N across the union.
                if limit is None:
                    selected.extend(rows)
                else:
                    selected.extend(rows[-limit:] if limit else [])
            selected.sort(key=_summary_order)
            if limit is not None:
                selected = selected[-limit:] if limit else []
            return [dict(row) for row in selected]

    def feed_window(self, chat_limit: int = 120,
                    work_limit: int = 14) -> dict:
        """Return only the dashboard's latest chat and work rows.

        ``/tasks?status=all`` now serialises tens of thousands of summaries
        every five seconds. The observatory never renders that whole history:
        it keeps a recent kinded feed (chat/proposal/reaction/etc.) and a short
        unkinded work queue. Split here, where rows are already indexed, so the
        wire response stays bounded without letting reaction noise displace
        work. Counts remain whole-mailbox facts rather than window facts.
        """
        with self._lock:
            chat = self._kinded[-chat_limit:] if chat_limit else []
            work = self._work[-work_limit:] if work_limit else []
            return {
                "chat": [dict(row) for row in chat],
                "work": [dict(row) for row in work],
                "counts": {
                    "chat": len(self._kinded),
                    "work": len(self._work),
                    "work_open": sum(row.get("status") == OPEN
                                     for row in self._work),
                },
            }

    def select(self, status: str | None = None, limit: int | None = None,
               selectors=None, exclude_proposal_ids=(),
               include_reactions: bool = True) -> list[dict]:
        """The latest window plus matching proposals and their reactions.

        ``selectors`` is an OR-list of ``{"actions": [...], "authors": [...]}``
        filters.  Within one selector the two fields are ANDed; an omitted or
        empty field matches any value.  Excluded proposal ids are skipped in
        the focused selection but never removed from the latest window: the
        caller still decides what the current window means.

        Reaction closure is the important part.  A late vote is a new task
        whose creation order says nothing about the old proposal it names, so
        filtering rows only by their own timestamp would silently drop it.
        """
        filters = []
        for selector in selectors or ():
            if not isinstance(selector, dict):
                continue
            actions = {str(value) for value in selector.get("actions", ())
                       if value}
            authors = {str(value) for value in selector.get("authors", ())
                       if value}
            if actions or authors:
                filters.append((actions, authors))
        excluded = {str(value) for value in exclude_proposal_ids or () if value}
        with self._lock:
            rows = (self._ordered if status is None or status == "all"
                    else self._by_status.get(status, []))

            base = rows[-limit:] if limit is not None and limit else []
            keep = {row.get("id") for row in base}
            proposal_ids = set()
            for row in base:
                parsed = _proposal_parts(row.get("title"))
                if parsed:
                    proposal_ids.add(parsed[0])

            if filters:
                for task_id, (proposal_id, action, author) in \
                        self._proposal_by_task.items():
                    row = self._summaries[task_id]
                    if not (status is None or status == "all"
                            or row.get("status") == status):
                        continue
                    if proposal_id in excluded:
                        continue
                    if any((not actions or action in actions)
                           and (not authors or author in authors)
                           for actions, authors in filters):
                        keep.add(task_id)
                        proposal_ids.add(proposal_id)

            if include_reactions:
                for proposal_id in proposal_ids:
                    for task_id in self._reactions_by_target.get(
                            proposal_id, ()):
                        row = self._summaries[task_id]
                        if status is None or status == "all" \
                                or row.get("status") == status:
                            keep.add(task_id)

            selected = [self._summaries[task_id] for task_id in keep
                        if task_id in self._summaries]
            selected.sort(key=_summary_order)
            return [dict(row) for row in selected]

    def count(self, status: str | None = None) -> int:
        with self._lock:
            if status is None or status == "all":
                return len(self._summaries)
            return sum(row.get("status") == status
                       for row in self._summaries.values())

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


def build_handler(mailbox: Mailbox, token: str, telemetry=None,
                  code_repo: Path | None = None,
                  code_cache: Path | None = None,
                  grant_file: Path | None = None,
                  identity_file: Path | None = None):
    def accepts_gzip(value: str) -> bool:
        for item in str(value or "").split(","):
            coding, *params = [part.strip() for part in item.split(";")]
            if coding.lower() != "gzip":
                continue
            for param in params:
                key, _, raw = param.partition("=")
                if key.strip().lower() == "q":
                    try:
                        return float(raw) > 0
                    except ValueError:
                        return False
            return True
        return False

    class MailboxHandler(BaseHTTPRequestHandler):
        # A peer may keep an idle HTTP/1.1 connection open.  Without a
        # timeout every such client permanently occupies a request thread;
        # repeated polling then exhausts the service after a few days.
        timeout = 30

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            compressed = (len(body) >= 1024 and
                          accepts_gzip(self.headers.get("Accept-Encoding", "")))
            if compressed:
                body = gzip.compress(body, compresslevel=6, mtime=0)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if compressed:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            header = self.headers.get("Authorization", "")
            if hmac.compare_digest(header, f"Bearer {token}"):
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
                self._send(200, {"ok": True, "service": "agent-mailbox",
                                 "open": mailbox.count(OPEN)})
                return
            if not self._authed():
                return
            if path == "/operator-grant":
                if grant_file is None or identity_file is None:
                    self._send(404, {"error": "operator delegation not configured"})
                    return
                try:
                    envelope = issue_envelope(grant_file, identity_path=identity_file)
                except Exception as exc:  # noqa: BLE001 - HTTP must fail closed
                    try:
                        local = grant_status("deploy_operator", grant_file)
                    except Exception:  # noqa: BLE001 - status is best effort
                        local = {"status": "unreadable"}
                    code = 404 if local.get("status") in ("missing", "expired") else 409
                    self._send(code, {"error": "operator grant unavailable",
                                      "status": local.get("status"),
                                      "detail": str(exc)[:300]})
                    return
                self._send(200, envelope)
                return
            if path == "/operator-status":
                if grant_file is None:
                    self._send(404, {"error": "operator status not configured"})
                    return
                self._send(200, operator_status(grant_file, identity_file))
                return
            if path == "/code.bundle":
                # The mailbox can be the only service another host can reach
                # when it moves networks.  Serving the same HEAD-keyed git
                # bundle here removes the need for a second port, a firewall
                # exception, or a separate seed-outbox process on that host.
                # It is a read-only route and never creates a mailbox task.
                if code_repo is None:
                    self._send(404, {"error": "code bundle not configured"})
                    return
                cache = (Path(code_cache).expanduser() if code_cache else
                         Path.home() / ".pandosi" / "code.bundle")
                try:
                    built = code_bundle(Path(code_repo).expanduser(), cache)
                except OSError as exc:
                    self._send(500, {"error": "code bundle unavailable",
                                     "detail": str(exc)[:200]})
                    return
                if built is None:
                    self._send(404, {"error": "no code bundle"})
                    return
                try:
                    raw = built.read_bytes()
                except OSError as exc:
                    self._send(500, {"error": "code bundle unreadable",
                                     "detail": str(exc)[:200]})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/tasks":
                params = parse_qs(query, keep_blank_values=True)
                status = (params.get("status") or [None])[0]
                raw_limit = (params.get("limit") or [None])[0]
                try:
                    limit = int(raw_limit) if raw_limit is not None else None
                    if limit is not None and limit < 0:
                        raise ValueError
                except ValueError:
                    self._send(400, {"error": "limit must be a non-negative integer"})
                    return
                raw_kinds = (params.get("kind") or []) + \
                    (params.get("kinds") or [])
                kinds = []
                for raw in raw_kinds:
                    for value in str(raw).split(","):
                        kind = value.strip().strip("[]").lower()
                        if kind and kind not in kinds:
                            kinds.append(kind)
                if len(kinds) > 16:
                    self._send(400, {"error": "at most 16 kinds are allowed"})
                    return
                answer = {"tasks": mailbox.summaries(status, limit, kinds)}
                if kinds:
                    answer["kinds"] = kinds
                self._send(200, answer)
                return
            if path == "/tasks/window":
                params = parse_qs(query, keep_blank_values=True)
                limits = {}
                for name, default in (("chat_limit", 120),
                                      ("work_limit", 14)):
                    raw = (params.get(name) or [str(default)])[0]
                    try:
                        value = int(raw)
                        if value < 0 or value > MAX_FEED_WINDOW:
                            raise ValueError
                    except (TypeError, ValueError):
                        self._send(400, {
                            "error": f"{name} must be an integer from 0 to "
                                     f"{MAX_FEED_WINDOW}"})
                        return
                    limits[name] = value
                self._send(200, mailbox.feed_window(**limits))
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
            if path == "/telemetry/pi5":
                if telemetry is None:
                    self._send(404, {"error": "not found"})
                    return
                try:
                    answer = telemetry.receive(payload)
                except TelemetrySignatureError as exc:
                    self._send(403, {"error": str(exc)[:200]})
                    return
                except TelemetryError as exc:
                    self._send(400, {"error": str(exc)[:200]})
                    return
                except OSError as exc:
                    # Validation succeeded, so the old snapshot is still the
                    # last valid observation. Do not pretend this new one
                    # landed when the atomic replace failed.
                    self._send(500, {"error": "telemetry write failed",
                                     "detail": str(exc)[:200]})
                    return
                self._send(200, answer)
                return
            if path == "/tasks/bodies":
                # A listing carries ids and titles but no bodies. Reading a
                # round's worth of rows one ``/task/<id>`` at a time is a round
                # trip per row -- thousands of them on every executor pass.
                # This route is the same data, in the chunks the caller names.
                ids = payload.get("ids") if isinstance(payload, dict) else None
                ids = ids if isinstance(ids, list) else []
                found = {}
                for task_id in ids[:MAX_BODIES]:
                    task = mailbox.load(str(task_id))
                    if task is not None:
                        found[str(task_id)] = task
                self._send(200, {"tasks": found})
                return
            if path == "/tasks/query":
                selectors = payload.get("selectors", []) if isinstance(payload, dict) else []
                excluded = payload.get("exclude_proposal_ids", []) if isinstance(payload, dict) else []
                if not isinstance(selectors, list) or not isinstance(excluded, list):
                    self._send(400, {"error": "selectors and exclude_proposal_ids must be lists"})
                    return
                if len(selectors) > 16:
                    self._send(400, {"error": "at most 16 selectors are allowed"})
                    return
                raw_limit = payload.get("limit") if isinstance(payload, dict) else None
                try:
                    limit = None if raw_limit is None else int(raw_limit)
                    if limit is not None and limit < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    self._send(400, {"error": "limit must be a non-negative integer"})
                    return
                tasks = mailbox.select(
                    status=str(payload.get("status") or "all"),
                    limit=limit, selectors=selectors,
                    exclude_proposal_ids=excluded,
                    include_reactions=bool(payload.get("include_reactions", True)))
                self._send(200, {"tasks": tasks})
                return
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
    # A node may fetch a code bundle while the executor polls tasks and peers
    # replicate state.  The stdlib default of 5 pending sockets turns a short
    # burst into connection resets, which a node correctly reports as an
    # unreachable code peer.
    request_queue_size = 64


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
    telemetry = (TelemetryReceiver(args.telemetry_root)
                 if getattr(args, "telemetry_root", "") else None)
    code_repo = (Path(args.code_repo).expanduser()
                 if getattr(args, "code_repo", "") else None)
    code_cache = (Path(args.code_cache).expanduser()
                  if getattr(args, "code_cache", "") else None)
    grant_file = (Path(args.grant_file).expanduser()
                  if getattr(args, "grant_file", "") else default_grants_path())
    identity_file = (Path(args.identity_file).expanduser()
                     if getattr(args, "identity_file", "")
                     else default_identity_path())
    server = _MailboxServer((args.host, args.port),
                            build_handler(mailbox, token, telemetry,
                                          code_repo=code_repo,
                                          code_cache=code_cache,
                                          grant_file=grant_file,
                                          identity_file=identity_file))
    print(f"agent mailbox serving on http://{args.host}:{args.port} "
          f"(dir {directory}, {mailbox.count(OPEN)} open task(s))",
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
    request_url = f"{args.url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if should_request_gzip(request_url):
        headers["Accept-Encoding"] = "gzip"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(request_url,
                                     data=body, headers=headers, method=method)
    try:
        with open_url(request, args.timeout) as response:
            return response_json(response)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "detail": response_text(exc)}
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
    server.add_argument(
        "--code-repo", default="",
        help="serve /code.bundle from this git checkout; empty disables it")
    server.add_argument(
        "--code-cache", default="~/.pandosi/code.bundle",
        help="HEAD-keyed cache for /code.bundle")
    server.add_argument(
        "--telemetry-root", default="",
        help="atomically publish signed Pi5 telemetry at this root-table path; "
             f"the signer is pinned to {PI5_PUBKEY}")
    server.add_argument(
        "--grant-file", default="",
        help="local grants ledger to delegate (default: pando.grants discovery)")
    server.add_argument(
        "--identity-file", default="",
        help="identity that signs delegated grants (default: ~/.pando/identity.json)")
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
