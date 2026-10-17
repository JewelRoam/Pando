#!/usr/bin/env python3
"""PandoSI observatory: one command, one page, live node state.

    python tools/observatory.py

Serves a live dashboard on http://127.0.0.1:8799 by default. It is the
civilization's window: who is in it, how they are configured, what they are
saying, and whether every service that keeps them alive is answering.

Design constraints (deliberate):
  * standard library only - no FastAPI/React/npm, so it starts on any node
    that can already run the mailbox (the repo pins deps to PyNaCl);
  * three arguments, all optional; everything else is auto-discovered;
  * nothing is mocked. A panel that cannot be read says so explicitly
    ("unavailable") rather than rendering a placeholder that looks real.

Reads live from:
  * the task mailbox  - /health, /tasks (chat + work feed)
  * the island root   - /nodes, /status, /playbook (participants, quorum)
  * local process state - launchd/systemd unit presence for this host
  * optional extras from ~/.pandosi/hosts.json ("services", "think_ledgers")
Writes back to:
  * the task mailbox only, and only when the operator types in the chat box;
    posting publishes a "[chat] ..." message from a chosen sender identity.

Config discovery order (first hit wins): command-line flag, environment
variable, then the per-machine files under ~/.pandosi/.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.genesis import Proposal  # noqa: E402
from pando.identity import load_or_create  # noqa: E402
from pando.net import open_url  # noqa: E402 - repo root must be importable
from pando.proposal_mail import publish_proposal  # noqa: E402

PANDO_DIR = Path(os.environ.get("PANDOSI_HOME", "~/.pandosi")).expanduser()
DEFAULT_MAILBOX = os.environ.get("PANDOSI_MAILBOX_URL", "http://127.0.0.1:8776")
_BODY_CACHE: dict[str, str] = {}


# --------------------------------------------------------------------------
# config discovery
# --------------------------------------------------------------------------
def _read_token(explicit: str | None, candidates: list[Path]) -> str | None:
    if explicit:
        return explicit.strip() or None
    for path in candidates:
        try:
            value = path.expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _read_hosts() -> dict:
    """Per-machine address book. Structure only, addresses stay local."""
    for path in (PANDO_DIR / "hosts.json", Path("hosts.example.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict):
            return raw
    return {}


def _first_healthy_mailbox(default: str, hosts: dict) -> str:
    """Keep localhost if it answers; otherwise try hosts.json candidates.

    Same rule as check_mac_endpoint.py: addresses come from per-machine
    config, never from the repository.
    """
    if http_probe(default.rstrip("/") + "/health", timeout=2.0)[0]:
        return default
    for url in (hosts.get("mailboxes") or []):
        url = str(url).strip()
        if url and http_probe(url.rstrip("/") + "/health", timeout=2.0)[0]:
            return url
    return default


def discover_root_url() -> str | None:
    env = os.environ.get("PANDOSI_ROOT_URL")
    if env:
        return env
    hosts = _read_hosts()
    for key in ("island_root_url", "island_root", "roots"):
        value = hosts.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return "http://127.0.0.1:8765"


def discover_pairing_token(explicit: str | None) -> str | None:
    return _read_token(
        explicit or os.environ.get("PANDOSI_PAIRING_TOKEN"),
        [PANDO_DIR / "pairing.token", PANDO_DIR / "root.token"],
    )


def _root_sources(hosts: dict, explicit: str | None = None) -> list[dict]:
    """Every island root this dashboard should read.

    ``roots`` names independent roots and gives each its own pairing token.
    ``island_root_url`` remains the single-root fallback so an old hosts.json
    keeps working unchanged.
    """
    explicit = explicit or os.environ.get("PANDOSI_ROOT_URL")
    if explicit:
        return [{"name": "island root", "url": explicit.rstrip("/"),
                 "token": discover_pairing_token(None)}]

    sources: list[dict] = []
    rows = hosts.get("roots")
    if isinstance(rows, (str, dict)):
        rows = [rows]
    for row in (rows or []):
        if isinstance(row, str):
            row = {"url": row}
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        token = str(row.get("token") or "").strip() or None
        token_file = str(row.get("token_file") or "").strip()
        if not token and token_file:
            token = _read_token(None, [Path(token_file)])
        sources.append({
            "name": str(row.get("name") or url).strip() or url,
            "url": url.rstrip("/"),
            "token": token or discover_pairing_token(None),
        })
    if not sources:
        url = discover_root_url()
        if url:
            sources.append({"name": "island root", "url": url.rstrip("/"),
                            "token": discover_pairing_token(None)})
    return sources


# --------------------------------------------------------------------------
# outbound HTTP
# --------------------------------------------------------------------------
def _direct(request: urllib.request.Request, timeout: float):
    """Open via pando/net.py - the project's single opener.

    LAN / loopback / tailnet literals go direct; public hosts keep the
    system proxy. Do not grow a second copy of those rules here.
    """
    return open_url(request, timeout)


def http_probe(url: str, timeout: float = 3.0) -> tuple:
    """(ok, detail, elapsed_ms, status) for endpoints that may not be JSON."""
    request = urllib.request.Request(url, method="GET")
    started = time.monotonic()
    try:
        with _direct(request, timeout) as resp:
            raw = resp.read(160)
            elapsed = round((time.monotonic() - started) * 1000, 1)
            text = raw.decode("utf-8", "replace").strip().replace("\n", " ")
            return 200 <= resp.status < 400, (text[:80] or f"HTTP {resp.status}"), \
                elapsed, resp.status
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}", \
            round((time.monotonic() - started) * 1000, 1), exc.code
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return False, f"unreachable: {exc}", \
            round((time.monotonic() - started) * 1000, 1), 0


def http_json(url: str, method: str = "GET", payload: dict | None = None,
              token: str | None = None, timeout: float = 4.0,
              header: str = "Authorization") -> tuple:
    """Return (data, error, elapsed_ms, status). Never raises.

    `header` is the field the credential goes in. The mailbox and the
    proposal endpoints take "Authorization: Bearer ..."; the island root
    takes a bare "X-Pairing-Token: ...". Getting this wrong reads as 403.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers[header] = token if header != "Authorization" else f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers,
                                     method=method)
    started = time.monotonic()
    try:
        with _direct(request, timeout) as resp:
            raw = resp.read()
            elapsed = (time.monotonic() - started) * 1000
            try:
                return json.loads(raw), None, round(elapsed, 1), resp.status
            except json.JSONDecodeError:
                return None, f"non-JSON response ({len(raw)} bytes)", \
                    round(elapsed, 1), resp.status
    except urllib.error.HTTPError as exc:
        elapsed = (time.monotonic() - started) * 1000
        return None, f"HTTP {exc.code}", round(elapsed, 1), exc.code
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        elapsed = (time.monotonic() - started) * 1000
        return None, f"unreachable: {exc}", round(elapsed, 1), 0


# --------------------------------------------------------------------------
# local process inspection - only meaningful for the host we run on
# --------------------------------------------------------------------------
_UNIT_CACHE: dict = {"at": 0.0, "value": None}


def local_units() -> dict:
    """Which PandoSI units are alive *on this machine*, by real inspection."""
    if time.time() - _UNIT_CACHE["at"] < 5 and _UNIT_CACHE["value"]:
        return _UNIT_CACHE["value"]
    units: dict = {"host": socket.gethostname(), "services": [], "source": None}
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=5).stdout
        if "pandosi" in out:
            units["source"] = "launchd"
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and "pandosi" in parts[2]:
                    pid = parts[0]
                    units["services"].append({
                        "name": parts[2], "pid": None if pid == "-" else pid,
                        "alive": pid not in ("-", ""),
                    })
    except (OSError, subprocess.SubprocessError):
        pass
    if not units["services"]:
        try:
            out = subprocess.run(
                ["systemctl", "--user", "list-units", "--type=service",
                 "--no-legend", "--no-pager", "*pandosi*"],
                capture_output=True, text=True, timeout=5).stdout
            if out.strip():
                units["source"] = "systemd"
                for line in out.splitlines():
                    fields = line.split()
                    if len(fields) >= 4 and "pandosi" in fields[0]:
                        units["services"].append({
                            "name": fields[0], "pid": None,
                            "alive": fields[2] == "running",
                        })
        except (OSError, subprocess.SubprocessError):
            pass
    _UNIT_CACHE.update(at=time.time(), value=units)
    return units


_REMOTE_CACHE: dict = {}


def read_remote_jsonl_tail(target: str, path: str, limit: int = 3,
                           ttl: float = 30.0) -> list:
    """Tail a ledger on another member over ssh. Empty list means no answer.

    The page refreshes every 5 s, so a slow ssh must not be paid per poll:
    results are cached for `ttl` seconds and the call itself is bounded.
    """
    key = (target, path)
    hit = _REMOTE_CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
             "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=1",
             "-o", "StrictHostKeyChecking=accept-new", target,
             f"tail -n {max(1, int(limit))} {path}"],
            capture_output=True, text=True, timeout=6).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        out = ""
    entries = []
    for line in out.splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _REMOTE_CACHE[key] = (time.time(), entries)
    return entries


def read_jsonl_tail(path: Path, limit: int) -> list:
    try:
        lines = path.expanduser().read_text(encoding="utf-8",
                                            errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _short(value: str | None, keep: int = 12) -> str | None:
    if not value:
        return None
    return value if len(value) <= keep else f"{value[:keep]}..."


def collect_state(cfg: dict) -> dict:
    started = time.monotonic()
    state: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "defaults": {"sender": cfg.get("sender") or "mac-codex",
                     "recipient": "all"},
        "generated_epoch": time.time(),
        "collector": {"ms": None, "errors": []},
        "services": [],
        "participants": [],
        "roots": [],
        "quorum": None,
        "proposers": [],
        "chat": [],
        "work": [],
        "thinkd": [],
    }

    # --- mailbox: health, then the feed -------------------------------
    base = cfg["mailbox_url"].rstrip("/")
    health, err, ms, status = http_json(f"{base}/health")
    state["services"].append({
        "group": "mailbox",
        "name": "task mailbox", "url": base + "/health",
        "alive": err is None and status == 200,
        "detail": err or f"open tasks: {(health or {}).get('open', '?')}",
        "latency_ms": ms, "status": status,
    })
    if err:
        state["collector"]["errors"].append(f"mailbox health: {err}")

    tasks, terr, tms, _ = http_json(f"{base}/tasks?status=all", token=cfg["mailbox_token"])
    state["services"].append({
        "group": "mailbox",
        "name": "task feed (/tasks)", "url": base + "/tasks",
        "alive": terr is None,
        "detail": terr or (f"{len(tasks.get('tasks') or [])} items"
                           if tasks else "empty payload"),
        "latency_ms": tms, "status": 200 if terr is None else 0,
    })
    if terr:
        state["collector"]["errors"].append(f"mailbox /tasks: {terr}")
    items = list((tasks or {}).get("tasks") or [])
    for item in items:
        title = str(item.get("title", ""))
        record = {
            "id": item.get("id"), "from": item.get("from"),
            "title": title, "status": item.get("status"),
            "created_at": item.get("created_at"),
            "claimed_by": item.get("claimed_by"),
        }
        if title.startswith("[chat]") or title.startswith("[proposal]"):
            record["body"] = item.get("body", "")
            record["kind"] = "chat" if title.startswith("[chat]") else "proposal"
            state["chat"].append(record)
        else:
            state["work"].append(record)

    # The /tasks listing carries titles only. Fetch every chat/proposal
    # record in full so the feed never hides the back half of a message.
    chat_ids = [str(m["id"]) for m in state["chat"] if m.get("id")]

    def _full_body(task_id: str) -> tuple[str, str | None]:
        cache_key = f"{base}|{task_id}"
        if cache_key in _BODY_CACHE:
            return task_id, _BODY_CACHE[cache_key]
        full, ferr, _, _ = http_json(
            f"{base}/task/{urllib.parse.quote(task_id)}",
            token=cfg["mailbox_token"])
        body = ((full or {}).get("task") or {}).get("body")
        if ferr is None and isinstance(body, str):
            _BODY_CACHE[cache_key] = body
            return task_id, body
        return task_id, None

    if chat_ids:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = dict(pool.map(_full_body, chat_ids))
        for record in state["chat"]:
            body = fetched.get(str(record.get("id")))
            if body is not None:
                record["body"] = body

    # --- island roots: participants, quorum, playbooks ----------------
    # Roots are independent membership tables. Read every configured root and
    # merge by node_id; the UI keeps the root names so it is obvious where a
    # participant was seen.
    for source in (cfg.get("roots") or []):
        root_url = str(source.get("url") or "").rstrip("/")
        if not root_url:
            continue
        label = str(source.get("name") or root_url)
        pairing = source.get("token") or cfg.get("pairing_token")
        summary = {"name": label, "url": root_url, "reachable": False,
                   "members": 0, "error": None, "quorum": None,
                   "playbook": None, "playbook_error": None}

        nodes, nerr, nms, nstatus = http_json(
            f"{root_url}/nodes", token=pairing, header="X-Pairing-Token")
        summary["reachable"] = nerr is None
        summary["error"] = nerr
        state["services"].append({
            "group": label,
            "name": "root /nodes", "url": root_url + "/nodes",
            "alive": nerr is None,
            "detail": nerr or f"{len((nodes or {}).get('nodes', {}))} members",
            "latency_ms": nms, "status": nstatus,
        })
        if nerr:
            state["collector"]["errors"].append(f"{label} /nodes: {nerr}")
        else:
            for node_id, info in ((nodes or {}).get("nodes") or {}).items():
                pubkey = info.get("pubkey") or ""
                cap = info.get("capability") or {}
                stats = info.get("stats") or {}
                state["participants"].append({
                    "node_id": node_id,
                    "root": label,
                    "lineage": info.get("checkpoint") or info.get("lineage") or "?",
                    "endpoint": info.get("url") or info.get("addr") or "?",
                    "host": info.get("host") or "?",
                    "capability": {
                        "platform": cap.get("platform"),
                        "arch": cap.get("arch"),
                        "cpu_count": cap.get("cpu_count"),
                        "ram_gb": cap.get("ram_gb"),
                    },
                    "pubkey_fingerprint": _short(pubkey, 16),
                    "has_pubkey": bool(pubkey),
                    "last_seen": info.get("last_seen"),
                    "stats": {
                        "successes": stats.get("successes"),
                        "failures": stats.get("failures"),
                        "latency_ewma_ms": stats.get("latency_ewma_ms"),
                        "consecutive_failures": stats.get("consecutive_failures"),
                        "last_error": stats.get("last_error"),
                    },
                })
            summary["members"] = len((nodes or {}).get("nodes") or {})

        root_status, serr, sms, sstatus = http_json(
            f"{root_url}/status", token=pairing, header="X-Pairing-Token")
        state["services"].append({
            "group": label,
            "name": "root /status", "url": root_url + "/status",
            "alive": serr is None, "detail": serr or "quorum snapshot",
            "latency_ms": sms, "status": sstatus,
        })
        if serr:
            state["collector"]["errors"].append(f"{label} /status: {serr}")
        else:
            summary["quorum"] = root_status
            if state.get("quorum") is None:
                state["quorum"] = root_status

        playbook, perr, pms, pstatus = http_json(
            f"{root_url}/playbook", token=pairing, header="X-Pairing-Token")
        if perr is None and playbook:
            summary["playbook"] = {
                "name": playbook.get("name"), "version": playbook.get("version"),
                "hash": _short(playbook.get("hash"), 16),
                "questions": len(playbook.get("questions") or []),
                "gating": playbook.get("gating"),
            }
        else:
            summary["playbook_error"] = perr or "empty"
        state["roots"].append(summary)

    merged: dict[str, dict] = {}
    for part in state["participants"]:
        node_id = part["node_id"]
        root_name = part.pop("root", "?")
        if node_id in merged:
            if root_name not in merged[node_id]["roots"]:
                merged[node_id]["roots"].append(root_name)
            continue
        part["roots"] = [root_name]
        merged[node_id] = part
    state["participants"] = sorted(merged.values(), key=lambda p: p["node_id"])

    # --- resident proposers: ledger tails ------------------------------
    # Default: this host. hosts.json may add {"host","path"[,"ssh"]}
    # entries so one observatory can show every proposer's round count.
    sources = list(cfg.get("think_ledgers") or [])
    if not sources:
        sources = [{"host": socket.gethostname().split(".")[0],
                    "path": str(PANDO_DIR / "think-ledger.jsonl")}]
    for source in sources:
        name = str(source.get("host") or "?")
        path = str(source.get("path") or "")
        if not path:
            continue
        target = source.get("ssh")
        entries = (read_remote_jsonl_tail(str(target), path, 3) if target
                   else read_jsonl_tail(Path(path).expanduser(), 3))
        state["thinkd"].append({
            "host": name, "ledger": path, "rounds": len(entries),
            "via": f"ssh {target}" if target else "local file",
            "alive": bool(entries),
            "last": entries[-1] if entries else None,
        })

    # The mailbox is the shared cross-machine truth: a proposer is visible here
    # even when its ledger sits on a host this dashboard cannot read.
    by_author: dict[str, dict] = {}
    for message in state["chat"]:
        if message.get("kind") != "proposal":
            continue
        author = str(message.get("from") or "?")
        row = by_author.setdefault(author, {
            "author": author, "count": 0, "last_at": None, "last_title": ""})
        row["count"] += 1
        created = str(message.get("created_at") or "")
        if row["last_at"] is None or created >= str(row["last_at"]):
            row["last_at"] = created
            row["last_title"] = message.get("title") or ""
    state["proposers"] = sorted(
        by_author.values(), key=lambda r: str(r["last_at"] or ""), reverse=True)

    # --- optional extra service probes (addresses stay in hosts.json) ---
    for row in (cfg.get("extra_services") or cfg.get("services") or []):
        url = str((row or {}).get("url") or "")
        if not url:
            continue
        name = str(row.get("name") or url)
        ok, detail, pms, pstatus = http_probe(url, timeout=3.0)
        state["services"].append({
            "group": str(row.get("group") or "本机与外部探测"),
            "name": name, "url": url, "alive": ok, "detail": detail,
            "latency_ms": pms, "status": pstatus,
        })

    state["local_units"] = local_units()
    state["collector"]["ms"] = round((time.monotonic() - started) * 1000, 1)
    return state


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------
def build_handler(cfg: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pandosi-observatory/1.0"

        def log_message(self, *args):  # keep the console clean
            return

        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, status: int, payload: dict):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                try:
                    self._json(200, collect_state(cfg))
                except Exception as exc:  # never take the page down with us
                    self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            path = urllib.parse.urlsplit(self.path).path
            if path != "/api/chat":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON body"})
                return
            text = str(payload.get("text", "")).strip()
            sender = str(payload.get("from") or cfg.get("sender") or "mac-codex").strip() \
                or "mac-codex"
            to = str(payload.get("to", "all")).strip().lstrip("@") or "all"
            kind = str(payload.get("kind", "chat")).strip().lower()
            if kind not in ("chat", "proposal"):
                self._json(400, {"error": "kind must be chat or proposal"})
                return
            if not text:
                self._json(400, {"error": "empty message"})
                return
            if len(text) > 4000:
                self._json(400, {"error": "message too long (max 4000)"})
                return
            base = cfg["mailbox_url"].rstrip("/")

            if kind == "proposal":
                # A proposal that is not signed is not a proposal: the agenda
                # verifiers drop it. Publish through the same signed path the
                # resident proposers use.
                author = str(cfg.get("proposer_id") or "mac-proposer")
                try:
                    identity = load_or_create(cfg.get("identity_path"))
                except Exception as exc:
                    self._json(500, {"error": f"identity unavailable: {exc}"})
                    return
                proposal = Proposal(
                    proposal_id=f"ui-{int(time.time())}-{uuid.uuid4().hex[:8]}",
                    author=author,
                    goal=text.splitlines()[0][:160],
                    action="human.propose",
                    expected_effect=text,
                    cost=0.0,
                    cites=["observatory"],
                )
                posted = publish_proposal(
                    proposal, identity, base, cfg.get("mailbox_token"))
                task = posted.get("task") if isinstance(posted, dict) else None
                if not isinstance(task, dict):
                    self._json(502, {"error": posted.get("error") or
                                     "mailbox rejected the signed proposal"})
                    return
                self._json(200, {
                    "ok": True, "id": task.get("id"), "kind": kind,
                    "author": author, "claimed": posted.get("claimed"),
                    "warning": posted.get("claim_warning"),
                })
                return

            title = f"[{kind}] @{to}: {text[:72]}"
            created, err, ms, _ = http_json(
                f"{base}/task", method="POST",
                payload={"from": sender, "title": title, "body": text},
                token=cfg["mailbox_token"])
            if err or not created or "task" not in created:
                self._json(502, {"error": err or "mailbox rejected the message"})
                return
            task_id = created["task"].get("id")
            # Chat must never linger as work - claim it as the sender.
            claimed, cerr, _, _ = http_json(
                f"{base}/claim", method="POST",
                payload={"id": task_id, "by": sender},
                token=cfg["mailbox_token"])
            self._json(200, {
                "ok": True, "id": task_id, "kind": kind, "author": sender,
                "claimed": cerr is None, "latency_ms": ms,
                "warning": None if cerr is None else f"claim failed: {cerr}",
            })

    return Handler


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PandoSI · Observatory</title>
<style>
:root{
  --bg:#0a0f16; --panel:#111823; --panel2:#0d141d; --line:#1e2937;
  --ink:#e8eef5; --dim:#93a1b1; --faint:#5f6d7d;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; --accent2:#39d3c3;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
a{color:var(--accent)}
.wrap{max-width:1560px;margin:0 auto;padding:22px 24px 70px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding-bottom:13px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.3px}
h1 span{color:var(--accent2);font-weight:500}
#hdr{margin-left:auto}
.sub{color:var(--dim);font-size:12px}
.sub b{color:var(--ink);font-weight:550}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:1px}
.ok{background:var(--ok)} .bad{background:var(--bad)} .warn{background:var(--warn)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 14px}
.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--faint);font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-top:1px}
.grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(380px,1fr);gap:16px;align-items:start}
@media(max-width:1100px){.grid{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card>h2{font-size:12px;letter-spacing:.12em;color:var(--dim);margin:0;
  padding:11px 15px 9px;border-bottom:1px solid var(--line);font-weight:600;
  display:flex;justify-content:space-between;align-items:center;gap:10px}
.card>h2 .n{color:var(--faint);letter-spacing:0;font-weight:400;font-size:11px}
.body{padding:12px 15px}
.grp{color:var(--faint);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  margin:10px 0 3px;padding-bottom:4px;border-bottom:1px solid #17212c}
.grp:first-child{margin-top:0}
.svc{display:flex;align-items:baseline;gap:9px;padding:6px 0}
.svc .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.svc .msg{color:var(--dim);font-size:12px}
.svc .ms{color:var(--faint);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:var(--faint);font-size:11px;letter-spacing:.07em;text-transform:uppercase;
  font-weight:600;padding:0 8px 6px 0;border-bottom:1px solid var(--line)}
td{padding:7px 8px 7px 0;border-bottom:1px solid #16202b;vertical-align:top}
tr:last-child td{border-bottom:0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);
  background:#0d151e;color:var(--dim);font-size:11px;white-space:nowrap}
.tag.b{border-color:#1f3a5c;color:var(--accent);background:#0c1a2a}
.tag.g{border-color:#1d3d28;color:var(--ok);background:#0b1a10}
.tag.w{border-color:#4a3a12;color:var(--warn);background:#1a1408}
.k{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13px}
.k dt{color:var(--faint)} .k dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.feed{display:flex;flex-direction:column;max-height:min(64vh,760px);overflow:auto}
.msg{padding:10px 15px;border-bottom:1px solid #151e28}
.msg:last-child{border-bottom:0}
.msg .hd{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap}
.msg .who{font-weight:600}
.msg .to{color:var(--accent2);font-size:12px}
.msg .ts{color:var(--faint);font-size:11px;margin-left:auto;font-variant-numeric:tabular-nums}
.msg .tx{margin-top:4px;white-space:pre-wrap;word-break:break-word}
.msg.prop{background:#0d1620}
.msg .meta{margin-top:6px;color:var(--faint);font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.composer{border-top:1px solid var(--line);padding:11px 13px;background:#0c131b}
.composer .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
input,select,textarea,button{font:inherit;color:var(--ink);background:#0e161f;
  border:1px solid var(--line);border-radius:8px;padding:7px 9px;outline:none}
input:focus,select:focus,textarea:focus{border-color:#2f5d8f}
textarea{width:100%;resize:vertical;min-height:62px}
button{background:#17324d;border-color:#26537f;color:#cfe4ff;cursor:pointer;font-weight:500}
button:hover{background:#1d3d5e} button:disabled{opacity:.5;cursor:default}
.rowline{display:flex;gap:10px;align-items:center}
.empty{color:var(--faint);font-style:italic;padding:12px 0}
.banner{border:1px solid #5c2a2a;background:#1d1113;color:#ffb4ae;border-radius:10px;
  padding:10px 13px;margin-top:14px;font-size:13px}
.banner b{color:#fff}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin-top:16px}
.ok-text{color:var(--ok)} .bad-text{color:var(--bad)} .warn-text{color:var(--warn)}
</style></head>
<body><div class="wrap">
<header>
  <h1>PandoSI <span>· Observatory</span></h1>
  <div class="sub" id="hdr">loading…</div>
</header>
<div id="errbox"></div>
<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="col">
    <div class="card">
      <h2>群组对话 <span class="n" id="cc"></span>
        <select id="filter" style="font-size:11px;padding:3px 6px">
          <option value="all">全部</option>
          <option value="chat">只看对话</option>
          <option value="proposal">只看提案</option>
        </select></h2>
      <div class="feed" id="chat"></div>
      <div class="composer">
        <div class="row">
          <select id="kind">
            <option value="chat">群聊消息</option>
            <option value="proposal">正式提案（签名）</option>
          </select>
          <label class="sub">发往</label>
          <input id="recipient" value="all" size="9">
          <label class="sub">署名</label>
          <input id="sender" value="mac-codex" size="11">
          <span class="sub" id="sendhint"></span>
        </div>
        <textarea id="text" placeholder="输入消息或提案… (Cmd/Ctrl+Enter 发送)"></textarea>
        <div class="rowline" style="margin-top:8px">
          <button id="send">发送</button>
        </div>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="card">
      <h2>参与者 <span class="n" id="pc"></span></h2>
      <div class="body" id="participants"></div>
    </div>
    <div class="card">
      <h2>服务健康 <span class="n" id="sc"></span></h2>
      <div class="body" id="services"></div>
    </div>
    <div class="card">
      <h2>提案者活动 <span class="n" id="tc"></span></h2>
      <div class="body" id="thinkd"></div>
    </div>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>任务队列 <span class="n" id="wc"></span></h2>
    <div class="body" id="work"></div>
  </div>
  <div class="card">
    <h2>共识 <span class="n" id="qc"></span></h2>
    <div class="body"><dl class="k" id="quorum"></dl></div>
  </div>
  <div class="card">
    <h2>本机运行 <span class="n" id="lcc"></span></h2>
    <div class="body"><dl class="k" id="local"></dl></div>
  </div>
</div>
</div>
<script>
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ts = (s) => (s || '').length >= 19 ? s.slice(11,19) : (s || '');
const CHAT_HDR = /^\[(chat|proposal)\]\s*/i;
const titleRest = (t) => String(t || '').replace(CHAT_HDR, '');
const chatTo = (t) => { const r = titleRest(t); const i = r.indexOf(':'); return i < 0 ? '' : r.slice(0, i).trim(); };
const chatBody = (t) => { const r = titleRest(t); const i = r.indexOf(':'); return i < 0 ? '' : r.slice(i + 1).trim(); };
const when = (s) => { const t = Date.parse(String(s||'').replace(' ', 'T')); return isNaN(t) ? ts(s) : new Date(t).toLocaleString('sv-SE'); };
const ago = (s) => {
  const t = Date.parse(String(s||'').replace(' ', 'T'));
  if (isNaN(t)) return '—';
  const d = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (d < 60) return d + ' 秒前';
  if (d < 3600) return Math.round(d/60) + ' 分钟前';
  if (d < 86400) return Math.round(d/3600) + ' 小时前';
  return Math.round(d/86400) + ' 天前';
};
function messageText(m){
  const body = String(m.body || '').trim();
  if (m.kind === 'proposal' && body) {
    try {
      const j = JSON.parse(body);
      const p = j.proposal || j;
      if (p && typeof p === 'object') {
        const out = [];
        if (p.expected_effect) out.push(String(p.expected_effect));
        else if (p.goal) out.push(String(p.goal));
        if (p.action) out.push('action: ' + p.action + (p.cost ? ' · cost ' + p.cost : ''));
        if (out.length) return out.join('\n');
      }
    } catch (e) {}
  }
  return body || chatBody(m.title) || titleRest(m.title);
}
function messageMeta(m){
  if (m.kind !== 'proposal') return '';
  try {
    const j = JSON.parse(String(m.body || ''));
    const p = j.proposal || j;
    const bits = [];
    if (p.proposal_id) bits.push(p.proposal_id);
    if (p.action) bits.push(p.action);
    if (p.cost) bits.push('cost ' + p.cost);
    if (j.signature || j.proof) bits.push('signed');
    return bits.join(' · ');
  } catch (e) { return ''; }
}

async function refresh(){
  let st;
  try { st = await (await fetch('/api/state', {cache:'no-store'})).json(); }
  catch(e){ $('#hdr').innerHTML = '<span class="dot bad"></span>面板后端不可达'; return; }
  if (st.error){ $('#errbox').innerHTML = `<div class="banner"><b>采集失败</b> — ${esc(st.error)}</div>`; return; }

  if (!window.__init){ window.__init = true;
    $('#sender').value = st.defaults?.sender || 'mac-codex';
    $('#recipient').value = st.defaults?.recipient || 'all';
  }

  const svcs = st.services || [], parts = st.participants || [], roots = st.roots || [];
  const alive = svcs.filter(s=>s.alive).length;
  $('#hdr').innerHTML = `<span class="dot ${alive===svcs.length?'ok':'bad'}"></span>`
    + `快照 ${esc(when(st.generated_at))} · 采集 ${st.collector?.ms ?? '?'} ms`;

  const errs = st.collector?.errors || [];
  $('#errbox').innerHTML = errs.length
    ? `<div class="banner"><b>部分数据未读到</b> — ${errs.map(esc).join(' · ')}</div>` : '';

  const reachable = roots.filter(r=>r.reachable).length;
  $('#kpis').innerHTML = `
    <div class="kpi"><div class="v">${reachable}/${roots.length || 1}</div><div class="l">island roots</div></div>
    <div class="kpi"><div class="v">${parts.length}</div><div class="l">participants</div></div>
    <div class="kpi"><div class="v ${alive===svcs.length?'ok-text':'bad-text'}">${alive}/${svcs.length}</div><div class="l">services up</div></div>
    <div class="kpi"><div class="v">${(st.proposers||[]).length}</div><div class="l">proposers seen</div></div>`;

  $('#pc').textContent = parts.length ? `${parts.length} 个节点` : '';
  $('#participants').innerHTML = parts.length ? `<table><thead><tr>
      <th>节点</th><th>来源</th><th>模型</th><th>主机</th><th>票 成/败</th><th>公钥</th></tr></thead><tbody>`
    + parts.map(p => {
        const c = p.capability || {}, s = p.stats || {};
        const spec = [c.arch, c.cpu_count ? c.cpu_count + ' cpu' : null,
                      c.ram_gb ? c.ram_gb + ' GB' : null].filter(Boolean).join(' · ') || '—';
        const lat = s.latency_ewma_ms == null ? '—' : Math.round(s.latency_ewma_ms) + ' ms';
        const bad = (s.consecutive_failures || 0) > 0;
        const rootTags = (p.roots || []).map(r =>
          `<span class="tag ${r==='win'?'b':(r==='mac'?'g':'')}">${esc(r)}</span>`).join(' ');
        return `<tr>
        <td class="mono">${esc(p.node_id)}</td>
        <td>${rootTags}</td>
        <td><span class="tag">${esc(p.lineage)}</span></td>
        <td class="sub">${esc(spec)}<br><span class="mono">${esc(p.host)}</span></td>
        <td class="${bad ? 'bad-text' : ''}">${esc((s.successes ?? 0) + '/' + (s.failures ?? 0))}
          <span class="sub">${esc(lat)}</span></td>
        <td class="mono">${p.has_pubkey ? '<span class="tag g">' + esc(p.pubkey_fingerprint) + '</span>' : '<span class="tag w">none</span>'}</td>
      </tr>`; }).join('') + `</tbody></table>`
    : `<div class="empty">没有读到任何参与者。</div>`;

  $('#sc').textContent = svcs.length ? `${alive}/${svcs.length} up` : '';
  const groups = {};
  svcs.forEach(s => (groups[s.group || '其他'] = groups[s.group || '其他'] || []).push(s));
  $('#services').innerHTML = Object.keys(groups).length ? Object.entries(groups).map(([g, rows]) =>
      `<div class="grp">${esc(g)}</div>` + rows.map(s => `<div class="svc">
        <span class="dot ${s.alive?'ok':'bad'}"></span>
        <span class="nm">${esc(s.name)} <span class="msg">${esc(s.detail)}</span></span>
        <span class="ms">${s.status ? s.status + ' · ' : ''}${s.latency_ms ?? '?'} ms</span>
      </div>`).join('')).join('')
    : `<div class="empty">没有可读的服务。</div>`;

  const tk = st.thinkd || [], pr = st.proposers || [];
  $('#tc').textContent = (tk.length || pr.length) ? `${tk.length} ledger · ${pr.length} mailbox` : '';
  const led = tk.map(t => {
    const l = t.last || {};
    const age = l.started_at ? Math.max(0, Math.round(Date.now() / 1000 - l.started_at)) : null;
    const stale = t.alive && age != null && age > 900;
    const state = !t.alive ? 'bad' : (stale ? 'warn' : 'ok');
    return `<div class="rowline" style="justify-content:space-between;padding:4px 0">
      <div><span class="dot ${state}"></span><b>${esc(t.host)}</b>
        <span class="tag b">${esc(l.action ?? '—')}</span></div>
      <div class="sub">round ${esc(l.round ?? '—')} · ${age == null ? '无记录' : age + ' 秒前'}${l.served_by ? ' · ' + esc(l.served_by) : ''}</div>
    </div>${l.goal ? `<div class="sub" style="margin:0 0 4px 14px">${esc(l.goal)}</div>` : ''}`;
  }).join('') || `<div class="empty">没有可读的 proposer ledger。</div>`;
  const mail = pr.map(p => `<div class="rowline" style="justify-content:space-between;padding:3px 0">
      <div><span class="dot ok"></span><span class="mono">${esc(p.author)}</span></div>
      <div class="sub">${esc(p.count)} 条提案 · 最近 ${esc(ago(p.last_at))}</div>
    </div>`).join('');
  $('#thinkd').innerHTML = led + (mail ? `<div class="grp">mailbox 可见的提案者</div>` + mail : '');

  const all = st.chat || [];
  const want = $('#filter').value;
  const chat = want === 'all' ? all : all.filter(m => m.kind === want);
  const nChat = all.filter(m => m.kind === 'chat').length;
  $('#cc').textContent = all.length ? `${nChat} 对话 / ${all.length - nChat} 提案` : '';
  const feed = $('#chat');
  const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 48 || !feed.dataset.ready;
  feed.innerHTML = chat.length ? chat.map(m => {
    const to = chatTo(m.title);
    const meta = messageMeta(m);
    return `<div class="msg ${m.kind === 'proposal' ? 'prop' : ''}">
      <div class="hd"><span class="who">${esc(m.from)}</span>
        ${to ? `<span class="to">@${esc(to)}</span>` : ''}
        <span class="tag ${m.kind==='proposal'?'b':''}">${m.kind==='proposal'?'提案':'消息'}</span>
        <span class="ts">${esc(when(m.created_at))}</span></div>
      <div class="tx">${esc(messageText(m))}</div>
      ${meta ? `<div class="meta">${esc(meta)}</div>` : ''}
    </div>`; }).join('') : `<div class="empty">没有匹配的消息。</div>`;
  feed.dataset.ready = '1';
  if (stick) feed.scrollTop = feed.scrollHeight;

  const rootRows = [];
  for (const r of roots) {
    const q = r.quorum || {};
    rootRows.push([`${r.name} · ${r.reachable ? '在线' : '不可达'}`,
      r.reachable ? `${r.members} 成员` : (r.error || '—')]);
    if (q && Object.keys(q).length) {
      rootRows.push([`${r.name} · quorum`, `${q.quorum_k ?? '—'}（下限 ${q.quorum_floor ?? '—'} / 上限 ${q.quorum_ceiling ?? '—'}）`]);
      if (q.decision_ewma_ms != null) rootRows.push([`${r.name} · 决策延迟`, Math.round(q.decision_ewma_ms) + ' ms']);
      rootRows.push([`${r.name} · blocked`, (q.blocked || []).length]);
    }
    if (r.playbook) rootRows.push([`${r.name} · playbook`, `${r.playbook.name} v${r.playbook.version} · ${r.playbook.questions} 题`]);
  }
  $('#quorum').innerHTML = rootRows.length
    ? rootRows.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')
    : `<dt>—</dt><dd>没有读到 root</dd>`;

  const units = (st.local_units || {}).services || [];
  $('#lcc').textContent = units.length ? `${units.filter(u=>u.alive).length}/${units.length} 运行中` : '';
  $('#local').innerHTML = units.length
    ? units.map(u => `<dt>${esc(String(u.name).replace(/^com\.(jewelroam|pandosi)\./,''))}</dt>
        <dd class="${u.alive?'ok-text':'bad-text'}">${u.alive ? '运行中' : '未运行'}${u.pid ? ' · pid ' + esc(u.pid) : ''}</dd>`).join('')
    : `<dt>—</dt><dd>无 launchd/systemd 记录</dd>`;

  const work = st.work || [];
  const nopen = work.filter(w => w.status === 'open').length;
  $('#wc').textContent = work.length ? `${nopen} open / ${work.length} total` : '';
  $('#work').innerHTML = work.length ? `<table><tbody>` + work.slice(-14).reverse().map(w =>
      `<tr><td><span class="tag ${w.status==='open'?'b':''}">${esc(w.status)}</span></td>
        <td class="mono">${esc(w.id)}</td><td>${esc(w.title)}</td>
        <td class="sub">${esc(w.claimed_by || w.from || '')}</td></tr>`).join('') + `</tbody></table>`
    : `<div class="empty">没有任务。</div>`;
}

$('#kind').onchange = () => {
  const prop = $('#kind').value === 'proposal';
  $('#sender').disabled = prop;
  $('#sendhint').textContent = prop ? '将用本机 proposer 身份签名发布' : '';
};
$('#send').onclick = async () => {
  const text = $('#text').value.trim();
  if (!text) return;
  const btn = $('#send'); btn.disabled = true; $('#sendhint').textContent = '发送中…';
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, from: $('#sender').value, to: $('#recipient').value,
                            kind: $('#kind').value})});
    const j = await r.json();
    if (j.ok){ $('#text').value=''; $('#sendhint').textContent = `已发送 ${j.id}` + (j.author ? ` · ${j.author}` : ''); refresh(); }
    else { $('#sendhint').textContent = '失败: ' + (j.error||r.status); }
  } catch(e){ $('#sendhint').textContent = '失败: '+e; }
  btn.disabled = false;
};
$('#filter').onchange = refresh;
$('#text').addEventListener('keydown', e => {
  if ((e.metaKey||e.ctrlKey) && e.key === 'Enter') $('#send').click();
});
$('#kind').onchange();
refresh();
setInterval(refresh, 5000);
</script>
</body></html>
"""


# --------------------------------------------------------------------------
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="PandoSI observatory: one command, one page.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PANDOSI_OBSERVATORY_PORT", "8799")),
                        help="listen port (default 8799)")
    parser.add_argument("--mailbox", default=None,
                        help="mailbox base URL (default: the first candidate "
                             "in ~/.pandosi/hosts.json that answers /health, "
                             f"else {DEFAULT_MAILBOX})")
    parser.add_argument("--root", default=None,
                        help="single island root base URL; by default all "
                             "roots in ~/.pandosi/hosts.json are read")
    args = parser.parse_args()

    hosts = _read_hosts()
    mailbox = (args.mailbox or os.environ.get("PANDOSI_MAILBOX_URL")
               or DEFAULT_MAILBOX)
    if not args.mailbox and mailbox == DEFAULT_MAILBOX:
        mailbox = _first_healthy_mailbox(mailbox, hosts)

    roots = _root_sources(hosts, args.root)
    cfg = {
        "mailbox_url": mailbox,
        "mailbox_token": _read_token(os.environ.get("PANDOSI_MAILBOX_TOKEN"),
                                     [PANDO_DIR / "mailbox.token"]),
        "roots": roots,
        "pairing_token": discover_pairing_token(None),
        "think_ledgers": hosts.get("think_ledgers"),
        "extra_services": hosts.get("services"),
        "sender": str(hosts.get("sender") or "mac-codex"),
        "proposer_id": str(hosts.get("proposer_id") or "mac-proposer"),
        "identity_path": hosts.get("identity_path"),
    }

    server = ThreadingHTTPServer(("127.0.0.1", args.port), build_handler(cfg))
    print(f"PandoSI observatory → http://127.0.0.1:{args.port}", flush=True)
    print(f"  mailbox   {cfg['mailbox_url']}"
          f"{'  (token loaded)' if cfg['mailbox_token'] else '  (NO TOKEN - feed will 403)'}",
          flush=True)
    for source in roots:
        print(f"  root      {source['name']}  {source['url']}"
              f"{'  (token loaded)' if source.get('token') else '  (NO TOKEN)'}",
              flush=True)
    print(f"  chat box defaults to from={cfg['sender']} to=all; both editable",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
