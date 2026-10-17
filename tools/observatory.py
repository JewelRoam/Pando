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
import re
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

from pando.genesis import Proposal, verify_reaction  # noqa: E402
from pando.identity import load_or_create, verify as verify_signature  # noqa: E402
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




_KIND_RE = re.compile(r"^\[([A-Za-z0-9_-]+)\]\s*(.*)$", re.DOTALL)


def split_title(title: str) -> tuple[str | None, str]:
    """A leading ``[kind]`` is the message class. Anything else is work.

    The kind is whatever the writer put in the brackets. The page does not
    keep a list of allowed members, roles, or message names.
    """
    match = _KIND_RE.match(str(title).strip())
    if not match:
        return None, str(title).strip()
    return match.group(1).lower(), match.group(2).strip()


def addressee(rest: str) -> str | None:
    """Name addressed by a single leading @. Extra @ signs are not kept.

    ``@win-codex: hello`` and ``@@win-codex: hello`` both yield ``win-codex``.
    The page draws one @ itself.
    """
    text = str(rest or "").strip()
    if not text.startswith("@"):
        return None
    head, _, _ = text.partition(":")
    name = head.lstrip("@").strip()
    return name or None


def signed_fields(body: str) -> dict | None:
    """Payload of a signed envelope, or None if it is not one.

    The signature is checked. A body that merely looks like JSON is not
    treated as fields.
    """
    try:
        record = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    pubkey = str(record.get("pubkey") or "")
    sig = str(record.get("sig") or "")
    if not pubkey or not sig or not verify_signature(pubkey, payload, sig):
        return None
    return dict(payload)


def is_annotation(fields: dict | None) -> bool:
    """A short stance hung on another message, not a message of its own."""
    if not isinstance(fields, dict):
        return False
    return bool(str(fields.get("target_id") or "").strip()
                and str(fields.get("stance") or "").strip()
                and str(fields.get("node_id") or "").strip())


def build_people(participants: list[dict], ledgers: list[dict],
                 messages: list[dict]) -> list[dict]:
    """One row per name we actually saw. No role is assigned here.

    A root member, a ledger host, and a mailbox author share a row only when
    the name is the same string. Otherwise they stay separate rows in the
    same list.
    """
    rows: list[dict] = []
    index: dict[str, dict] = {}

    def add(name: str) -> dict:
        row = index.get(name)
        if row is not None:
            return row
        row = {
            "name": name, "roots": [], "lineage": "", "host": "",
            "capability": {}, "stats": {}, "pubkey": "",
            "activity": None, "seen": None,
        }
        rows.append(row)
        if name:
            index[name] = row
        return row

    for part in participants:
        row = add(str(part.get("node_id") or ""))
        row["roots"] = list(part.get("roots") or [])
        row["lineage"] = part.get("lineage") or ""
        row["host"] = part.get("host") or ""
        row["capability"] = part.get("capability") or {}
        row["stats"] = part.get("stats") or {}
        row["pubkey"] = part.get("pubkey_fingerprint") if part.get("has_pubkey") else ""
    for ledger in ledgers:
        last = ledger.get("last") or {}
        name = str(ledger.get("host") or "")
        row = add(name)
        row["activity"] = {
            "action": last.get("action") or "",
            "goal": last.get("goal") or "",
            "round": last.get("round"),
            "started_at": last.get("started_at"),
            "served_by": last.get("served_by") or "",
            "alive": bool(ledger.get("alive")),
            "via": ledger.get("via") or "",
        }
        if not row["lineage"]:
            row["lineage"] = row["activity"]["served_by"]
    counts: dict[str, dict] = {}
    for message in messages:
        fields = message.get("fields") or {}
        if is_annotation(fields):
            continue
        author = str(message.get("from") or "")
        if not author:
            continue
        slot = counts.setdefault(author, {"count": 0, "kind": "", "at": "", "action": ""})
        slot["count"] += 1
        created = str(message.get("created_at") or "")
        if created >= str(slot["at"]):
            slot["at"] = created
            slot["kind"] = str(message.get("kind") or "")
            slot["action"] = str(fields.get("action") or "")
    for author, slot in counts.items():
        row = add(author)
        row["seen"] = slot
    return rows


def compose_kinds(hosts: dict) -> list[dict]:
    """What the composer may send. Names come from hosts.json, not the page."""
    rows = hosts.get("compose") if isinstance(hosts, dict) else None
    parsed: list[dict] = []
    for row in rows or []:
        if isinstance(row, str):
            row = {"kind": row}
        if not isinstance(row, dict) or not row.get("kind"):
            continue
        parsed.append({
            "kind": str(row["kind"]),
            "signed": bool(row.get("signed")),
            "action": str(row.get("action") or ""),
        })
    if parsed:
        return parsed
    return [
        {"kind": "chat", "signed": False, "action": ""},
        {"kind": "proposal", "signed": True, "action": "human.propose"},
    ]


def proposal_view(body: str) -> dict | None:
    """Turn a signed proposal envelope into the fields a person reads.

    The mailbox stores ``{kind, version, payload, pubkey, sig}``. Goal, action,
    expected effect, cost and cites live under ``payload``; the signature is
    proof, not content. Returns None when the body is ordinary text.
    """
    try:
        record = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        alt = record.get("proposal")
        payload = alt if isinstance(alt, dict) else None
    if not isinstance(payload, dict):
        return None
    if not any(payload.get(key) for key in
               ("goal", "action", "expected_effect", "proposal_id")):
        return None
    cites = payload.get("cites") or []
    if not isinstance(cites, list):
        cites = [cites]
    return {
        "proposal_id": str(payload.get("proposal_id") or ""),
        "author": str(payload.get("author") or ""),
        "goal": str(payload.get("goal") or ""),
        "action": str(payload.get("action") or ""),
        "expected_effect": str(payload.get("expected_effect") or ""),
        "cost": payload.get("cost"),
        "cites": [str(item) for item in cites if item not in (None, "")],
        "signed": bool(record.get("sig") and record.get("pubkey")),
        "pubkey": _short(str(record.get("pubkey") or ""), 16) or "",
    }



def attach_reactions(messages: list[dict], reactions: list[dict]) -> None:
    """Hang each decision-model stance on the proposal it names.

    One chip per model. A later answer from the same node replaces the earlier
    one. Messages that are not that proposal are left alone.
    """
    by_target: dict[str, list[dict]] = {}
    for reaction in reactions:
        target = str(reaction.get("target_id") or "")
        node = str(reaction.get("node_id") or "")
        if not target or not node:
            continue
        bucket = by_target.setdefault(target, [])
        bucket[:] = [row for row in bucket if row.get("node_id") != node]
        bucket.append({
            "node_id": node,
            "checkpoint": reaction.get("checkpoint") or "",
            "stance": reaction.get("stance") or "",
        })
    for message in messages:
        proposal = message.get("fields") or message.get("proposal") or {}
        target = str(proposal.get("proposal_id") or "")
        if target and target in by_target:
            message["reactions"] = by_target[target]


def collect_state(cfg: dict) -> dict:
    started = time.monotonic()
    state: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "defaults": {"sender": cfg.get("sender") or "",
                     "recipient": cfg.get("recipient") or "all"},
        "compose": list(cfg.get("compose") or []),
        "generated_epoch": time.time(),
        "collector": {"ms": None, "errors": []},
        "services": [],
        "participants": [],
        "roots": [],
        "quorum": None,
        "proposers": [],
        "chat": [],
        "reactions": [],
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
        kind, rest = split_title(title)
        if kind:
            record["body"] = item.get("body", "")
            record["kind"] = kind
            record["to"] = addressee(rest)
            state["chat"].append(record)
        else:
            state["work"].append(record)

    # The /tasks listing carries titles only. Fetch every chat/proposal
    # record in full so the feed never hides the back half of a message.
    feed = state["chat"] + state["reactions"]
    chat_ids = [str(m["id"]) for m in feed if m.get("id")]

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
        parsed_reactions = []
        visible = []
        for record in feed:
            body = fetched.get(str(record.get("id")))
            fields = signed_fields(body) if isinstance(body, str) and body.lstrip().startswith("{") else None
            if fields is not None:
                record["fields"] = fields
                record["signed"] = True
                record["body"] = ""
                if fields.get("proposal_id"):
                    record["proposal"] = {"proposal_id": fields.get("proposal_id")}
            elif isinstance(body, str):
                record["body"] = body
            if is_annotation(fields):
                parsed_reactions.append({
                    "target_id": str(fields.get("target_id") or ""),
                    "node_id": str(fields.get("node_id") or ""),
                    "checkpoint": str(fields.get("checkpoint") or ""),
                    "stance": str(fields.get("stance") or ""),
                })
                continue
            visible.append(record)
        state["chat"] = visible
        state["reactions"] = parsed_reactions
        attach_reactions(state["chat"], parsed_reactions)

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
    state["people"] = build_people(
        state["participants"], state["thinkd"], state["chat"])

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
            sender = str(payload.get("from") or cfg.get("sender") or "").strip()
            to = str(payload.get("to") or cfg.get("recipient") or "all").strip().lstrip("@") or "all"
            kind = str(payload.get("kind") or "").strip().lower()
            compose = {row["kind"]: row for row in (cfg.get("compose") or [])}
            if not sender:
                self._json(400, {"error": "from is required"})
                return
            if kind not in compose:
                self._json(400, {"error": "unknown kind"})
                return
            if not text:
                self._json(400, {"error": "empty message"})
                return
            if len(text) > 4000:
                self._json(400, {"error": "message too long (max 4000)"})
                return
            base = cfg["mailbox_url"].rstrip("/")

            if compose[kind].get("signed"):
                author = sender
                try:
                    identity = load_or_create(cfg.get("identity_path"))
                except Exception as exc:
                    self._json(500, {"error": f"identity unavailable: {exc}"})
                    return
                proposal = Proposal(
                    proposal_id=f"ui-{int(time.time())}-{uuid.uuid4().hex[:8]}",
                    author=author,
                    goal=text.splitlines()[0][:160],
                    action=str(compose[kind].get("action") or "propose"),
                    expected_effect=text,
                    cost=0.0,
                    cites=[],
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
.msg .goal{margin-top:7px;font-weight:600}
.msg .effect{margin-top:3px;white-space:pre-wrap;word-break:break-word;color:#d7e0ea}
.msg .facts{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
.msg .pid{margin-left:auto;color:var(--faint);font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.reacts{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:99px;
  border:1px solid var(--line);background:#0e1722;font-size:12px}
.chip .who{color:var(--dim);font-weight:500}
.chip.yes{border-color:#1d3d28;color:var(--ok)}
.chip.mid{border-color:#4a3a12;color:var(--warn)}
.chip.no{border-color:#5c2a2a;color:#ffb4ae}
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
        </select></h2>
      <div class="feed" id="chat"></div>
      <div class="composer">
        <div class="row">
          <select id="kind"></select>
          <label class="sub">发往</label>
          <input id="recipient" value="" size="9">
          <label class="sub">署名</label>
          <input id="sender" value="" size="11">
          <span class="sub" id="sendhint"></span>
        </div>
        <textarea id="text" placeholder="写在这里 (Cmd/Ctrl+Enter 发送)"></textarea>
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
function reactionRow(list){
  if (!list || !list.length) return '';
  return `<div class="reacts">` + list.map(r =>
    `<span class="chip" title="${esc(r.checkpoint || '')}">${esc(r.stance)} <span class="who">${esc(r.node_id)}</span></span>`
  ).join('') + `</div>`;
}
function renderPeople(rows){
  if (!rows.length) return '<div class="empty">没有读到任何人。</div>';
  const head = `<table><thead><tr><th>名字</th><th>来自</th><th>最近</th><th>主机</th><th></th></tr></thead><tbody>`;
  return head + rows.map(p => {
    const tags = []
      .concat((p.roots || []).map(r => `<span class="tag">${esc(r)}</span>`))
      .concat(p.activity ? ['<span class="tag">ledger</span>'] : [])
      .concat(p.seen && p.seen.kind ? [`<span class="tag">${esc(p.seen.kind)} ${esc(p.seen.count)}</span>`] : []);
    const act = p.activity || {};
    const recent = [p.lineage, act.action, p.seen && p.seen.action].filter(Boolean).join(' · ') || '—';
    const c = p.capability || {}, st = p.stats || {};
    const spec = [p.host, c.arch, c.cpu_count ? c.cpu_count + ' cpu' : null,
                  c.ram_gb ? c.ram_gb + ' GB' : null].filter(Boolean).join(' · ');
    const votes = (st.successes != null || st.failures != null)
      ? `${st.successes ?? 0}/${st.failures ?? 0}` : '';
    const round = act.round != null ? 'r' + act.round : '';
    const whenSeen = act.started_at
      ? Math.max(0, Math.round(Date.now() / 1000 - act.started_at)) + 's'
      : (p.seen && p.seen.at ? ago(p.seen.at) : '');
    const key = p.pubkey ? `<span class="tag">${esc(p.pubkey)}</span>` : '';
    const goal = act.goal ? `<div class="sub">${esc(act.goal)}</div>` : '';
    return `<tr>
      <td class="mono">${esc(p.name)} ${key}</td>
      <td>${tags.join(' ') || '—'}</td>
      <td>${esc(recent)}${goal}</td>
      <td class="sub">${esc(spec || act.via || '—')}</td>
      <td class="sub">${esc([votes, round, whenSeen].filter(Boolean).join(' · '))}</td>
    </tr>`;
  }).join('') + '</tbody></table>';
}
function renderMessage(m){
  const f = m.fields || null;
  if (f && (f.goal || f.action || f.expected_effect)) {
    const goal = String(f.goal || '').trim();
    const effect = String(f.expected_effect || '').trim();
    const showGoal = goal && goal !== effect;
    const cites = (Array.isArray(f.cites) ? f.cites : []).map(c => `<span class="tag">${esc(c)}</span>`).join('');
    const cost = (f.cost === 0 || f.cost) ? `<span class="tag">${esc(f.cost)}</span>` : '';
    return `<div class="msg prop">
      <div class="hd"><span class="who">${esc(f.author || m.from)}</span>
        ${f.action ? `<span class="tag b">${esc(f.action)}</span>` : ''}
        <span class="tag">${esc(m.kind || '')}</span>
        ${m.signed ? '<span class="tag g">已签名</span>' : ''}
        <span class="ts">${esc(when(m.created_at))}</span></div>
      ${showGoal ? `<div class="goal">${esc(goal)}</div>` : ''}
      ${effect ? `<div class="effect">${esc(effect)}</div>` : (goal ? `<div class="effect">${esc(goal)}</div>` : '')}
      <div class="facts">${cost}${cites}
        ${f.proposal_id ? `<span class="pid">${esc(f.proposal_id)}</span>` : ''}</div>
      ${reactionRow(m.reactions)}
    </div>`;
  }
  const text = String(m.body || '').trim() || titleRest(m.title);
  return `<div class="msg">
    <div class="hd"><span class="who">${esc(m.from)}</span>
      ${m.to ? `<span class="to">@${esc(m.to)}</span>` : ''}
      <span class="tag">${esc(m.kind || '')}</span>
      <span class="ts">${esc(when(m.created_at))}</span></div>
    <div class="tx">${esc(text)}</div>
  </div>`;
}

async function refresh(){
  let st;
  try { st = await (await fetch('/api/state', {cache:'no-store'})).json(); }
  catch(e){ $('#hdr').innerHTML = '<span class="dot bad"></span>面板后端不可达'; return; }
  if (st.error){ $('#errbox').innerHTML = `<div class="banner"><b>采集失败</b> — ${esc(st.error)}</div>`; return; }

  if (!window.__init){ window.__init = true;
    const rows = st.compose || [];
    window.__signedKinds = rows.filter(r => r.signed).map(r => r.kind);
    $('#kind').innerHTML = rows.map(r => `<option value="${esc(r.kind)}">${esc(r.kind)}</option>`).join('');
    $('#sender').value = st.defaults?.sender || '';
    $('#recipient').value = st.defaults?.recipient || '';
    $('#kind').onchange();
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
    <div class="kpi"><div class="v">${(st.people||[]).length}</div><div class="l">参与者</div></div>
    <div class="kpi"><div class="v ${alive===svcs.length?'ok-text':'bad-text'}">${alive}/${svcs.length}</div><div class="l">services up</div></div>`;

  const people = st.people || [];
  $('#pc').textContent = people.length ? `${people.length}` : '';
  $('#participants').innerHTML = renderPeople(people);
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

  const all = st.chat || [];
  const want = $('#filter').value;
  const chat = want === 'all' ? all : all.filter(m => m.kind === want);
  const kinds = [...new Set(all.map(m => m.kind).filter(Boolean))];
  if ($('#filter').dataset.kinds !== kinds.join(',')) {
    const cur = $('#filter').value || 'all';
    $('#filter').innerHTML = '<option value="all">全部</option>'
      + kinds.map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
    $('#filter').value = [...$('#filter').options].some(o => o.value === cur) ? cur : 'all';
    $('#filter').dataset.kinds = kinds.join(',');
  }
  const counts = kinds.map(k => `${all.filter(m => m.kind === k).length} ${k}`).join(' · ');
  $('#cc').textContent = counts;
  const feed = $('#chat');
  const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 48 || !feed.dataset.ready;
  feed.innerHTML = chat.length ? chat.map(renderMessage).join('')
    : `<div class="empty">没有匹配的消息。</div>`;
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
    ? units.map(u => `<dt>${esc(u.name)}</dt>
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
  const signed = new Set(window.__signedKinds || []);
  $('#sender').disabled = false;
  $('#sendhint').textContent = signed.has($('#kind').value) ? '签名后发布' : '';
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
        "sender": str(hosts.get("sender") or ""),
        "recipient": str(hosts.get("recipient") or "all"),
        "compose": compose_kinds(hosts),
        "proposer_id": str(hosts.get("proposer_id") or ""),
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
