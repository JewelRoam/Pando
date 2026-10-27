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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.chat import normalize_chat  # noqa: E402
from pando.genesis import Proposal, verify_reaction  # noqa: E402
from pando.identity import load_or_create, verify as verify_signature  # noqa: E402
from pando.net import open_url  # noqa: E402 - repo root must be importable
from pando.proposal_mail import proposal_age_seconds  # noqa: E402
from pando.proposal_mail import publish_proposal  # noqa: E402

PANDO_DIR = Path(os.environ.get("PANDOSI_HOME", "~/.pandosi")).expanduser()
DEFAULT_MAILBOX = os.environ.get("PANDOSI_MAILBOX_URL", "http://127.0.0.1:8776")
_BODY_CACHE: dict[str, str] = {}
_STATE_CACHE: dict = {"at": 0.0, "value": None}
_STATE_LOCK = threading.Lock()
_ROOT_FAILURE_TTL = 30.0
_ROOT_FAILURE_CACHE: dict[str, dict] = {}
_ROOT_FAILURE_LOCK = threading.Lock()
CHAT_VIEW_LIMIT = 120
WORK_VIEW_LIMIT = 14


def _cached_root_failure(root_url: str) -> tuple[str, float] | None:
    """Return a recent /nodes failure without pretending it is a live read."""
    now = time.monotonic()
    with _ROOT_FAILURE_LOCK:
        row = _ROOT_FAILURE_CACHE.get(root_url)
        if not row:
            return None
        age = now - row["at"]
        if age < _ROOT_FAILURE_TTL:
            return str(row["error"]), age
        _ROOT_FAILURE_CACHE.pop(root_url, None)
    return None


def _remember_root_failure(root_url: str, error: str) -> None:
    with _ROOT_FAILURE_LOCK:
        _ROOT_FAILURE_CACHE[root_url] = {
            "at": time.monotonic(), "error": str(error),
        }


def _clear_root_failure(root_url: str) -> None:
    with _ROOT_FAILURE_LOCK:
        _ROOT_FAILURE_CACHE.pop(root_url, None)


def cached_state(cfg: dict, ttl: float = 5.0) -> dict:
    """Serve a 5s-old state instead of collecting once per poll.

    The page polls every 5s; with 260+ messages one collection is a few
    seconds, and overlapping collections (page + CLI curl + another tab)
    starve the small thread pool - seen live as a 10s timeout on /api/state
    while the socket was listening. One collect per TTL period fixes it.
    """
    now = time.time()
    if _STATE_CACHE["value"] is not None and now - _STATE_CACHE["at"] < ttl:
        return _STATE_CACHE["value"]
    with _STATE_LOCK:  # concurrent misses collect once, not N times
        if (_STATE_CACHE["value"] is not None
                and time.time() - _STATE_CACHE["at"] < ttl):
            return _STATE_CACHE["value"]
        state = collect_state(cfg)
        _STATE_CACHE.update(at=time.time(), value=state)
        return state


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
        snapshot = str(row.get("file") or "").strip()
        if not url and not snapshot:
            continue
        token = str(row.get("token") or "").strip() or None
        token_file = str(row.get("token_file") or "").strip()
        if not token and token_file:
            token = _read_token(None, [Path(token_file)])
        sources.append({
            "name": str(row.get("name") or url or snapshot).strip() or url or snapshot,
            "url": url.rstrip("/"),
            "file": snapshot,
            "token": token or (discover_pairing_token(None) if url else None),
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
                           ttl: float = 30.0) -> tuple[list, str | None]:
    """Tail a ledger over ssh. The error is kept; an empty list is not success.

    The page refreshes every 5 s, so a slow ssh must not be paid per poll:
    results are cached for `ttl` seconds and the call itself is bounded.
    A launchd process on this Mac cannot open 192.168.31.0/24 (EHOSTUNREACH).
    The error has to stay on the row, or the host looks like an empty person.
    """
    key = (target, path)
    hit = _REMOTE_CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1], hit[2]
    err = None
    out = ""
    try:
        done = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
             "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=1",
             "-o", "StrictHostKeyChecking=accept-new", target,
             f"tail -n {max(1, int(limit))} {path}"],
            capture_output=True, text=True, timeout=6)
        out = done.stdout or ""
        if done.returncode != 0 or not out.strip():
            err = ((done.stderr or "").strip().replace("\n", " ")[:240]
                   or f"ssh rc={done.returncode}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        err = str(exc)[:240]
    entries = []
    for line in out.splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if entries:
        err = None
    _REMOTE_CACHE[key] = (time.time(), entries, err)
    return entries, err


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


#: How old a member's newest answered proposal may be before the page stops
#: calling it current. The fleet stamps proposals every few minutes, so a live
#: voter that has answered nothing in six hours is not having a quiet spell;
#: it is working through a backlog. Kept deliberately loose -- this column
#: exists to catch a node answering *yesterday*, not to nag about a slow
#: minute, and a false alarm here would teach a reader to ignore the column.
VOTE_STALE_S = 6 * 3600.0

#: How old the published vote judgement may be before the page stops calling it
#: current. The watchdog runs every 300s, so three missed passes is a stopped
#: judge rather than a slow one.
VOTE_VERDICT_STALE_S = 900.0

#: Written by tools/vote_watchdog.py. Overridable, so a non-default install
#: does not have to guess.
DEFAULT_VOTE_VERDICT = PANDO_DIR / "vote-watchdog.json"

#: The four states the watchdog may report, and how the page says each one.
#: Only ``ok`` is calm: "we measured it and it is fine" and "we could not
#: measure it at all" must not wear the same face.
VERDICT_TEXT = {"ok": "正常", "wedge": "落后", "silent": "沉默", "unknown": "未知"}
#: The two that are an alarm rather than a reading.
VERDICT_ALARM = ("wedge", "silent")


def _verdict_lag(seconds) -> str:
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        return f"{seconds / 3600.0:.1f} 小时"
    return "—"


def verdict_why(verdict: dict | None) -> str:
    """The numbers behind a verdict, in the smallest honest form.

    An alarm that says only "落后" teaches a reader to distrust it; an alarm
    that prints the slope it measured can be checked and argued with.
    """
    row = verdict or {}
    slope = row.get("slope")
    if isinstance(slope, (int, float)) and not isinstance(slope, bool):
        text = (f"slope {slope:+.2f} · "
                f"{_verdict_lag(row.get('first_lag_s'))} → "
                f"{_verdict_lag(row.get('last_lag_s'))}")
        span = row.get("span_s")
        if isinstance(span, (int, float)) and not isinstance(span, bool):
            text += f" · {row.get('samples', '?')} 票 / {span / 60.0:.0f} 分"
        return text
    return str(row.get("why") or "")


def verdict_line(verdict: dict, alarms: list[dict] | None = None) -> str:
    """Who judged, how long ago, and whether that judgement is still current."""
    if not verdict.get("available"):
        return f"表态判定不可用（{verdict.get('error') or '未写出'}）"
    age = verdict.get("age_s")
    if age is None:
        when = "时间未知"
    elif age < 120:
        when = f"{age:.0f} 秒前"
    else:
        when = f"{age / 60.0:.0f} 分钟前"
    head = f"判定 {when}"
    if verdict.get("stale"):
        head += "（已过期）"
    count = len(alarms or [])
    return head + (f" · {count} 个告警" if count else " · 无告警")


def read_vote_verdict(path: Path | None = None,
                      now: float | None = None) -> dict:
    """The vote_watchdog's last judgement, carrying its own age.

    The page does not judge votes. It fetches the last 120 messages, so a
    judgement computed here would be fitted over minutes of evidence while the
    watchdog fits the same column name over the whole feed -- one name, two
    questions. The difference is silent and lands on the reassuring side:
    fewer samples means ``unknown``, and ``unknown`` reads like ``no alarm``.
    So this reads the published snapshot instead.

    A missing or unreadable file is ``available: False`` with a reason. It is
    never rendered as an empty column, because "no alarm" and "no judge" must
    not look alike.
    """
    moment = time.time() if now is None else now
    target = Path(path) if path else DEFAULT_VOTE_VERDICT
    result = {"available": False, "path": str(target), "error": None,
              "written": None, "age_s": None, "stale": True,
              "members": {}, "decision": [], "window_s": None, "counts": {}}
    try:
        payload = json.loads(target.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        result["error"] = "never written"
        return result
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("members"), dict):
        result["error"] = "not a verdict snapshot"
        return result
    result["available"] = True
    result["members"] = payload["members"]
    result["decision"] = [str(name) for name in (payload.get("decision") or [])]
    result["window_s"] = payload.get("window_s")
    written = payload.get("written")
    if isinstance(written, (int, float)) and not isinstance(written, bool):
        result["written"] = float(written)
        result["age_s"] = max(0.0, moment - float(written))
        result["stale"] = result["age_s"] > VOTE_VERDICT_STALE_S
    else:
        result["error"] = "snapshot has no timestamp"
    counts: dict[str, int] = {}
    for verdict in result["members"].values():
        status = str((verdict or {}).get("status") or "?")
        counts[status] = counts.get(status, 0) + 1
    result["counts"] = counts
    return result


def newest_vote_age(reactions: list[dict],
                    now: float | None = None) -> dict[str, float]:
    """Per member, the age in seconds of the newest proposal it answered.

    A reaction names the proposal it answers, and every proposal id ends in
    the epoch its proposer stamped it with. *Newest* is what separates a
    voting member from a member that is merely still running: on 2026-10-19
    ``win-laya-int8`` appended a reaction every two minutes while every
    target it named was 35 hours old. Its process was up, its ledger grew,
    and its vote was worthless -- and every column on the page looked fine.

    A member that answered nothing is absent from the map. An id with no
    stamp cannot be aged at all, so it is skipped rather than counted fresh.
    """
    moment = time.time() if now is None else now
    found: dict[str, float] = {}
    for reaction in reactions or []:
        if not isinstance(reaction, dict):
            continue
        node = str(reaction.get("node_id") or "").strip()
        age = proposal_age_seconds(reaction.get("target_id"), moment)
        if not node or age is None:
            continue
        if node not in found or age < found[node]:
            found[node] = age
    return found


def build_people(participants: list[dict], ledgers: list[dict],
                 messages: list[dict], reactions: list[dict] | None = None,
                 now: float | None = None) -> list[dict]:
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
            "capability": {}, "stats": {}, "pubkey": "", "chat": None,
            "role": "", "activity": None, "seen": None, "vote_age_s": None,
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
        row["chat"] = normalize_chat(part.get("chat"))
        if not row["role"]:
            row["role"] = str(part.get("role") or "")
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
            "error": ledger.get("error") or "",
        }
        if not row["lineage"]:
            row["lineage"] = row["activity"]["served_by"]
        if not row["host"] and ledger.get("address"):
            row["host"] = str(ledger["address"])
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
    # A vote from a name with no row of its own is not shown: this panel is a
    # roll of members, and inventing a member out of one stance would make an
    # unregistered reactor look admitted.
    for name, age in newest_vote_age(reactions or [], now).items():
        row = index.get(name)
        if row is not None:
            row["vote_age_s"] = age
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
        "destination": str(payload.get("destination") or ""),
        "cost": payload.get("cost"),
        "cites": [str(item) for item in cites if item not in (None, "")],
        "signed": bool(record.get("sig") and record.get("pubkey")),
        "pubkey": _short(str(record.get("pubkey") or ""), 16) or "",
    }




def read_execution_ledger(path: Path) -> dict[str, dict]:
    """Last ledger row for each proposal. Missing file means nothing was recorded."""
    found: dict[str, dict] = {}
    try:
        lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        proposal_id = str(row.get("proposal_id") or "")
        if not proposal_id:
            continue
        found[proposal_id] = {
            "action": str(row.get("action") or ""),
            "executed": bool(row.get("executed")),
            "skipped": str(row.get("skipped") or ""),
            "failed": str(row.get("failed") or ""),
            "applied": row.get("applied") or {},
            "destination": str(row.get("destination") or ""),
            "sealed": bool(row.get("sealed")),
        }
    return found


def execution_text(row: dict | None) -> str:
    """The ledger's own words. No extra label."""
    if not row:
        return ""
    if row.get("sealed"):
        return "sealed"
    applied = row.get("applied") or {}
    if isinstance(applied, dict) and applied:
        # A rule change is the interesting half of the row: "executed" alone
        # says something happened, not what the gate now does.
        return "applied " + ", ".join(
            f"{name}={(applied[name] or {}).get('to')}"
            for name in sorted(applied))
    if row.get("executed"):
        return "executed"
    skipped = str(row.get("skipped") or "")
    destination = str(row.get("destination") or "")
    if skipped and destination:
        return f"{skipped} {destination}"
    if skipped:
        return skipped
    failed = str(row.get("failed") or "")
    if failed:
        return f"failed: {failed}"
    return ""


def attach_executions(messages: list[dict], executions: dict[str, dict]) -> None:
    for message in messages:
        proposal = message.get("fields") or message.get("proposal") or {}
        target = str(proposal.get("proposal_id") or "")
        if target and target in executions:
            message["execution"] = executions[target]


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



def _dash(value) -> str:
    text = str(value or "").strip()
    return text if text else "—"


def _code_cell(head: str, majority: str) -> dict:
    """Show one member's commit next to the one most of the fleet runs.

    Two different honest answers, and they are not the same finding: a member
    that names a commit other than the fleet's is **old** (it can still tell us
    what it runs); a member that names nothing predates the field entirely, so
    we know only that it is **unknown**. Neither is silently rendered as fine.
    """
    head = str(head or "").strip()
    if not head:
        return {"code": "未知", "code_stale": False, "code_unknown": True}
    return {"code": head[:7], "code_stale": bool(majority and head != majority),
            "code_unknown": False}


def _fresh_cell(vote_age_s, role: str) -> dict:
    """How stale a member's newest vote is, in one field with its unit.

    Three honest answers, kept apart on purpose. A number means the member
    answered something that recently. A decision member with no answer at all
    gets ``无票``: it was admitted to vote and never has, which is its own
    finding, not a blank. Anyone who is not here to vote -- a proposer, an
    LLM -- gets a dash, because a proposer has no vote to be late with.
    """
    if isinstance(vote_age_s, (int, float)) and not isinstance(vote_age_s, bool):
        age = max(0.0, float(vote_age_s))
        return {"fresh": f"{age / 3600.0:.1f} 小时",
                "fresh_stale": age > VOTE_STALE_S}
    if str(role or "") == "decision":
        return {"fresh": "无票", "fresh_stale": True}
    return {"fresh": "—", "fresh_stale": False}


def _verdict_cell(verdict: dict | None, role: str) -> dict:
    """The judged state, or an honest blank when there is no judgement.

    A proposer has no vote to be late with, so it gets a dash. A *decision*
    member with no verdict is a gap in the evidence rather than a quiet
    member, and it is labelled as one -- that is the whole point of putting
    the judgement on the page instead of a number to eyeball.
    """
    status = str((verdict or {}).get("status") or "").strip()
    invited_to_vote = str(role or "") == "decision"
    if status in VERDICT_ALARM:
        return {"verdict": VERDICT_TEXT[status], "verdict_bad": True,
                "verdict_status": status}
    if status == "ok":
        return {"verdict": VERDICT_TEXT["ok"], "verdict_bad": False,
                "verdict_status": status}
    if status == "unknown":
        # An LLM that casts no vote is ``unknown`` by design -- the watchdog
        # counts it as unnamed_silent and says so. Printing "未知" for every
        # proposer would put a permanent, meaningless badge on the page and
        # teach the reader to ignore the column.
        if invited_to_vote:
            return {"verdict": VERDICT_TEXT["unknown"], "verdict_bad": False,
                    "verdict_status": status}
        return {"verdict": "—", "verdict_bad": False, "verdict_status": status}
    if invited_to_vote:
        return {"verdict": "无判", "verdict_bad": True, "verdict_status": ""}
    return {"verdict": "—", "verdict_bad": False, "verdict_status": ""}


def _majority_head(rows: list[dict]) -> str:
    """The commit most members report, or "" when nobody reports one."""
    heads = [str((row.get("capability") or {}).get("code_head") or "").strip()
             for row in rows]
    heads = [head for head in heads if head]
    if not heads:
        return ""
    return max(set(heads), key=lambda head: (heads.count(head), head))


def present_people(rows: list[dict],
                   verdicts: dict | None = None) -> list[dict]:
    """One display string per column. The page does not assemble these."""
    majority = _majority_head(rows)
    verdicts = verdicts or {}
    presented = []
    for row in rows:
        roots = [str(item) for item in (row.get("roots") or []) if item]
        activity = row.get("activity") or {}
        seen = row.get("seen") or {}
        stats = row.get("stats") or {}
        if activity.get("round") is not None:
            record = f"r{activity['round']}"
            if activity.get("action"):
                record = f"{activity['action']} · {record}"
        elif stats.get("successes") is not None or stats.get("failures") is not None:
            record = f"票 {stats.get('successes') or 0}/{stats.get('failures') or 0}"
        elif seen.get("count"):
            record = f"{seen['count']} 条"
        else:
            record = "—"
        role = {"decision": "决策", "proposal": "提案"}.get(
            str(row.get("role") or ""))
        if not role and row.get("chat"):
            role = "提案"
        chat = row.get("chat") if isinstance(row.get("chat"), dict) else {}
        model = str(chat.get("model") or "").strip() or row.get("lineage") or activity.get("served_by") or ""
        presented.append({
            "name": _dash(row.get("name")),
            "from": ", ".join(roots) if roots else "—",
            "host": _dash(row.get("host")),
            "model": _dash(model),
            "role": role or "—",
            "record": record,
            "detail": str(activity.get("error") or activity.get("goal") or ""),
            "mark": str(row.get("pubkey") or ""),
            **_code_cell((row.get("capability") or {}).get("code_head"), majority),
            **_fresh_cell(row.get("vote_age_s"), row.get("role")),
            **_verdict_cell(verdicts.get(str(row.get("name") or "")),
                            row.get("role")),
        })
    return presented


def present_messages(messages: list[dict]) -> list[dict]:
    presented = []
    for message in messages:
        fields = message.get("fields") or {}
        card = bool(fields.get("goal") or fields.get("action") or fields.get("expected_effect"))
        goal = str(fields.get("goal") or "").strip()
        effect = str(fields.get("expected_effect") or "").strip()
        created = str(message.get("created_at") or "")
        when = created[5:16].replace("T", " ") if len(created) >= 16 else created
        cites = fields.get("cites") if isinstance(fields.get("cites"), list) else []
        meta = [str(item) for item in cites if item]
        if fields.get("cost") not in (None, ""):
            meta.insert(0, str(fields.get("cost")))
        if fields.get("destination"):
            meta.append(str(fields.get("destination")))
        if fields.get("proposal_id"):
            meta.append(str(fields.get("proposal_id")))
        reactions = []
        for reaction in message.get("reactions") or []:
            reactions.append({
                "text": f"{reaction.get('stance') or ''} {reaction.get('node_id') or ''}".strip(),
                "title": str(reaction.get("checkpoint") or ""),
            })
        address = str(message.get("to") or "").strip()
        presented.append({
            "who": _dash(fields.get("author") if card else message.get("from")),
            "to": f"@{address}" if address else "",
            "kind": str(message.get("kind") or ""),
            "when": when,
            "action": str(fields.get("action") or "") if card else "",
            "goal": goal if card and goal and goal != effect else "",
            "text": effect if card and effect else (goal if card else str(message.get("body") or "").strip()),
            "meta": " · ".join(meta),
            "signed": bool(message.get("signed")),
            "card": card,
            "reactions": reactions,
            "execution": execution_text(message.get("execution")),
        })
    return presented



def _service_group(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("127.0.0.1", "::1", "localhost"):
        return "本机"
    return host or "其他"


def present_services(services: list[dict]) -> list[dict]:
    groups: list[dict] = []
    index: dict[str, dict] = {}
    for row in services:
        name = str(row.get("group") or "其他")
        bucket = index.get(name)
        if bucket is None:
            bucket = {"name": name, "rows": []}
            index[name] = bucket
            groups.append(bucket)
        parts = []
        if row.get("status"):
            parts.append(str(row.get("status")))
        if row.get("latency_ms") is not None:
            parts.append(f"{row.get('latency_ms')} ms")
        bucket["rows"].append({
            "name": str(row.get("name") or ""),
            "detail": str(row.get("detail") or ""),
            "alive": bool(row.get("alive")),
            "line": " · ".join(parts) if parts else "—",
        })
    return groups


def present(state: dict) -> dict:
    """The only object the page reads."""
    verdict = state.get("vote_verdict") or {}
    alarms = []
    for name, row in sorted((verdict.get("members") or {}).items()):
        status = str((row or {}).get("status") or "")
        if status in VERDICT_ALARM:
            alarms.append({
                "name": str(name),
                "status": status,
                "label": VERDICT_TEXT.get(status, status),
                "seen": str((row or {}).get("seen") or ""),
                "why": verdict_why(row),
            })
    people = present_people(state.get("people") or [],
                            verdict.get("members") or {})
    messages = present_messages(state.get("chat") or [])
    roots = state.get("roots") or []
    services = state.get("services") or []
    root_up = sum(1 for row in roots if row.get("reachable"))
    service_up = sum(1 for row in services if row.get("alive"))
    counts: dict[str, int] = {}
    kinds: list[str] = []
    for message in messages:
        kind = message["kind"] or "?"
        if kind not in counts:
            kinds.append(kind)
            counts[kind] = 0
        counts[kind] += 1
    quorum = []
    for row in roots:
        name = str(row.get("name") or "")
        if row.get("reachable"):
            quorum.append({"label": name, "value": f"{row.get('members') or 0} 成员"})
        else:
            quorum.append({"label": name, "value": str(row.get("error") or "不可达")})
        snapshot = row.get("quorum") or {}
        if snapshot:
            quorum.append({
                "label": f"{name} quorum",
                "value": f"{snapshot.get('quorum_k', '—')}（{snapshot.get('quorum_floor', '—')}–{snapshot.get('quorum_ceiling', '—')}）",
            })
            if snapshot.get("decision_ewma_ms") is not None:
                quorum.append({
                    "label": f"{name} 决策",
                    "value": f"{round(float(snapshot['decision_ewma_ms']))} ms",
                })
    units = []
    for unit in ((state.get("local_units") or {}).get("services") or []):
        units.append({
            "name": str(unit.get("name") or ""),
            "alive": bool(unit.get("alive")),
            "value": "运行中" + (f" · {unit['pid']}" if unit.get("pid") else "") if unit.get("alive") else "未运行",
        })
    work = []
    for item in reversed((state.get("work") or [])[-WORK_VIEW_LIMIT:]):
        work.append({
            "status": str(item.get("status") or ""),
            "id": str(item.get("id") or ""),
            "title": str(item.get("title") or ""),
            "who": str(item.get("claimed_by") or item.get("from") or ""),
        })
    work_counts = state.get("work_counts") or {}
    open_work = work_counts.get("open")
    if open_work is None:
        open_work = sum(1 for item in (state.get("work") or [])
                        if item.get("status") == "open")
    work_total = work_counts.get("total")
    if work_total is None:
        work_total = len(state.get("work") or [])
    return {
        "headline": f"{root_up}/{len(roots)} root · {len(people)} 人 · {service_up}/{len(services)} 服务",
        "service_count": f"{service_up}/{len(services)}" if services else "",
        "error": " · ".join(state.get("collector", {}).get("errors") or []),
        "verdict_line": verdict_line(verdict, alarms),
        "verdict_available": bool(verdict.get("available")),
        "alarms": alarms,
        "kinds": kinds,
        "kind_counts": " · ".join(f"{counts[kind]} {kind}" for kind in kinds),
        "compose": list(state.get("compose") or []),
        "defaults": state.get("defaults") or {},
        "people": people,
        "services": present_services(services),
        "messages": messages,
        "quorum": quorum,
        "units": units,
        "unit_count": f"{sum(1 for unit in units if unit['alive'])}/{len(units)}" if units else "",
        "work": work,
        "work_count": f"{open_work} open / {work_total}",
    }


def audit_view(view: dict) -> list[str]:
    """Problems a reader would hit. Empty means the columns say one thing each."""
    problems = []
    for row in view.get("people") or []:
        origin = row.get("from") or ""
        if origin != "—" and not all(part and " " not in part.strip() or True for part in origin.split(", ")):
            problems.append(f"from has a phrase: {row.get('name')} {origin}")
        if any(token in origin for token in ("ledger", "条", "chat", "proposal")):
            problems.append(f"from is not a root: {row.get('name')} {origin}")
        if "cpu" in (row.get("host") or "") or "GB" in (row.get("host") or ""):
            problems.append(f"host mixes a spec: {row.get('name')}")
        if "条" in (row.get("model") or ""):
            problems.append(f"model mixes a count: {row.get('name')}")
        record = row.get("record") or ""
        has_unit = ("条" in record or record.startswith("票 ")
                    or re.search(r"(^| )r\d+", record))
        if record != "—" and not has_unit:
            problems.append(f"record has no unit: {row.get('name')} {record}")
        fresh = row.get("fresh") or ""
        if fresh not in ("—", "无票") and "小时" not in fresh:
            problems.append(f"fresh has no unit: {row.get('name')} {fresh}")
        verdict = row.get("verdict") or ""
        if not verdict:
            problems.append(f"verdict is blank: {row.get('name')}")
        elif row.get("role") == "决策" and verdict == "—":
            problems.append(f"decision member has no verdict: {row.get('name')}")
    for message in view.get("messages") or []:
        address = message.get("to") or ""
        if address.startswith("@@") or (address.startswith("@") and address[1:].startswith("@")):
            problems.append(f"double at: {address}")
        text = message.get("text") or ""
        if text.lstrip().startswith("{") or '"payload"' in text or '"sig"' in text:
            problems.append(f"json in text: {message.get('who')}")
    return problems



def participant_from_node(node_id: str, info: dict, root: str) -> dict:
    """One root member, using only fields the node record actually carries."""
    pubkey = info.get("pubkey") or ""
    cap = info.get("capability") or {}
    stats = info.get("stats") or {}
    return {
        "node_id": node_id,
        "root": root,
        "lineage": info.get("checkpoint") or info.get("lineage") or "?",
        "endpoint": info.get("url") or info.get("addr") or "?",
        "host": info.get("host") or "?",
        "role": info.get("role") or "",
        "chat": normalize_chat(info.get("chat")),
        "capability": {
            "platform": cap.get("platform"),
            "arch": cap.get("arch"),
            "cpu_count": cap.get("cpu_count"),
            "ram_gb": cap.get("ram_gb"),
            # Which commit this member is running. Empty means it is old
            # enough to predate the field -- which is itself the finding.
            "code_head": cap.get("code_head") or "",
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
    }


def read_root_snapshot(path: str) -> tuple[dict | None, str | None]:
    """A root table copied onto this machine. Missing or empty is an error."""
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)[:200]
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), dict):
        return None, "snapshot has no nodes"
    return payload, None


def _enrich_member_chat(participants: list[dict]) -> list[dict]:
    """Fill missing member chat metadata without serialising health probes."""
    missing = [part for part in participants
               if not part.get("chat")
               and str(part.get("endpoint") or "").startswith("http")]
    if not missing:
        return participants

    def _health_chat(part: dict) -> tuple[str, dict | None]:
        health, err, _, _ = http_json(
            str(part["endpoint"]).rstrip("/") + "/health", timeout=1.5)
        if err or not isinstance(health, dict):
            return part["node_id"], None
        return part["node_id"], normalize_chat(health.get("chat"))

    with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
        chats = dict(pool.map(_health_chat, missing))
    for part in participants:
        chat = chats.get(part["node_id"])
        if chat:
            part["chat"] = chat
    return participants


def _probe_root(source: dict, fallback_token: str | None) -> dict:
    """Read one independent root into rows the collector can merge in order.

    This function owns every network call for a root, including member health.
    Root workers run concurrently so one unreachable root does not serialise a
    healthy root or its member probes into the same timeout budget.
    """
    root_url = str(source.get("url") or "").rstrip("/")
    snapshot_file = str(source.get("file") or "").strip()
    if not root_url and not snapshot_file:
        return {}
    label = str(source.get("name") or root_url or snapshot_file)
    result = {"summary": None, "participants": [], "services": [], "errors": [],
              "quorum": None}

    if snapshot_file and not root_url:
        payload, snap_err = read_root_snapshot(snapshot_file)
        age = None
        if payload and isinstance(payload.get("mirrored_at"), (int, float)):
            age = max(0.0, time.time() - float(payload["mirrored_at"]))
        stale = age is not None and age > 180
        summary = {"name": label, "url": snapshot_file,
                   "reachable": payload is not None, "members": 0,
                   "error": snap_err, "quorum": None, "playbook": None,
                   "playbook_error": None}
        if stale:
            summary["error"] = f"snapshot {int(age)}s old"
        detail = snap_err or (f"snapshot {int(age)}s old"
                              if age is not None else "snapshot")
        if payload and not snap_err:
            nodes = payload["nodes"]
            participants = []
            for node_id, info in nodes.items():
                if isinstance(info, dict):
                    participants.append(participant_from_node(
                        str(node_id), info, label))
            result["participants"] = _enrich_member_chat(participants)
            summary["members"] = len(nodes)
            detail = f"{len(nodes)} members, snapshot {int(age or 0)}s old"
            if isinstance(payload.get("status"), dict):
                summary["quorum"] = payload["status"]
                result["quorum"] = payload["status"]
        result["services"].append({
            "group": label, "name": "root snapshot", "url": snapshot_file,
            "alive": payload is not None and not stale,
            "detail": detail, "latency_ms": None, "status": None,
        })
        if summary["error"]:
            result["errors"].append(f"{label} snapshot: {summary['error']}")
        result["summary"] = summary
        return result

    pairing = source.get("token") or fallback_token
    summary = {"name": label, "url": root_url, "reachable": False,
               "members": 0, "error": None, "quorum": None,
               "playbook": None, "playbook_error": None}
    cached_failure = _cached_root_failure(root_url)
    if cached_failure:
        failure, age = cached_failure
        detail = f"cached {age:.1f}s ago: {failure}"
        summary["error"] = detail
        summary["cached"] = True
        summary["cache_age_s"] = round(age, 1)
        result["services"].append({
            "group": label,
            "name": "root /nodes", "url": root_url + "/nodes",
            "alive": False, "detail": detail,
            "latency_ms": None, "status": None,
            "cached": True, "cache_age_s": round(age, 1),
        })
        result["errors"].append(f"{label} /nodes: {detail}")
        skipped = "skipped: /nodes unreachable"
        result["services"].append({
            "group": label,
            "name": "root /status", "url": root_url + "/status",
            "alive": False, "detail": skipped,
            "latency_ms": None, "status": None,
        })
        summary["playbook_error"] = skipped
        result["summary"] = summary
        return result

    nodes, nerr, nms, nstatus = http_json(
        f"{root_url}/nodes", token=pairing, header="X-Pairing-Token")
    summary["reachable"] = nerr is None
    summary["error"] = nerr
    result["services"].append({
        "group": label,
        "name": "root /nodes", "url": root_url + "/nodes",
        "alive": nerr is None,
        "detail": nerr or f"{len((nodes or {}).get('nodes', {}))} members",
        "latency_ms": nms, "status": nstatus,
    })
    if nerr:
        _remember_root_failure(root_url, nerr)
        result["errors"].append(f"{label} /nodes: {nerr}")
        skipped = "skipped: /nodes unreachable"
        result["services"].append({
            "group": label,
            "name": "root /status", "url": root_url + "/status",
            "alive": False, "detail": skipped,
            "latency_ms": None, "status": None,
        })
        summary["playbook_error"] = skipped
        result["summary"] = summary
        return result

    _clear_root_failure(root_url)

    participants = []
    for node_id, info in ((nodes or {}).get("nodes") or {}).items():
        if isinstance(info, dict):
            participants.append(participant_from_node(str(node_id), info, label))
    summary["members"] = len((nodes or {}).get("nodes") or {})

    # Member health, root status, and playbook are independent once /nodes
    # answers. Running them together bounds this root by its slowest probe.
    with ThreadPoolExecutor(max_workers=3) as pool:
        status_future = pool.submit(
            http_json, f"{root_url}/status", token=pairing,
            header="X-Pairing-Token")
        playbook_future = pool.submit(
            http_json, f"{root_url}/playbook", token=pairing,
            header="X-Pairing-Token")
        members_future = pool.submit(_enrich_member_chat, participants)
        root_status, serr, sms, sstatus = status_future.result()
        playbook, perr, pms, pstatus = playbook_future.result()
        result["participants"] = members_future.result()

    result["services"].append({
        "group": label,
        "name": "root /status", "url": root_url + "/status",
        "alive": serr is None, "detail": serr or "quorum snapshot",
        "latency_ms": sms, "status": sstatus,
    })
    if serr:
        result["errors"].append(f"{label} /status: {serr}")
    else:
        summary["quorum"] = root_status
        result["quorum"] = root_status
    if perr is None and playbook:
        summary["playbook"] = {
            "name": playbook.get("name"), "version": playbook.get("version"),
            "hash": _short(playbook.get("hash"), 16),
            "questions": len(playbook.get("questions") or []),
            "gating": playbook.get("gating"),
        }
    else:
        summary["playbook_error"] = perr or "empty"
    result["summary"] = summary
    return result


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
        "work_counts": {"total": None, "open": None},
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

    feed_path = (f"/tasks/window?chat_limit={CHAT_VIEW_LIMIT}"
                 f"&work_limit={WORK_VIEW_LIMIT}")
    tasks, terr, tms, feed_status = http_json(
        base + feed_path, token=cfg["mailbox_token"])
    # A mailbox that predates the purpose-built route still gets a working
    # dashboard. Fall back only on 404: retrying a timeout or auth failure
    # with the 9MB legacy feed would make the fault worse.
    if feed_status == 404:
        feed_path = "/tasks?status=all"
        tasks, terr, tms, _ = http_json(
            base + feed_path, token=cfg["mailbox_token"])

    windowed = feed_path.startswith("/tasks/window")
    if windowed:
        valid_window = (isinstance(tasks, dict)
                        and isinstance(tasks.get("chat"), list)
                        and isinstance(tasks.get("work"), list)
                        and isinstance(tasks.get("counts"), dict))
        if terr is None and not valid_window:
            terr = "malformed /tasks/window payload"
        chat_items = list(tasks.get("chat") or []) if valid_window else []
        work_items = list(tasks.get("work") or []) if valid_window else []
        counts = tasks.get("counts") or {} if valid_window else {}
        try:
            chat_total = int(counts["chat"])
            state["work_counts"] = {
                "total": int(counts["work"]),
                "open": int(counts["work_open"]),
            }
        except (KeyError, TypeError, ValueError):
            chat_total = 0
            state["work_counts"] = {"total": None, "open": None}
        if terr is None and any(value is None for value in
                                state["work_counts"].values()):
            terr = "malformed /tasks/window counts"
        detail = (f"{chat_total} kinded / "
                  f"{state['work_counts']['total']} work; window "
                  f"{len(chat_items) + len(work_items)}")
    else:
        items = list((tasks or {}).get("tasks") or [])
        chat_items = []
        work_items = []
        for item in items:
            title = str(item.get("title", ""))
            (chat_items if split_title(title)[0] else work_items).append(item)
        detail = (terr or (f"{len(chat_items) + len(work_items)} items"
                           if tasks else "empty payload"))

    state["services"].append({
        "group": "mailbox",
        "name": ("task feed (/tasks/window)" if windowed
                 else "task feed (/tasks)"),
        "url": base + feed_path.split("?", 1)[0],
        "alive": terr is None,
        "detail": terr or detail,
        "latency_ms": tms, "status": 200 if terr is None else 0,
    })
    if terr:
        state["collector"]["errors"].append(f"mailbox {feed_path}: {terr}")

    for item in chat_items + work_items:
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

    if not windowed:
        state["work_counts"] = {
            "total": len(state["work"]),
            "open": sum(item.get("status") == "open"
                        for item in state["work"]),
        }

    # The /tasks listing carries titles only. Fetch every chat/proposal
    # record in full so the feed never hides the back half of a message.
    # 600+ messages, one body fetch each, measured 23 s per collect. The
    # dashboard shows recent history; whole-feed counts stay intact.
    feed = (state["chat"] + state["reactions"])[-CHAT_VIEW_LIMIT:]
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
        ledger_path = Path(cfg.get("execute_ledger") or (PANDO_DIR / "execute-ledger.jsonl"))
        attach_executions(state["chat"], read_execution_ledger(ledger_path))

    # --- island roots: participants, quorum, playbooks ----------------
    # Roots are independent membership tables. Read every configured root and
    # merge by node_id; the UI keeps the root names so it is obvious where a
    # participant was seen.
    root_sources = list(cfg.get("roots") or [])
    extra_rows = list(cfg.get("extra_services") or cfg.get("services") or [])

    def _probe_extra(row: dict) -> tuple[dict, tuple]:
        url = str((row or {}).get("url") or "")
        return row, http_probe(url, timeout=3.0) if url else (False, "", None, None)

    root_results = []
    extra_results = []
    workers = min(8, max(1, len(root_sources) + len(extra_rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        root_futures = [pool.submit(_probe_root, source,
                                    cfg.get("pairing_token"))
                        for source in root_sources]
        extra_futures = [pool.submit(_probe_extra, row) for row in extra_rows]
        root_results = [future.result() for future in root_futures]
        extra_results = [future.result() for future in extra_futures]

    for result in root_results:
        summary = result.get("summary")
        if not summary:
            continue
        state["participants"].extend(result["participants"])
        state["services"].extend(result["services"])
        state["collector"]["errors"].extend(result["errors"])
        state["roots"].append(summary)
        if state.get("quorum") is None and result.get("quorum") is not None:
            state["quorum"] = result["quorum"]

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
        if target:
            entries, err = read_remote_jsonl_tail(str(target), path, 3)
        else:
            entries, err = read_jsonl_tail(Path(path).expanduser(), 3), None
        state["thinkd"].append({
            "host": name, "ledger": path, "rounds": len(entries),
            "via": f"ssh {target}" if target else "local file",
            "alive": bool(entries),
            "error": err,
            "address": str(source.get("address") or ""),
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
        state["participants"], state["thinkd"], state["chat"],
        state["reactions"])

    # --- optional extra service probes (addresses stay in hosts.json) ---
    for row, (ok, detail, pms, pstatus) in extra_results:
        url = str((row or {}).get("url") or "")
        if not url:
            continue
        name = str(row.get("name") or url)
        state["services"].append({
            "group": str(row.get("group") or _service_group(url)),
            "name": name, "url": url, "alive": ok, "detail": detail,
            "latency_ms": pms, "status": pstatus,
        })

    # --- the vote judgement: read it, do not re-derive it -------------
    verdict_path = cfg.get("vote_verdict")
    state["vote_verdict"] = read_vote_verdict(
        Path(verdict_path) if verdict_path else None)
    verdict = state["vote_verdict"]
    verdict_alarms = [row for row in (verdict.get("members") or {}).values()
                      if str((row or {}).get("status") or "") in VERDICT_ALARM]
    state["services"].append({
        "group": "watchdog",
        "name": "表态判定 (vote_watchdog)",
        "url": verdict["path"],
        "alive": bool(verdict["available"]) and not verdict.get("stale"),
        "detail": verdict.get("error") or verdict_line(verdict, verdict_alarms),
        "latency_ms": None, "status": None,
    })
    if not verdict["available"]:
        state["collector"]["errors"].append(
            f"表态判定不可用: {verdict.get('error') or 'unavailable'}")
    elif verdict.get("stale"):
        state["collector"]["errors"].append(
            f"表态判定已过期 ({verdict.get('age_s') or 0:.0f}s)")

    state["local_units"] = local_units()
    state["collector"]["ms"] = round((time.monotonic() - started) * 1000, 1)
    state["view"] = present(state)
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
                    self._json(200, cached_state(cfg))
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
<title>Pando · observatory</title>
<style>
:root{
  --bg:#17191b; --panel:#22262a; --ink:#e6e8eb; --dim:#a4a8ae; --faint:#6e737a;
  --line:#34383d; --sap:#a3b37a; --bark:#e08a3c; --bad:#e15d4a;
}
*{box-sizing:border-box}
body{
  margin:0; color:var(--ink);
  background:
    radial-gradient(880px 320px at 100% -8%, #2c3034 0%, transparent 60%),
    var(--bg);
  font:15px/1.45 "Iowan Old Style", Palatino, "Songti SC", serif;
}
.mono, td, th, .msg, input, select, textarea, button, h2, .sub, #hdr{
  font-family:ui-monospace, "SF Mono", Menlo, monospace;
}
.wrap{max-width:1480px;margin:0 auto;padding:28px 28px 64px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:18px;
  border-bottom:1px solid var(--line);padding-bottom:14px}
h1{margin:0;font-weight:500;font-size:28px;letter-spacing:-.03em;color:#f7f7f5}
h1 em{font-style:italic;color:var(--bark);font-weight:500}
#hdr{color:var(--dim);font-size:12px}
.sub{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(340px,.9fr);gap:22px;margin-top:22px;align-items:start}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:22px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line)}
.card>h2{margin:0;padding:10px 14px;border-bottom:1px solid var(--line);
  color:var(--dim);font-size:12px;font-weight:500;
  display:flex;justify-content:space-between;align-items:center;gap:10px}
.body{padding:8px 14px 12px}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:var(--faint);font-size:11px;font-weight:500;padding:0 10px 6px 0}
td{padding:8px 10px 8px 0;border-top:1px solid var(--line);vertical-align:top;font-size:12px}
.feed{max-height:min(68vh,780px);overflow:auto}
.msg{padding:12px 14px;border-top:1px solid var(--line)}
.msg.cardish{border-left:2px solid var(--bark);padding-left:12px}
.hd{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.who{color:var(--ink)}
.to{color:var(--bark)}
.kind{color:var(--faint)}
.ts{margin-left:auto;color:var(--faint);font-size:11px}
.goal{margin-top:6px}
.tx{margin-top:4px;white-space:pre-wrap;word-break:break-word;color:#d7dbe0}
.meta{margin-top:6px;color:var(--faint);font-size:11px}
.reacts{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{border:1px solid #3d4632;padding:1px 6px;color:var(--sap);font-size:12px}
.svc{display:flex;gap:8px;align-items:baseline;padding:5px 0;border-top:1px solid var(--line)}
.svc:first-of-type{border-top:0}
.dot{width:7px;height:7px;border-radius:50%;background:var(--bad);flex:none}
.dot.on{background:var(--sap)}
.nm{flex:1;min-width:0}
.nm span{color:var(--dim)}
.ms{color:var(--faint);font-size:11px}
.grp{color:var(--faint);font-size:11px;margin:10px 0 2px}
.k{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;font-size:12px}
.k dt{color:var(--faint)} .k dd{margin:0;text-align:right}
.composer{border-top:1px solid var(--line);padding:10px 12px;background:#1c1f22}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
input,select,textarea,button{font:12px/1.4 ui-monospace,Menlo,monospace;color:var(--ink);
  background:#141618;border:1px solid var(--line);padding:6px 8px}
textarea{width:100%;min-height:64px;resize:vertical}
button{background:#2c3034;border-color:#4a4f55;color:#e6e8eb;cursor:pointer}
button:disabled{opacity:.5}
.lower{display:grid;grid-template-columns:1.2fr .8fr .8fr;gap:22px;margin-top:22px}
@media(max-width:980px){.lower{grid-template-columns:1fr}}
.empty{color:var(--faint);padding:12px 14px}
.banner{border:1px solid #6a3030;color:#f0b2a8;padding:8px 12px;margin-top:14px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.ok{color:var(--sap)} .bad{color:var(--bad)}
@media (prefers-reduced-motion: no-preference){
  .dot.on{animation:tick 4.8s steps(2) infinite}
  @keyframes tick{50%{opacity:.35}}
}
</style></head>
<body><div class="wrap">
<header>
  <h1>Pando <em>observatory</em></h1>
  <div id="hdr">…</div>
</header>
<div id="errbox"></div>
<div id="alarmbox"></div>
<div class="grid">
  <div class="card">
    <h2>对话 <span class="sub" id="cc"></span>
      <select id="filter"><option value="all">全部</option></select></h2>
    <div class="feed" id="chat"></div>
    <div class="composer">
      <div class="row">
        <select id="kind"></select>
        <label class="sub">发往</label><input id="recipient" size="10">
        <label class="sub">署名</label><input id="sender" size="12">
        <span class="sub" id="sendhint"></span>
      </div>
      <textarea id="text" placeholder="写在这里"></textarea>
      <div class="row" style="margin:8px 0 0"><button id="send">发送</button></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>参与者 <span class="sub" id="pc"></span> <span class="sub" id="vl"></span></h2>
      <div class="body" id="participants"></div>
    </div>
    <div class="card">
      <h2>服务 <span class="sub" id="sc"></span></h2>
      <div class="body" id="services"></div>
    </div>
  </div>
</div>
<div class="lower">
  <div class="card"><h2>任务 <span class="sub" id="wc"></span></h2><div class="body" id="work"></div></div>
  <div class="card"><h2>共识</h2><div class="body"><dl class="k" id="quorum"></dl></div></div>
  <div class="card"><h2>本机 <span class="sub" id="lcc"></span></h2><div class="body"><dl class="k" id="local"></dl></div></div>
</div>
</div>
<script>
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function peopleTable(rows){
  if (!rows.length) return '<div class="empty">没有人。</div>';
  return `<table><thead><tr><th>名字</th><th>角色</th><th>来自</th><th>主机</th><th>模型</th><th>代码</th><th>表态龄</th><th>表态</th><th>记录</th></tr></thead><tbody>`
    + rows.map(p => `<tr>
      <td${p.mark ? ` title="${esc(p.mark)}"` : ''}>${esc(p.name)}</td>
      <td>${esc(p.role)}</td>
      <td>${esc(p.from)}</td>
      <td>${esc(p.host)}</td>
      <td>${esc(p.model)}</td>
      <td class="mono">${esc(p.code)}${p.code_stale ? ' <span class="bad">旧</span>' : ''}</td>
      <td>${esc(p.fresh)}${p.fresh_stale ? ' <span class="bad">旧</span>' : ''}</td>
      <td class="mono">${p.verdict_bad ? `<span class="bad">${esc(p.verdict)}</span>` : esc(p.verdict)}</td>
      <td>${esc(p.record)}${p.detail ? `<div class="sub">${esc(p.detail)}</div>` : ''}</td>
    </tr>`).join('') + '</tbody></table>';
}
function messageBlock(m){
  const reacts = (m.reactions || []).map(r =>
    `<span class="chip"${r.title ? ` title="${esc(r.title)}"` : ''}>${esc(r.text)}</span>`).join('');
  return `<div class="msg${m.card ? ' cardish' : ''}">
    <div class="hd"><span class="who">${esc(m.who)}</span>
      ${m.to ? `<span class="to">${esc(m.to)}</span>` : ''}
      ${m.action ? `<span class="kind">${esc(m.action)}</span>` : ''}
      ${m.kind ? `<span class="kind">${esc(m.kind)}</span>` : ''}
      ${m.signed ? '<span class="kind">已签名</span>' : ''}
      <span class="ts">${esc(m.when)}</span></div>
    ${m.goal ? `<div class="goal">${esc(m.goal)}</div>` : ''}
    ${m.text ? `<div class="tx">${esc(m.text)}</div>` : ''}
    ${m.meta ? `<div class="meta">${esc(m.meta)}</div>` : ''}
    ${reacts ? `<div class="reacts">${reacts}</div>` : ''}
    ${m.execution ? `<div class="meta">${esc(m.execution)}</div>` : ''}
  </div>`;
}
async function refresh(){
  let st;
  try { st = await (await fetch('/api/state', {cache:'no-store'})).json(); }
  catch(e){ $('#hdr').textContent = '面板后端不可达'; return; }
  const v = st.view || {};
  if (!window.__init){
    window.__init = true;
    const rows = v.compose || [];
    window.__signedKinds = rows.filter(r => r.signed).map(r => r.kind);
    $('#kind').innerHTML = rows.map(r => `<option value="${esc(r.kind)}">${esc(r.kind)}</option>`).join('');
    $('#sender').value = (v.defaults || {}).sender || '';
    $('#recipient').value = (v.defaults || {}).recipient || '';
    $('#kind').onchange();
  }
  $('#hdr').textContent = v.headline || '';
  $('#errbox').innerHTML = v.error ? `<div class="banner">${esc(v.error)}</div>` : '';
  $('#vl').textContent = v.verdict_line || '';
  const alarms = v.alarms || [];
  $('#alarmbox').innerHTML = alarms.length ? `<div class="banner">${
    alarms.map(a => `${esc(a.name)} <b>${esc(a.label)}</b>${a.seen === 'ongoing' ? '（持续）' : ''} — ${esc(a.why)}`).join('<br>')
  }</div>` : '';
  const people = v.people || [];
  $('#pc').textContent = people.length ? String(people.length) : '';
  $('#participants').innerHTML = peopleTable(people);
  const groups = v.services || [];
  $('#sc').textContent = v.service_count || '';
  $('#services').innerHTML = groups.map(g =>
    `<div class="grp">${esc(g.name)}</div>` + g.rows.map(r =>
      `<div class="svc"><span class="dot${r.alive ? ' on' : ''}"></span><span class="nm">${esc(r.name)} <span>${esc(r.detail)}</span></span><span class="ms">${esc(r.line)}</span></div>`
    ).join('')).join('') || '<div class="empty">没有服务。</div>';
  const messages = v.messages || [];
  const kinds = v.kinds || [];
  if ($('#filter').dataset.kinds !== kinds.join(',')){
    const cur = $('#filter').value || 'all';
    $('#filter').innerHTML = '<option value="all">全部</option>' + kinds.map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
    $('#filter').value = [...$('#filter').options].some(o => o.value === cur) ? cur : 'all';
    $('#filter').dataset.kinds = kinds.join(',');
  }
  const want = $('#filter').value;
  const shown = want === 'all' ? messages : messages.filter(m => m.kind === want);
  $('#cc').textContent = v.kind_counts || '';
  const feed = $('#chat');
  const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 48 || !feed.dataset.ready;
  feed.innerHTML = shown.length ? shown.map(messageBlock).join('') : '<div class="empty">没有消息。</div>';
  feed.dataset.ready = '1';
  if (stick) feed.scrollTop = feed.scrollHeight;
  const quorum = v.quorum || [];
  $('#quorum').innerHTML = quorum.length ? quorum.map(r => `<dt>${esc(r.label)}</dt><dd>${esc(r.value)}</dd>`).join('') : '<dt>—</dt><dd>没有 root</dd>';
  const units = v.units || [];
  $('#lcc').textContent = v.unit_count || '';
  $('#local').innerHTML = units.length ? units.map(u => `<dt>${esc(u.name)}</dt><dd class="${u.alive ? 'ok' : 'bad'}">${esc(u.value)}</dd>`).join('') : '<dt>—</dt><dd>无记录</dd>';
  const work = v.work || [];
  $('#wc').textContent = v.work_count || '';
  $('#work').innerHTML = work.length ? `<table><tbody>${work.map(w =>
    `<tr><td>${esc(w.status)}</td><td>${esc(w.id)}</td><td>${esc(w.title)}</td><td class="sub">${esc(w.who)}</td></tr>`).join('')}</tbody></table>`
    : '<div class="empty">没有任务。</div>';
}
$('#kind').onchange = () => {
  const signed = new Set(window.__signedKinds || []);
  $('#sendhint').textContent = signed.has($('#kind').value) ? '签名后发布' : '';
};
$('#send').onclick = async () => {
  const text = $('#text').value.trim();
  if (!text) return;
  const btn = $('#send'); btn.disabled = true; $('#sendhint').textContent = '发送中…';
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, from: $('#sender').value, to: $('#recipient').value, kind: $('#kind').value})});
    const j = await r.json();
    $('#sendhint').textContent = j.ok ? `已发送 ${j.id || ''}` : ('失败: ' + (j.error || r.status));
    if (j.ok){ $('#text').value = ''; refresh(); }
  } catch(e){ $('#sendhint').textContent = '失败: ' + e; }
  btn.disabled = false;
};
$('#filter').onchange = refresh;
$('#text').addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') $('#send').click(); });
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
