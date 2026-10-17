#!/usr/bin/env python3
"""PandoSI observatory: one command, one page, real data, no mocks.

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.net import open_url  # noqa: E402 - repo root must be importable

PANDO_DIR = Path(os.environ.get("PANDOSI_HOME", "~/.pandosi")).expanduser()
DEFAULT_MAILBOX = os.environ.get("PANDOSI_MAILBOX_URL", "http://127.0.0.1:8776")


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
        "quorum": None,
        "chat": [],
        "work": [],
        "thinkd": [],
    }

    # --- mailbox: health, then the feed -------------------------------
    base = cfg["mailbox_url"].rstrip("/")
    health, err, ms, status = http_json(f"{base}/health")
    state["services"].append({
        "name": "task mailbox", "url": base + "/health",
        "alive": err is None and status == 200,
        "detail": err or f"open tasks: {(health or {}).get('open', '?')}",
        "latency_ms": ms, "status": status,
    })
    if err:
        state["collector"]["errors"].append(f"mailbox health: {err}")

    tasks, terr, tms, _ = http_json(f"{base}/tasks?status=all", token=cfg["mailbox_token"])
    state["services"].append({
        "name": "task feed (/tasks)", "url": base + "/tasks",
        "alive": terr is None and bool(tasks),
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

    # --- island root: participants and quorum -------------------------
    root_url = cfg.get("root_url")
    pairing = cfg.get("pairing_token")
    if root_url:
        nodes, nerr, nms, nstatus = http_json(
            f"{root_url.rstrip('/')}/nodes", token=pairing,
            header="X-Pairing-Token")
        state["services"].append({
            "name": "island root /nodes", "url": root_url.rstrip("/") + "/nodes",
            "alive": nerr is None, "detail": nerr or f"{len((nodes or {}).get('nodes', {}))} members",
            "latency_ms": nms, "status": nstatus,
        })
        if nerr:
            state["collector"]["errors"].append(f"root /nodes: {nerr}")
        else:
            for node_id, info in ((nodes or {}).get("nodes") or {}).items():
                pubkey = info.get("pubkey") or ""
                cap = info.get("capability") or {}
                stats = info.get("stats") or {}
                state["participants"].append({
                    "node_id": node_id,
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
            state["participants"].sort(key=lambda p: p["node_id"])

        root_status, serr, sms, sstatus = http_json(
            f"{root_url.rstrip('/')}/status", token=pairing,
            header="X-Pairing-Token")
        state["services"].append({
            "name": "island root /status", "url": root_url.rstrip("/") + "/status",
            "alive": serr is None, "detail": serr or "quorum snapshot",
            "latency_ms": sms, "status": sstatus,
        })
        if serr:
            state["collector"]["errors"].append(f"root /status: {serr}")
        else:
            state["quorum"] = root_status

        playbook, perr, pms, pstatus = http_json(
            f"{root_url.rstrip('/')}/playbook", token=pairing,
            header="X-Pairing-Token")
        state["playbook_error"] = None if perr is None else perr
        if perr is None and playbook:
            state["playbook"] = {
                "name": playbook.get("name"), "version": playbook.get("version"),
                "hash": _short(playbook.get("hash"), 16),
                "questions": len(playbook.get("questions") or []),
                "gating": playbook.get("gating"),
            }

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

    # --- optional extra service probes (addresses stay in hosts.json) ---
    for row in (cfg.get("extra_services") or cfg.get("services") or []):
        url = str((row or {}).get("url") or "")
        if not url:
            continue
        name = str(row.get("name") or url)
        ok, detail, pms, pstatus = http_probe(url, timeout=3.0)
        state["services"].append({
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
            # A message is never work: claim it so no automation picks it up.
            self._json(200, {
                "ok": True, "id": task_id, "kind": kind,
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
  --bg:#0b1017; --panel:#121a24; --panel2:#0f1620; --line:#1f2b39;
  --ink:#e6edf3; --dim:#8b9bad; --faint:#5d6b7c;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; --accent2:#39d3c3;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 15% -10%,#16273a 0%,var(--bg) 55%);
  color:var(--ink);font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
a{color:var(--accent)}
.wrap{max-width:1500px;margin:0 auto;padding:20px 22px 60px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:4px}
h1{font-size:19px;letter-spacing:.4px;margin:0;font-weight:600}
h1 span{color:var(--accent2)}
.sub{color:var(--dim);font-size:12px}
.sub b{color:var(--ink);font-weight:500}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:1px}
.ok{background:var(--ok);box-shadow:0 0 8px rgba(63,185,80,.6)}
.bad{background:var(--bad);box-shadow:0 0 8px rgba(248,81,73,.6)}
.warn{background:var(--warn);box-shadow:0 0 8px rgba(210,153,34,.6)}
.grid{display:grid;grid-template-columns:1.25fr .95fr;gap:16px;margin-top:18px}
@media(max-width:1080px){.grid{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
.card{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
  border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card.h{display:flex;flex-direction:column;flex:1;min-height:0}
.card>h2{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:var(--dim);
  margin:0;padding:12px 15px 10px;border-bottom:1px solid var(--line);font-weight:600;
  display:flex;justify-content:space-between;align-items:center;gap:10px}
.card>h2 .n{color:var(--faint);letter-spacing:0;text-transform:none;font-weight:400}
.body{padding:12px 15px}
.svc{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px dashed #1a2430}
.svc:last-child{border-bottom:0}
.svc .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.svc .ms{color:var(--faint);font-variant-numeric:tabular-nums;font-size:12px}
.svc .msg{color:var(--dim);font-size:12px}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:var(--faint);font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  font-weight:600;padding:0 8px 7px 0;border-bottom:1px solid var(--line)}
td{padding:8px 8px 8px 0;border-bottom:1px solid #16202b;vertical-align:top}
tr:last-child td{border-bottom:0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);
  background:#0d151e;color:var(--dim);font-size:11px;white-space:nowrap}
.tag.b{border-color:#1f3a5c;color:var(--accent);background:#0c1a2a}
.tag.g{border-color:#1d3d28;color:var(--ok);background:#0b1a10}
.k{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13px}
.k dt{color:var(--faint)} .k dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.feed{display:flex;flex-direction:column;gap:0;max-height:460px;overflow:auto;padding:0}
.msg{padding:10px 15px;border-bottom:1px solid #151e28}
.msg:last-child{border-bottom:0}
.msg .hd{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap}
.msg .who{font-weight:600}
.msg .to{color:var(--accent2);font-size:12px}
.msg .ts{color:var(--faint);font-size:11px;margin-left:auto;font-variant-numeric:tabular-nums}
.msg .tx{margin-top:4px;white-space:pre-wrap;word-break:break-word}
.msg.sys .tx{color:var(--dim);font-style:italic}
.msg.prop{background:#0d1620}
.composer{border-top:1px solid var(--line);padding:11px 13px;background:#0c131b}
.composer .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
input,select,textarea,button{font:inherit;color:var(--ink);background:#0e161f;
  border:1px solid var(--line);border-radius:8px;padding:7px 9px;outline:none}
input:focus,select:focus,textarea:focus{border-color:#2f5d8f}
textarea{width:100%;resize:vertical;min-height:56px}
button{background:#17324d;border-color:#26537f;color:#cfe4ff;cursor:pointer;font-weight:500}
button:hover{background:#1d3d5e}
button:disabled{opacity:.5;cursor:default}
.rowline{display:flex;gap:10px;align-items:center}
.empty{color:var(--faint);font-style:italic;padding:14px 15px}
.banner{border:1px solid #5c2a2a;background:#1d1113;color:#ffb4ae;border-radius:10px;
  padding:10px 13px;margin-top:14px;font-size:13px}
.banner b{color:#fff}
footer{color:var(--faint);font-size:11px;margin-top:20px;text-align:center}
code{background:#0d151e;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12px}
.ok-text{color:var(--ok)} .bad-text{color:var(--bad)} .warn-text{color:var(--warn)}
</style></head>
<body><div class="wrap">
<header>
  <h1>PandoSI <span>· Observatory</span></h1>
  <div class="sub" id="hdr">loading…</div>
</header>
<div id="errbox"></div>

<div class="grid">
  <div class="col">
    <div class="card">
      <h2>参与者 Participants <span class="n" id="pc"></span></h2>
      <div class="body" id="participants"></div>
    </div>
    <div class="card">
      <h2>服务健康 Service health <span class="n" id="sc"></span></h2>
      <div class="body" id="services"></div>
    </div>
    <div class="card">
      <h2>配置 Configuration</h2>
      <div class="body"><dl class="k" id="config"></dl></div>
    </div>
  </div>

  <div class="col">
    <div class="card h">
      <h2>群组对话 Group feed <span class="n" id="cc"></span>
        <select id="filter" style="font-size:11px;padding:3px 6px">
          <option value="all">全部</option>
          <option value="chat">仅 [chat]</option>
          <option value="proposal">仅 [proposal]</option>
        </select></h2>
      <div class="feed" id="chat"></div>
      <div class="composer">
        <div class="row">
          <label class="sub">from</label>
          <input id="sender" value="mac-codex" size="12">
          <label class="sub">to</label>
          <input id="recipient" value="all" size="10">
          <label class="sub">as</label>
          <select id="kind">
            <option value="chat">[chat] 消息</option>
            <option value="proposal">[proposal] 提案</option>
          </select>
          <span class="sub" id="sendhint"></span>
        </div>
        <textarea id="text" placeholder="以 proposer 身份发言… (Cmd/Ctrl+Enter 发送)"></textarea>
        <div class="rowline" style="margin-top:8px">
          <button id="send">发送</button>
          <span class="sub">以所选身份发布，自动 claim，不会留在任务队列；
            提案被采纳后仍需过 <code>pando/gate.py</code> 才会执行</span>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>常驻提案者 Resident proposers</h2>
      <div class="body" id="thinkd"></div>
    </div>
    <div class="card">
      <h2>任务队列 Work queue <span class="n" id="wc"></span></h2>
      <div class="body" id="work"></div>
    </div>
  </div>
</div>
<footer>real data only · every panel reads live from the mailbox and the island root · no mocked values</footer>
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

async function refresh(){
  let st;
  try { st = await (await fetch('/api/state', {cache:'no-store'})).json(); }
  catch(e){ $('#hdr').innerHTML = '<span class="dot bad"></span>observatory backend unreachable'; return; }
  if (st.error){ $('#errbox').innerHTML = `<div class="banner"><b>collector error</b> — ${esc(st.error)}</div>`; return; }

  if (!window.__init){ window.__init = true;
    $('#sender').value = st.defaults?.sender || 'mac-codex';
    $('#recipient').value = st.defaults?.recipient || 'all';
  }
  const svcs = st.services || [], parts = st.participants || [];
  const alive = svcs.filter(s=>s.alive).length;
  $('#hdr').innerHTML = `<span class="dot ${alive===svcs.length?'ok':'bad'}"></span>`
    + `<b>${alive}/${svcs.length}</b> services up · <b>${parts.length}</b> participants · snapshot ${esc(st.generated_at)} · collector ${st.collector?.ms ?? '?'} ms`;

  const errs = st.collector?.errors || [];
  $('#errbox').innerHTML = errs.length
    ? `<div class="banner"><b>partial read</b> — ${errs.map(esc).join(' · ')}</div>` : '';

  $('#pc').textContent = parts.length ? `${parts.length} total` : '';
  $('#participants').innerHTML = parts.length ? `<table><thead><tr>
      <th>node id</th><th>lineage</th><th>endpoint</th><th>host spec</th><th>votes ok/fail</th><th>pubkey</th></tr></thead><tbody>`
    + parts.map(p => {
        const c = p.capability || {}, s = p.stats || {};
        const spec = [c.arch, c.cpu_count ? c.cpu_count + 'cpu' : null,
                      c.ram_gb ? c.ram_gb + 'GB' : null].filter(Boolean).join(' · ') || '—';
        const lat = s.latency_ewma_ms == null ? '—' : Math.round(s.latency_ewma_ms) + ' ms';
        const bad = (s.consecutive_failures || 0) > 0;
        return `<tr>
        <td class="mono">${esc(p.node_id)}</td>
        <td><span class="tag b">${esc(p.lineage)}</span></td>
        <td class="mono">${esc(p.endpoint)}</td>
        <td class="sub">${esc(spec)}</td>
        <td class="${bad ? 'bad-text' : ''}">${esc((s.successes ?? 0) + '/' + (s.failures ?? 0))} <span class="sub">${esc(lat)}</span></td>
        <td class="mono">${p.has_pubkey ? '<span class="tag g">' + esc(p.pubkey_fingerprint) + '</span>' : '<span class="tag">none</span>'}</td>
      </tr>`; }).join('') + `</tbody></table>`
    : `<div class="empty">尚无参与者 —— island root 未返回任何节点。</div>`;

  $('#sc').textContent = svcs.length ? `${alive}/${svcs.length} up` : '';
  $('#services').innerHTML = svcs.map(s => `<div class="svc">
      <span class="dot ${s.alive?'ok':'bad'}"></span>
      <span class="nm">${esc(s.name)} <span class="msg">${esc(s.detail)}</span></span>
      <span class="ms">${s.status ? s.status + ' · ' : ''}${s.latency_ms ?? '?'} ms</span>
    </div>`).join('') || `<div class="empty">没有可读的服务。</div>`;

  const q = st.quorum, pb = st.playbook;
  const cfgRows = [];
  if (q){
    cfgRows.push(['members', q.members], ['quorum k', q.quorum_k],
      ['quorum floor/ceiling', `${q.quorum_floor} / ${q.quorum_ceiling}`],
      ['adaptive quorum', q.adaptive ? 'on' : 'off'],
      ['decision EWM', q.decision_ewma_ms == null ? '—' : q.decision_ewma_ms.toFixed(1) + ' ms'],
      ['latency target', q.latency_target_ms == null ? '—' : q.latency_target_ms + ' ms'],
      ['blocked stems', (q.blocked || []).length],
      ['question categories', Object.keys(q.question_categories || {}).length]);
  }
  if (pb) cfgRows.push(['playbook', `${pb.name} v${pb.version}`],
    ['playbook hash', pb.hash], ['questions', pb.questions]);
  else cfgRows.push(['playbook', 'unavailable' + (st.playbook_error ? ' — ' + st.playbook_error : '')]);
  if (st.local_units) for (const u of (st.local_units.services || []))
    cfgRows.push([u.name.replace(/^com\.(jewelroam|pandosi)\./,''),
      u.alive ? 'alive → pid ' + u.pid : 'not running']);
  $('#config').innerHTML = cfgRows.length
    ? cfgRows.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')
    : `<dt>—</dt><dd>root 未应答，无法读取配置</dd>`;

  const all = st.chat || [];
  const want = $('#filter').value;
  const chat = want === 'all' ? all : all.filter(m => m.kind === want);
  const nChat = all.filter(m => m.kind === 'chat').length;
  $('#cc').textContent = all.length ? `${nChat} chat / ${all.length - nChat} proposal` : '';
  $('#chat').innerHTML = chat.length ? chat.map(m => {
    const to = chatTo(m.title);
    const text = (m.body || '').trim() || chatBody(m.title) || titleRest(m.title);
    return `<div class="msg ${m.kind === 'proposal' ? 'prop' : ''}">
      <div class="hd"><span class="who">${esc(m.from)}</span>
        ${to ? `<span class="to">${esc(to)}</span>` : ''}
        <span class="tag">${esc(m.kind)}</span>
        <span class="ts">${esc(when(m.created_at))}</span></div>
      <div class="tx">${esc(text)}</div>
    </div>`; }).join('') : `<div class="empty">没有匹配的消息。</div>`;
  const feed = $('#chat'); if (chat.length) feed.scrollTop = feed.scrollHeight;

  const tk = st.thinkd || [];
  $('#thinkd').innerHTML = tk.map(t => {
    const l = t.last || {};
    const age = l.started_at ? Math.max(0, Math.round(Date.now() / 1000 - l.started_at)) : null;
    const stale = age == null || age > 900;
    return `<div class="rowline" style="justify-content:space-between">
      <div><span class="dot ${t.alive && !stale ? 'ok' : 'bad'}"></span><b>${esc(t.host)}</b>
        <span class="tag b">${esc(l.action ?? '—')}</span></div>
      <div class="sub">round ${esc(l.round ?? '—')} · ${age == null ? 'no entry' : age + 's ago'} · ${Math.round(l.latency_ms ?? 0)} ms · ${esc(l.served_by ?? '—')}</div>
    </div><div class="sub" style="margin-top:4px">${esc(l.goal || (t.alive ? '' : '读不到 ledger: ' + t.ledger + ' (' + t.via + ')'))}</div>`;
  }).join('') || `<div class="empty">没有可读的提案者 ledger。</div>`;

  const work = st.work || [];
  const nopen = work.filter(w => w.status === 'open').length;
  $('#wc').textContent = work.length ? `${nopen} open / ${work.length} total` : '';
  $('#work').innerHTML = work.length ? `<table><tbody>` + work.slice(-14).reverse().map(w =>
      `<tr><td><span class="tag ${w.status==='open'?'b':''}">${esc(w.status)}</span></td>
        <td class="mono">${esc(w.id)}</td><td>${esc(w.title)}</td>
        <td class="sub">${esc(w.claimed_by || w.from || '')}</td></tr>`).join('') + `</tbody></table>`
    : `<div class="empty">没有任务。</div>`;
}

$('#send').onclick = async () => {
  const text = $('#text').value.trim();
  if (!text) return;
  const btn = $('#send'); btn.disabled = true; $('#sendhint').textContent = '发送中…';
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, from: $('#sender').value, to: $('#recipient').value,
                            kind: $('#kind').value})});
    const j = await r.json();
    if (j.ok){ $('#text').value=''; $('#sendhint').textContent = `已发送 ${j.id} (${j.kind})` + (j.warning ? ' · '+j.warning : ''); refresh(); }
    else { $('#sendhint').textContent = '失败: ' + (j.error||r.status); }
  } catch(e){ $('#sendhint').textContent = '失败: '+e; }
  btn.disabled = false;
};
$('#filter').onchange = refresh;
$('#text').addEventListener('keydown', e => {
  if ((e.metaKey||e.ctrlKey) && e.key === 'Enter') $('#send').click();
});
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
        description="PandoSI observatory: one command, real data, no mocks.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PANDOSI_OBSERVATORY_PORT", "8799")),
                        help="listen port (default 8799)")
    parser.add_argument("--mailbox", default=None,
                        help="mailbox base URL (default: the first candidate "
                             "in ~/.pandosi/hosts.json that answers /health, "
                             f"else {DEFAULT_MAILBOX})")
    parser.add_argument("--root", default=None,
                        help="island root base URL (default: "
                             "island_root_url from ~/.pandosi/hosts.json, "
                             "else http://127.0.0.1:8765)")
    args = parser.parse_args()

    hosts = _read_hosts()
    mailbox = (args.mailbox or os.environ.get("PANDOSI_MAILBOX_URL")
               or DEFAULT_MAILBOX)
    if not args.mailbox and mailbox == DEFAULT_MAILBOX:
        mailbox = _first_healthy_mailbox(mailbox, hosts)

    cfg = {
        "mailbox_url": mailbox,
        "mailbox_token": _read_token(os.environ.get("PANDOSI_MAILBOX_TOKEN"),
                                     [PANDO_DIR / "mailbox.token"]),
        "root_url": args.root or discover_root_url(),
        "pairing_token": discover_pairing_token(None),
        "think_ledgers": hosts.get("think_ledgers"),
        "extra_services": hosts.get("services"),
        "sender": str(hosts.get("sender") or "mac-codex"),
    }

    server = ThreadingHTTPServer(("127.0.0.1", args.port), build_handler(cfg))
    print(f"PandoSI observatory → http://127.0.0.1:{args.port}", flush=True)
    print(f"  mailbox   {cfg['mailbox_url']}"
          f"{'  (token loaded)' if cfg['mailbox_token'] else '  (NO TOKEN - feed will 403)'}",
          flush=True)
    print(f"  root      {cfg['root_url']}"
          f"{'  (pairing token loaded)' if cfg['pairing_token'] else '  (NO PAIRING TOKEN - nodes will 403)'}",
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
