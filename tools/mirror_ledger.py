"""Copy a remote ledger onto this machine.

The observatory runs under launchd, and launchd on this Mac cannot open
the LAN. A process that can ssh writes the file locally. The
observatory then reads a normal path. A failed copy leaves the previous
file in place, so one outage does not blank the row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def ssh_cat(host: str, remote: str, timeout: float = 8.0) -> str:
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host,
         f"cat {remote}"],
        capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        err = (done.stderr or "").strip() or f"ssh rc={done.returncode}"
        raise RuntimeError(err[:240])
    return done.stdout


def mirror_once(host: str, remote: str, dest: Path, fetch=ssh_cat) -> dict:
    """Write ``dest`` only when the fetch returns a non-empty ledger."""
    try:
        text = fetch(host, remote)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        return {"wrote": False, "error": str(exc)[:240]}
    if not text.strip():
        return {"wrote": False, "error": "empty"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return {"wrote": True, "bytes": len(text.encode("utf-8"))}


REMOTE_SNAPSHOT = """
import json, os, urllib.request
root = os.environ["PANDO_ROOT"].rstrip("/")
token = open(os.environ["PANDO_TOKEN"], encoding="utf-8").read().strip()
headers = {"X-Pairing-Token": token}

def get(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=4) as resp:
        return json.load(resp)

document = get(root + "/nodes")
nodes = document.get("nodes") if isinstance(document, dict) else None
if not isinstance(nodes, dict) or not nodes:
    raise SystemExit("nodes missing")
for info in nodes.values():
    if not isinstance(info, dict) or info.get("chat"):
        continue
    url = str(info.get("url") or "")
    if not url.startswith("http"):
        continue
    try:
        health = get(url.rstrip("/") + "/health")
    except Exception:
        continue
    chat = health.get("chat") if isinstance(health, dict) else None
    model = chat.get("model") if isinstance(chat, dict) else None
    if isinstance(model, str) and model.strip():
        info["chat"] = {"model": model.strip()}
out = {"nodes": nodes}
try:
    status = get(root + "/status")
except Exception as exc:
    out["status_error"] = str(exc)[:200]
else:
    if isinstance(status, dict):
        out["status"] = status
print(json.dumps(out, ensure_ascii=False))
"""


def ssh_snapshot(host: str, root_url: str, token_path: str, timeout: float = 12.0) -> str:
    """Ask the remote host for its own root table. The copy runs there."""
    command = (
        f"PANDO_ROOT={root_url} PANDO_TOKEN={token_path} python3 -"
    )
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host, command],
        input=REMOTE_SNAPSHOT, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        err = (done.stderr or done.stdout or "").strip() or f"ssh rc={done.returncode}"
        raise RuntimeError(err[:240])
    return done.stdout


def snapshot_once(dest: Path, fetch) -> dict:
    """Write a root snapshot. A failed or empty fetch keeps the previous file."""
    try:
        text = fetch()
        payload = json.loads(text)
    except (OSError, subprocess.SubprocessError, RuntimeError,
            ValueError, json.JSONDecodeError) as exc:
        return {"wrote": False, "error": str(exc)[:240]}
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, dict) or not nodes:
        return {"wrote": False, "error": "no nodes"}
    payload["mirrored_at"] = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)
    return {"wrote": True, "members": len(nodes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote", default="",
                        help="path on the remote host; pass an absolute path, a local shell expands ~")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--interval", type=float, default=0,
                        help="seconds between copies; 0 copies once and exits")
    parser.add_argument("--snapshot", action="store_true",
                        help="copy a root /nodes table instead of a ledger file")
    parser.add_argument("--root-url", default="")
    parser.add_argument("--token-path", default="")
    args = parser.parse_args()
    dest = Path(args.dest).expanduser()
    if args.snapshot and (not args.root_url or not args.token_path):
        raise SystemExit("snapshot needs --root-url and --token-path")
    if not args.snapshot and not args.remote:
        raise SystemExit("ledger copy needs --remote")
    while True:
        if args.snapshot:
            report = snapshot_once(
                dest, lambda: ssh_snapshot(args.host, args.root_url, args.token_path))
        else:
            report = mirror_once(args.host, args.remote, dest)
        print(time.strftime("%H:%M:%S"), report, flush=True)
        if args.interval <= 0:
            return 0 if report.get("wrote") else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
