#!/usr/bin/env python3
"""Keep a node's checkout on main without asking a cloud agent to run git.

The Windows node has no cloud operator any more. Its logon supervisor keeps
local processes alive, but nothing there fetches code, so a pushed commit sits
unused until a human logs in with a shell. This is the missing local half:
every interval it fast-forwards the checkout -- GitHub first, the Mac's
``/code.bundle`` peer when GitHub cannot be reached -- and then records what it
did in a ledger a peer can read.

What happens after a fast-forward is explicit, never implied:

  none     record the new HEAD and change nothing (default)
  clients  stop the client processes so the logon supervisor respawns them;
           a model server is never touched
  node     run the full node restart script, models included

``node`` is used automatically when the update changed the supervisor's own
roster file: a supervisor that is already running cannot learn a new service
name, so only a restart of the whole stack can add one.

A tree with local edits, a diverged history, or a failed fetch is left exactly
as it was. No dependency is installed and no model is started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.code_sync import sync_repo  # noqa: E402


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _append(ledger: Path | None, row: dict) -> None:
    if ledger is None:
        return
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _spawn(command: str) -> dict:
    """Start a restart command and do not wait for it.

    Both restart commands stop the process table this watcher lives in, so
    waiting would mean never writing the ledger line that explains the
    restart. Fire, record, exit.
    """
    if not command:
        return {"ran": False, "reason": "no-command"}
    kwargs: dict = {}
    if os.name == "nt":  # stay alive when the parent tree is killed
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        process = subprocess.Popen(command, shell=True, **kwargs)
    except OSError as exc:
        return {"ran": False, "reason": f"spawn-failed: {exc}"}
    return {"ran": True, "pid": process.pid}


def watch_round(*, repo, remote="origin", peer="", peer_token="",
                on_update="none", client_cmd="", node_cmd="",
                supervisor="", ledger=None) -> dict:
    """One sync. Restart only if the fast-forward actually moved HEAD."""
    repo = Path(repo).expanduser()
    supervisor_path = Path(supervisor).expanduser() if supervisor else None
    roster_before = _sha256(supervisor_path) if supervisor_path else ""
    result = sync_repo(repo, remote=remote, peer=peer, peer_token=peer_token)
    roster_after = _sha256(supervisor_path) if supervisor_path else ""
    roster_changed = bool(roster_before) and roster_before != roster_after

    row = dict(result)
    row["at"] = round(time.time(), 3)
    row["roster_changed"] = roster_changed
    row["action"] = "none"
    if result.get("updated"):
        if on_update == "none":
            row["action"] = "recorded"
        else:
            use_node = roster_changed or on_update == "node"
            row["action"] = "node-restart" if use_node else "client-restart"
            row["command"] = _spawn(node_cmd if use_node else client_cmd)
    _append(ledger, row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--peer", default="",
                        help="used only when --remote cannot be fetched")
    parser.add_argument("--peer-token-file", default="")
    parser.add_argument("--on-update", default="none",
                        choices=("none", "clients", "node"))
    parser.add_argument("--client-cmd", default="",
                        help="stop the clients; the supervisor respawns them")
    parser.add_argument("--node-cmd", default="",
                        help="full node restart, models included")
    parser.add_argument("--supervisor", default="",
                        help="roster file; a change in it forces a node restart")
    parser.add_argument("--ledger", default="")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    token = ""
    if args.peer_token_file:
        token = Path(args.peer_token_file).expanduser().read_text(
            encoding="utf-8").strip()
    ledger = Path(args.ledger).expanduser() if args.ledger else None

    def run(announce: bool) -> int:
        row = watch_round(repo=args.repo, remote=args.remote, peer=args.peer,
                          peer_token=token, on_update=args.on_update,
                          client_cmd=args.client_cmd, node_cmd=args.node_cmd,
                          supervisor=args.supervisor, ledger=ledger)
        # An idle round is the normal case; only say something when it was not.
        if announce or row.get("updated") or row.get("reason") != "already":
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.once:
        return run(True)
    while True:
        try:
            run(False)
        except Exception as exc:  # noqa: BLE001 - a bad round must not kill it
            print(json.dumps({"error": str(exc)[:200]}, ensure_ascii=False),
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
