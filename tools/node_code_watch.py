#!/usr/bin/env python3
"""Keep a node's checkout on main without asking a cloud agent to run git.

The Windows node has no cloud operator any more. Its logon supervisor keeps
local processes alive, but nothing there fetches code, so a pushed commit sits
unused until a human logs in with a shell. This is the missing local half:
every interval it fast-forwards the checkout -- GitHub first, the Mac's
``/code.bundle`` peer when GitHub cannot be reached -- and then records what it
did in a ledger a peer can read.

What a round does when the checkout is clean and stands somewhere other
than the head the clients were last restarted onto is the same as what it
does after a fast-forward: stop them and let their supervisors respawn
them. That covers the head which moves without a fetch -- a commit made in
this very checkout, or a hand-applied update -- which ``merge --ff-only``
reports as "already" and which used to leave every client on old code
forever. What happens after an update is explicit, never implied:

  none     record the new HEAD and change nothing (default)
  clients  stop the client processes so whatever supervises them respawns
           them; a model server is never touched
  node     run the full node restart script, models included

``clients`` needs no per-platform command. With ``--client-cmd`` it runs that
string, which is how this was first deployed on Windows; without it -- the
default, and the only form the Mac can use -- it hands the *names* of the
clients to ``code_sync.restart_units``, which picks ``systemctl``,
``launchctl`` or the Windows process table from the running interpreter. One
list of clients, three hosts, no third copy of the deployment to keep in sync.

``node`` is used automatically when the update changed the supervisor's own
roster file: a supervisor that is already running cannot learn a new service
name, so only a restart of the whole stack can add one.

A tree with local edits, a diverged history, or a failed fetch is left exactly
as it was, and nothing is restarted. A tree that does not even compile is
refused for the same reason and in the same way: it is the one failure worth
catching before a restart on a node nobody is watching, because the
supervisor would otherwise respawn a client that cannot start, every ten
seconds, with nothing on the box able to read the failure. No dependency is
installed and no model is started here.
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

from tools.code_sync import CLIENT_UNITS, restart_units, sync_repo  # noqa: E402


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


def _acted_head(ledger: Path | None) -> str:
    """The head the clients were last stopped-and-respawned onto.

    A fetch that changes nothing still has to be reconciled: a Mac that
    commits in place, or a node that was updated by hand, leaves the
    checkout on new code while every supervised client keeps running the
    old interpreter -- ``git merge --ff-only`` would say "already" and
    nothing would ever restart. Recording the head we acted on is what
    makes "already" mean "and the clients are already running it".
    """
    if ledger is None:
        return ""
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        head = str(row.get("acted_head") or "")
        if head:
            return head
    return ""


def _parses(repo: Path, python: str = "", timeout: float = 300.0) -> dict:
    """Does the tree we just moved to still compile?

    A syntax check, not a proof. It is deliberately the cheapest thing that
    is still true: it catches the commit that cannot start at all, and it has
    no side effects, so refusing on it cannot itself damage a node. Import
    errors and bad arguments still get through, and the honest name for that
    is a smaller gate, not a stronger claim.
    """
    command = [python or sys.executable, "-m", "compileall", "-q", str(repo)]
    try:
        got = subprocess.run(command, capture_output=True, text=True,
                             timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": f"check-failed: {exc}"[:200]}
    if got.returncode != 0:
        detail = (got.stderr or got.stdout or "").strip()
        return {"ok": False, "reason": "does-not-parse",
                "error": detail[-600:]}
    return {"ok": True}


def _restart_happened(outcome: dict) -> bool:
    """Did a driver say, in its own words, that it restarted something?

    ``nothing-running`` is not that word. A pattern that matches nothing is
    how a roster goes wrong quietly, and recording it as "the clients are
    already on this head" is exactly the silence this file exists to end.
    """
    return str(outcome.get("reason") or "") == "restarted"


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
                supervisor="", ledger=None, platform="") -> dict:
    """One sync. Restart whenever the running clients are not on HEAD.

    Acting on ``updated`` alone is not enough -- it misses the head that
    moved without a fetch (a commit made in this checkout, or an update
    somebody applied by hand). Acting on every round is too much. So the
    ledger remembers the head the clients were last restarted onto, and a
    round restarts them when the checkout is clean and stands somewhere
    else. A dirty tree or a failed fetch acts on nothing.
    """
    repo = Path(repo).expanduser()
    supervisor_path = Path(supervisor).expanduser() if supervisor else None
    roster_before = _sha256(supervisor_path) if supervisor_path else ""
    result = sync_repo(repo, remote=remote, peer=peer, peer_token=peer_token)
    roster_after = _sha256(supervisor_path) if supervisor_path else ""
    roster_changed = bool(roster_before) and roster_before != roster_after

    head = str(result.get("after") or "")
    clean = result.get("reason") in ("already", "fast-forward")
    behind = ledger is not None and bool(head) and head != _acted_head(ledger)

    row = dict(result)
    row["at"] = round(time.time(), 3)
    row["head"] = head
    row["roster_changed"] = roster_changed
    row["action"] = "none"
    if result.get("updated") and on_update == "none":
        row["action"] = "recorded"
    elif clean and on_update != "none" and (result.get("updated") or behind):
        use_node = roster_changed or on_update == "node"
        row["action"] = "node-restart" if use_node else "client-restart"
        check = _parses(repo)
        if not check["ok"]:
            # Refusing leaves the node on the code that works, and it leaves
            # the round visible: no ``acted_head`` is written, so the next
            # round tries again instead of deciding it already converged.
            row["action"] = "refused"
            row["blocked_by"] = check["reason"]
            row["error"] = check.get("error", "")
        else:
            command = node_cmd if use_node else client_cmd
            if command:
                row["command"] = _spawn(command)
                acted = bool(row["command"].get("ran"))
            else:
                # No shell string: the clients are named, and the platform
                # underneath this process decides what "restart" means. This
                # is the branch the Mac uses and the branch Windows uses.
                row["restart"] = restart_units(list(CLIENT_UNITS),
                                               platform=platform)
                acted = _restart_happened(row["restart"])
            row["acted"] = acted
            if acted:
                row["acted_head"] = head
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
                        help="stop the clients with this command; omit it to "
                             "use the built-in per-platform restart")
    parser.add_argument("--platform", default="",
                        help="service manager to use (default: this host's)")
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
                          supervisor=args.supervisor, ledger=ledger,
                          platform=args.platform)
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
