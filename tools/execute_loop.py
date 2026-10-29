"""Resident executor loop for a node that is not the Mac.

Every INTERVAL seconds, run `pando.execute --once`: judge the mailbox, seal at
most one adopted seed.export that THIS host signed (the executor refuses
foreign proposals), and exit. Idle rounds are free; a broken mailbox or an
empty queue is recorded, never raised.

The Mac does not need this: launchd already fires `pando.execute --once` on an
interval. A node whose only watcher is a logon supervisor has no such clock, so
this process is the clock -- and on purpose not a timer unit, because the
roster's rule is that no host gains a new scheduled task.

Address resolution is local-only: the host's own `hosts.json` (`mailboxes`),
then the localhost conventions. The file is resolved here and handed to the
child with `--hosts`, so the loop and the round can never disagree about which
table describes this host -- a loop that reads one table and an executor that
reads another is how a host drains nobody's backlog and looks idle.

The mailbox is a *list*, not an address. `mailbox_urls` reuses the same order
the round does (command line, then every entry the table names, then
`--mailbox-fallback`), so a node whose peer moved keeps draining its own
mailbox instead of polling an address that no longer answers. Each extra
shelf is handed down as `--mailbox-fallback`, which `pando.execute` grew on
2026-10-19 for exactly this: before it, a moved Mac address was a silent,
healthy-looking halt on every node that read its queue.

Usage: python tools/execute_loop.py [interval-seconds] [--repo DIR]
                                   [--hosts FILE] [--state-dir DIR] ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # The service starts this file by path (not ``-m tools.execute_loop``), so
    # the repository root is not guaranteed to be importable.  The Windows
    # dispatcher in particular runs it from ``D:\PandoRSI\tools``.
    sys.path.insert(0, str(ROOT))

from pando.grant_delegation import refresh_and_install  # noqa: E402
from pando.grants import default_path as default_grants_path  # noqa: E402

#: Tried in order when `--hosts` is not given: the Mac's directory first, then
#: the shorter one the Pi and the Windows node were set up under.
HOST_FILES = ("~/.pandosi/hosts.json", "~/.pando/hosts.json")


def host_file(explicit: str = "") -> Path | None:
    """The first hosts.json that exists, or the preferred path when none does.

    Returning the preferred path when nothing exists keeps the child's error
    ("missing table") pointing at the place a person would go and write one.
    """
    if explicit:
        return Path(explicit).expanduser()
    candidates = [Path(name).expanduser() for name in HOST_FILES]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def read_hosts(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def mailbox_url(hosts: dict, fallback: str = "http://127.0.0.1:8776") -> str:
    urls = hosts.get("mailboxes") or []
    if isinstance(urls, list) and urls:
        return str(urls[0])
    return fallback


#: The localhost conventions, in the order the connectivity doctrine lists
#: them: the Mac's mailbox port first, the win node's second. Two of them
#: because a node with no table has to find its own door: 8776 is the Mac's
#: convention, 8778 the win node's, and neither is a machine address.
LOCAL_MAILBOXES = ("http://127.0.0.1:8776", "http://127.0.0.1:8778")


def mailbox_urls(hosts: dict, mailbox: str = "", fallbacks=()) -> list[str]:
    """The ordered shelves one round may read, deduplicated.

    Mirrors `pando.execute.executor_mailboxes` -- deliberately, because the
    loop prints the list the round will walk. Two implementations of one order
    is a loop that logs one address while the round drains another.
    """
    listed = hosts.get("mailboxes") if isinstance(hosts, dict) else None
    named = [str(url).strip() for url in listed
             if str(url or "").strip()] if isinstance(listed, list) else []
    primary = str(mailbox or "").strip() or (named[0] if named else "")
    seen: set[str] = set()
    out: list[str] = []
    for url in [primary] + named + [str(u).strip() for u in (fallbacks or [])]:
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out or list(LOCAL_MAILBOXES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("interval", nargs="?", type=float, default=300.0,
                        help="seconds between rounds (default: 300)")
    parser.add_argument("--repo", default=str(ROOT),
                        help="the checkout to run the round from")
    parser.add_argument("--hosts", default="",
                        help="this host's table (default: %s)"
                             % ", ".join(HOST_FILES))
    parser.add_argument("--state-dir", default="",
                        help="passed through as the round's state directory")
    parser.add_argument("--mailbox", default="",
                        help="overrides hosts.json mailboxes[0]")
    parser.add_argument("--mailbox-fallback", action="append", default=[],
                        help="extra shelf to try after the table's own "
                             "(repeatable); passed to the round as "
                             "--mailbox-fallback")
    parser.add_argument("--token-file",
                        default="metrics/lan-check/mac-mailbox.token",
                        help="mailbox token, relative to --repo unless absolute. "
                             "The default is the Windows node's path and is kept "
                             "so an already-running roster does not silently "
                             "start failing to find its own token")
    parser.add_argument("--grant-url", action="append", default=[],
                        help="signed operator delegation shelf to refresh from "
                             "(repeatable); network failure leaves a local, "
                             "unexpired grant in place")
    parser.add_argument("--limit", type=int, default=40,
                        help="rows the round may consider (default: 40)")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="kill a round that hangs (default: 180s)")
    parser.add_argument("--extra", action="append", default=[],
                        help="extra argument for pando.execute; repeatable")
    return parser


def round_command(args: argparse.Namespace, url: str, hosts: Path) -> list[str]:
    """The one round this loop runs, as an argv a person could paste."""
    command = [sys.executable, "-m", "pando.execute", "--once",
               "--mailbox", url,
               "--hosts", str(hosts),
               "--token-file", args.token_file,
               "--limit", str(args.limit)]
    if args.state_dir:
        command += ["--state-dir", args.state_dir]
    for url in getattr(args, "mailbox_fallback", []) or []:
        command += ["--mailbox-fallback", url]
    return command + list(args.extra)


def grants_path(args: argparse.Namespace) -> Path:
    """The same grants ledger the child uses for its state directory."""
    if args.state_dir:
        return Path(args.state_dir).expanduser() / "grants.toml"
    return default_grants_path()


def _repo_path(repo: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else repo / candidate


def refresh_grants(args: argparse.Namespace) -> dict:
    """Refresh the local signed grant, never turning transport into a crash."""
    urls = list(getattr(args, "grant_url", []) or [])
    if not urls:
        return {"status": "disabled", "changed": False}
    repo = Path(args.repo).expanduser()
    token_file = _repo_path(repo, args.token_file)
    try:
        return refresh_and_install(
            urls, token_file=token_file, grants_path=grants_path(args))
    except Exception as exc:  # noqa: BLE001 - this loop is the node's clock
        return {"status": "error", "changed": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def run_round(args: argparse.Namespace, urls: list[str], hosts: Path,
              *, runner=subprocess.run) -> dict:
    """Refresh authorization, then run exactly one executor round.

    The grant refresh is deliberately advisory to the child: a sleeping Mac,
    a dead LAN address, a malformed response, or a local audit error is
    recorded, while an already-installed unexpired grant remains usable.
    """
    try:
        grant = refresh_grants(args)
    except Exception as exc:  # noqa: BLE001 - authorization refresh is advisory
        grant = {"status": "error", "changed": False,
                 "reason": f"{type(exc).__name__}: {exc}"}
    result = {
        "grant_status": grant.get("status"),
        "grant_changed": bool(grant.get("changed")),
    }
    if grant.get("source"):
        result["grant_source"] = grant["source"]
    if grant.get("reason"):
        result["grant_reason"] = str(grant["reason"])[:300]
    try:
        completed = runner(
            round_command(args, urls[0], hosts),
            cwd=str(Path(args.repo).expanduser()), timeout=args.timeout)
        result["executed"] = True
        result["returncode"] = int(getattr(completed, "returncode", 0))
    except Exception as exc:  # noqa: BLE001 - the loop must never die
        result["executed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hosts = host_file(args.hosts)
    urls = mailbox_urls(read_hosts(hosts), args.mailbox, args.mailbox_fallback)
    print(f"execute-loop: every {args.interval:.0f}s against "
          f"{', '.join(urls)} (hosts {hosts})", flush=True)
    while True:
        run_round(args, urls, hosts)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
