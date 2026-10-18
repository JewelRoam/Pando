"""Resident executor loop for a node that is not the Mac.

Every INTERVAL seconds, run `pando.execute --once`: judge the mailbox, seal at
most one adopted seed.export that THIS host signed (the executor refuses
foreign proposals), and exit. Idle rounds are free; a broken mailbox or an
empty queue is recorded, never raised.

The Mac does not need this: launchd already fires `pando.execute --once` on an
interval. A node whose only watcher is a logon supervisor has no such clock, so
this process is the clock -- and on purpose not a timer unit, because the
roster's rule is that no host gains a new scheduled task.

Address resolution is local-only: the host's own `hosts.json` (`mailboxes[0]`),
then the localhost convention. The file is resolved here and handed to the
child with `--hosts`, so the loop and the round can never disagree about which
table describes this host -- a loop that reads one table and an executor that
reads another is how a host drains nobody's backlog and looks idle.

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
    parser.add_argument("--token-file",
                        default="metrics/lan-check/mac-mailbox.token",
                        help="mailbox token, relative to --repo unless absolute. "
                             "The default is the Windows node's path and is kept "
                             "so an already-running roster does not silently "
                             "start failing to find its own token")
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
    return command + list(args.extra)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hosts = host_file(args.hosts)
    url = args.mailbox or mailbox_url(read_hosts(hosts))
    command = round_command(args, url, hosts)
    print(f"execute-loop: every {args.interval:.0f}s against {url} "
          f"(hosts {hosts})", flush=True)
    while True:
        try:
            subprocess.run(command, cwd=str(Path(args.repo).expanduser()),
                           timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 - the loop must never die
            print(f"execute-loop: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
