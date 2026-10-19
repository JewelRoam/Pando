#!/usr/bin/env python3
"""code_drift_check: catch the member that is thinking with last week's code.

KeepAlive answers "did the process exit?". ``think_watchdog`` answers "is it
still producing thoughts?". ``vote_watchdog`` answers "is the thought still
about *now*?". This answers the fourth question, which is the one that cost
the colony a night:

    was the code that produced the thought the code we think is running?

On 2026-10-19 ``win-laya-int8`` answered stale proposals for 35 hours because
its checkout was frozen ~28 commits behind ``main``, and the commit that
ignores stale proposals landed after that freeze. Nobody noticed for four
hours. ``capability.code_head`` exists for exactly this - a member stamps the
commit it booted from - and it has been sitting on the observatory panel since
``dbe78c3`` while nothing *paged* on it. A column nobody is told to look at is
a docstring; that is the defect this file closes.

Why not an age threshold: there is no honest one. A push makes the *whole*
fleet behind for as long as one watcher interval. So being behind is not the
signal - being behind *and staying* behind is. This tool records when it first
saw each member behind, and only raises after ``--min-persist-seconds``. A
member that catches up is forgotten, so a later drift pages again.

What it cannot see: a node whose checkout is current but whose *process* was
started before it. ``code_head`` is stamped at boot, so this reads the running
code, not the tree - which is the point, and is also why "the tree is at main"
is never accepted as evidence here.

Usage:
  python tools/code_drift_check.py --repo . \\
      --root mac http://127.0.0.1:8765 ~/.pandosi/pairing.token \\
      --root win http://192.168.31.100:8790 ~/.pandosi/win-pairing.token --json

Exit codes: 0 = every member is on the tip, 2 = at least one member has been
behind for longer than the grace window, 1 = a root table could not be read or
the local history could not be interrogated (that is our fault, not a
member's, and it must not be dressed up as somebody being wrong).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT = "current"
BEHIND = "behind"
UNKNOWN = "unknown"
DIVERGED = "diverged"

#: Verdicts. Only STALE is somebody's fault; LAG is the grace window doing its
#: job, and is reported so a reader can tell "just pushed" from "stuck".
OK = "ok"
LAG = "lag"
STALE = "stale"
#: The statuses the caller in think_watchdog.py turns into a task.
CODE_STATUSES = (STALE,)

#: Three polls of the 300s watchdog. One interval is the time a watcher on
#: another host needs to notice a push, so a shorter grace would page the whole
#: fleet every time anybody commits.
DEFAULT_MIN_PERSIST_S = 900.0
DEFAULT_TIMEOUT_S = 20.0


def _read_token(path: str | None) -> str | None:
    if not path:
        return None
    try:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def fetch_nodes(url: str, token: str | None, timeout: float = DEFAULT_TIMEOUT_S):
    """The root table at ``url``, or raise. The pairing header, not Bearer."""
    headers = {"X-Pairing-Token": token} if token else {}
    request = urllib.request.Request(url.rstrip("/") + "/nodes", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=30)


def head_of(repo: Path) -> str | None:
    """The commit this checkout is on, or None when it is not a git tree."""
    try:
        proc = _git(repo, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "").strip()
    return text if proc.returncode == 0 and text else None


def resolve(repo: Path, commit: str) -> str | None:
    """The full hash of ``commit`` if this repo knows it, else None.

    A member reporting a commit this machine has never fetched is *not*
    "behind" - it is a member running something this machine cannot describe,
    and saying "behind" about it would be a guess dressed as a measurement.
    """
    commit = str(commit or "").strip()
    if not commit:
        return None
    try:
        proc = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    try:
        proc = _git(repo, "merge-base", "--is-ancestor", older, newer)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def commits_behind(repo: Path, older: str, newer: str) -> int | None:
    try:
        proc = _git(repo, "rev-list", "--count", f"{older}..{newer}")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip())
    except ValueError:
        return None


def classify(repo: Path, tip: str, member_head: str) -> tuple[str, int | None, str]:
    """(raw status, commits behind, why) for one member's reported head."""
    reported = str(member_head or "").strip()
    if not reported:
        return UNKNOWN, None, ("no code_head reported - this member's process "
                               "is old enough to predate the field")
    resolved = resolve(repo, reported)
    if resolved is None:
        return DIVERGED, None, ("reports a commit this checkout does not know; "
                                "it is not a version of this history")
    if resolved == tip:
        return CURRENT, 0, "on the tip"
    if is_ancestor(repo, resolved, tip):
        behind = commits_behind(repo, resolved, tip)
        return BEHIND, behind, f"{behind} commit(s) behind the tip"
    if is_ancestor(repo, tip, resolved):
        return DIVERGED, None, "reports a commit *ahead* of this checkout"
    return DIVERGED, None, "shares no ancestry with this checkout"


def survey(repo: Path, tip: str, roots, timeout: float = DEFAULT_TIMEOUT_S):
    """(rows, unreadable). One pass over every root; never raises for a root."""
    rows: dict[str, dict] = {}
    unreadable: list[dict] = []
    for name, url, token_file in roots:
        try:
            payload = fetch_nodes(url, _read_token(token_file), timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            unreadable.append({"root": name, "url": url, "error": str(exc)[:200]})
            continue
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, dict):
            unreadable.append({"root": name, "url": url,
                               "error": "root table has no nodes"})
            continue
        for node_id, info in nodes.items():
            if not isinstance(info, dict):
                continue
            capability = info.get("capability")
            reported = ""
            if isinstance(capability, dict):
                reported = str(capability.get("code_head") or "")
            status, behind, why = classify(repo, tip, reported)
            rows[str(node_id)] = {
                "member": str(node_id),
                "root": name,
                "reported": reported,
                "head": reported[:12],
                "behind": behind,
                "raw": status,
                "why": why,
                "role": str(info.get("role") or ""),
            }
    return rows, unreadable


def track(state: dict, rows: dict, now: float,
          min_persist_s: float) -> dict:
    """Set each row's verdict, using and updating the first-seen record.

    The clock restarts whenever the member's reported commit *changes*: moving
    to a different commit is progress, and a member that moved but is still
    behind deserves a fresh grace window rather than an inherited one.
    """
    for name, row in rows.items():
        if row["raw"] == CURRENT:
            state.pop(name, None)
            row["status"] = OK
            continue
        entry = state.get(name)
        if not isinstance(entry, dict):
            entry = {"first_seen": now, "head": row["reported"]}
        if str(entry.get("head") or "") != row["reported"]:
            entry["first_seen"] = now
            entry["head"] = row["reported"]
        # ``or now`` would re-date a legitimate 0.0 stamp forever, which is a
        # grace clock that can never expire - a watchdog that never pages. The
        # state file is written by machines and read back here, so read the
        # value as absent only when it is genuinely missing.
        recorded = entry.get("first_seen")
        first_seen = now if recorded is None else float(recorded)
        row["first_seen"] = first_seen
        row["behind_for_s"] = max(0.0, now - first_seen)
        if row["behind_for_s"] >= float(min_persist_s):
            row["status"] = STALE
        else:
            row["status"] = LAG
        state[name] = entry
    return rows


def load_state(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        parsed = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    try:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True),
                        encoding="utf-8")
    except OSError as exc:
        print(f"code-drift-check state not written: {exc}", file=sys.stderr)


def write_snapshot(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    try:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, sort_keys=True),
                        encoding="utf-8")
    except OSError as exc:
        print(f"code-drift-check snapshot not written: {exc}", file=sys.stderr)


def short(head: str) -> str:
    return str(head or "")[:12] or "(none)"


def verdict_line(row: dict) -> str:
    if row["status"] == OK:
        return f"{row['member']}: ok, on {short(row['reported'])}"
    where = f"on {short(row['reported'])}" if row["reported"] else "reports no head"
    tail = (f", behind for {row['behind_for_s'] / 60:.0f}m"
            if "behind_for_s" in row else "")
    return (f"{row['member']}: {row['status'].upper()} {where} - "
            f"{row['why']}{tail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=str(ROOT),
                        help="the checkout whose HEAD is the tip to compare against")
    parser.add_argument("--root", nargs=3, action="append", default=[],
                        metavar=("NAME", "URL", "TOKEN_FILE"),
                        help="a root table to read (repeat for each host)")
    parser.add_argument("--min-persist-seconds", type=float,
                        default=DEFAULT_MIN_PERSIST_S,
                        help="how long a member may stay behind before it pages")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--state-file", default="~/.pandosi/code-drift-state.json",
                        help="where the first-seen record lives (empty=stateless, "
                             "which makes every poll a fresh grace window)")
    parser.add_argument("--snapshot-file",
                        default="~/.pandosi/code-drift-check.json",
                        help="where the verdict is published for other readers")
    parser.add_argument("--json", action="store_true",
                        help="print the payload the caller parses")
    parser.add_argument("--now", type=float, default=None, help="override clock")
    args = parser.parse_args(argv)

    now = time.time() if args.now is None else args.now
    repo = Path(args.repo).expanduser().resolve()

    tip = head_of(repo)
    if tip is None:
        print(f"code-drift-check UNREADABLE: {repo} is not a git checkout",
              file=sys.stderr)
        return 1

    rows, unreadable = survey(repo, tip, args.root, args.timeout)
    state_path = Path(args.state_file) if args.state_file else None
    state = load_state(state_path)
    rows = track(state, rows, now, args.min_persist_seconds)
    save_state(state_path, state)

    stale = sorted(name for name, row in rows.items() if row["status"] == STALE)
    payload = {
        "head": tip,
        "now": now,
        "min_persist_s": float(args.min_persist_seconds),
        "members": rows,
        "unreadable": unreadable,
        "stale": stale,
    }
    write_snapshot(Path(args.snapshot_file) if args.snapshot_file else None, payload)

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        stamp = time.strftime("%F %T %Z", time.localtime(now))
        print(f"{stamp} code-drift-check tip {short(tip)}")
        for name in sorted(rows):
            print(f"  {verdict_line(rows[name])}")
        for bad in unreadable:
            print(f"  {bad['root']}: UNREADABLE - {bad['error']}")

    if unreadable and not stale:
        # A root we cannot read is our fault, and it is *also* the only way a
        # whole host's members vanish from this report. Say so with rc=1 so the
        # caller logs it without paging a human about a member that may be fine.
        print(f"code-drift-check: {len(unreadable)} root(s) unreadable",
              file=sys.stderr)
        return 1
    if stale:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
