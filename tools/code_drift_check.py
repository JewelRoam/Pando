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
      --root win http://192.168.31.100:8790 ~/.pandosi/win-pairing.token \\
      --root-json pi5 ~/.pandosi/remote/pi5-root.json --json

``--root NAME URL TOKEN_FILE`` dials a root table. ``--root-json NAME FILE``
reads one that another process copied here, which is how a host this job may
not dial stays measured instead of merely mentioned. A mirror is evidence
only while it is fresh; see ``load_root_json``.

Exit codes: 0 = every member is on the tip, 2 = at least one member has been
behind for longer than the grace window, 1 = a root table could not be read or
the local history could not be interrogated (that is our fault, not a
member's, and it must not be dressed up as somebody being wrong).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
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
#: A root table that does not answer is measured as nothing at all, and the
#: members behind it silently drop out of the page. On 2026-10-19 the first
#: drift page named only the Mac's own two members because a two-minute LAN
#: blip made win and pi5 unreachable at that instant - the three frozen
#: members the tool existed to name were the ones missing from it. So a
#: transport failure is retried before it is believed. HTTP errors are not
#: retried: a 403 is an answer, not a dropped packet.
DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY_S = 2.0
#: A mirrored root table is a copy of somebody else's answer, so its own age
#: is part of the measurement. tools/mirror_ledger.py refreshes pi5's every
#: 60s; five missed refreshes is where the copy stops being evidence about
#: the host and starts being evidence about the copier.
DEFAULT_ROOT_JSON_MAX_AGE_S = 300.0


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


def fetch_nodes_retrying(url: str, token: str | None,
                         timeout: float = DEFAULT_TIMEOUT_S,
                         retries: int = DEFAULT_RETRIES,
                         delay: float = DEFAULT_RETRY_DELAY_S):
    """``fetch_nodes``, but a dropped packet is not the same as a silent host.

    Retries only transport failures. An ``HTTPError`` is a definitive answer
    (the token is wrong, the path moved) and retrying it three times only makes
    the job slower; a ``URLError`` without a status is usually the wifi.
    ``sleep`` is injected so the tests do not wait in real time.
    """
    attempt = 0
    while True:
        try:
            return fetch_nodes(url, token, timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError):
            if attempt >= max(0, int(retries)):
                raise
            attempt += 1
            if delay > 0:
                time.sleep(delay)


def load_root_json(path: str, now: float,
                   max_age: float = DEFAULT_ROOT_JSON_MAX_AGE_S) -> dict:
    """A root table that reached us as a file, or raise ``ValueError``.

    This exists because macOS 27 gates local-network reachability per
    *responsible process*, and a launchd job led by the Homebrew interpreter
    may not open 192.168.31.0/24 at all - a refusal that follows it into its
    children, so it cannot be worked around inside this tool. A process that
    *can* reach the peer (an ssh from a job led by ``/bin/sh``; see
    tools/mirror_ledger.py) writes the peer's ``/nodes`` here instead.

    The copy is the peer's own answer, so it is as good as the peer for reading
    ``capability.code_head`` - but only while it is fresh. A mirror older than
    ``max_age`` is this machine's blind spot, not a member falling behind, and
    the caller reports it as unreadable rather than paging a member.
    ``max_age <= 0`` reads the file without any freshness claim.
    """
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read mirror: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"mirror is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("mirror is not a root table")
    if max_age > 0:
        stamped = payload.get("mirrored_at")
        if not isinstance(stamped, (int, float)) or isinstance(stamped, bool):
            raise ValueError("mirror carries no mirrored_at; it cannot say how "
                             "old it is")
        age = now - float(stamped)
        if age > max_age:
            raise ValueError(f"mirror is {age / 60:.0f}m old, past the "
                             f"{max_age / 60:.0f}m freshness window")
    return payload


def _read_source(source, now: float, max_age: float, timeout: float,
                 retries: int, retry_delay: float):
    """(name, where, payload, blind). ``blind`` is why we saw nothing."""
    if isinstance(source, dict):
        name = str(source.get("name") or "")
        if source.get("file"):
            path = str(source["file"])
            where = {"path": path}
            try:
                return name, where, load_root_json(path, now, max_age), None
            except (OSError, ValueError) as exc:
                return name, where, {}, str(exc)[:200]
        url = str(source.get("url") or "")
        token_file = source.get("token_file")
        where = {"url": url}
    else:
        name, url, token_file = source
        name, url = str(name), str(url)
        where = {"url": url}
    try:
        payload = fetch_nodes_retrying(url, _read_token(token_file),
                                       timeout, retries, retry_delay)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return name, where, {}, str(exc)[:200]
    return name, where, payload, None


def _clean_git_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment with every GIT_* variable that names a repository gone.

    ``git`` hands a pre-push hook ``GIT_DIR``, and ``git -C <dir>`` obeys that
    variable *instead of* ``-C``. On 2026-10-19 this probe ran from the push
    hook, inherited ``GIT_DIR`` pointing at the real worktree, and answered
    about that worktree no matter which member it was handed - three drift
    tests read the wrong repository and failed. Naming the repo in argv is only
    half a fix; the other half is not handing git a second, contradictory one.
    """
    env = dict(os.environ if environ is None else environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return env


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=30,
                          env=_clean_git_env())


def head_of(repo: Path) -> str | None:
    """The commit this checkout is on, or None when it is not a git tree."""
    try:
        proc = _git(repo, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "").strip()
    return text if proc.returncode == 0 and text else None


def deploy_tip(repo: Path) -> str | None:
    """The commit the fleet can actually pull, falling back to local HEAD.

    A developer may commit locally and wait before pushing. Comparing members
    to that private HEAD starts the fifteen-minute grace window before the
    commit is deployable, so a healthy fleet can be paged for not pulling an
    object that was never published. Prefer the branch upstream (for this
    checkout, ``origin/main``); a repository without an upstream keeps the old
    local-HEAD behaviour.
    """
    try:
        proc = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
                    "@{upstream}")
    except (OSError, subprocess.SubprocessError):
        return head_of(repo)
    upstream = (proc.stdout or "").strip()
    if proc.returncode == 0 and upstream:
        resolved = resolve(repo, upstream)
        if resolved is not None:
            return resolved
    return head_of(repo)


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


def survey(repo: Path, tip: str, roots, timeout: float = DEFAULT_TIMEOUT_S,
           retries: int = DEFAULT_RETRIES,
           retry_delay: float = DEFAULT_RETRY_DELAY_S,
           now: float | None = None,
           max_age: float = DEFAULT_ROOT_JSON_MAX_AGE_S):
    """(rows, unreadable). One pass over every root; never raises for a root.

    Each entry of ``roots`` is either the historical ``(name, url,
    token_file)`` triple or a dict naming either ``url``+``token_file`` or
    ``file``. A root that could not be read lands in ``unreadable`` with the
    reason - never as a member that fell behind.
    """
    rows: dict[str, dict] = {}
    unreadable: list[dict] = []
    stamp = time.time() if now is None else now
    for source in roots:
        name, where, payload, blind = _read_source(
            source, stamp, max_age, timeout, retries, retry_delay)
        if blind is not None:
            unreadable.append({"root": name, **where, "error": blind})
            continue
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, dict):
            unreadable.append({"root": name, **where,
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
    parser.add_argument("--root-json", nargs=2, action="append", default=[],
                        metavar=("NAME", "FILE"),
                        help="a root table that a process which *can* reach "
                             "the host copied here (repeat for each host)")
    parser.add_argument("--root-json-max-age", type=float,
                        default=DEFAULT_ROOT_JSON_MAX_AGE_S,
                        help="how old a --root-json mirror may be before it "
                             "counts as a blind spot instead of evidence "
                             "(0 = make no freshness claim)")
    parser.add_argument("--min-persist-seconds", type=float,
                        default=DEFAULT_MIN_PERSIST_S,
                        help="how long a member may stay behind before it pages")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="extra attempts after a root drops a packet "
                             "(HTTP errors are never retried)")
    parser.add_argument("--retry-delay", type=float,
                        default=DEFAULT_RETRY_DELAY_S,
                        help="seconds between those attempts")
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

    tip = deploy_tip(repo)
    if tip is None:
        print(f"code-drift-check UNREADABLE: {repo} is not a git checkout",
              file=sys.stderr)
        return 1

    sources = [{"name": name, "url": url, "token_file": token_file}
               for name, url, token_file in args.root]
    sources += [{"name": name, "file": path} for name, path in args.root_json]
    rows, unreadable = survey(repo, tip, sources, args.timeout,
                              args.retries, args.retry_delay, now,
                              args.root_json_max_age)
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
