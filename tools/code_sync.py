#!/usr/bin/env python3
"""Fast-forward a checkout to main. Do not restart a model.

A dirty tree or a history that is not a fast-forward is left untouched.
GitHub is the first remote. ``--peer`` is tried only when that fetch fails,
and it must be the same branch, not a second history.

``--self-source``/``--self-target`` exist for the copy of this file that lives
*outside* the checkout on a headless node. See ``refresh_self``.
``--install-units`` is the same hop one level up: the units are also files
outside the checkout, and nothing rewrote them either. See ``install_units``.

One file, three service managers. The clients are named once, in
``CLIENT_UNITS``, and each name carries a handle per platform: a ``systemd``
unit on Linux, a ``launchd`` label (or a supervised process) on macOS, a
process pattern under the logon supervisor on Windows. Everything above that
table -- the allow-list, the model refusal, the ledger row -- is shared, so a
fix to the guard is a fix on every host instead of a fix on the host that
happened to be edited. See ``platform_id`` and ``restart_units``. ``--install-python`` is the same hop again: a pip-installed
member cannot probe ``git`` for its commit, so the install leaves a stamp
beside the copy. See ``stamp_installed_head``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Client processes only. A model server is not on this list, on purpose.
#
# ``pandosi-react`` belongs here: pi5 installs this checkout into its venv, and
# a fast-forward rewrites that copy on disk. If the consumer is not restarted
# the running process keeps the old module in memory and silently works with
# superseded logic -- the exact drift that left the voter without
# ``--mailbox-fallback`` for four hours after ``a5a7ff5`` landed.
#
# ``pandosi-execute`` belongs here for the same reason and with more at stake:
# a voter that reloads stale code casts a stale vote, but an executor that
# reloads stale code *acts* stale -- it would keep sealing seeds and staging
# drafts through actuators whose contracts this repo has since changed. A
# restart is how the loop's next round picks up the new module.
#
# The three ``pandosi-stem-*`` units belong here, and their absence was a
# measured bug, not a caution. A *stem* is a client: it is the process that
# imports ``pando.graft`` and calls ``capability()``, while ``pandosi-llama``
# is the model server the stem talks to. On 2026-10-19 pi5's checkout and venv
# were at ``9e69dea`` while all three stems had been running since Oct 18
# 16:42 -- so every one of them reported ``code_head: None``, because
# ``10f5043`` added that key to ``graft.capability()`` and the running process
# was holding the previous module in memory. Restarting a stem re-registers it
# with rootd; it does not reload a model and costs no memory.
#
# ``pandosi-llama`` is the exception the boundary is really about: restarting
# it would reload a 2.33 GB model on a 4 GB board.
#
# ``rootd`` is deliberately absent: it registers members at startup only and
# restarting it would drop the live registry for every local stem.
#: The clients, named once, with a handle for each service manager that may
#: be hosting them. A name is what a human and a ledger row say; a handle is
#: what this platform can act on.
#:
#: ``launchd:<label>``  -- a job launchd itself keeps alive (macOS).
#: ``proc:<pattern>``   -- a process something else respawns: launchd
#:                         KeepAlive, the Mac's ``mac-node.sh`` loop, or the
#:                         Windows logon supervisor. Killing it is the whole
#:                         restart; the supervisor is the half that knows how
#:                         to bring it back, and it survives on purpose.
#: ``""``               -- this host does not run that unit. Absent is not the
#:                         same fact as refused, and the ledger says which.
#:
#: A model server appears in no handle anywhere: not ``pandosi-llama.service``
#: on Linux, not ``com.jewelroam.pandosi-llm8b`` on macOS, not
#: ``llama-server.exe`` on Windows. The Windows roster is the one place this
#: list is *wider* than Linux's -- ``pando.rootd`` is stopped there because the
#: supervisor's comment records the measurement that a half-registered
#: registry is worse than a rebuilt one -- so the Mac's ``rootd`` is left
#: alone here: it is started by ``mac-node.sh`` and restarting it would drop
#: the live registry for every local stem, which is the Linux rule too.
CLIENT_UNITS: dict[str, dict[str, str]] = {
    "think": {
        "linux": "pandosi-think.service",
        "darwin": "launchd:com.jewelroam.pandosi-think",
        "windows": "proc",
    },
    "chat": {
        "linux": "pandosi-chat.service",
        "darwin": "launchd:com.jewelroam.pandosi-chat",
        "windows": "proc",
    },
    "react": {
        "linux": "pandosi-react.service",
        "darwin": "launchd:com.jewelroam.pandosi-react",
        "windows": "proc",
    },
    "execute": {
        "linux": "pandosi-execute.service",
        "darwin": "launchd:com.jewelroam.pandosi-execute",
        "windows": "proc",
    },
    "seed-pull": {
        "linux": "pandosi-seed-pull.service",
        "darwin": "",
        "windows": "proc",
    },
    # One pattern for all three stems on macOS: the Mac's text stem is
    # respawned by ``mac-node.sh`` and its two model stems by launchd
    # ``KeepAlive``, so the driver that works for one works for all three, and
    # ``rootd`` -- which is a different process -- is not matched.
    "stem-text": {
        "linux": "pandosi-stem-text.service",
        "darwin": "proc:pando.stemd",
        "windows": "proc",
    },
    "stem-laya": {
        "linux": "pandosi-stem-laya.service",
        "darwin": "proc:pando.stemd",
        "windows": "proc",
    },
    "stem-llm": {
        "linux": "pandosi-stem-llm.service",
        "darwin": "proc:pando.stemd",
        "windows": "proc",
    },
}

#: The Linux handles, kept because callers and tests already speak systemd
#: names and a rename would make a guard silently stop guarding.
SAFE_UNITS = frozenset(handles["linux"] for handles in CLIENT_UNITS.values())

#: What the Windows supervisor respawns. Mirrors the pattern in
#: ``tools/win_restart_clients.ps1``; the two are the same edge and a test
#: pins them together so editing one and not the other fails loudly.
WINDOWS_CLIENT_PATTERN = (
    r"pando\.stemd|win_worker|agent_mailbox|pando\.thinkd|observatory\.py"
    r"|pando\.chatd|pando\.reactd|execute_loop\.py|pando\.rootd"
    r"|seed_pull\.py|node_code_watch\.py"
)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def _git(repo: Path, args: list[str], env: dict[str, str],
         timeout: int = 90) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A stalled fetch must become a failed fetch, so the peer is tried.
        return subprocess.CompletedProcess(
            ["git", "-C", str(repo), *args], 124, "", "timeout")


def _head(repo: Path, env: dict[str, str]) -> str:
    got = _git(repo, ["rev-parse", "HEAD"], env)
    return got.stdout.strip() if got.returncode == 0 else ""


def _dirty(repo: Path, env: dict[str, str]) -> bool:
    """True only when the fast-forward could destroy a person's *tracked* edit.

    ``--untracked-files=no`` is the whole point of this function. Plain
    ``git status --porcelain`` also lists untracked files, but the question
    here is not "is the tree pristine" -- it is "would moving HEAD forward
    throw away work". A stray untracked artifact is not work the fast-forward
    destroys, and counting it turns that artifact into a permanent, silent
    wedge.

    That is not hypothetical. ``node_code_watch`` runs this guard every five
    minutes on the Windows node, and the bootstrap command that installs that
    watcher downloads ``code.bundle`` **into the checkout root**. With
    untracked files counted, every later round would report ``dirty`` and
    never move -- a node whose ledger shows a live, healthy updater, frozen
    forever. The updater failing in its own uniform is the one failure mode
    it exists to make impossible.

    The protection is relocated, not dropped: a merge that would overwrite an
    untracked file is refused by git itself, and that refusal now surfaces as
    ``not-fast-forward`` with git's stderr recorded, never as silence.
    """
    got = _git(repo, ["status", "--porcelain", "--untracked-files=no"], env)
    return bool(got.stdout.strip())


def _download(url: str, token: str, dest: Path) -> bool:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(30 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError):
        return False
    if len(data) > 30 * 1024 * 1024:
        return False
    dest.write_bytes(data)
    return True


def _fetch(repo: Path, env: dict[str, str], remote: str, token: str = "") -> bool:
    """Fetch ``main``. A URL is downloaded as a bundle first.

    ``http.lowSpeedTime`` stops a stalled GitHub fetch instead of leaving
    the timer hung.
    """
    source = remote
    temporary = None
    if remote.startswith(("http://", "https://")):
        handle, name = tempfile.mkstemp(prefix="pandosi-", suffix=".bundle")
        os.close(handle)
        temporary = Path(name)
        if not _download(remote, token, temporary):
            temporary.unlink(missing_ok=True)
            return False
        source = str(temporary)
    try:
        ref = "HEAD" if str(source).endswith(".bundle") else "main"
        got = _git(repo, ["-c", "http.lowSpeedLimit=1000",
                          "-c", "http.lowSpeedTime=20",
                          "fetch", source, ref], env, timeout=25)
        return got.returncode == 0
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_repo(repo, *, remote: str = "origin", peer: str = "",
              peer_token: str = "") -> dict:
    """Fetch ``main`` and fast-forward. Never discards local edits."""
    repo = Path(repo).expanduser()
    env = _env()
    before = _head(repo, env)
    result = {"updated": False, "before": before, "after": before,
              "reason": "no-git", "restarted": []}
    if not (repo / ".git").exists():
        return result
    if _dirty(repo, env):
        result["reason"] = "dirty"
        return result
    if not _fetch(repo, env, remote):
        if not peer or not _fetch(repo, env, peer, peer_token):
            result["reason"] = "fetch-failed"
            return result
        result["via"] = "peer"
    else:
        result["via"] = "origin"
    merged = _git(repo, ["merge", "--ff-only", "FETCH_HEAD"], env)
    after = _head(repo, env)
    result["after"] = after
    if merged.returncode != 0:
        result["reason"] = "not-fast-forward"
        # Keep git's own words. "Refused" and "nothing to do" are different
        # facts, and the ledger is the only place a headless node can say
        # which one happened -- e.g. an untracked file the incoming commit
        # now tracks.
        result["error"] = (merged.stderr or merged.stdout or "").strip()[:300]
        return result
    result["reason"] = "fast-forward" if after != before else "already"
    result["updated"] = after != before
    return result


def platform_id(explicit: str = "") -> str:
    """Which service manager this process should talk to.

    Take the platform from the running interpreter, never from a flag a
    caller passed by hand: ``--restart pandosi-think.service`` is a
    ``systemd`` name, and the same request on a laptop has to become
    ``launchctl kickstart`` without the operator knowing that. ``explicit``
    exists for tests, which cannot change ``sys.platform``.
    """
    if explicit:
        return explicit
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def handles_for(platform: str) -> dict[str, str]:
    """Logical unit name -> the handle ``platform`` can act on, if any."""
    return {name: per.get(platform, "") for name, per in CLIENT_UNITS.items()}


def resolve_units(units: list[str], platform: str) -> tuple[list[str], list[str]]:
    """Resolve names or native handles into handles for ``platform``.

    Accepts both vocabularies on purpose. A pi5 unit file says
    ``pandosi-react.service`` and a Mac ledger says ``think``; refusing either
    would mean one caller had to learn the other's dialect, and the dialect
    is the thing this function exists to hide.

    The lookup runs over *every* platform's handles, not this one's, and that
    is the point: the pi5 unit file says ``pandosi-react.service`` and the Mac
    ledger says ``think``, and both mean the same client. A name that matches
    no platform is refused, which is the only case where "we do not know what
    you mean" and "you may not touch that" can honestly be the same answer --
    ``pandosi-llama.service``, ``com.jewelroam.pandosi-llm8b`` and
    ``llama-server.exe`` all land here.

    Returns ``(handles, refused)``. A unit that is a client but is not
    deployed on this host resolves to *no* handle and is not refused -- "the
    Mac has no seed-pull" is not an attack, it is a Tuesday.
    """
    index: dict[str, list[str]] = {}
    for name, per in CLIENT_UNITS.items():
        for handle in per.values():
            if handle:
                index.setdefault(handle, []).append(name)
    handles: list[str] = []
    refused: list[str] = []
    for unit in units:
        if not unit:
            continue
        names = [unit] if unit in CLIENT_UNITS else index.get(unit, [])
        if not names:
            refused.append(unit)
            continue
        for name in names:
            handle = CLIENT_UNITS[name].get(platform, "")
            if handle and handle not in handles:
                handles.append(handle)
    return handles, refused


def _restart_systemd(handles: list[str], run) -> dict:
    """``systemctl --user restart``, one unit at a time, stopping at the first
    failure so the ledger can name the unit that failed."""
    restarted = []
    for handle in handles:
        got = run(["systemctl", "--user", "restart", handle],
                  capture_output=True, text=True, timeout=30)
        if got.returncode != 0:
            return {"restarted": restarted, "reason": "restart-failed",
                    "unit": handle, "driver": "systemd"}
        restarted.append(handle)
    return {"restarted": restarted, "reason": "restarted", "driver": "systemd"}


def _restart_launchd(handles: list[str], run) -> dict:
    """``launchctl kickstart -k``, or a kill for a supervised process.

    ``kickstart -k`` is launchd's restart: it is the one verb that both stops
    and starts, and it works on a job whose ``KeepAlive`` would otherwise race
    the stop. A ``proc:`` handle is the other half of the Mac's deployment --
    the text stem belongs to ``mac-node.sh`` and not to launchd, so the kill
    *is* the restart and the loop that started it does the respawn.
    """
    uid = getattr(os, "getuid", lambda: 0)()
    restarted: list[str] = []
    killed: list[int] = []
    for handle in handles:
        if handle.startswith("proc:"):
            pattern = handle.split(":", 1)[1]
            pids, err = _pgrep(pattern, run)
            if err:
                return {"restarted": restarted, "reason": "restart-failed",
                        "unit": handle, "driver": "launchd", "error": err}
            for pid in pids:
                armed = run(["kill", "-TERM", str(pid)],
                            capture_output=True, text=True, timeout=15)
                if armed.returncode == 0:
                    killed.append(pid)
            continue
        label = handle.split(":", 1)[1] if handle.startswith("launchd:") else handle
        got = run(["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
                  capture_output=True, text=True, timeout=30)
        if got.returncode != 0:
            return {"restarted": restarted, "reason": "restart-failed",
                    "unit": label, "driver": "launchd",
                    "error": (got.stderr or "").strip()[:200]}
        restarted.append(label)
    return {"restarted": restarted, "killed": killed, "reason": "restarted",
            "driver": "launchd"}


#: ``pgrep -f`` prints one PID per line and, unlike the Python-side
#: enumeration on Windows, needs no interpreter start-up per round.
def _pgrep(pattern: str, run) -> tuple[list[int], str]:
    try:
        got = run(["pgrep", "-f", pattern], capture_output=True, text=True,
                  timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)[:120]
    if got.returncode not in (0, 1):  # 1 is "no match", not a failure
        return [], (got.stderr or "pgrep-failed").strip()[:120]
    pids = []
    for line in got.stdout.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids, ""


#: One PowerShell that lists candidates; the *decision* is made in Python so
#: there is one pattern and one test for it rather than a regex copied into a
#: script.
WINDOWS_LIST_COMMAND = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
)


def windows_client_pids(rows, pattern: str = WINDOWS_CLIENT_PATTERN) -> list[int]:
    """The PIDs of the supervisor's clients, given process rows.

    Stopping is the whole restart: ``win_node_supervisor.ps1`` starts any
    child that has exited within 10 seconds, and it is not itself matched, so
    it survives to do the respawning.
    """
    import re
    if isinstance(rows, dict):  # ConvertTo-Json unwraps a single row
        rows = [rows]
    matched = re.compile(pattern)
    pids = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        line = row.get("CommandLine") or ""
        pid = row.get("ProcessId")
        if isinstance(pid, int) and matched.search(line):
            pids.append(pid)
    return pids


def _restart_windows(handles: list[str], run) -> dict:
    """Stop the client processes; the logon supervisor respawns them."""
    try:
        listing = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                       WINDOWS_LIST_COMMAND],
                      capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"restarted": [], "reason": "restart-failed",
                "driver": "windows", "error": str(exc)[:120]}
    if listing.returncode != 0:
        return {"restarted": [], "reason": "restart-failed",
                "driver": "windows",
                "error": (listing.stderr or "listing-failed").strip()[:200]}
    try:
        rows = json.loads(listing.stdout or "[]")
    except ValueError:
        return {"restarted": [], "reason": "restart-failed",
                "driver": "windows", "error": "unparsable-process-list"}
    pids = windows_client_pids(rows)
    if not pids:
        return {"restarted": [], "killed": [], "reason": "nothing-running",
                "driver": "windows"}
    ids = ",".join(str(pid) for pid in pids)
    got = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
               f"Stop-Process -Id {ids} -Force -ErrorAction SilentlyContinue"],
              capture_output=True, text=True, timeout=60)
    if got.returncode != 0:
        return {"restarted": [], "reason": "restart-failed",
                "driver": "windows",
                "error": (got.stderr or "").strip()[:200]}
    return {"restarted": [], "killed": pids, "reason": "restarted",
            "driver": "windows", "units": handles}


def restart_units(units: list[str], *, platform: str = "",
                  runner=None) -> dict:
    """Restart only the clients. A model is refused on every platform.

    ``units`` may be logical names (``think``), systemd handles
    (``pandosi-think.service``), launchd labels, or ``proc:`` patterns; the
    resolver accepts whatever vocabulary the caller already has and the driver
    below it speaks only the platform's own verbs.

    The refusal is all-or-nothing and comes *before* any driver runs. A batch
    containing one model unit restarts nothing, because the alternative -- the
    good units moving while the ledger records a failure -- is how a node ends
    up half-updated with no record of which half.
    """
    plat = platform_id(platform)
    handles, refused = resolve_units(units, plat)
    if refused:
        return {"restarted": [], "reason": "refused", "units": refused,
                "platform": plat}
    run = runner or subprocess.run
    if plat == "linux":
        return _restart_systemd(handles, run)
    if plat == "darwin":
        return _restart_launchd(handles, run)
    if plat == "windows":
        return _restart_windows(handles, run)
    return {"restarted": [], "reason": "unknown-platform", "platform": plat}


#: Only units this project ships are ever written. A name outside the prefix is
#: not copied, so ``--install-units`` cannot be pointed at an arbitrary file.
UNIT_PREFIX = "pandosi-"
UNIT_SUFFIXES = (".service", ".timer")

#: The same boundary, spelled for launchd -- and tighter, because the Mac's
#: deployment is a superset of what a fast-forward may restart. The rule this
#: list encodes: **this file writes exactly the agents it can also restart,
#: and nothing else.** A plist that were adopted but never reloaded would
#: recreate the pi5 trap one level up -- the file on disk right, the running
#: process stale -- so the agents outside this set (``mailbox``,
#: ``mailbox-replicate``, ``observatory``, ``seed-outbox``, ``tick``, ``mac``)
#: are left to the human who owns their restart. ``llm4b``/``llm8b``,
#: ``cloudflared.*`` and ``seedserve`` are not here at all: they start a model
#: or a tunnel, and no updater may restart either.
#:
#: ``com.pandosi.stem.llm8b`` is the one that looks like a model and is not:
#: it runs ``pando.stemd``, the client that *calls* the 8B server, which is
#: why its name carries the model it talks to.
DARWIN_CLIENT_AGENTS = frozenset({
    "com.jewelroam.pandosi-think",
    "com.jewelroam.pandosi-chat",
    "com.jewelroam.pandosi-react",
    "com.jewelroam.pandosi-execute",
    "com.pandosi.stem.laya",
    "com.pandosi.stem.llm8b",
})


def install_units(source, target, names: list[str] | None = None, *,
                  platform: str = "", runner=None) -> dict:
    """Adopt the checkout's service definitions, then reload the manager.

    One caller, three targets. The pi5 units are systemd files in
    ``~/.config/systemd/user``, the Mac's are plists in
    ``~/Library/LaunchAgents``, and Windows has no such directory at all: its
    roster is ``tools/win_node_supervisor.ps1`` *inside* the checkout, so the
    fast-forward already carries it and ``node_code_watch`` restarts the stack
    when the roster's hash changes. That asymmetry is a fact about the three
    hosts, not a gap in the updater, and reporting it beats pretending to have
    written something.
    """
    plat = platform_id(platform)
    if plat == "darwin":
        return _install_launchd(source, target, names, runner=runner)
    if plat == "windows":
        return {"changed": [], "reloaded": False, "installed": False,
                "reason": "roster-inside-checkout", "platform": plat}
    return _install_systemd(source, target, names, runner=runner)


def _install_systemd(source, target, names: list[str] | None = None, *,
                     runner=None) -> dict:
    """Adopt the checkout's systemd units, then reload systemd.

    The units live outside the checkout for the same reason the updater does --
    they describe how the checkout is *run*, so they have to survive a tree that
    is mid-fast-forward or broken. And they share its trap: nothing rewrites
    them. On 2026-10-19 the pi5 code-sync unit **on disk** carried
    ``--self-source``/``--self-target`` while the **loaded** unit did not -- the
    file had been overwritten with ``cat >`` and ``systemctl --user
    daemon-reload`` was never run. The timer then fired, fast-forwarded the
    checkout, and silently declined to adopt its own successor, because the
    process that would have done the adopting was never given the flag. On a
    host with no journal that round left no trace at all.

    Three bounds, each answering a specific way this could go wrong:

    * The input is the checkout's own directory -- a commit someone reviewed --
      never a name, path, or body taken from a proposal.
    * Only ``pandosi-*`` files with a ``.service``/``.timer`` suffix are
      written. Everything else in the directory is ignored, not copied.
    * Nothing here enables, disables, starts, or stops a unit. The whole action
      is "write the file, reload systemd", so a template that changes
      ``ExecStart`` cannot restart a model as a side effect of an update.

    A previous copy is left beside the new one as ``<name>.bak``: the one
    artifact that makes a bad template recoverable without another commit.

    The reload runs even when nothing was rewritten. "The file matches" and
    "systemd loaded the file" are different facts and only the second one runs;
    a round that wrote a file and then failed to reload is otherwise
    indistinguishable from a round that was already in sync -- which is exactly
    how the pi5 unit stayed stale. This removes the cause we have actually
    observed. It does not *prove* systemd took the new bytes: a reload that
    returns 0 is not a read-back, so a template that fails to take effect still
    needs the ledger, not this function, to say so.
    """
    run = runner or subprocess.run
    source = Path(source).expanduser()
    target = Path(target).expanduser()
    if not source.is_dir():
        return {"changed": [], "reloaded": False, "reason": "no-source"}
    if names is None:
        names = sorted(entry.name for entry in source.iterdir()
                       if entry.is_file() and entry.suffix in UNIT_SUFFIXES)
    changed = []
    for name in names:
        if not name.startswith(UNIT_PREFIX) or name.endswith(".bak"):
            continue
        if "/" in name or "\\" in name or name in (".", ".."):
            continue
        template = source / name
        if template.suffix not in UNIT_SUFFIXES or not template.is_file():
            continue
        try:
            text = template.read_text(encoding="utf-8")
        except OSError as exc:
            return {"changed": changed, "reloaded": False,
                    "reason": "unreadable-source", "unit": name,
                    "error": str(exc)[:120]}
        deployed = target / name
        try:
            if deployed.read_text(encoding="utf-8") == text:
                continue
            previous = deployed.read_text(encoding="utf-8")
        except OSError:
            previous = None
        try:
            target.mkdir(parents=True, exist_ok=True)
            if previous is not None:
                (target / (name + ".bak")).write_text(previous, encoding="utf-8")
            deployed.write_text(text, encoding="utf-8")
        except OSError as exc:
            return {"changed": changed, "reloaded": False,
                    "reason": "write-failed", "unit": name,
                    "error": str(exc)[:120]}
        changed.append(name)
    try:
        got = run(["systemctl", "--user", "daemon-reload"],
                  capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        # A host without systemctl must report the gap, not lose the updater.
        return {"changed": changed, "reloaded": False,
                "reason": "reload-failed", "error": str(exc)[:120]}
    if got.returncode != 0:
        return {"changed": changed, "reloaded": False,
                "reason": "reload-failed"}
    return {"changed": changed, "reloaded": True,
            "reason": "installed" if changed else "same", "platform": "linux"}


#: ``<key>Label</key><string>name</string>`` with any whitespace between the
#: two tags, which is how every plist in this repo is written.
_LABEL_RE = None


def _plist_label(text: str) -> str:
    """The ``Label`` a plist claims, or ``""`` if it does not claim one.

    Deliberately a regex and not ``plistlib``: ``plistlib`` rejects the XML
    comments this project's plists carry (a comment containing ``--`` is legal
    to launchd and, strictly, illegal to a conforming XML parser). A validator
    that refuses the files the OS is running is a validator that would refuse
    to fix them, so the check here is structural and the *real* verdict comes
    from ``plutil -lint`` below.
    """
    global _LABEL_RE
    if _LABEL_RE is None:
        import re
        _LABEL_RE = re.compile(
            r"<key>\s*Label\s*</key>\s*<string>([^<]*)</string>")
    found = _LABEL_RE.search(text)
    return found.group(1).strip() if found else ""


def _install_launchd(source, target, names: list[str] | None = None, *,
                     runner=None) -> dict:
    """Adopt the checkout's launchd agents, then reload only what changed.

    The Mac keeps its agents in ``~/Library/LaunchAgents`` -- outside the
    checkout, for the same reason the updater lives outside it -- and until
    now nothing in this repo rewrote them. The measurement that makes that a
    bug and not a preference: on 2026-10-19 the *live*
    ``com.jewelroam.pandosi-think`` carried two ``--mailbox-fallback``
    arguments that the checkout's copy did not, so running this function
    against a stale template would have silently rolled a working node's
    mailbox failover back to a single address.

    Two places this diverges from ``systemd``, both deliberate:

    * A reload is not free. ``daemon-reload`` re-reads text; ``bootout`` plus
      ``bootstrap`` stops and starts the agent, which for the Mac's mailbox
      means a listener that is down for the length of a round. So an
      *unchanged* plist is not reloaded -- it is only *checked*, with
      ``launchctl print``, and bootstrapped when the manager does not have it.
      That keeps the pi5 trap closed (a file written and never loaded) without
      turning every round into an outage.
    * The verdict comes from the platform. ``plutil -lint`` is what launchd
      itself uses, so a file that fails it is not written at all, where
      ``systemd`` had to accept whatever bytes the commit contained.

    The one bound that does not move: a name outside
    ``DARWIN_CLIENT_AGENTS`` is never written. ``--install-units`` cannot be
    pointed at a model, at a tunnel, or at an agent whose restart this file
    does not own.
    """
    run = runner or subprocess.run
    source = Path(source).expanduser()
    target = Path(target).expanduser()
    if not source.is_dir():
        return {"changed": [], "reloaded": False, "reason": "no-source"}
    if names is None:
        names = sorted(entry.name for entry in source.iterdir()
                       if entry.is_file() and entry.suffix == ".plist")
    uid = getattr(os, "getuid", lambda: 0)()
    changed: list[str] = []
    booted: list[str] = []
    #: Templates this repo ships that this round could not adopt. A skip is
    #: reported, never silent: "the commit is right and the host is wrong" is
    #: the fact that used to leave no trace anywhere.
    skipped: list[dict] = []
    for name in names:
        if not name.endswith(".plist") or name.endswith(".bak"):
            continue
        if "/" in name or "\\" in name or name in (".", ".."):
            continue
        if name[: -len(".plist")] not in DARWIN_CLIENT_AGENTS:
            continue
        template = source / name
        if not template.is_file():
            continue
        try:
            text = template.read_text(encoding="utf-8")
        except OSError as exc:
            return {"changed": changed, "reloaded": False,
                    "reason": "unreadable-source", "unit": name,
                    "error": str(exc)[:120]}
        label = _plist_label(text)
        if not label or label != name[: -len(".plist")]:
            # A plist whose Label and filename disagree installs a job whose
            # name is not the one the ledger would report, and the next round
            # writes it again forever. The other templates are still adopted:
            # one broken file must not stop the fleet's updater.
            skipped.append({"unit": name, "reason": "label-mismatch",
                            "label": label})
            continue
        deployed = target / name
        try:
            if deployed.read_text(encoding="utf-8") == text:
                continue
            previous = deployed.read_text(encoding="utf-8")
        except OSError:
            previous = None
        linted = run(["plutil", "-lint", "-", "-"], capture_output=True,
                     text=True, timeout=30, input=text)
        if linted.returncode != 0:
            skipped.append({"unit": name, "reason": "plist-does-not-parse",
                            "error": (linted.stdout
                                      or linted.stderr).strip()[:160]})
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            if previous is not None:
                (target / (name + ".bak")).write_text(previous,
                                                      encoding="utf-8")
            deployed.write_text(text, encoding="utf-8")
        except OSError as exc:
            return {"changed": changed, "reloaded": False,
                    "reason": "write-failed", "unit": name,
                    "error": str(exc)[:120]}
        changed.append(name)
    wanted = changed or [
        name for name in names
        if name.endswith(".plist") and not name.endswith(".bak")
        and name[: -len(".plist")] in DARWIN_CLIENT_AGENTS
        and (target / name).is_file()
        and _not_loaded(name[: -len(".plist")], uid, run)]
    for name in wanted:
        label = name[: -len(".plist")]
        run(["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True, text=True, timeout=30)
        got = run(["launchctl", "bootstrap", f"gui/{uid}", str(target / name)],
                  capture_output=True, text=True, timeout=30)
        if got.returncode != 0:
            return {"changed": changed, "reloaded": False, "booted": booted,
                    "skipped": skipped, "reason": "bootstrap-failed",
                    "unit": label, "error": (got.stderr or "").strip()[:200]}
        booted.append(label)
    return {"changed": changed, "booted": booted, "skipped": skipped,
            "reloaded": True,
            "reason": "installed" if changed else "same", "platform": "darwin"}


def _not_loaded(label: str, uid: int, run) -> bool:
    """True when launchd has no job by this name.

    The check that distinguishes "on disk and loaded" from "on disk and never
    loaded" -- the pi5 failure, translated. ``launchctl print`` exits non-zero
    for a job launchd does not know about, so its failure *is* the signal here
    and is not reported as an error.
    """
    try:
        got = run(["launchctl", "print", f"gui/{uid}/{label}"],
                  capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return got.returncode != 0




#: Must equal ``pando.graft.CODE_HEAD_STAMP``; a test pins the pair. Duplicated
#: rather than imported because the deployed copy of this file runs outside the
#: checkout, under an interpreter that need not have pando installed.
INSTALLED_HEAD_STAMP = "_code_head.txt"


def installed_package_dir(python: str) -> Path | None:
    """Where ``python`` imports pando from, or None if it cannot say."""
    got = subprocess.run(
        [python, "-c",
         "import pathlib,pando;"
         "print(pathlib.Path(pando.__file__).resolve().parent)"],
        capture_output=True, text=True, timeout=60)
    out = got.stdout.strip()
    return Path(out) if got.returncode == 0 and out else None


def stamp_installed_head(python: str, repo: Path,
                         package_dir: Path | None = None) -> str:
    """Record the installed copy's commit beside the copy itself.

    ``graft.code_head()`` probes ``git`` where the module lives, and a pip
    install copies the tree without ``.git``. pi5 runs that way, so every stem
    there answered ``code_head: ""`` -- which makes a stale member and a
    current one indistinguishable from every peer. The stamp is written after
    a successful install, so it describes the bytes that were copied, not
    whatever the checkout has moved on to since.
    """
    head = _head(repo, _env())
    if not head:
        return ""
    where = package_dir if package_dir is not None else installed_package_dir(python)
    if where is None:
        return ""
    try:
        (where / INSTALLED_HEAD_STAMP).write_text(head + "\n", encoding="utf-8")
    except OSError:
        return ""
    return head


def install_checkout(python: str, repo: Path) -> bool:
    """Install this checkout into ``python`` without fetching new dependencies."""
    got = subprocess.run(
        [python, "-m", "pip", "install", "--no-deps",
         "--disable-pip-version-check", str(repo)],
        capture_output=True, text=True, timeout=180)
    if got.returncode != 0:
        return False
    stamp_installed_head(python, repo)
    return True


def refresh_self(source: Path, target: Path) -> dict:
    """Let the deployed updater adopt the checkout's copy of itself.

    The copy that runs on a headless node sits outside the checkout on purpose:
    it has to run while the tree is mid-fast-forward, dirty, or broken. The trap
    is that nothing rewrites it. On 2026-10-19 the copy on pi5 was current only
    because a human had copied it by hand (sha256 equal to this file, mtime
    00:53), so any later fix *here* would have landed in the repo and never
    taken effect on the node that needed it. That is the same one-level-up trap
    that left win's supervisor without its ``code-watch`` service: an updater
    that cannot receive its own fixes.

    Adoption is guarded, because the opposite failure is worse than a stale
    updater: a candidate that does not compile leaves the working copy exactly
    where it is, so a bad commit cannot leave a node with no updater at all.
    """
    source = Path(source).expanduser()
    target = Path(target).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        return {"adopted": False, "reason": "no-source",
                "error": str(exc)[:120]}
    try:
        compile(text, str(source), "exec")
    except SyntaxError as exc:
        return {"adopted": False, "reason": "candidate-does-not-compile",
                "line": exc.lineno}
    try:
        if target.read_text(encoding="utf-8") == text:
            return {"adopted": False, "reason": "same"}
    except OSError:
        pass
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        return {"adopted": False, "reason": "write-failed",
                "error": str(exc)[:120]}
    return {"adopted": True, "reason": "adopted",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--peer", default="",
                        help="used only when --remote cannot be fetched")
    parser.add_argument("--peer-token-file", default="",
                        help="bearer token for an http peer bundle")
    parser.add_argument("--install-python", default="",
                        help="after a fast-forward, pip install --no-deps into this interpreter")
    parser.add_argument("--restart", action="append", default=[],
                        help="client unit to restart after a fast-forward; repeatable")
    parser.add_argument("--platform", default="",
                        choices=("", "linux", "darwin", "windows"),
                        help="service manager to use (default: this host's)")
    parser.add_argument("--self-source", default="",
                        help="updater inside the checkout; adopted by --self-target")
    parser.add_argument("--self-target", default="",
                        help="the copy of this file that is running on the node")
    parser.add_argument("--install-units", default="",
                        help="checkout directory of systemd unit templates")
    parser.add_argument("--install-units-target", default="",
                        help="where this node's units are deployed")
    args = parser.parse_args(argv)
    token = ""
    if args.peer_token_file:
        token = Path(args.peer_token_file).expanduser().read_text(encoding="utf-8").strip()
    result = sync_repo(args.repo, remote=args.remote, peer=args.peer, peer_token=token)
    if result.get("updated") and args.install_python:
        result["installed"] = install_checkout(args.install_python, Path(args.repo))
        if result["installed"] and args.restart:
            done = restart_units(args.restart, platform=args.platform)
            result["restarted"] = done["restarted"]
            result["restart"] = done["reason"]
            result["restart_driver"] = done.get("driver", "")
    elif args.restart and result.get("updated"):
        done = restart_units(args.restart, platform=args.platform)
        result["restarted"] = done["restarted"]
        result["restart"] = done["reason"]
        result["restart_driver"] = done.get("driver", "")
    # Adopt our own successor, and our own units, from a checkout we just
    # confirmed is current. A fetch that failed leaves the checkout at an older
    # commit; adopting from *that* copy would roll the updater (and the units)
    # backwards, taking the very flags this line needs with it -- which is how a
    # node ends up with no updater at all. "Already at the tip" is still a
    # confirmation, so a checkout that merely sat still is not a reason to keep
    # running yesterday's bytes.
    confirmed = result.get("reason") in ("fast-forward", "already")
    unconfirmed = {"reason": "checkout-not-confirmed",
                   "checkout": result.get("reason")}
    if args.self_source and args.self_target:
        if confirmed:
            result["self"] = refresh_self(args.self_source, args.self_target)
        else:
            result["self"] = {"adopted": False, **unconfirmed}
    if args.install_units and args.install_units_target:
        if confirmed:
            result["units"] = install_units(args.install_units,
                                            args.install_units_target,
                                            platform=args.platform)
        else:
            result["units"] = {"changed": [], "reloaded": False, **unconfirmed}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["reason"] in ("already", "fast-forward") else 2


if __name__ == "__main__":
    raise SystemExit(main())
