#!/usr/bin/env python3
"""Fast-forward a checkout to main. Do not restart a model.

A dirty tree or a history that is not a fast-forward is left untouched.
GitHub is the first remote. ``--peer`` is tried only when that fetch fails,
and it must be the same branch, not a second history.

``--self-source``/``--self-target`` exist for the copy of this file that lives
*outside* the checkout on a headless node. See ``refresh_self``.
``--install-units`` is the same hop one level up: the systemd units are also
files outside the checkout, and nothing rewrote them either. See
``install_units``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
SAFE_UNITS = frozenset({
    "pandosi-think.service",
    "pandosi-chat.service",
    "pandosi-react.service",
    "pandosi-seed-pull.service",
    "pandosi-execute.service",
    "pandosi-stem-text.service",
    "pandosi-stem-laya.service",
    "pandosi-stem-llm.service",
})


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
    got = _git(repo, ["status", "--porcelain"], env)
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
        return result
    result["reason"] = "fast-forward" if after != before else "already"
    result["updated"] = after != before
    return result


def restart_units(units: list[str]) -> dict:
    """Restart only the client units. A model unit is refused."""
    wanted = [unit for unit in units if unit]
    refused = [unit for unit in wanted if unit not in SAFE_UNITS]
    if refused:
        return {"restarted": [], "reason": "refused", "units": refused}
    restarted = []
    for unit in wanted:
        got = subprocess.run(
            ["systemctl", "--user", "restart", unit],
            capture_output=True, text=True, timeout=30)
        if got.returncode != 0:
            return {"restarted": restarted, "reason": "restart-failed", "unit": unit}
        restarted.append(unit)
    return {"restarted": restarted, "reason": "restarted"}


#: Only units this project ships are ever written. A name outside the prefix is
#: not copied, so ``--install-units`` cannot be pointed at an arbitrary file.
UNIT_PREFIX = "pandosi-"
UNIT_SUFFIXES = (".service", ".timer")


def install_units(source, target, names: list[str] | None = None) -> dict:
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
        got = subprocess.run(["systemctl", "--user", "daemon-reload"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        # A host without systemctl must report the gap, not lose the updater.
        return {"changed": changed, "reloaded": False,
                "reason": "reload-failed", "error": str(exc)[:120]}
    if got.returncode != 0:
        return {"changed": changed, "reloaded": False,
                "reason": "reload-failed"}
    return {"changed": changed, "reloaded": True,
            "reason": "installed" if changed else "same"}




def install_checkout(python: str, repo: Path) -> bool:
    """Install this checkout into ``python`` without fetching new dependencies."""
    got = subprocess.run(
        [python, "-m", "pip", "install", "--no-deps",
         "--disable-pip-version-check", str(repo)],
        capture_output=True, text=True, timeout=180)
    return got.returncode == 0


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
            done = restart_units(args.restart)
            result["restarted"] = done["restarted"]
            result["restart"] = done["reason"]
    elif args.restart and result.get("updated"):
        done = restart_units(args.restart)
        result["restarted"] = done["restarted"]
        result["restart"] = done["reason"]
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
                                            args.install_units_target)
        else:
            result["units"] = {"changed": [], "reloaded": False, **unconfirmed}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["reason"] in ("already", "fast-forward") else 2


if __name__ == "__main__":
    raise SystemExit(main())
