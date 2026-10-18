#!/usr/bin/env python3
"""Fast-forward a checkout to main. Do not restart a model.

A dirty tree or a history that is not a fast-forward is left untouched.
GitHub is the first remote. ``--peer`` is tried only when that fetch fails,
and it must be the same branch, not a second history.
"""
from __future__ import annotations

import argparse
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
# ``rootd`` is deliberately absent: it registers members at startup only and
# restarting it would drop the live registry for every local stem.
SAFE_UNITS = frozenset({
    "pandosi-think.service",
    "pandosi-chat.service",
    "pandosi-react.service",
    "pandosi-seed-pull.service",
    "pandosi-execute.service",
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


def install_checkout(python: str, repo: Path) -> bool:
    """Install this checkout into ``python`` without fetching new dependencies."""
    got = subprocess.run(
        [python, "-m", "pip", "install", "--no-deps",
         "--disable-pip-version-check", str(repo)],
        capture_output=True, text=True, timeout=180)
    return got.returncode == 0


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
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["reason"] in ("already", "fast-forward") else 2


if __name__ == "__main__":
    raise SystemExit(main())
