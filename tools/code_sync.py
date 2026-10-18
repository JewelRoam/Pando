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
from pathlib import Path

# Client processes only. A model server is not on this list, on purpose.
SAFE_UNITS = frozenset({
    "pandosi-think.service",
    "pandosi-chat.service",
    "pandosi-seed-pull.service",
})


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def _git(repo: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, env=env, timeout=90)


def _head(repo: Path, env: dict[str, str]) -> str:
    got = _git(repo, ["rev-parse", "HEAD"], env)
    return got.stdout.strip() if got.returncode == 0 else ""


def _dirty(repo: Path, env: dict[str, str]) -> bool:
    got = _git(repo, ["status", "--porcelain"], env)
    return bool(got.stdout.strip())


def _fetch(repo: Path, env: dict[str, str], remote: str) -> bool:
    got = _git(repo, ["fetch", remote, "main"], env)
    return got.returncode == 0


def sync_repo(repo, *, remote: str = "origin", peer: str = "") -> dict:
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
        if not peer or not _fetch(repo, env, peer):
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
    parser.add_argument("--install-python", default="",
                        help="after a fast-forward, pip install --no-deps into this interpreter")
    parser.add_argument("--restart", action="append", default=[],
                        help="client unit to restart after a fast-forward; repeatable")
    args = parser.parse_args(argv)
    result = sync_repo(args.repo, remote=args.remote, peer=args.peer)
    if result.get("updated") and args.install_python:
        result["installed"] = install_checkout(args.install_python, Path(args.repo))
        if result["installed"] and args.restart:
            result.update(restart_units(args.restart))
    elif args.restart and result.get("updated"):
        result.update(restart_units(args.restart))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["reason"] in ("already", "fast-forward") else 2


if __name__ == "__main__":
    raise SystemExit(main())
