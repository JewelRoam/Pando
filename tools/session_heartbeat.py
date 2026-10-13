"""Idempotent local heartbeat: snapshot transcript changes, commit, push.

Runs every 30 minutes via the user-level Windows task. It does not invoke an
LLM and never sends the transcript anywhere: backup_session.py only writes a
sanitized snapshot into this repository, and git push is operator-controlled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "session-heartbeat.log"


def run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def log(message: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {message}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    latest = run(["git", "log", "-1", "--pretty=%h %s"])
    if latest.returncode != 0:
        log(f"git log failed: {latest.stderr.strip()}")
        return 1
    note = f"automatic snapshot after {latest.stdout.strip()}"
    backup = run([sys.executable, str(ROOT / "tools" / "backup_session.py"),
                  "--note", note])
    log(backup.stdout.strip() or backup.stderr.strip() or "backup returned no output")
    if backup.returncode != 0 or "wrote " not in backup.stdout:
        return backup.returncode

    run(["git", "add", "docs/session-logs"])
    staged = run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        log("no staged backup changes")
        return 0

    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    commit = run(["git", "commit", "-m", f"chore: session backup {stamp}"])
    if commit.returncode != 0:
        log(f"commit failed: {commit.stderr.strip()}")
        return commit.returncode
    log(f"committed {stamp}")

    if args.no_push:
        return 0
    pushed = run(["git", "push", "origin", "main"])
    if pushed.returncode != 0:
        log(f"push failed: {pushed.stderr.strip()}")
        return pushed.returncode
    log("pushed origin/main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
