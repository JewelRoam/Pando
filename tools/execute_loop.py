"""Resident executor loop for the win node.

Every INTERVAL seconds, run `pando.execute --once`: judge the mailbox,
seal at most one adopted seed.export that THIS host signed (the executor
refuses foreign proposals), and exit. Idle rounds are free; a broken
mailbox or an empty queue is recorded, never raised. Address resolution
is local-only (hosts.json mailboxes[0], then localhost convention) - no
git lookups, per doctrine.

Usage: python tools/execute_loop.py [interval-seconds]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mailbox_url() -> str:
    try:
        hosts = json.loads((Path.home() / ".pandosi" / "hosts.json")
                           .read_text(encoding="utf-8"))
        urls = hosts.get("mailboxes") or []
        if urls:
            return str(urls[0])
    except (OSError, json.JSONDecodeError):
        pass
    return "http://127.0.0.1:8776"


def main() -> int:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    url = mailbox_url()
    print(f"execute-loop: every {interval:.0f}s against {url}", flush=True)
    while True:
        try:
            subprocess.run(
                [sys.executable, "-m", "pando.execute", "--once",
                 "--mailbox", url,
                 "--token-file", "metrics/lan-check/mac-mailbox.token"],
                cwd=str(ROOT), timeout=180)
        except Exception as exc:  # noqa: BLE001 - the loop must never die
            print(f"execute-loop: {exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
