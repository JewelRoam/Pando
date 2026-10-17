#!/usr/bin/env python3
"""think_watchdog: catch the proposer that is alive but no longer thinking.

launchd `KeepAlive` and systemd `Restart=always` both answer the same question:
"did the process exit?" Neither answers the question we actually care about:
"is it still producing thoughts?" A proposer wedged on an unread socket, a
model server that accepts the connection and never replies, a deadlocked
retry - all of these stay *running* while the ledger stops growing. That is
the exact silence this civilisation has already been burned by twice (a
double-forked backend that vanished at 12:46, a stale lock that made the
takeover say "skip: another tick is running" forever).

So the signal is not liveness, it is *progress*: the mtime of the thinker's
ledger. If it has not moved in longer than any round should plausibly take,
kick the supervisor. Recording the restart is the point - an unlogged rescue
is indistinguishable from nothing having gone wrong.

Deliberately dumb and stateless apart from a cooldown file: it runs every few
minutes forever, so it must not itself become a thing that can hang, wedge, or
need a human. Every failure path exits 0 and prints a line.

Usage:
  python tools/think_watchdog.py --ledger ~/.pandosi/think-ledger.jsonl \\
      --stall-seconds 900 --restart-cmd "launchctl kickstart -k gui/501/com.jewelroam.pandosi-think"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_COOLDOWN_S = 300.0


def ledger_age(now: float, ledger: Path) -> float | None:
    """Seconds since the ledger last grew, or None if there is no ledger yet."""
    try:
        return max(0.0, now - ledger.stat().st_mtime)
    except OSError:
        return None


def decide(now: float, ledger: Path, stall_seconds: float,
           cooldown_file: Path | None = None,
           cooldown_seconds: float = DEFAULT_COOLDOWN_S) -> tuple[str, str]:
    """Return (verdict, why). verdict is "ok" | "stall" | "cooldown"."""
    age = ledger_age(now, ledger)
    if age is None:
        # No ledger at all. Either it was never created (a real problem) or it
        # was just deleted on purpose. Either way a kick is cheap and safe.
        return "stall", f"no ledger at {ledger}"
    if age <= stall_seconds:
        return "ok", f"ledger age {age:.0f}s <= {stall_seconds:.0f}s"
    if cooldown_file is not None:
        try:
            last = float(cooldown_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            last = 0.0
        since = now - last
        if since < cooldown_seconds:
            # We already kicked recently and it has not helped yet. Kicking
            # again in a loop would hide the problem behind restart noise.
            return "cooldown", (f"ledger age {age:.0f}s but kicked {since:.0f}s "
                                f"ago (< {cooldown_seconds:.0f}s)")
    return "stall", f"ledger age {age:.0f}s > {stall_seconds:.0f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", required=True,
                        help="the thinker's ledger; its mtime is the heartbeat")
    parser.add_argument("--stall-seconds", type=float, default=900.0,
                        help="silence longer than this is treated as wedged")
    parser.add_argument("--restart-cmd", required=True,
                        help="shell command that kicks the supervisor")
    parser.add_argument("--cooldown-file", default="",
                        help="where to remember the last restart (empty=never cooldown)")
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--now", type=float, default=None,
                        help="override clock (testing only)")
    args = parser.parse_args(argv)

    now = time.time() if args.now is None else args.now
    ledger = Path(args.ledger).expanduser()
    cooldown = Path(args.cooldown_file).expanduser() if args.cooldown_file else None

    verdict, why = decide(now, ledger, args.stall_seconds, cooldown,
                          args.cooldown_seconds)
    stamp = time.strftime("%F %T", time.localtime(now))
    if verdict == "ok":
        print(f"{stamp} think-watchdog ok: {why}", flush=True)
        return 0
    if verdict == "cooldown":
        print(f"{stamp} think-watchdog STILL STALLED: {why}", flush=True)
        return 0

    print(f"{stamp} think-watchdog kick: {why}", flush=True)
    try:
        proc = subprocess.run(args.restart_cmd, shell=True,
                              capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            print(f"{stamp} think-watchdog restart failed rc={proc.returncode}: "
                  f"{(proc.stderr or proc.stdout or '').strip()[:400]}", flush=True)
        else:
            print(f"{stamp} think-watchdog restarted supervisor", flush=True)
    except Exception as exc:  # noqa: BLE001 - the watchdog must never die
        print(f"{stamp} think-watchdog restart raised: {exc}", flush=True)
        return 0

    if cooldown is not None:
        try:
            cooldown.parent.mkdir(parents=True, exist_ok=True)
            cooldown.write_text(f"{now}\n", encoding="utf-8")
        except OSError as exc:
            print(f"{stamp} think-watchdog cooldown not written: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
