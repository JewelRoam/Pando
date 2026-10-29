"""Fail-closed duplicate-actor guard for the launchd failover tick.

The failover agent may only start when the desktop app is absent and there
has been no ``mac-codex`` activity in the last ten minutes.  A fixed chat
``--limit 10`` is not a time window: under normal proposer traffic it can
cover only a few minutes and miss a nearby primary actor.

This tool makes the coverage requirement explicit.  A bounded mailbox tail is
safe to interpret only when it includes the full requested time window; if
the oldest returned row is still newer than the cutoff, the result is
``unknown`` and the caller must not take over.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import agent_chat  # noqa: E402


SAFE = "safe"
RECENT = "recent"
UNKNOWN = "unknown"

EXIT_SAFE = 0
EXIT_RECENT = 10
EXIT_UNKNOWN = 11


@dataclass(frozen=True)
class Assessment:
    status: str
    reason: str
    rows: int = 0
    oldest_seen: str | None = None
    latest_match: str | None = None


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _parse_created(value, fallback_tz) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=fallback_tz)
    return parsed


def assess_activity(tasks: Iterable[dict], now: datetime, *,
                    window_seconds: int = 600,
                    identity: str = "mac-codex",
                    limit: int = 2000) -> Assessment:
    """Assess whether a bounded task tail covers a recent-actor window.

    ``tasks`` must be the newest ``limit`` rows in mailbox order.  Equality at
    the cutoff counts as recent (the rule is "within the last ten minutes").
    Any malformed timestamp in the returned tail makes the result unknown
    rather than silently shrinking the window.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if now.tzinfo is None:
        now = now.replace(tzinfo=_local_now().tzinfo)
    cutoff = now - timedelta(seconds=window_seconds)

    rows: list[dict] = []
    for row in tasks:
        if not isinstance(row, dict):
            return Assessment(UNKNOWN, "mailbox returned a non-object row")
        rows.append(row)

    parsed: list[tuple[dict, datetime]] = []
    for row in rows:
        created = _parse_created(row.get("created_at"), now.tzinfo)
        if created is None:
            return Assessment(
                UNKNOWN,
                "mailbox returned a row with invalid created_at: "
                f"{row.get('id', '?')}",
                rows=len(rows),
            )
        parsed.append((row, created))

    if not parsed:
        return Assessment(SAFE, "mailbox has no activity", rows=0)

    ordered = sorted(parsed, key=lambda item: item[1])
    oldest = ordered[0][1]
    newest = ordered[-1][1]
    # A response shorter than the requested limit proves the whole mailbox was
    # returned.  At the limit, the oldest row must reach back past the cutoff.
    complete = len(rows) < limit or oldest <= cutoff
    if not complete:
        return Assessment(
            UNKNOWN,
            "bounded mailbox tail does not cover the full activity window "
            f"(oldest={oldest.isoformat(timespec='seconds')}, "
            f"cutoff={cutoff.isoformat(timespec='seconds')})",
            rows=len(rows),
            oldest_seen=oldest.isoformat(timespec="seconds"),
        )

    matches = [(row, created) for row, created in parsed
               if str(row.get("from", "")) == identity and created >= cutoff]
    if matches:
        latest_row, latest_at = max(matches, key=lambda item: item[1])
        return Assessment(
            RECENT,
            f"{identity} activity at "
            f"{latest_at.isoformat(timespec='seconds')} "
            f"({max(0, int((now - latest_at).total_seconds()))}s ago): "
            f"{latest_row.get('title', '')}",
            rows=len(rows),
            oldest_seen=oldest.isoformat(timespec="seconds"),
            latest_match=latest_at.isoformat(timespec="seconds"),
        )

    return Assessment(
        SAFE,
        f"no {identity} activity since "
        f"{cutoff.isoformat(timespec='seconds')} "
        f"(covered back to {oldest.isoformat(timespec='seconds')}; "
        f"newest={newest.isoformat(timespec='seconds')})",
        rows=len(rows),
        oldest_seen=oldest.isoformat(timespec="seconds"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get(
        "PANDOSI_MAILBOX_URL", "http://127.0.0.1:8776"))
    parser.add_argument("--token", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN"))
    parser.add_argument("--token-file", default=os.environ.get(
        "PANDOSI_MAILBOX_TOKEN_FILE", "~/.pando/mailbox.token"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=int, default=600)
    parser.add_argument("--identity", default="mac-codex")
    parser.add_argument("--limit", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_args = SimpleNamespace(
        url=args.url,
        token=args.token,
        token_file=os.path.expanduser(args.token_file),
        timeout=args.timeout,
    )
    try:
        tasks = agent_chat.fetch_tasks(request_args, limit=args.limit)
    except Exception as exc:  # This gate must fail closed on transport drift.
        print(f"UNKNOWN mailbox query failed: {exc}")
        return EXIT_UNKNOWN

    assessment = assess_activity(
        tasks,
        _local_now(),
        window_seconds=args.window_seconds,
        identity=args.identity,
        limit=args.limit,
    )
    print(f"{assessment.status.upper()} {assessment.reason}")
    return {
        SAFE: EXIT_SAFE,
        RECENT: EXIT_RECENT,
        UNKNOWN: EXIT_UNKNOWN,
    }[assessment.status]


if __name__ == "__main__":
    raise SystemExit(main())
