#!/usr/bin/env python3
"""vote_watchdog: catch the reactor that is answering yesterday and looking fine.

KeepAlive answers "did the process exit?". ``think_watchdog`` answers "is it
still producing thoughts?". Neither answers the third question, and this one
cost the colony a night of votes:

    is the thought it is producing still about *now*?

On 2026-10-19 ``win-laya-int8`` appended a reaction every two minutes for
35 hours. Its process was up, its ledger grew, its stances varied (赞同/反对/
搁置), and every column on the observatory said fine -- but every target it
named was stamped 2026-10-17 13:00-14:48. It was not voting late. It was
voting about a fixed point in the past, and the gap only widened.

An absolute age threshold cannot catch that honestly. ``VOTE_STALE_S`` is six
hours, a number nobody measured, and a member that wedges at 07:00 reads
"fresh" on the panel until 13:00 -- five hours of a green light over a dead
voter. This tool measures the *slope* instead: how fast a member's answer-lag
grows.

    A member keeping up answers proposals that sit some roughly fixed
    distance behind the tip, so its lag is flat (slope ~ 0).
    A member answering a frozen point in the past spends all of wall-clock
    falling behind, so its lag climbs 1:1 with the clock (slope -> 1).

The discriminator needs no magic age, and it separates the two cases inside
one window. Measured live, 2026-10-19 02:24, 90-minute window, mailbox :8776,
delivered_at minus proposal stamp:

    mac-text            slope -15.17   (draining a backlog, end state bounded)
    pandosi-pi5-x       slope -22.49   (same)
    mac-llm-8b-local    slope  -0.01   (steady state: the real noise floor)
    win-laya-int8       slope  +0.49   (gap widening half a second per second)

Only the positive one is a wedge. The default threshold (0.20) is a margin
over that measured -0.01 noise floor, not a measured constant; a member is
accused only when it spends more than a fifth of wall-clock falling behind.

Usage:
  python tools/vote_watchdog.py --url http://127.0.0.1:8776 \\
      --token-file ~/.pandosi/mailbox.token \\
      --decision mac-text --decision win-laya-int8

A Mac voter that may sleep is named twice: once in ``--decision``
(it must be voting) and once in ``--sleep-gated-member`` (but only
while this host was awake during the window). ``--ignore-member``
remains the unconditional omission for hosts outside the always-on
set. Judging a sleeping laptop as if it were an always-on voter is
what this split exists to stop.

Exit codes: 0 = every member bounded, 2 = somebody is falling behind or a
named decision member has gone quiet, 1 = the feed could not be read.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pando.proposal_mail import is_reaction_task  # noqa: E402

#: Same stamp rule the rest of the colony uses (pando/proposal_mail._ID_STAMP):
#: every proposal id ends in the epoch its proposer stamped it with.
STAMP_RE = re.compile(r"-(\d{9,})$")

#: Wall-clock span to measure the slope over. Long enough that a member voting
#: once per two minutes yields ~45 samples, short enough that a fresh wedge is
#: caught within the hour rather than the following morning.
DEFAULT_WINDOW_S = 5400.0
#: Below this many reactions in the window there is no slope to speak of, and
#: the tool says so rather than guessing.
DEFAULT_MIN_SAMPLES = 6
#: Require the samples to span real time. Six reactions inside four seconds is
#: a burst, not a trend.
DEFAULT_MIN_SPAN_S = 600.0
#: Slope above this means more than a fifth of wall-clock is being spent
#: falling further behind. See the module docstring for the measured margin.
DEFAULT_SLOPE = 0.20

#: Where the judgement is published for readers that must not re-derive it.
DEFAULT_SNAPSHOT = "~/.pandosi/vote-watchdog.json"

OK = "ok"
WEDGED = "wedge"
SILENT = "silent"
UNKNOWN = "unknown"


def _read_token(path: str | None) -> str | None:
    if not path:
        return None
    try:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


SYSCTL_SEC_RE = re.compile(r"sec\s*=\s*(\d+)")


def _sysctl_epoch(text: str) -> float:
    """The ``sec`` field of a ``kern.*time`` sysctl value, or 0 when absent.

    ``sysctl -n kern.sleeptime`` prints ``{ sec = 0, usec = 0 } ...``; the
    number is an absolute epoch, and 0 means the kernel has no such event
    since boot.
    """
    match = SYSCTL_SEC_RE.search(str(text or ""))
    return float(match.group(1)) if match else 0.0


def read_host_sleep() -> tuple[float, float]:
    """(last sleep epoch, last wake epoch) of this host, from sysctl.

    ``kern.sleeptime`` is when the kernel last entered sleep and
    ``kern.waketime`` when it last woke; each is 0 if that never happened.
    This is a subprocess only because the stdlib has no sysctl binding, and it
    runs at most once per watchdog pass. Any failure (not macOS, a sandbox that
    cannot fork) reads as "never slept", which keeps a member judged rather
    than silently dropped.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.sleeptime", "kern.waketime"],
            capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return (0.0, 0.0)
    if out.returncode != 0:
        return (0.0, 0.0)
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    sleep_start = _sysctl_epoch(lines[0]) if lines else 0.0
    wake = _sysctl_epoch(lines[1]) if len(lines) > 1 else 0.0
    return (sleep_start, wake)


def sleep_overlaps_window(sleep_start: float, sleep_end: float,
                          now: float, window_s: float) -> bool:
    """True when [sleep_start, sleep_end) touches [now - window_s, now].

    ``sleep_end <= sleep_start`` means "still asleep now"; the caller passes
    that open-ended case as ``sleep_end = now``. A zero ``sleep_start`` means
    the host has never slept, i.e. awake the whole window.
    """
    if sleep_start <= 0:
        return False
    end = now if sleep_end <= sleep_start else sleep_end
    return sleep_start <= now and end >= now - window_s


def host_was_asleep(now: float, window_s: float) -> bool:
    """True when this host spent part of [now-window_s, now] asleep.

    ``kern.sleeptime`` only remembers the *last* sleep. Because sleeps are
    ordered, the last sleep overlapping the window is exactly "some sleep
    overlapped the window" - an earlier sleep cannot be the last one and fall
    outside it. So two sysctls are sufficient; no pmset log parsing.
    """
    sleep_start, wake = read_host_sleep()
    if sleep_start > 0 and wake <= sleep_start:
        sleep_end = now  # last transition was into sleep (or never woke)
    else:
        sleep_end = wake
    return sleep_overlaps_window(sleep_start, sleep_end, now, window_s)


def _http_json(url: str, token: str | None, timeout: float = 20.0):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def delivered_epoch(created_at: str) -> float | None:
    """The mailbox's naive local timestamp as an epoch, or None if unparseable.

    The mailbox writes ``%Y-%m-%dT%H:%M:%S`` in the server's local zone and
    nothing else; reading it as anything but local would shift every lag by the
    UTC offset and make a wedge look like a healthy evening.
    """
    text = str(created_at or "").strip()
    if len(text) < 19:
        return None
    try:
        return time.mktime(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return None


def proposal_stamp(target_id: str) -> int | None:
    """The epoch a proposal id carries, or None when it carries none.

    Anchored at the end exactly like ``pando.proposal_mail``: an id with no
    stamp cannot be aged, and is skipped rather than counted fresh.
    """
    match = STAMP_RE.search(str(target_id or ""))
    return int(match.group(1)) if match else None


def lag_rows(tasks: list) -> dict[str, list[tuple[float, float]]]:
    """Per member: ``(delivered_epoch, answer_lag_seconds)``, oldest first.

    Only ``[reaction]`` rows count -- same rule as ``[chat]`` and ``[proposal]``
    elsewhere: a reaction is a message, never work.
    """
    rows: dict[str, list[tuple[float, float]]] = {}
    for task in tasks or []:
        if not isinstance(task, dict):
            continue
        if not is_reaction_task(task.get("title")):
            continue
        member = str(task.get("from") or "").strip()
        if not member:
            continue
        delivered = delivered_epoch(task.get("created_at"))
        if delivered is None:
            continue
        stamp = proposal_stamp(_target_of(task))
        if stamp is None:
            continue
        rows.setdefault(member, []).append((delivered, delivered - stamp))
    for member in rows:
        rows[member].sort()
    return rows


def _target_of(task: dict) -> str:
    """The proposal id a reaction task names.

    The listing carries it in the title (``[reaction] <id> <stance> by
    <author>``); the body's ``payload.target_id`` is the signed original. Both
    are accepted so the tool works from a listing alone and from full bodies.
    """
    title = str(task.get("title") or "").lstrip()
    if title.startswith("[reaction]"):
        rest = title[len("[reaction]"):].strip()
        if rest:
            return rest.split()[0]
    body = task.get("body")
    if isinstance(body, str) and body.lstrip().startswith("{"):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return ""
        payload = parsed.get("payload") if isinstance(parsed, dict) else None
        if isinstance(payload, dict):
            return str(payload.get("target_id") or "")
    return ""


def exclude_members(rows: dict[str, list[tuple[float, float]]],
                    ignored: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Drop members whose host is explicitly outside the always-on set.

    A laptop that is allowed to sleep accumulates stale reactions while it is
    suspended. Those reactions are real mailbox traffic but not evidence that
    an always-on deployment is falling behind, so callers must be able to
    remove those members before the slope survey sees them.
    """
    ignored_names = {name for name in ignored if name}
    return {member: member_rows for member, member_rows in rows.items()
            if member not in ignored_names}


def resolve_ignored(ignore_member: list[str], sleep_gated_member: list[str],
                    host_slept: bool) -> set[str]:
    """Members to drop: the unconditional set, plus the sleep-gated set only
    while this host was asleep during the window."""
    ignored = {name for name in ignore_member if name}
    if host_slept:
        ignored.update(name for name in sleep_gated_member if name)
    return ignored


def member_slope(rows: list[tuple[float, float]], now: float,
                 window_s: float = DEFAULT_WINDOW_S,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 min_span_s: float = DEFAULT_MIN_SPAN_S,
                 slope_max: float = DEFAULT_SLOPE) -> dict:
    """Fit answer-lag against wall-clock over the tail of a member's votes.

    Returns a dict carrying ``status`` (ok/wedge/unknown) and, when a slope
    could be estimated, ``slope``, ``samples``, the span it was measured over,
    and the lag at each end so a human can see the number, not just the label.
    """
    window = [(at, lag) for at, lag in rows if at >= now - window_s]
    window.sort()
    if len(window) < min_samples:
        return {"status": UNKNOWN, "samples": len(window),
                "why": f"only {len(window)} vote(s) in the last "
                       f"{window_s / 3600.0:.1f}h (need {min_samples})"}
    first_at, first_lag = window[0]
    last_at, last_lag = window[-1]
    span = last_at - first_at
    if span < min_span_s:
        return {"status": UNKNOWN, "samples": len(window), "span_s": span,
                "why": f"{len(window)} votes span only {span:.0f}s "
                       f"(need {min_span_s:.0f}s to see a trend)"}
    slope = (last_lag - first_lag) / span
    result = {
        "samples": len(window), "span_s": span, "slope": slope,
        "first_lag_s": first_lag, "last_lag_s": last_lag,
        "window_s": window_s,
    }
    result["slope_max"] = slope_max
    if slope > slope_max:
        result["status"] = WEDGED
    else:
        result["status"] = OK
    return result


def survey(rows: dict[str, list[tuple[float, float]]], decision: list[str],
           now: float, **kwargs) -> dict[str, dict]:
    """One verdict per member under suspicion, plus every named decision member.

    A ``decision`` name that produced no reactions in the window is reported
    ``silent`` -- that is its own failure ("admitted to vote and not voting"),
    and it must not be folded into "ok" just because it has no slope to
    compute. Members seen in the feed are reported too, so a wedge cannot hide
    behind a name the caller forgot to list.
    """
    verdicts: dict[str, dict] = {}
    for member in sorted(set(list(rows) + [d for d in decision if d])):
        verdict = member_slope(rows.get(member, []), now, **kwargs)
        if verdict["status"] == UNKNOWN and member in decision \
                and not rows.get(member):
            verdict = {"status": SILENT, "samples": 0,
                       "why": "named as a decision member but cast no vote "
                              "in the window"}
        verdicts[member] = verdict
    return verdicts


def track_first_seen(state_file: Path | None, verdicts: dict, now: float) -> dict:
    """Mark each alarm "new" or "ongoing" and remember when it was first seen.

    A member that is wedged stays wedged until somebody fixes it, so a watchdog
    that pages on every poll turns one real fault into an alarm every ten
    minutes until it is buried under its own noise. Recording first-seen lets a
    caller page once on ``new`` and stay quiet on ``ongoing`` -- and a member
    that recovers is forgotten, so a *second* wedge pages again.

    Without a state file this is a no-op and every alarm reads "new".
    """
    for verdict in verdicts.values():
        if verdict["status"] in (WEDGED, SILENT):
            verdict["seen"] = "new"
    if state_file is None:
        return verdicts
    try:
        known = json.loads(state_file.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        known = {}
    members = known.get("members") if isinstance(known, dict) else None
    members = dict(members) if isinstance(members, dict) else {}

    live = set()
    for member, verdict in verdicts.items():
        if verdict["status"] not in (WEDGED, SILENT):
            continue
        live.add(member)
        first = members.get(member)
        if isinstance(first, dict) and isinstance(first.get("first_seen"), (int, float)):
            verdict["seen"] = "ongoing"
            verdict["first_seen"] = first["first_seen"]
        else:
            members[member] = {"first_seen": now,
                               "status": verdict["status"]}
            verdict["first_seen"] = now
    # Forget members that came back: a recovery must not immunise them.
    for member in list(members):
        if member not in live:
            members.pop(member, None)
    try:
        state_file.expanduser().parent.mkdir(parents=True, exist_ok=True)
        state_file.expanduser().write_text(
            json.dumps({"written": now, "members": members},
                       ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
    except OSError as exc:
        print(f"vote-watchdog: state not written: {exc}", flush=True)
    return verdicts



def write_snapshot(snapshot_file: Path | None, verdicts: dict, now: float,
                   decision: list[str], window_s: float) -> None:
    """Publish the whole judgement so that nobody has to re-derive it.

    The verdict has exactly one owner: this module. A reader that refit the
    same numbers over a shorter feed would be answering a different question
    under the same name, and the two answers would drift -- in practice toward
    the reassuring one, because less history means fewer samples and a tool
    that runs out of samples says "unknown", which reads like "no alarm". So
    the page *reads* this file instead of judging.

    Written on every pass, including the passes where every member is fine:
    "nothing is wrong" is itself a finding, and a file that only appears when
    something breaks cannot be told apart from a watchdog that stopped running.

    The first-seen memory kept by ``track_first_seen`` is deliberately a
    *different* file. That one forgets a member the moment it recovers, because
    a second wedge must page again; this one has to go on saying ``ok``.
    """
    if snapshot_file is None:
        return
    payload = {
        "written": now,
        "written_at": time.strftime("%F %T", time.localtime(now)),
        "window_s": window_s,
        "decision": sorted(d for d in decision if d),
        "members": verdicts,
    }
    try:
        path = snapshot_file.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                   sort_keys=True), encoding="utf-8")
    except OSError as exc:
        print(f"vote-watchdog: snapshot not written: {exc}", flush=True)


def _fmt_lag(seconds: float) -> str:
    return f"{seconds / 3600.0:.1f}h"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8776")
    parser.add_argument("--token-file", default="~/.pandosi/mailbox.token")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-span-seconds", type=float, default=DEFAULT_MIN_SPAN_S)
    parser.add_argument("--slope", type=float, default=DEFAULT_SLOPE,
                        help="lag growth per second of wall-clock that counts "
                             "as falling behind (default: %(default)s)")
    parser.add_argument("--decision", action="append", default=[], metavar="NAME",
                        help="repeatable: a member that must be voting")
    parser.add_argument("--ignore-member", action="append", default=[],
                        metavar="NAME",
                        help="repeatable: omit a member entirely; use for a "
                             "host outside the always-on set")
    parser.add_argument("--sleep-gated-member", action="append", default=[],
                        metavar="NAME",
                        help="repeatable: judge NAME normally, but omit it "
                             "when this host was asleep during the window "
                             "(a laptop voter checked only while awake)")
    parser.add_argument("--now", type=float, default=None,
                        help="override clock (testing only)")
    parser.add_argument("--json", action="store_true",
                        help="emit the verdicts as JSON instead of lines")
    parser.add_argument("--state-file", default="",
                        help="remember which alarms were already reported, so "
                             "a caller can page once instead of every poll")
    parser.add_argument("--snapshot-file", default=DEFAULT_SNAPSHOT,
                        help="publish the verdicts here for readers that must "
                             "not re-derive them (default: %(default)s); pass "
                             "an empty string to publish nothing")
    parser.add_argument("--page-new-only", action="store_true",
                        help="exit 2 only for an alarm that is new since the "
                             "state file was last written")
    args = parser.parse_args(argv)

    now = time.time() if args.now is None else args.now
    base = args.url.rstrip("/")
    token = _read_token(args.token_file)
    try:
        payload = _http_json(f"{base}/tasks?status=all", token)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"vote-watchdog: cannot read {base}/tasks: {exc}", flush=True)
        return 1
    tasks = (payload or {}).get("tasks") or []

    host_slept = bool(args.sleep_gated_member) and host_was_asleep(
        now, args.window_seconds)
    ignored = resolve_ignored(args.ignore_member, args.sleep_gated_member,
                              host_slept)
    decision = [d for d in args.decision if d and d not in ignored]
    rows = exclude_members(lag_rows(tasks), sorted(ignored))
    verdicts = survey(rows, decision, now,
                      window_s=args.window_seconds,
                      min_samples=args.min_samples,
                      min_span_s=args.min_span_seconds,
                      slope_max=args.slope)
    state_file = Path(args.state_file) if args.state_file else None
    track_first_seen(state_file, verdicts, now)
    write_snapshot(Path(args.snapshot_file) if args.snapshot_file else None,
                   verdicts, now, decision, args.window_seconds)

    unnamed_silent = [m for m, v in verdicts.items()
                      if v["status"] == UNKNOWN and m not in decision]
    omitted = set(unnamed_silent)

    if args.json:
        print(json.dumps({"now": now, "members": verdicts,
                          "unnamed_silent": unnamed_silent},
                         ensure_ascii=False, sort_keys=True), flush=True)
    else:
        stamp = time.strftime("%F %T", time.localtime(now))
        for member, verdict in sorted(verdicts.items()):
            if member in omitted:
                continue
            status = verdict["status"]
            if status in (WEDGED, SILENT) and verdict.get("seen") == "ongoing":
                status = f"{status}-ongoing"
            # Branch on the *base* status: a suppressed repeat ("ongoing")
            # still owes the reader the numbers that made it an alarm.
            if verdict["status"] in (OK, WEDGED):
                line = (f"{stamp} vote-watchdog {status.upper():6} {member:22} "
                        f"slope {verdict['slope']:+.2f}  "
                        f"lag {_fmt_lag(verdict['first_lag_s'])} -> "
                        f"{_fmt_lag(verdict['last_lag_s'])}  "
                        f"({verdict['samples']} votes / "
                        f"{verdict['span_s'] / 60.0:.0f}m)")
            else:
                line = (f"{stamp} vote-watchdog {status.upper():6} {member:22} "
                        f"{verdict.get('why', '')}")
            print(line, flush=True)
        if unnamed_silent:
            print(f"{stamp} vote-watchdog NOTE   no vote in the window from "
                  f"{len(unnamed_silent)}: {', '.join(sorted(unnamed_silent))}"
                  f" - proposers by design, or dead voters. Pass --decision "
                  f"NAME to turn any of them into a hard alarm.", flush=True)

    alarms = [v for v in verdicts.values() if v["status"] in (WEDGED, SILENT)]
    if not alarms:
        return 0
    if args.page_new_only and not any(v.get("seen") == "new" for v in alarms):
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
