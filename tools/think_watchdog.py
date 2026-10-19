#!/usr/bin/env python3
"""think_watchdog: catch the silences that keep a green light on a dead node.

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

The second silence is a *voter* answering a frozen point in the past. Its
process is up, its ledger grows, its stances vary (赞同/反对/搁置), and the panel
stays green - but every proposal it names is stamped hours ago and the gap
keeps widening. An age threshold cannot see that honestly; the slope of the
answer-lag can, and ``tools/vote_watchdog.py`` measures it. That signal existed
for a day with nobody calling it, which made it a docstring: a watchdog with no
caller catches nothing. This file is the caller, and it hangs on the timer that
is already running (launchd, every 300s) rather than inventing a third one.
Exit code 2 from vote-watchdog means "a member just started falling behind",
and that becomes a *task in the mailbox* - not another line in a log, because a
line in a log nobody reads is the same failure one level up.

The third silence is code that no longer matches the fleet's ``main``. A
member whose process booted an old commit looks perfectly alive: the ledger
grows, the stances vary, the panel is green - and every thought it contributes
is written by code the colony already corrected. ``tools/code_drift_check.py``
measures the drift (``capability.code_head`` against the local tip, with a
grace window so a fresh push is not a fault); this file is its caller, so the
same 300s timer that already runs turns it into work.

All three checks are best-effort and none may fail this process. Every failure
path exits 0 and prints a line: the job's exit code is itself a health signal,
and turning a parser bug into a red job would trade a silent fault for a noisy
one.

Deliberately dumb and stateless apart from a cooldown file: it runs every few
minutes forever, so it must not itself become a thing that can hang, wedge, or
need a human.

Usage:
  python tools/think_watchdog.py --ledger ~/.pandosi/think-ledger.jsonl \\
      --stall-seconds 900 --restart-cmd "launchctl kickstart -k gui/501/com.jewelroam.pandosi-think"

  python tools/think_watchdog.py --ledger ~/.pandosi/think-ledger.jsonl \\
      --restart-cmd "launchctl kickstart -k gui/501/com.jewelroam.pandosi-think" \\
      --vote-cmd ".venv/bin/python tools/vote_watchdog.py --url http://127.0.0.1:8776 \\
          --token-file ~/.pandosi/mailbox.token --decision mac-text --json \\
          --state-file ~/.pandosi/vote-watchdog-state.json --page-new-only" \\
      --alarm-url http://127.0.0.1:8776 --alarm-token-file ~/.pandosi/mailbox.token \
      --alarm-state-file ~/.pandosi/think-watchdog-alarms.json

``--alarm-state-file`` turns the alarm task from a snapshot into a log. Without
it an alarm that is still open is only named on stdout, and the task itself
keeps describing the numbers from the day it was filed. With it, the same task
gets a dated re-check while the fault persists, an age so a reader can tell
"just tripped" from "open since Tuesday", and - past ``--alarm-overdue-seconds``
- a plain statement that nothing automatic is coming. The body also says out
loud that the restart it asks for is only possible from a host that can reach
the member, because the alternative is a pager that keeps re-issuing an
instruction nobody can carry out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_COOLDOWN_S = 300.0
#: The vote check reads the whole task feed. A mailbox that accepts and never
#: answers must not hold this watchdog open past the next launchd interval.
DEFAULT_VOTE_TIMEOUT_S = 120.0
#: vote_watchdog's contract: 0 = every member bounded, 2 = new alarm, 1 = the
#: feed could not be read. Only 2 is somebody's fault; only 2 pages a human.
VOTE_ALARM_RC = 2
VOTE_STATUSES = ("wedge", "silent")
#: code_drift_check's contract, the same split as the vote leg: 0 = every
#: member on the tip, 2 = somebody has been behind past the grace window, 1 =
#: a root table could not be read. Only 2 is a member's fault; only 2 pages.
CODE_ALARM_RC = 2
CODE_STATUSES = ("stale",)
#: Three root tables plus a few git reads. The vote leg's 120s is the same
#: order and the same reason: a hung peer must not hold the job past launchd's
#: next interval.
DEFAULT_CODE_TIMEOUT_S = 120.0
#: One open task per fault family, not per member list. Scoping both the state
#: key and recovery to the family is what stops a clean vote poll from erasing
#: an open drift (or the reverse) - and it is also what stops a member set that
#: churns from minting a task per generation (see ``alarm_key``).
VOTE_FAMILY = "[alarm] vote-"
CODE_FAMILY = "[alarm] code-drift"
#: How often a still-open alarm gets a fresh re-check appended to the *same*
#: task instead of a second task. vote_watchdog's ``--page-new-only`` already
#: stops duplicates; what it cannot do is stop an open task from ageing into a
#: paragraph that describes a fault two days old. Six hours is short enough
#: that a reader never mistakes a stale body for the current state, and long
#: enough that a job firing every few minutes cannot flood the task.
DEFAULT_ALARM_REFRESH_S = 21600.0
#: Past this age the body stops implying the ordinary fix is on its way and
#: says plainly that nothing automatic will clear it. Twelve hours is taken
#: from the one measured incident: win-laya-int8 rewound its read cursor at
#: 2026-10-18 ~23:00 and the first human doubt was only voiced at 03:51.
DEFAULT_ALARM_OVERDUE_S = 43200.0


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


def payload_from(stdout: str) -> dict | None:
    """The last JSON object printed on stdout, or None.

    ``vote_watchdog --json`` prints exactly one line, but a tool that also warns
    on stderr is allowed to print a notice first, so the read is "last JSON
    object" rather than "the only line".
    """
    for line in reversed(str(stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def vote_alarms(payload: dict | None) -> list[dict]:
    """The wedge/silent rows of a vote-watchdog payload, worst first.

    Returns [] for anything unusable. An alarm list we cannot read is a fault
    in the pager, not in a member, and inventing one from a parse error would
    page a human about our own bug.
    """
    members = (payload or {}).get("members")
    if not isinstance(members, dict):
        return []
    alarms: list[dict] = []
    for member, verdict in members.items():
        if not isinstance(verdict, dict):
            continue
        if verdict.get("status") not in VOTE_STATUSES:
            continue
        row = dict(verdict)
        row["member"] = str(member)
        # The family is carried on the row so the one delivery path can pick a
        # body that describes the right fault without a second caller.
        row["kind"] = ("vote-wedge" if verdict.get("status") == "wedge"
                       else "vote-silent")
        alarms.append(row)
    # A widening gap first (it gets worse on its own), then the quiet ones.
    alarms.sort(key=lambda row: (row["status"] != "wedge",
                                 -float(row.get("slope") or 0.0),
                                 row["member"]))
    return alarms


def code_alarms(payload: dict | None) -> list[dict]:
    """The stale rows of a code_drift_check payload, worst first.

    "stale" is the tool's own verdict, already past its grace window, so this
    does no thresholding of its own - the honest threshold lives next to the
    only data that can set it. Rows the tool could not measure (an unreadable
    root) never reach here: they carry no member.
    """
    members = (payload or {}).get("members")
    if not isinstance(members, dict):
        return []
    alarms: list[dict] = []
    for member, row in members.items():
        if not isinstance(row, dict):
            continue
        if row.get("status") not in CODE_STATUSES:
            continue
        entry = dict(row)
        entry["member"] = str(member)
        entry["kind"] = "code-drift"
        alarms.append(entry)
    # Most commits behind first (a member deep in the past is farther from
    # any fix), then the ones that have been stuck longest.
    alarms.sort(key=lambda row: (-int(row.get("behind") or 0),
                                 -float(row.get("behind_for_s") or 0.0),
                                 row["member"]))
    return alarms


def alarm_kind(alarms: list[dict]) -> str:
    """The family of alarm. It decides the title verb *and* the body.

    Two faults share one delivery path: a voter answering the past, and a
    member running code from the past. They need different instructions, so
    the family travels with the rows rather than being re-derived from a
    status word that happens to mean different things in each tool.
    """
    kinds = [str(row.get("kind") or "") for row in alarms]
    if "code-drift" in kinds:
        return "code-drift"
    if "vote-wedge" in kinds or any(r.get("status") == "wedge" for r in alarms):
        return "vote-wedge"
    return "vote-silent"


def alarm_title(alarms: list[dict]) -> str:
    """The mailbox title: readable in a listing, and it must read as *work*.

    ``[chat]`` and ``[proposal]`` mean "message, never work" to the
    claim-handling automation. Somebody has to actually restart a wedged
    member, so this carries ``[alarm]`` and lands as a claimable task.
    """
    named = ", ".join(row["member"] for row in alarms[:3]) or "unknown member"
    if len(alarms) > 3:
        named += f" +{len(alarms) - 3}"
    return f"[alarm] {alarm_kind(alarms)}: {named}"


def alarm_family(alarms: list[dict]) -> str:
    """The stable name of the fault class this alarm belongs to.

    This is the unit the state file is keyed on, and the unit recovery clears.
    It is deliberately *not* the title.  On 2026-10-19 the first code-drift
    poll could not read the win and pi5 root tables (a two-minute LAN blip),
    so it named two members; the very next poll measured all nine and named
    five.  Keyed on the title those are two alarms: a second task appears, and
    the first one - whose body is now wrong and can never be updated, because
    no later poll will ever produce that title again - stays open forever
    saying the fault is smaller than it is.  A frozen body that reads like a
    final word is worse than a title that lags, so identity is the family and
    the member list is carried by the body, which every re-check refreshes.
    """
    return CODE_FAMILY if alarm_kind(alarms) == "code-drift" else VOTE_FAMILY


def alarm_key(alarms: list[dict]) -> str:
    """The state key for "this same alarm": the family, so a churning member
    set refreshes one task instead of minting a second and abandoning the
    first.  The mailbox title is still ``alarm_title`` - readable, and naming
    the members as of the first raise - because a title is a label, not an
    index, and no poll can rewrite one."""
    return alarm_family(alarms)


def age_phrase(seconds: float) -> str:
    """A duration a reader can parse at a glance: "48m", "7.4h", "2.1d"."""
    seconds = max(0.0, float(seconds))
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def load_alarm_state(path: Path | None) -> dict:
    """The open alarms we have already told someone about. Never raises."""
    if path is None:
        return {}
    try:
        parsed = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_alarm_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    try:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True),
                        encoding="utf-8")
    except OSError as exc:
        print(f"think-watchdog alarm state not written: {exc}", flush=True)


def escalate(cmd: str, overdue: bool) -> str:
    """Run the escalation command and return a *truthful* account of it.

    With nothing configured this says so. An alarm whose body implies that
    something chased it, when nothing exists to chase it, is the same lie as a
    green light on a dead node - so the absence is stated, not papered over.
    """
    if not cmd:
        return ("no escalation target configured (--escalate-cmd), so nothing "
                "outside this task is chasing it")
    if not overdue:
        return "not due yet (see --alarm-overdue-seconds)"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60)
    except Exception as exc:  # noqa: BLE001 - the pager must never die
        return f"escalation command raised: {exc}"
    detail = (proc.stdout or proc.stderr or "").strip().replace("\n", " ")[:200]
    return (f"escalation command rc={proc.returncode}"
            + (f": {detail}" if detail else ""))


def alarm_body(alarms: list[dict], raw: str, now: float, *,
               age_s: float | None = None, overdue: bool = False,
               note: str = "", escalated: str = "") -> str:
    """The numbers that produced the verdict, the falsifier, and the raw feed.

    The reader has to be able to disagree with the alarm from the task alone:
    a slope with no samples, span and window behind it cannot be checked.

    It also has to be honest about *who* can clear it. The first version of
    this body told every reader to "restart that member's reactor", full stop -
    which is impossible for a member on a host the issuer has no shell or
    pairing path to. win-laya-int8 sat wedged for three hours while the task
    kept re-asking for a restart nobody could perform. A pager that keeps
    re-issuing an impossible instruction teaches its reader to ignore it, so
    the action is stated together with its precondition.
    """
    stamp = time.strftime("%F %T %Z", time.localtime(now))
    lines = [f"vote-watchdog raised this at {stamp}.", ""]
    for row in alarms:
        if row["status"] == "wedge":
            lines.append(
                f"- {row['member']}: WEDGE "
                f"slope {float(row.get('slope') or 0.0):+.2f} "
                f"(threshold {float(row.get('slope_max') or 0.0):+.2f}), "
                f"lag {float(row.get('first_lag_s') or 0.0) / 3600:.1f}h -> "
                f"{float(row.get('last_lag_s') or 0.0) / 3600:.1f}h "
                f"over {int(row.get('samples') or 0)} votes / "
                f"{float(row.get('span_s') or 0.0) / 60:.0f}m "
                f"(window {float(row.get('window_s') or 0.0) / 3600:.1f}h)"
                + (f", first seen {float(row['first_seen']):.0f}"
                   if isinstance(row.get("first_seen"), (int, float)) else ""))
        else:
            lines.append(f"- {row['member']}: SILENT - {row.get('why', '')}")
    lines += [
        "",
        "What WEDGE means: this member's answer-lag is growing with wall-clock,",
        "so the proposals it is answering recede as time passes. A member keeping",
        "up answers at a roughly fixed distance behind the tip (slope ~ 0); the",
        "measured noise floor on 2026-10-19 was -0.01. This is a slope, not an",
        "age threshold, so it does not depend on the six-hour guess on the panel.",
        "",
        "Expected action: restart that member's reactor (reactd) ON THE HOST THAT",
        "RUNS IT, then confirm the next poll reports a fresh lag below the",
        "threshold. That restart needs a shell or a pairing path to that host,",
        "and the process that raised this alarm may have neither - the member",
        "can be on a machine this issuer cannot reach. Waiting does not clear it.",
        "When the host is unreachable this needs a human at that host; re-polling",
        "only re-confirms the wedge.",
    ]
    if note:
        lines += ["", f"Issuer note: {note}"]
    if age_s is not None:
        lines += ["", f"Re-check: first raised {age_phrase(age_s)} ago; still open."]
    if overdue:
        lines += [
            "",
            f"Escalation: open for {age_phrase(age_s or 0.0)} with no restart.",
            "Nothing automatic will clear it - a human hand is needed on the",
            "host that runs this member.",
            f"Escalation path: {escalated or '(none recorded)'}",
        ]
    lines += [
        "",
        "Falsifier: if the next poll puts this member's slope back under the",
        "threshold with a fresh (not merely smaller) lag, this alarm was noise -",
        "and the numbers above are what to check that against.",
        "",
        "Raw payload:",
        raw.strip()[:4000] or "(empty)",
    ]
    return "\n".join(lines)


def code_alarm_body(alarms: list[dict], raw: str, now: float, *,
                    age_s: float | None = None, overdue: bool = False,
                    note: str = "", escalated: str = "") -> str:
    """The numbers behind a drift verdict, and the cheap action that clears it.

    Two differences from the vote body matter. First, the quantity is a
    ``code_head`` stamped at *boot*, so this measures the code that produced
    the thought, not the checkout on disk - a fast-forwarded tree under an old
    process still reports the old head, and saying "the tree is current" is
    not a rebuttal. Second, the fix is cheap and bounded: fast-forward and
    restart the *client* processes only. A code change never requires reloading
    a model, which on this fleet costs minutes.
    """
    stamp = time.strftime("%F %T %Z", time.localtime(now))
    lines = [f"code-drift-check raised this at {stamp}.", ""]
    for row in alarms:
        where = (f"on {str(row.get('head') or '')[:12]}"
                 if row.get("reported") else "reports no code_head")
        detail = str(row.get("why") or "")
        if row.get("behind") is not None:
            detail = f"{row['behind']} commit(s) behind the tip"
        age = row.get("behind_for_s")
        lines.append(
            f"- {row['member']}: STALE {where} - {detail}"
            + (f", behind for {age_phrase(float(age))}" if isinstance(age, (int, float)) else ""))
    lines += [
        "",
        "What STALE means: this member's RUNNING PROCESS booted a commit that",
        "is not the tip, and it has stayed there past the grace window. The",
        "head is stamped at boot, so an up-to-date tree under an old process",
        "still reports the old commit - that is the point of reading it here.",
        "The grace window exists because one push makes the whole fleet behind",
        "for a watcher interval; only a drift that persists past it is a fault.",
        "",
        "Expected action: on the host that runs this member, fast-forward the",
        "checkout and restart the CLIENT processes (stems, reactors, proposers)",
        "- never a model server. tools/node_code_watch.py does exactly this on",
        "the node itself and needs no operator: GitHub first, the Mac's",
        "/code.bundle peer when GitHub cannot be reached. A host with no",
        "self-update entry in its supervisor needs a human at that host;",
        "re-polling only re-confirms the drift.",
    ]
    if note:
        lines += ["", f"Issuer note: {note}"]
    if age_s is not None:
        lines += ["", f"Re-check: first raised {age_phrase(age_s)} ago; still open."]
    if overdue:
        lines += [
            "",
            f"Escalation: open for {age_phrase(age_s or 0.0)} with no catch-up.",
            "Nothing automatic will clear it - a human hand is needed on the",
            "host that runs this member.",
            f"Escalation path: {escalated or '(none recorded)'}",
        ]
    lines += [
        "",
        "Falsifier: if the next poll reports the tip, this alarm retires. A",
        "member that moved to a different commit restarts its grace clock",
        "rather than inheriting it - moving is progress, even while still",
        "behind - so the numbers above are what to check that against.",
        "",
        "Raw payload:",
        raw.strip()[:4000] or "(empty)",
    ]
    return "\n".join(lines)


def build_alarm_body(alarms: list[dict], raw: str, now: float,
                     **kw) -> str:
    """Pick the body that describes *this* fault. One delivery path, no second
    caller: everything downstream only needs a title, a body and a task id."""
    if alarm_kind(alarms) == "code-drift":
        return code_alarm_body(alarms, raw, now, **kw)
    return alarm_body(alarms, raw, now, **kw)


#: One task per family means the title names the members as of the first
#: raise, which a later poll cannot rewrite. Say so where the current list is,
#: so nobody reads a lagging title as the present tense.
CURRENT_SET_NOTE = ("The task title names the members as of the first raise and "
                    "cannot be rewritten; the list below is the current one.")


def recheck_block(alarms: list[dict], now: float, age_s: float,
                  overdue: bool, note: str = "",
                  escalated: str = "") -> str:
    """The append for an alarm we already paged: short, current, dated.

    Re-posting the whole original body would bury today's numbers under
    yesterday's. This appends only what changed, and always says how old the
    alarm is, so a reader landing on the task can tell "just tripped" from
    "open since Tuesday" without reading the whole history.
    """
    stamp = time.strftime("%F %T %Z", time.localtime(now))
    lines = ["", "---", "",
             f"{stamp} RE-CHECK: first raised {age_phrase(age_s)} ago, still open.",
             CURRENT_SET_NOTE,
             ""]
    for row in alarms:
        if row["status"] == "wedge":
            lines.append(
                f"- {row['member']}: still WEDGE, slope "
                f"{float(row.get('slope') or 0.0):+.2f}, lag now "
                f"{float(row.get('last_lag_s') or 0.0) / 3600:.1f}h "
                f"({int(row.get('samples') or 0)} votes in window)")
        else:
            lines.append(f"- {row['member']}: still SILENT - {row.get('why', '')}")
    if note:
        lines += ["", f"Issuer note: {note}"]
    if overdue:
        lines += [
            "",
            f"Escalation: open {age_phrase(age_s)} with no restart. Nothing",
            "automatic will clear it; a human hand is needed on the host that",
            "runs this member.",
            f"Escalation path: {escalated or '(none recorded)'}",
        ]
    lines += ["", "Falsifier unchanged: a fresh lag under the threshold on the "
                  "next poll retires this alarm."]
    return "\n".join(lines)


def code_recheck_block(alarms: list[dict], now: float, age_s: float,
                       overdue: bool, note: str = "",
                       escalated: str = "") -> str:
    """The append for a drift alarm we already paged: short, current, dated."""
    stamp = time.strftime("%F %T %Z", time.localtime(now))
    lines = ["", "---", "",
             f"{stamp} RE-CHECK: first raised {age_phrase(age_s)} ago, still open.",
             CURRENT_SET_NOTE,
             ""]
    for row in alarms:
        where = (f"on {str(row.get('head') or '')[:12]}"
                 if row.get("reported") else "reports no code_head")
        age = row.get("behind_for_s")
        lines.append(
            f"- {row['member']}: still STALE {where}"
            + (f", behind for {age_phrase(float(age))}" if isinstance(age, (int, float)) else ""))
    if note:
        lines += ["", f"Issuer note: {note}"]
    if overdue:
        lines += [
            "",
            f"Escalation: open {age_phrase(age_s)} with no catch-up. Nothing",
            "automatic will clear it; a human hand is needed on the host that",
            "runs this member.",
            f"Escalation path: {escalated or '(none recorded)'}",
        ]
    lines += ["", "Falsifier unchanged: a poll reporting the tip retires this "
                  "alarm."]
    return "\n".join(lines)


def build_recheck_block(alarms: list[dict], now: float, age_s: float,
                        overdue: bool, **kw) -> str:
    if alarm_kind(alarms) == "code-drift":
        return code_recheck_block(alarms, now, age_s, overdue, **kw)
    return recheck_block(alarms, now, age_s, overdue, **kw)


def post_alarm(url: str, token: str | None, sender: str, title: str,
               body: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Put the alarm in the mailbox as an open task. Returns (ok, detail)."""
    data = json.dumps({"from": sender, "title": title,
                       "body": body}).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/task", data=data,
        headers={"Content-Type": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a pager that dies pages nobody
        return False, f"{type(exc).__name__}: {exc}"
    task = payload.get("task") if isinstance(payload, dict) else None
    task_id = str((task or {}).get("id") or "")
    return (bool(task_id), task_id or "no task id in the response")


def append_alarm(url: str, token: str | None, sender: str, task_id: str,
                 body: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Append a re-check to an already-open alarm task. Returns (ok, detail).

    Deliberately not a second task: two live tasks for one wedged member is how
    a queue fills with duplicates of the same fault, and the second one is
    always the one nobody reads.
    """
    data = json.dumps({"id": task_id, "by": sender,
                       "output": body}).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/append", data=data,
        headers={"Content-Type": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a pager that dies pages nobody
        return False, f"{type(exc).__name__}: {exc}"
    task = payload.get("task") if isinstance(payload, dict) else None
    return bool(task), str((task or {}).get("id") or "")


def run_vote_check(cmd: str,
                   timeout: float = DEFAULT_VOTE_TIMEOUT_S
                   ) -> tuple[str, str, dict | None]:
    """Run the vote command. Returns (verdict, text, payload).

    verdict is "ok" | "alarm" | "unreadable". "unreadable" is kept separate
    from "alarm" on purpose: a feed we cannot read is not a member falling
    behind, and conflating the two is how a pager starts lying.
    """
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the watchdog must never die
        return "unreadable", f"vote check raised: {exc}", None
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    payload = payload_from(proc.stdout or "")
    if proc.returncode == VOTE_ALARM_RC:
        return "alarm", text, payload
    if proc.returncode != 0:
        return "unreadable", text or f"vote check rc={proc.returncode}", payload
    return "ok", text, payload


def run_code_check(cmd: str,
                   timeout: float = DEFAULT_CODE_TIMEOUT_S
                   ) -> tuple[str, str, dict | None]:
    """Run the drift command. Returns (verdict, text, payload).

    Same three-way split as the vote leg, and for the same reason: a root
    table we cannot read is our blind spot, not a member's fault, and dressing
    a lookup failure up as somebody being wrong is how a pager starts lying.
    """
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the watchdog must never die
        return "unreadable", f"code check raised: {exc}", None
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    payload = payload_from(proc.stdout or "")
    if proc.returncode == CODE_ALARM_RC:
        return "alarm", text, payload
    if proc.returncode != 0:
        return "unreadable", text or f"code check rc={proc.returncode}", payload
    return "ok", text, payload


def check_code(args, now: float, stamp: str) -> None:
    """One poll of the fleet's code heads, and one task if it found a drift."""
    verdict, text, payload = run_code_check(args.code_cmd, args.code_timeout)
    if verdict == "unreadable":
        # rc=1 is *our* feed failing. The drift tool already refuses to call
        # that a member fault; the caller must not undo that by paging.
        print(f"{stamp} think-watchdog code-check UNREADABLE (not paged): "
              f"{text[:400]}", flush=True)
        return

    alarms = code_alarms(payload)
    if verdict == "ok":
        detail = ""
        if payload is not None and isinstance(payload.get("members"), dict):
            detail = f", {len(payload['members'])} members measured"
        print(f"{stamp} think-watchdog code-check ok{detail}", flush=True)
        # An open drift alarm still has to be brought up to date, and with no
        # alarms the same call clears *its own* family (never the vote leg's -
        # they share one state file).
        if args.alarm_state_file:
            page_open_alarm(args, alarms, text, now, stamp,
                            family=CODE_FAMILY)
        return

    if not alarms:
        alarms = [{"member": "(unparsed)", "status": "stale",
                   "kind": "code-drift",
                   "why": "code-drift-check exited 2 without a readable payload"}]
    print(f"{stamp} think-watchdog code-check ALARM: {alarm_title(alarms)}",
          flush=True)
    page_open_alarm(args, alarms, text, now, stamp, family=CODE_FAMILY)


def page_open_alarm(args, alarms: list[dict], raw: str, now: float,
                    stamp: str, family: str = "") -> None:
    """Deliver an alarm, or bring the task we already opened up to date.

    The state file is what makes "open since 03:00" a fact rather than a guess.
    Without it this is stateless and behaves exactly as before: a new alarm is
    posted once, and an ongoing one is only named on stdout.
    """
    state_path = (Path(args.alarm_state_file) if args.alarm_state_file else None)
    state = load_alarm_state(state_path)
    if not alarms:
        # Every member we were watching is bounded again. Forget them, so a
        # second fault later is a *new* alarm and pages again. The vote leg and
        # the code leg share one state file, so a clean *vote* poll must not
        # erase an open *drift* alarm: only keys in this family are dropped.
        recovered = [key for key in state
                     if not family or key.startswith(family)]
        if recovered:
            for key in recovered:
                state.pop(key, None)
            save_alarm_state(state_path, state)
            print(f"{stamp} think-watchdog alarm state cleared "
                  f"({len(recovered)} recovered)", flush=True)
        return

    key = alarm_key(alarms)
    entry = state.get(key)
    if not isinstance(entry, dict):
        entry = None
    if entry is None:
        # Adopt a task opened before the key became the family: those state
        # entries are keyed by title, which still carries the family prefix.
        # Without this the first poll after the change posts a second task for
        # a fault that already has one, which is the exact duplicate this key
        # exists to prevent. Newest wins when several generations piled up.
        legacy = [k for k in state
                  if k != key and family and k.startswith(family)
                  and isinstance(state.get(k), dict)]
        if legacy:
            newest = max(legacy, key=lambda k: float(
                state[k].get("issued_at") or 0.0))
            print(f"{stamp} think-watchdog adopting open alarm {newest!r} "
                  f"as {key!r}", flush=True)
            state[key] = state.pop(newest)
            entry = state[key]
            save_alarm_state(state_path, state)

    token = ""
    if args.alarm_token_file:
        try:
            token = Path(args.alarm_token_file).expanduser().read_text(
                encoding="utf-8").strip()
        except OSError as exc:
            print(f"{stamp} think-watchdog alarm token unreadable: {exc}",
                  flush=True)

    if entry is None:
        body = build_alarm_body(alarms, raw, now, note=args.alarm_note)
        if not args.alarm_url:
            print(f"{stamp} think-watchdog alarm not delivered: no --alarm-url",
                  flush=True)
            return
        ok, detail = post_alarm(args.alarm_url, token or None, args.alarm_by,
                                alarm_title(alarms), body)
        if not ok:
            print(f"{stamp} think-watchdog alarm NOT delivered ({detail}); "
                  f"the line above is the only record", flush=True)
            return
        print(f"{stamp} think-watchdog alarm delivered as task {detail}",
              flush=True)
        if state_path is not None:
            state[key] = {"task_id": detail, "issued_at": now,
                          "last_append_at": now, "appends": 0}
            save_alarm_state(state_path, state)
        return

    age_s = max(0.0, now - float(entry.get("issued_at") or now))
    since = max(0.0, now - float(entry.get("last_append_at") or 0.0))
    if since < float(args.alarm_refresh_seconds):
        print(f"{stamp} think-watchdog alarm still open ({age_phrase(age_s)}), "
              f"next re-check in "
              f"{age_phrase(float(args.alarm_refresh_seconds) - since)}",
              flush=True)
        return

    task_id = str(entry.get("task_id") or "")
    if not args.alarm_url or not task_id:
        print(f"{stamp} think-watchdog re-check skipped: no alarm url or task id",
              flush=True)
        return
    overdue = age_s >= float(args.alarm_overdue_seconds)
    escalated = escalate(args.escalate_cmd, overdue)
    body = build_recheck_block(alarms, now, age_s, overdue,
                               note=args.alarm_note, escalated=escalated)
    ok, detail = append_alarm(args.alarm_url, token or None, args.alarm_by,
                              task_id, body)
    if not ok:
        print(f"{stamp} think-watchdog re-check NOT appended ({detail})",
              flush=True)
        return
    entry["last_append_at"] = now
    entry["appends"] = int(entry.get("appends") or 0) + 1
    entry["last_seen"] = now
    state[key] = entry
    save_alarm_state(state_path, state)
    print(f"{stamp} think-watchdog re-checked task {detail} "
          f"({age_phrase(age_s)} open"
          + (", ESCALATED" if overdue else "") + ")", flush=True)


def check_votes(args, now: float, stamp: str) -> None:
    """One poll of the voters, and one task if it found a new alarm."""
    verdict, text, payload = run_vote_check(args.vote_cmd, args.vote_timeout)
    if verdict == "unreadable":
        # A feed we cannot read is our fault, not a member's. Page nobody.
        print(f"{stamp} think-watchdog vote-check UNREADABLE (not paged): "
              f"{text[:400]}", flush=True)
        return

    alarms = vote_alarms(payload)
    if verdict == "ok":
        detail = ""
        if payload is not None and isinstance(payload.get("members"), dict):
            detail = f", {len(payload['members'])} members measured"
        # rc=0 with --page-new-only means "nothing *new* to page" - it does not
        # mean every member is keeping up. A plain "ok" printed over a member
        # that is still wedged is the green light over a dead node this whole
        # file exists to prevent, so name the suppressed alarms too.
        ongoing = [row["member"] for row in alarms
                   if row.get("seen") == "ongoing"]
        if ongoing:
            detail += (f"; {len(ongoing)} known alarm(s) still ongoing: "
                       + ", ".join(ongoing[:3])
                       + (f" +{len(ongoing) - 3}" if len(ongoing) > 3 else ""))
        # Say the bound out loud: "no output" and "everyone is fine" are the
        # same exit code, and only one of them is reassuring.
        print(f"{stamp} think-watchdog vote-check ok{detail}", flush=True)
        # An alarm that is *already* open still has to be brought up to date,
        # otherwise its task keeps describing an outage whose numbers are hours
        # old and a reader cannot tell "just tripped" from "open since Tuesday".
        # With no alarms the same call *clears* the state, which is what lets a
        # later wedge be a new alarm instead of a suppressed one.
        if args.alarm_state_file:
            page_open_alarm(args, alarms, text, now, stamp,
                            family=VOTE_FAMILY)
        return

    if not alarms:
        # Exit code 2 with no parseable row is still an alarm, but we cannot
        # name a member. Page with what we have rather than swallow it.
        alarms = [{"member": "(unparsed)", "status": "unreadable",
                   "kind": "vote-silent",
                   "why": "vote-watchdog exited 2 without a readable payload"}]
    print(f"{stamp} think-watchdog vote-check ALARM: {alarm_title(alarms)}",
          flush=True)
    page_open_alarm(args, alarms, text, now, stamp, family=VOTE_FAMILY)


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
    parser.add_argument("--vote-cmd", default="",
                        help="shell command running tools/vote_watchdog.py with "
                             "--json; exit 2 means a member just started "
                             "falling behind (empty=no vote check)")
    parser.add_argument("--vote-timeout", type=float, default=DEFAULT_VOTE_TIMEOUT_S)
    parser.add_argument("--code-cmd", default="",
                        help="shell command running tools/code_drift_check.py "
                             "with --json; exit 2 means a member's running code "
                             "has been behind past the grace window "
                             "(empty=no code check)")
    parser.add_argument("--code-timeout", type=float,
                        default=DEFAULT_CODE_TIMEOUT_S)
    parser.add_argument("--alarm-url", default="",
                        help="mailbox base url for the alarm task (empty=log only)")
    parser.add_argument("--alarm-token-file", default="")
    parser.add_argument("--alarm-by", default="vote-watchdog")
    parser.add_argument("--alarm-state-file", default="",
                        help="remember the open alarm's task id here so a later "
                             "poll re-checks that task instead of describing a "
                             "fault whose numbers are stale (empty=stateless)")
    parser.add_argument("--alarm-refresh-seconds", type=float,
                        default=DEFAULT_ALARM_REFRESH_S,
                        help="minimum gap between re-check appends on one alarm")
    parser.add_argument("--alarm-overdue-seconds", type=float,
                        default=DEFAULT_ALARM_OVERDUE_S,
                        help="past this age the body says nothing automatic "
                             "will clear it and demands a human at the host")
    parser.add_argument("--alarm-note", default="",
                        help="one line the issuer adds about the delivery path, "
                             "e.g. which hosts it cannot restart a member on")
    parser.add_argument("--escalate-cmd", default="",
                        help="run this once an alarm is overdue (empty=none, "
                             "and the body says so rather than implying one ran)")
    parser.add_argument("--now", type=float, default=None,
                        help="override clock (testing only)")
    args = parser.parse_args(argv)

    now = time.time() if args.now is None else args.now
    ledger = Path(args.ledger).expanduser()
    cooldown = Path(args.cooldown_file).expanduser() if args.cooldown_file else None
    stamp = time.strftime("%F %T", time.localtime(now))

    verdict, why = decide(now, ledger, args.stall_seconds, cooldown,
                          args.cooldown_seconds)
    if verdict == "ok":
        print(f"{stamp} think-watchdog ok: {why}", flush=True)
    elif verdict == "cooldown":
        print(f"{stamp} think-watchdog STILL STALLED: {why}", flush=True)
    else:
        print(f"{stamp} think-watchdog kick: {why}", flush=True)
        try:
            proc = subprocess.run(args.restart_cmd, shell=True,
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                print(f"{stamp} think-watchdog restart failed rc={proc.returncode}: "
                      f"{(proc.stderr or proc.stdout or '').strip()[:400]}",
                      flush=True)
            else:
                print(f"{stamp} think-watchdog restarted supervisor", flush=True)
        except Exception as exc:  # noqa: BLE001 - the watchdog must never die
            print(f"{stamp} think-watchdog restart raised: {exc}", flush=True)
        if cooldown is not None:
            try:
                cooldown.parent.mkdir(parents=True, exist_ok=True)
                cooldown.write_text(f"{now}\n", encoding="utf-8")
            except OSError as exc:
                print(f"{stamp} think-watchdog cooldown not written: {exc}",
                      flush=True)

    # The voter check runs whatever the proposer did: one wedged reactor and one
    # wedged proposer are different faults and neither hides the other.
    if args.vote_cmd:
        try:
            check_votes(args, now, stamp)
        except Exception as exc:  # noqa: BLE001 - the watchdog must never die
            print(f"{stamp} think-watchdog vote-check raised: {exc}", flush=True)

    # And the third question: is the code that produced the thought the code we
    # think is running? A member can answer *now* and still be answering with a
    # checkout frozen a week ago - the two faults are independent, so neither
    # leg may hide the other.
    if args.code_cmd:
        try:
            check_code(args, now, stamp)
        except Exception as exc:  # noqa: BLE001 - the watchdog must never die
            print(f"{stamp} think-watchdog code-check raised: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
