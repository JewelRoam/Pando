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

Both checks are best-effort and neither may fail this process. Every failure
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
      --alarm-url http://127.0.0.1:8776 --alarm-token-file ~/.pandosi/mailbox.token
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
        alarms.append(row)
    # A widening gap first (it gets worse on its own), then the quiet ones.
    alarms.sort(key=lambda row: (row["status"] != "wedge",
                                 -float(row.get("slope") or 0.0),
                                 row["member"]))
    return alarms


def alarm_title(alarms: list[dict]) -> str:
    """The mailbox title: readable in a listing, and it must read as *work*.

    ``[chat]`` and ``[proposal]`` mean "message, never work" to the
    claim-handling automation. Somebody has to actually restart a wedged
    member, so this carries ``[alarm]`` and lands as a claimable task.
    """
    named = ", ".join(row["member"] for row in alarms[:3]) or "unknown member"
    if len(alarms) > 3:
        named += f" +{len(alarms) - 3}"
    kind = "wedge" if any(row["status"] == "wedge" for row in alarms) else "silent"
    return f"[alarm] vote-{kind}: {named}"


def alarm_body(alarms: list[dict], raw: str, now: float) -> str:
    """The numbers that produced the verdict, the falsifier, and the raw feed.

    The reader has to be able to disagree with the alarm from the task alone:
    a slope with no samples, span and window behind it cannot be checked.
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
        "Expected action: restart that member's reactor (reactd) and confirm the",
        "next poll reports a fresh lag below the threshold. A member that recovers",
        "is forgotten by the state file, so a second wedge pages again.",
        "",
        "Falsifier: if the next poll puts this member's slope back under the",
        "threshold with a fresh (not merely smaller) lag, this alarm was noise -",
        "and the numbers above are what to check that against.",
        "",
        "Raw payload:",
        raw.strip()[:4000] or "(empty)",
    ]
    return "\n".join(lines)


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


def check_votes(args, now: float, stamp: str) -> None:
    """One poll of the voters, and one task if it found a new alarm."""
    verdict, text, payload = run_vote_check(args.vote_cmd, args.vote_timeout)
    if verdict == "ok":
        alarms = vote_alarms(payload)
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
        return
    if verdict == "unreadable":
        print(f"{stamp} think-watchdog vote-check UNREADABLE (not paged): "
              f"{text[:400]}", flush=True)
        return

    alarms = vote_alarms(payload)
    if not alarms:
        # Exit code 2 with no parseable row is still an alarm, but we cannot
        # name a member. Page with what we have rather than swallow it.
        alarms = [{"member": "(unparsed)", "status": "unreadable",
                   "why": "vote-watchdog exited 2 without a readable payload"}]
    title = alarm_title(alarms)
    body = alarm_body(alarms, text, now)
    print(f"{stamp} think-watchdog vote-check ALARM: {title}", flush=True)

    if not args.alarm_url:
        print(f"{stamp} think-watchdog alarm not delivered: no --alarm-url",
              flush=True)
        return
    token = ""
    if args.alarm_token_file:
        try:
            token = Path(args.alarm_token_file).expanduser().read_text(
                encoding="utf-8").strip()
        except OSError as exc:
            print(f"{stamp} think-watchdog alarm token unreadable: {exc}",
                  flush=True)
    ok, detail = post_alarm(args.alarm_url, token or None, args.alarm_by,
                            title, body)
    if ok:
        print(f"{stamp} think-watchdog alarm delivered as task {detail}",
              flush=True)
    else:
        print(f"{stamp} think-watchdog alarm NOT delivered ({detail}); "
              f"the line above is the only record", flush=True)


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
    parser.add_argument("--alarm-url", default="",
                        help="mailbox base url for the alarm task (empty=log only)")
    parser.add_argument("--alarm-token-file", default="")
    parser.add_argument("--alarm-by", default="vote-watchdog")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
