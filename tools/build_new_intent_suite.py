"""Build the breadth-axis suite (split F): six brand-new intent families.

Task 013 asks for a *new-intent-family* held-out suite.  The existing suites
(heldout / heldout2 / heldout3 / big_test) all draw on the same eight service
intents (refund, chargeback, exchange, complaint, praise, greeting,
login_issue, invoice).  This builder introduces six intents those suites never
contain, each with its own cue inventory, so a learner fitted on the eight old
families faces genuinely unseen classes.

By-construction labelling: every row carries exactly one cue phrase, drawn
from exactly one new family, and ``expected`` is that family.  The generator
asserts this programmatically and also runs a wording-overlap audit against the
old suites, so "zero overlap" is measured, not asserted.

Usage:
    python tools/build_new_intent_suite.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The eight families every older suite uses.  Kept here only so the overlap
# audit can assert the new suite shares no label with them.
OLD_LABELS = [
    "chargeback", "complaint", "exchange", "greeting",
    "invoice", "login_issue", "praise", "refund",
]

NEW_CUES = {
    "reschedule": [
        "move my appointment to a later slot",
        "shift the meeting to Thursday",
        "push my booking back a week",
        "the time no longer works, change it",
        "can we find another time",
        "I need to move our call earlier",
        "please reschedule the session",
        "swap my booking to the afternoon",
        "the date clashes with something, change it",
        "can you move this to next month",
        "I would like a different slot",
        "please put the meeting later in the day",
        "move my visit to the morning instead",
        "my plans changed, shift the time",
        "the appointment needs a new date",
        "bump the booking to Friday",
        "can we do it a different day",
        "I have a clash, move the meeting",
        "change the time for my slot",
        "the schedule needs to move",
    ],
    "billing_clarify": [
        "what is this charge for",
        "there are two identical lines on my bill",
        "this amount does not match the price",
        "break down the fees please",
        "why was I billed twice",
        "an unfamiliar merchant appears on my account",
        "explain these line items",
        "the total is higher than I agreed",
        "I do not recognise this charge",
        "what does this fee mean",
        "the bill shows something I did not buy",
        "tell me what I am being charged for",
        "these two charges look the same",
        "why is the amount different from the quote",
        "please clarify this line on the account",
        "the price changed and I cannot tell why",
        "I was billed twice for one order",
        "something on here is unexplained",
        "which of these is the actual total",
        "the fees are not clear",
    ],
    "permission_escalation": [
        "grant me access to the admin console",
        "I need higher privileges",
        "give me write permission on this project",
        "please upgrade my role to editor",
        "my access level is too limited",
        "add me to the security group",
        "I need rights to modify settings",
        "please authorise my account for this tool",
        "requesting elevated access for deployment",
        "give my login the ability to publish",
        "I need the new scopes added",
        "raise my permissions on the dashboard",
        "unlock the controls for my user",
        "I need to be allowed into the back end",
        "switch my role to administrator",
        "authorise me for the restricted area",
        "my team needs extra access rights",
        "enable the advanced controls for me",
        "assign me the higher role",
        "I need permission to change the config",
    ],
    "data_export": [
        "send me all the data you hold on me",
        "export my account history",
        "I want a copy of my personal information",
        "give me my records machine-readable",
        "provide everything you store about me",
        "I am exercising my portability right",
        "download all my profile data",
        "I need the full archive of my activity",
        "hand over my personal records",
        "send my information in a portable format",
        "a subject access request for my files",
        "let me take my data elsewhere",
        "give me an export of everything linked to my account",
        "I need a copy of all my stored details",
        "provide my records as a spreadsheet",
        "release my data to me under privacy rules",
        "requesting the data you keep on me",
        "I want to move my records to another provider",
        "dump my activity log to a file",
        "send a complete copy of my information",
    ],
    "content_appeal": [
        "my post was taken down unfairly",
        "reinstate my account, the ban is wrong",
        "review the removal of my comment",
        "I am appealing the takedown",
        "restore the content you deleted",
        "why was my message removed",
        "lift the restriction on my account",
        "please reconsider the moderation decision",
        "my upload was flagged and I disagree",
        "undo the suspension on my profile",
        "give me my posting rights back",
        "the removal of my video is a mistake",
        "appeal this content strike",
        "reinstate the listing you hid",
        "review the flag on my submission",
        "restore the post and my account",
        "my comment was hidden without reason",
        "overturn the content warning",
        "unblock my profile after the takedown",
        "I want the deleted thread reviewed",
    ],
    "device_triage": [
        "my laptop keeps overheating",
        "the screen flickers when I open it",
        "my phone will not hold a charge",
        "the printer says it is offline",
        "the fan is loud all the time",
        "my monitor shows no signal",
        "the machine shuts down at random",
        "the keyboard keys are unresponsive",
        "my tablet gets very hot",
        "the camera does not turn on",
        "the device beeps and stops working",
        "my mouse disconnects constantly",
        "the battery drains in an hour",
        "the laptop boots to a black screen",
        "the speaker crackles during calls",
        "my headset is not recognised",
        "the drive makes a clicking noise",
        "the tablet screen is unresponsive",
        "my charger only works sometimes",
        "the desktop restarts on its own",
    ],
}

# Fresh filler inventory that shares no 3-gram with any spike in the older
# suites (checked by the overlap audit below).
FILLERS = [
    "I would rather sort this out today.",
    "No hurry, whenever you get a moment.",
    "Let me know if you need more from me.",
    "I have the reference handy.",
    "This is somewhat time-sensitive.",
    "I can share screenshots on request.",
    "I am writing this from my desk.",
    "Thanks for looking into it.",
    "I have tried the obvious steps.",
    "Flagging it here so it is on record.",
    "I would appreciate a quick reply.",
    "It started happening this week.",
    "Nothing else seems affected.",
    "I have not changed anything recently.",
    "Reaching out before it gets worse.",
    "I am happy to run any checks you suggest.",
    "Point me at the correct team.",
    "First time raising this.",
    "Sorry to be a bother.",
    "Just want this captured.",
]

TASK_ID_PREFIX = "big7-"
CHECKPOINT = "general"
OLD_SUITE_PATHS = [
    "assets/heldout_test.jsonl",
    "assets/heldout2_test.jsonl",
    "assets/heldout3_test.jsonl",
    "assets/big_test.jsonl",
]


def norm_tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9']+", text.lower()) if len(tok) > 2]


def ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_suite(seed: int) -> list[dict]:
    rng = random.Random(seed)
    labels = sorted(NEW_CUES)
    cues = {label: list(NEW_CUES[label]) for label in labels}
    for group in cues.values():
        rng.shuffle(group)
    tasks: list[dict] = []
    index = 0
    while len(tasks) < 120:
        label = labels[index % len(labels)]
        index += 1
        if not cues[label]:
            cues[label] = list(NEW_CUES[label])
            rng.shuffle(cues[label])
        cue = cues[label].pop()
        filler = rng.choice(FILLERS)
        frame = rng.choice([
            f"User says: {cue}.",
            f"Message: {cue}.",
            f"Customer writes: {cue}.",
            f"User says: {cue}. {filler}",
            f"Message: {cue}. {filler}",
        ])
        distractors = [c for c in labels if c != label]
        old = rng.choice(OLD_LABELS if rng.random() < 0.5 else distractors)
        candidates = [label]
        pool = [c for c in labels if c != label]
        rng.shuffle(pool)
        while len(candidates) < 3:
            c = pool.pop()
            if c not in candidates:
                candidates.append(c)
        if old not in candidates:
            candidates.append(old)
        rng.shuffle(candidates)
        tasks.append({
            "task_id": f"{TASK_ID_PREFIX}{len(tasks) + 1:04d}",
            "state": frame,
            "question": "Choose the intent that best matches the message.",
            "candidates": candidates,
            "expected": label,
            "qtype": "choice",
            "checkpoint": CHECKPOINT,
        })
    return tasks


def validate_by_construction(rows: list[dict]) -> None:
    """Each row must contain exactly one new-family cue, and it must match."""
    for row in rows:
        state = row["state"].lower()
        hits = {
            label for label, group in NEW_CUES.items()
            for cue in group if cue.lower() in state
        }
        if len(hits) != 1:
            raise SystemExit(
                f"row {row['task_id']} matched {len(hits)} families: {hits}")
        if hits != {row["expected"]}:
            raise SystemExit(
                f"row {row['task_id']} cue family {hits} != "
                f"expected {row['expected']}")
        if row["candidates"][0] is None or len(row["candidates"]) not in (3, 4):
            raise SystemExit(f"row {row['task_id']} candidate arity wrong")
        if row["expected"] not in row["candidates"]:
            raise SystemExit(f"row {row['task_id']} expected not in candidates")
        if len(set(row["candidates"])) != len(row["candidates"]):
            raise SystemExit(f"row {row['task_id']} has duplicate candidates")


def overlap_audit(rows: list[dict]) -> dict:
    """Measure label and wording overlap against the four old suites."""
    old_rows: list[dict] = []
    for rel in OLD_SUITE_PATHS:
        old_rows.extend(load_rows(ROOT / rel))

    new_labels = set(NEW_CUES)
    old_labels = set(OLD_LABELS)
    label_intersection = sorted(new_labels & old_labels)

    new_expected = {row["expected"] for row in rows}
    old_expected = {row["expected"] for row in old_rows}
    expected_intersection = sorted(new_expected & old_expected)

    # Internal wording reuse: longest shared word run between two cues that
    # belong to different new families.
    max_inter_run = 0
    for a in NEW_CUES:
        for b in NEW_CUES:
            if a >= b:
                continue
            for left in NEW_CUES[a]:
                lt = norm_tokens(left)
                for right in NEW_CUES[b]:
                    rt = norm_tokens(right)
                    for size in range(1, min(len(lt), len(rt)) + 1):
                        if ngrams(lt, size) & ngrams(rt, size):
                            max_inter_run = max(max_inter_run, size)
                            break

    # Cross-suite wording reuse: longest shared word run between a new cue and
    # any old-suite state.
    old_text = [row["state"] for row in old_rows]
    max_cross_run = 0
    cross_hits = []
    for label, group in NEW_CUES.items():
        for cue in group:
            ct = norm_tokens(cue)
            for state in old_text:
                st = norm_tokens(state)
                for size in range(1, min(len(ct), len(st)) + 1):
                    if ngrams(ct, size) & ngrams(st, size):
                        if size > max_cross_run:
                            max_cross_run = size
                        if size >= 3:
                            cross_hits.append((label, cue, size))
                        break

    # Filler leakage: fillers must not reuse a 3-gram from any old state.
    filler_leaks = []
    for filler in FILLERS:
        ft = norm_tokens(filler)
        for state in old_text:
            st = norm_tokens(state)
            if ngrams(ft, 3) & ngrams(st, 3):
                filler_leaks.append(filler)
                break

    return {
        "old_suite_rows": len(old_rows),
        "new_suite_rows": len(rows),
        "label_intersection": label_intersection,
        "label_intersection_count": len(label_intersection),
        "expected_intersection": expected_intersection,
        "expected_intersection_count": len(expected_intersection),
        "max_internal_cue_run_words": max_inter_run,
        "max_cross_suite_cue_run_words": max_cross_run,
        "cross_suite_cue_runs_ge_3": len(cross_hits),
        "cross_suite_cue_hits_ge_3": cross_hits[:20],
        "filler_leak_count": len(filler_leaks),
        "filler_leaks": filler_leaks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "assets" / "heldout5_test.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = build_suite(args.seed)
    validate_by_construction(rows)
    audit = overlap_audit(rows)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} rows={len(rows)}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["expected"]] = counts.get(row["expected"], 0) + 1
    print("class distribution " + json.dumps(counts, sort_keys=True))
    print("overlap audit " + json.dumps(audit, ensure_ascii=False,
                                        sort_keys=True))
    if args.report:
        args.report.write_text(json.dumps(audit, ensure_ascii=False, indent=2)
                               + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
