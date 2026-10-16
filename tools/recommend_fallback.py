#!/usr/bin/env python3
"""Compare advisor and veto fallbacks on a set of recorded votes.

This is a small offline planner for the policy slot described in round 161.
It does not call a model or a network service: give it the English, ft-v3,
and LLM vote files, and it replays the gate plus either fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.combiner import Combiner  # noqa: E402
from pando.common import Vote, aggregate  # noqa: E402

COMBINER = ROOT / "assets" / "combiner.json"
SUITES = {
    "D": ROOT / "assets" / "heldout3_test.jsonl",
    "E": ROOT / "assets" / "heldout4_test.jsonl",
    "F": ROOT / "assets" / "heldout5_test.jsonl",
}


def load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["task_id"]: row for row in payload.get("rows", [])}
    return payload, rows


def load_suite(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["task_id"]] = item
    return rows


def accuracy(rows: dict[str, dict]) -> float:
    if not rows:
        return 0.0
    return sum(row.get("choice") == row.get("expected")
               for row in rows.values()) / len(rows)


def replay(suite: dict[str, dict], members: dict[str, dict[str, dict]],
           fallback: str, witness: str, neutral: bool) -> tuple[int, int]:
    advisor = Combiner.load(COMBINER) if fallback == "advisor" else None
    hits = 0
    decided = 0
    for task_id, task in suite.items():
        votes = []
        choices = {}
        for member, rows in members.items():
            row = rows.get(task_id)
            if row is None:
                continue
            probability = row.get("probability", 0.5)
            if probability is None:
                probability = 0.5
            vote_neutral = bool(row.get("neutral", False))
            if neutral and member.startswith("mac-llm"):
                vote_neutral = probability == 0.5
            votes.append(Vote(member, member, row["choice"],
                              float(probability), neutral=vote_neutral))
            choices[member] = (row["choice"], float(probability))
        if not votes:
            continue
        result = aggregate(votes, min_quorum=len(votes))
        answer = result.choice if result.tier == "auto" else None
        if answer is None and fallback == "veto":
            answer = choices[witness][0]
        elif answer is None:
            suggestion = advisor.suggest(
                [(member, choices[member][0], choices[member][1])
                 for member in members if member in choices],
                task.get("candidates"), task.get("state"),
                task.get("question"))
            answer = suggestion["choice"] if suggestion else None
        if answer is not None:
            decided += 1
            hits += answer == task["expected"]
    return hits, decided


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", required=True, type=Path,
                        help="recorded English vote JSON")
    parser.add_argument("--ft-v3", required=True, type=Path,
                        help="recorded ft-v3 vote JSON")
    parser.add_argument("--llm", required=True, type=Path,
                        help="recorded LLM vote JSON")
    parser.add_argument("--suite", type=Path,
                        help="suite JSONL; inferred from the LLM file when omitted")
    parser.add_argument("--witness", choices=["english", "ft-v3", "llm", "auto"],
                        default="auto",
                        help="member used by veto; auto uses highest solo accuracy")
    parser.add_argument("--neutral", action="store_true",
                        help="treat p=0.5 LLM votes as neutral confidence")
    args = parser.parse_args()

    payloads = {}
    rows = {}
    for member, path in (("english", args.english), ("ft-v3", args.ft_v3),
                          ("llm", args.llm)):
        payloads[member], rows[member] = load_rows(path)
    suite_path = args.suite
    if suite_path is None:
        suite_name = Path(str(payloads["llm"].get("suite", ""))).name
        suite_path = ROOT / "assets" / suite_name
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    suite = load_suite(suite_path)
    if not suite:
        raise SystemExit(f"empty or missing suite: {suite_path}")

    witness = args.witness
    if witness == "auto":
        witness = max(rows, key=lambda member: accuracy(rows[member]))
    member_rows = dict(rows)
    member_rows["llm"] = rows["llm"]
    advisor_hits, advisor_decided = replay(suite, member_rows, "advisor",
                                           witness, args.neutral)
    veto_hits, veto_decided = replay(suite, member_rows, "veto",
                                     witness, args.neutral)
    total = len(suite)
    print(f"suite: {suite_path}")
    print(f"witness: {witness}")
    print(f"  advisor decided: {advisor_hits}/{total} = {advisor_hits / total:.4f}")
    print(f"  veto    decided: {veto_hits}/{total} = {veto_hits / total:.4f}")
    if veto_hits > advisor_hits:
        print("recommendation: veto (witness answers every non-auto row)")
    elif advisor_hits > veto_hits:
        print("recommendation: advisor (fitted combiner resolves non-auto rows)")
    else:
        print("recommendation: either; the two fallbacks tie on this evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
