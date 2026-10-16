#!/usr/bin/env python3
"""Recompute PandoSI headline decisions from the in-repo recorded votes.

The tool is intentionally offline: it reads only ``assets/votes/`` and the
repository's own gate/advisor artifacts.  Missing vote files are reported,
not synthesized.  A mismatch on a row that *can* be recomputed is an error
unless ``--no-fail`` is passed.
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

VOTES = ROOT / "assets" / "votes"
MANIFEST = VOTES / "provenance.json"
COMBINER = ROOT / "assets" / "combiner.json"
SUITES = {
    "D": ROOT / "assets" / "heldout3_test.jsonl",
    "E": ROOT / "assets" / "heldout4_test.jsonl",
    "F": ROOT / "assets" / "heldout5_test.jsonl",
}

# (member, split, expected accuracy)
SOLO_EXPECTED = [
    ("english", "D", 0.7292), ("english", "E", 0.8375),
    ("english", "F", 0.8417),
    ("ft-v3", "D", 0.9417), ("ft-v3", "E", 0.9917),
    ("ft-v3", "F", 0.8667),
    ("mac-llm", "D", 0.8167), ("mac-llm", "E", 0.9167),
    ("mac-llm", "F", 0.9333),
    ("mac-llm-8b", "D", 0.8583), ("mac-llm-8b", "E", 0.9250),
    ("mac-llm-8b", "F", 0.9833),
]

# (label, split, required members, fallback, trusted member,
#  expected accuracy, neutral variant)
COMPOSITIONS = [
    ("D core + advisor", "D", ("english", "ft-v3"), "advisor", None,
     0.9750, False),
    ("D trio + assent/advisor", "D",
     ("english", "ft-v3", "mac-llm"), "advisor", "ft-v3", 0.9833, False),
    # Assent + veto has an identity worth knowing: every row's answer ends
    # up being the witness's choice (agreement keeps the majority, which
    # includes the witness; dissent demotes to the witness). decided is
    # therefore exactly the witness's own accuracy - here ft-v3, the
    # registry's selected witness for the known family (E 0.9917).
    ("E trio + assent/veto (witness ft-v3)", "E",
     ("english", "ft-v3", "mac-llm-8b"), "veto", "ft-v3",
     0.9917, True),
    ("F core + 8B witness/veto", "F",
     ("english", "ft-v3", "mac-llm-8b"), "veto", "mac-llm-8b",
     0.9833, False),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest() -> dict:
    try:
        return load_json(MANIFEST)
    except FileNotFoundError:
        raise SystemExit(f"missing manifest: {MANIFEST}")


def load_suite(split: str) -> dict[str, dict]:
    path = SUITES[split]
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows[item["task_id"]] = item
    return rows


def load_recorded(manifest: dict) -> dict[tuple[str, str], dict]:
    """Return vote files keyed by (split, member)."""

    recorded = {}
    for entry in manifest.get("files", []):
        path = VOTES / entry["file"]
        payload = load_json(path)
        rows = {}
        for row in payload.get("rows", []):
            rows[row["task_id"]] = row
        recorded[(entry["split"], entry["member"])] = {
            "entry": dict(entry, path=path),
            "rows": rows,
        }
    return recorded


def run_composition(recorded: dict, suite_rows: dict, split: str,
                    members: tuple[str, ...], fallback: str,
                    trusted: str | None, neutral: bool) -> int:
    """Return the number of correct decided rows for one composition."""

    if fallback not in {"advisor", "veto"}:
        raise ValueError(fallback)
    advisor = Combiner.load(COMBINER) if fallback == "advisor" else None
    hits = 0
    total = 0
    for task_id, task in suite_rows.items():
        present = []
        choices = {}
        for member in members:
            record = recorded.get((split, member))
            if record is None or task_id not in record["rows"]:
                return -1
            row = record["rows"][task_id]
            choice = row.get("choice")
            probability = row.get("probability", 0.5)
            if probability is None:
                probability = 0.5
            is_neutral = bool(row.get("neutral", False))
            if neutral and member.startswith("mac-llm"):
                is_neutral = probability == 0.5
            present.append(Vote(member, member, choice, float(probability),
                                neutral=is_neutral))
            choices[member] = (choice, float(probability))

        result = aggregate(present, min_quorum=len(present))
        answer = result.choice if result.tier == "auto" else None
        if (answer is not None and trusted is not None
                and choices[trusted][0] != result.choice):
            answer = None

        if answer is None:
            if fallback == "veto":
                if trusted is None:
                    return -2
                answer = choices[trusted][0]
            else:
                suggestion = advisor.suggest(
                    [(member, choices[member][0], choices[member][1])
                     for member in members],
                    task.get("candidates"), task.get("state"),
                    task.get("question"))
                answer = suggestion["choice"] if suggestion else None
        total += 1
        hits += answer == task["expected"]
    return hits if total == len(suite_rows) else -1


def fmt(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fail", action="store_true",
                        help="print mismatches but return zero")
    parser.add_argument("--strict", action="store_true",
                        help="return non-zero when a required source is missing")
    args = parser.parse_args()

    manifest = load_manifest()
    recorded = load_recorded(manifest)
    failures = []
    missing = []

    print("PandoSI recorded-vote reproduction")
    print(f"vote directory: {VOTES}")
    print("solo accuracy (choice == expected):")
    print(f"  {'member':12s} {'split':5s} {'expected':>9s} {'observed':>9s}")
    for member, split, expected in SOLO_EXPECTED:
        record = recorded.get((split, member))
        if record is None:
            observed = None
            missing.append(f"solo {member}/{split}")
        else:
            rows = record["rows"].values()
            observed = sum(row["choice"] == row["expected"] for row in rows)
            observed = observed / len(record["rows"]) if record["rows"] else None
        print(f"  {member:12s} {split:5s} {expected:9.4f} {fmt(observed):>9s}")
        if observed is None:
            continue
        expected_hits = expected * len(record["rows"])
        if abs(sum(row["choice"] == row["expected"]
                   for row in record["rows"].values()) - expected_hits) > 1.0 + 1e-9:
            failures.append(f"suite {split}: {member} observed {observed:.4f}")

    print("\ncomposition (decided accuracy; missing prerequisites are explicit):")
    print(f"  {'configuration':34s} {'expected':>9s} {'observed':>9s}")
    for label, split, members, fallback, trusted, expected, neutral in COMPOSITIONS:
        suite_rows = load_suite(split)
        observed_hits = run_composition(recorded, suite_rows, split, members,
                                         fallback, trusted, neutral)
        observed = None if observed_hits < 0 else observed_hits / len(suite_rows)
        print(f"  {label:34s} {expected:9.4f} {fmt(observed):>9s}")
        if observed is None:
            missing.append(label)
        elif abs(observed_hits - expected * len(suite_rows)) > 1.0 + 1e-9:
            failures.append(
                f"{label}: observed {observed:.4f}, expected {expected:.4f}")

    if missing:
        print("\nMISSING prerequisites:")
        for item in missing:
            print(f"  - {item}")
        print("  See assets/votes/README.md for the known missing files.")
    if failures:
        print("\nFAILED checks:")
        for item in failures:
            print(f"  - {item}")
        return 0 if args.no_fail else 1
    if missing and args.strict:
        return 2
    print("\nall available checks passed"
          + ("; missing sources were reported" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
