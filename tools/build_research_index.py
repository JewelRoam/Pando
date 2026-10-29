"""Rebuild docs/archive/research/INDEX.md from the round files themselves.

One row per file: the round number parsed from the file name and the title
taken from the file's own first line. Titles are read, never inferred from
file names, so the index cannot claim something a round does not say.

Usage:
    python tools/build_research_index.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"
HEADING_PREFIX = re.compile(r"^#\s*Research Round\s+\S+\s*[-\u2014\u2013]\s*")
ROUND_IN_NAME = re.compile(r"round(\d+)([a-z]*)", re.IGNORECASE)


def describe(path: Path) -> tuple[str, str]:
    """Return (round, title) for one round file."""
    match = ROUND_IN_NAME.search(path.name)
    number = f"{match.group(1)}{match.group(2)}" if match else "-"
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0].strip() if text.strip() else ""
    if not first.startswith("#"):
        title = "(no heading)"
    else:
        title = HEADING_PREFIX.sub("", first).lstrip("# ").strip()
        title = title or "(empty heading)"
    return number, title


def sort_key(round_id: str) -> tuple[int, int, str]:
    match = re.match(r"(\d+)([a-z]*)", round_id)
    if not match:
        return (1, 0, round_id)
    return (0, int(match.group(1)), match.group(2))


def main() -> int:
    files = sorted(path for path in RESEARCH.glob("*.md")
                   if path.name != "INDEX.md")
    rows = [(path.name, *describe(path)) for path in files]
    rows.sort(key=lambda row: sort_key(row[1]))

    lines = ["# PandoSI research round index",
             "Rebuilt by tools/build_research_index.py; one row per file.",
             "",
             "| file | round | title |",
             "| --- | --- | --- |"]
    for name, number, title in rows:
        lines.append(f"| {name} | {number} | {title} |")

    without = [name for name, number, _ in rows if number == "-"]
    lines += ["", "## Files without a round number", ""]
    lines += [f"- {name}" for name in without] or ["(none)"]
    lines.append("")

    target = RESEARCH / "INDEX.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}: {len(rows)} rows, "
          f"{len(without)} without a round number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
