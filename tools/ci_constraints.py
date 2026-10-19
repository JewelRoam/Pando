#!/usr/bin/env python3
"""Keep the runner's dependency set identical to the one measured locally.

The workflow used to say ``pip install pynacl zeroconf onnxruntime onnx``.
That is a different test every morning: a new transitive release can turn a
green commit red without a code change. This tool makes the drift visible in
one place and, more importantly, fails on the machine that can still fix it.

    .venv/bin/python tools/ci_constraints.py --installed --workflows

``--installed`` compares the active interpreter with ``constraints.txt``.
``--workflows`` refuses a workflow that installs Python packages without
pointing pip at the same constraints file.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "constraints.txt"
WORKFLOWS = ROOT / ".github" / "workflows"
REQUIRED_ENV = "PIP_CONSTRAINT"
REQUIRED_VALUE = "${{ github.workspace }}/constraints.txt"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_constraints(path: Path = CONSTRAINTS) -> dict[str, str]:
    """Parse exact ``name==version`` pins; ranges are not constraints here."""
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"{path.name}:{number}: expected name==version, got {line!r}")
        name, expected = (part.strip() for part in line.split("==", 1))
        if not name or not expected:
            raise ValueError(f"{path.name}:{number}: empty name or version")
        key = _canonical(name)
        if key in pins:
            raise ValueError(f"{path.name}:{number}: duplicate pin for {name}")
        pins[key] = expected
    if not pins:
        raise ValueError(f"{path.name}: no pins")
    return pins


def installed_problems(pins: dict[str, str]) -> list[str]:
    """Every pin must be present at exactly that version in this interpreter."""
    problems: list[str] = []
    for name, expected in sorted(pins.items()):
        try:
            actual = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}=={expected} is pinned but not installed")
            continue
        if actual != expected:
            problems.append(f"{name} is {actual}, constraints.txt says {expected}")
    return problems


def workflow_problems(workflows: Path = WORKFLOWS) -> list[str]:
    """A pip install is reproducible only if its job imports the constraints."""
    problems: list[str] = []
    for path in sorted(workflows.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        installs = [line.strip() for line in text.splitlines()
                    if "pip install" in line and not line.lstrip().startswith("#")]
        if not installs:
            continue
        if not re.search(rf"^\s*{REQUIRED_ENV}:\s*{re.escape(REQUIRED_VALUE)}\s*$",
                         text, re.MULTILINE):
            problems.append(
                f"{path.name} installs packages without {REQUIRED_ENV}="
                f"{REQUIRED_VALUE}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--installed", action="store_true",
                        help="compare the active interpreter with constraints.txt")
    parser.add_argument("--workflows", action="store_true",
                        help="require every pip-installing workflow to use it")
    args = parser.parse_args(argv)
    if not args.installed and not args.workflows:
        parser.error("choose --installed and/or --workflows")

    problems: list[str] = []
    try:
        pins = read_constraints()
    except (OSError, ValueError) as exc:
        print(f"ci-constraints: {exc}", file=sys.stderr)
        return 1
    if args.installed:
        problems.extend(installed_problems(pins))
    if args.workflows:
        problems.extend(workflow_problems())
    if problems:
        for problem in problems:
            print(f"ci-constraints: {problem}", file=sys.stderr)
        return 1
    print(f"ci-constraints: {len(pins)} pins match and every workflow uses them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
