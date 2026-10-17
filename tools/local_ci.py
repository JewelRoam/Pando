"""The checks in .github/workflows/ci.yml, run on this machine before a push.

CI is a second copy of this list, not the first time it runs. A red remote
run means this command was skipped.

    .venv/bin/python tools/local_ci.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# marker is a substring of one `run:` line in ci.yml. The argv is what this
# machine executes. pip installs are the runner image, not checks.
CHECKS: list[tuple[str, list[str]]] = [
    ("py_compile", ["-m", "py_compile"]),
    ("tools/check_no_weights.py", ["tools/check_no_weights.py"]),
    ("unittest discover -s tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ("tools/genesis_check.py", ["tools/genesis_check.py"]),
    ("tools/genesis_log_check.py", ["tools/genesis_log_check.py"]),
    ("tools/genesis_transport_check.py", ["tools/genesis_transport_check.py"]),
    ("tools/topology_check.py", ["tools/topology_check.py"]),
    ("pando.doctor --json", ["-m", "pando.doctor", "--json"]),
    ("-m pando --help", ["-m", "pando", "--help"]),
    ("pando doctor --plan", ["-m", "pando", "doctor", "--plan"]),
    ("-m build", ["-m", "build"]),
    ("tools/verify_wheel.py", ["tools/verify_wheel.py", "dist"]),
    ("tools/verify_sdist.py", ["tools/verify_sdist.py", "dist"]),
    ("tools/demo.py", ["tools/demo.py"]),
]


def run_lines(text: str) -> list[str]:
    """Shell commands under `run:` keys. A `run: |` block contributes each line."""
    lines = text.splitlines()
    found: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped.startswith("run: |"):
            indent = len(raw) - len(raw.lstrip())
            index += 1
            while index < len(lines):
                block = lines[index]
                if block.strip() == "":
                    index += 1
                    continue
                if (len(block) - len(block.lstrip())) <= indent:
                    break
                found.append(block.strip())
                index += 1
            continue
        if stripped.startswith("run:"):
            found.append(stripped[len("run:"):].strip())
        index += 1
    return found


def _image_setup(line: str) -> bool:
    return line.startswith("pip ") or line.startswith("python -m pip ")


def drift(workflow_text: str) -> list[str]:
    """Empty when every CI check is in CHECKS and every CHECKS marker is in CI."""
    lines = [line for line in run_lines(workflow_text) if not _image_setup(line)]
    problems: list[str] = []
    for line in lines:
        if not any(marker in line for marker, _argv in CHECKS):
            problems.append(f"ci runs {line!r}; local_ci does not")
    for marker, _argv in CHECKS:
        if not any(marker in line for line in lines):
            problems.append(f"local_ci runs {marker!r}; ci.yml does not")
    return problems


def _argv(marker: str, args: list[str]) -> list[str]:
    if marker == "py_compile":
        files = sorted(
            str(path.relative_to(ROOT))
            for path in [*ROOT.glob("pando/*.py"), *ROOT.glob("tools/*.py")]
        )
        return [sys.executable, *args, *files]
    if marker == "-m build":
        # A clean tree is what CI checks. Leftover wheels would be verified too.
        for name in ("dist", "build"):
            shutil.rmtree(ROOT / name, ignore_errors=True)
        try:
            import build  # noqa: F401
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "build"],
                cwd=ROOT, check=True)
    return [sys.executable, *args]


def main() -> int:
    problems = drift(WORKFLOW.read_text(encoding="utf-8"))
    if problems:
        print("local_ci does not match .github/workflows/ci.yml:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    for marker, args in CHECKS:
        command = _argv(marker, args)
        print(f"\n== {marker} ==", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"\nlocal_ci failed at {marker} (exit {completed.returncode})",
                  file=sys.stderr)
            return completed.returncode
    print("\nlocal_ci passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
