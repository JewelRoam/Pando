"""The checks in .github/workflows/ci.yml, run on this machine before a push.

CI is a second copy of this list, not the first time it runs. A red remote
run means this command was skipped. The list mirrors the workflow, and one
step goes further: the suite also runs from a depth-1 checkout, because that
is what ``actions/checkout@v4`` hands the runner and a full local history
cannot see the difference (see ``shallow_parity``).

    .venv/bin/python tools/local_ci.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
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


def _shallow_checkout(dest: Path) -> None:
    """A depth-1 copy of this tree, including what is not committed yet.

    ``git clone`` only carries commits, so the checkout lands on HEAD while
    the working tree may be ahead of it. Every path git tracks or would track
    is copied over that HEAD, so the depth-1 copy is the code about to be
    pushed rather than the code last committed.
    """
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", f"file://{ROOT}", str(dest)],
        check=True)
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z",
         "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True).stdout.decode("utf-8")
    for rel in listing.split("\0"):
        if not rel:
            continue
        src, dst = ROOT / rel, dest / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


def shallow_parity() -> int:
    """Run the suite where CI runs it: a checkout with no ancestry.

    ``actions/checkout@v4`` fetches one commit. On 2026-10-19 ten drift tests
    reached for ``HEAD~1`` and errored on the runner, ten tests at a time, for
    twenty consecutive pushes, while every push was green here -- this working
    tree has the whole history and the runner has a single commit. The
    variable was never the code; it was the depth. A depth-1 clone has the
    same blindness, so the same discovery CI runs also runs here, on the
    machine that can still see the problem cheaply.
    """
    with tempfile.TemporaryDirectory(prefix="pandosi-shallow-") as tmp:
        dest = Path(tmp) / "checkout"
        _shallow_checkout(dest)
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=dest)
    return completed.returncode


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
    print("\n== depth-1 checkout (the tree CI is actually handed) ==",
          flush=True)
    parity = shallow_parity()
    if parity != 0:
        print("\nlocal_ci failed at the depth-1 checkout "
              f"(exit {parity}): a test leaned on history the runner does not "
              "have", file=sys.stderr)
        return parity
    print("\nlocal_ci passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
