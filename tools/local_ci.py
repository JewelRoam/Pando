"""Every check the workflows run, run on this machine before a push.

CI is a second copy of this list, not the first time it runs. A red remote
run means this command was skipped. It mirrors **all** the push-triggered
workflows, not just ``ci.yml``: ``swarm.yml`` fires when a push touches
``pando/rootd.py`` and runs nine checks that used to be invisible here, so a
green gate could still be followed by a red run. Two steps go further than
any runner:

* the suite also runs from a depth-1 checkout, because that is what
  ``actions/checkout@v7.0.1`` hands the runner and a full local history cannot
  see the difference (see ``shallow_parity``);
* ``service.yml`` is Linux-systemd-only and is skipped here with a printed
  reason instead of silently claiming coverage.

    .venv-ci312/bin/python tools/local_ci.py

The child checks always run under the interpreter the runner uses (3.12),
never the shell's default interpreter. Resolution order is
``PANDOSI_CI_PYTHON``, ``.venv-ci312``, then the current interpreter when it is
already 3.12. A mismatch fails before any check with setup instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "ci.yml"
SWARM_WORKFLOW = WORKFLOWS / "swarm.yml"
CI_PYTHON = "3.12"

#: A workflow this machine cannot execute, and why. ``service.yml`` runs a
#: real user systemd bus; macOS has launchd and no ``systemctl --user``, so
#: claiming to have run it here would be a lie in the one place this gate is
#: supposed to be trustworthy. The reason is printed on every gate run.
PLATFORM_SKIPS: dict[str, str] = {
    "service.yml": "Linux systemd user bus; macOS has launchd (no systemctl --user)",
}


def _python_version(executable: str) -> str | None:
    """Return ``major.minor`` for an interpreter, or ``None`` if it cannot run."""
    try:
        completed = subprocess.run(
            [executable, "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, env=_ci_env())
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _ci_python() -> str:
    """The interpreter CI uses, or a loud error rather than a false green.

    The workflows pin Python 3.12. Testing here under the developer's newer
    default can hide syntax/stdlib differences and, worse, make a green local
    run meaningless for the runner. The first matching source wins.
    """
    configured = os.environ.get("PANDOSI_CI_PYTHON")
    if configured:
        candidate = shutil.which(configured) or configured
        version = _python_version(candidate)
        if version != CI_PYTHON:
            actual = version or "unavailable"
            raise RuntimeError(
                f"PANDOSI_CI_PYTHON={configured!r} is Python {actual}, "
                f"but CI uses {CI_PYTHON}")
        return candidate

    project_python = ROOT / ".venv-ci312"
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = project_python / relative
        if not candidate.exists():
            continue
        version = _python_version(str(candidate))
        if version != CI_PYTHON:
            actual = version or "unavailable"
            raise RuntimeError(
                f"{candidate} is Python {actual}, but CI uses {CI_PYTHON}; "
                "recreate .venv-ci312 with a 3.12 interpreter")
        return str(candidate)

    current = _python_version(sys.executable)
    if current == CI_PYTHON:
        return sys.executable
    actual = current or "unavailable"
    raise RuntimeError(
        f"local_ci must run child checks under Python {CI_PYTHON}, but the "
        f"current interpreter is {actual}. Set PANDOSI_CI_PYTHON to a 3.12 "
        "executable or create .venv-ci312 from one. Example: "
        "python3.12 -m venv .venv-ci312 && "
        "PIP_CONSTRAINT=constraints.txt .venv-ci312/bin/python -m pip install "
        "pynacl zeroconf onnxruntime onnx build")


def _ci_env() -> dict[str, str]:
    """The environment CI has: no inherited repository pointer.

    This script is the pre-push hook, and in a linked worktree git hands that
    hook ``GIT_DIR``. Every git call here names its repository in argv, but an
    inherited ``GIT_DIR``/``GIT_WORK_TREE`` wins over argv: on 2026-10-19 the
    depth-1 fixture committed into the real worktree that way, and with both
    variables set even ``git clone`` refuses to run ("working tree already
    exists"). A runner has neither variable, so removing them here is not a
    workaround - it is the difference between measuring this machine and
    measuring CI.
    """
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return env

# marker is a substring of one `run:` line in ci.yml. The argv is what this
# machine executes. pip installs are the runner image, not checks.
CHECKS: list[tuple[str, list[str]]] = [
    ("tools/ci_constraints.py", ["tools/ci_constraints.py", "--installed", "--workflows"]),
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

# marker is a substring of one `run:` line in swarm.yml. These fire whenever a
# push touches pando/rootd.py, pando/stemd.py and friends -- which is most
# pushes that touch the mesh. They were absent from this gate until
# 2026-10-19, so the gate could pass and the push still go red.
SWARM_CHECKS: list[tuple[str, list[str]]] = [
    ("tools/swarm_check.py", ["tools/swarm_check.py", "--yes"]),
    ("tools/slow_stem_check.py", ["tools/slow_stem_check.py", "--yes"]),
    ("tools/lan_http_check.py", ["tools/lan_http_check.py", "--yes"]),
    ("tools/remote_latency_check.py", ["tools/remote_latency_check.py", "--yes"]),
    ("tools/two_group_check.py", ["tools/two_group_check.py", "--yes"]),
    ("tools/spore_chain.py", ["tools/spore_chain.py"]),
    ("tools/spore_dag_check.py", ["tools/spore_dag_check.py"]),
    ("tools/spore_root_check.py", ["tools/spore_root_check.py"]),
    ("tools/autonomy_check.py", ["tools/autonomy_check.py", "--yes"]),
    # Ring 0 must import without site-packages, and numpy must be absent
    # under -S. Both are asserted in one CI step; both are asserted here.
    ("import numpy", ["-S", "-c", "import numpy"]),
    ("from pando.backend import MockBackend",
     ["-S", "-c", "from pando.backend import MockBackend"]),
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
    """Lines that prepare a runner rather than assert something.

    pip installs are the runner image, and ``set -euxo pipefail`` is shell
    strictness around the assertions in the same block. Neither is a check
    this gate can mirror.
    """
    return (line.startswith("pip ") or line.startswith("python -m pip ")
            or line.startswith("set -euxo pipefail"))


def drift(workflow_text: str,
          checks: "list[tuple[str, list[str]]] | None" = None,
          workflow_name: str = "ci.yml") -> list[str]:
    """Empty when every workflow check is mirrored and nothing is orphaned.

    ``checks`` defaults to ``CHECKS`` so the original single-workflow call
    keeps working; the swarm workflow passes ``SWARM_CHECKS``. A check present
    in one direction only is a problem in both directions: a remote-only step
    is the red-run this gate exists to prevent, and a local-only step means
    the gate is testing something CI never sees.
    """
    checks = CHECKS if checks is None else checks
    lines = [line for line in run_lines(workflow_text) if not _image_setup(line)]
    problems: list[str] = []
    for line in lines:
        if not any(marker in line for marker, _argv in checks):
            problems.append(f"{workflow_name} runs {line!r}; local_ci does not")
    for marker, _argv in checks:
        if not any(marker in line for line in lines):
            problems.append(f"local_ci runs {marker!r}; {workflow_name} does not")
    return problems


def _argv(marker: str, args: list[str], python: str | None = None) -> list[str]:
    """Build a check's argv, always using the CI interpreter."""
    python = python or _ci_python()
    if marker == "import numpy":
        # CI asserts numpy is *absent*: `python -S -c "import numpy" && exit 1
        # || true`. Reproduced exactly, because "the mock swarm has no numpy
        # dependency" is a property that only holds if numpy cannot load.
        return [python, "-c",
                "import subprocess, sys;"
                "rc = subprocess.run([sys.executable, '-S', '-c', 'import numpy'],"
                " capture_output=True).returncode;"
                "print('numpy absent under -S: ok' if rc else 'numpy imported under -S: FAIL');"
                "sys.exit(0 if rc else 1)"]
    if marker == "py_compile":
        files = sorted(
            str(path.relative_to(ROOT))
            for path in [*ROOT.glob("pando/*.py"), *ROOT.glob("tools/*.py")]
        )
        return [python, *args, *files]
    if marker == "-m build":
        # A clean tree is what CI checks. Leftover wheels would be verified too.
        for name in ("dist", "build"):
            shutil.rmtree(ROOT / name, ignore_errors=True)
        has_build = subprocess.run(
            [python, "-c", "import build"], cwd=ROOT, env=_ci_env(),
            capture_output=True).returncode == 0
        if not has_build:
            env = _ci_env()
            env.setdefault("PIP_CONSTRAINT", str(ROOT / "constraints.txt"))
            subprocess.run(
                [python, "-m", "pip", "install", "build"],
                cwd=ROOT, env=env, check=True)
    return [python, *args]


def _shallow_checkout(dest: Path) -> None:
    """A depth-1 copy of this tree, including what is not committed yet.

    ``git clone`` only carries commits, so the checkout lands on HEAD while
    the working tree may be ahead of it. Every path git tracks or would track
    is copied over that HEAD, so the depth-1 copy is the code about to be
    pushed rather than the code last committed.
    """
    env = _ci_env()
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", f"file://{ROOT}", str(dest)],
        check=True, env=env)
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z",
         "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True, env=env).stdout.decode("utf-8")
    for rel in listing.split("\0"):
        if not rel:
            continue
        src, dst = ROOT / rel, dest / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


def shallow_parity(python: str | None = None) -> int:
    """Run the suite where CI runs it: a checkout with no ancestry.

    ``actions/checkout@v7.0.1`` fetches one commit. On 2026-10-19 ten drift tests
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
            [python or _ci_python(), "-m", "unittest", "discover", "-s", "tests"],
            cwd=dest, env=_ci_env())
    return completed.returncode


def main() -> int:
    try:
        python = _ci_python()
    except RuntimeError as exc:
        print(f"local_ci: {exc}", file=sys.stderr)
        return 2
    print(f"local_ci: child interpreter is {python} (Python {CI_PYTHON})",
          flush=True)
    problems = drift(WORKFLOW.read_text(encoding="utf-8"))
    if problems:
        print("local_ci does not match .github/workflows/ci.yml:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    swarm = drift(SWARM_WORKFLOW.read_text(encoding="utf-8"),
                  SWARM_CHECKS, "swarm.yml")
    if swarm:
        print("local_ci does not match .github/workflows/swarm.yml:",
              file=sys.stderr)
        for problem in swarm:
            print(f"  {problem}", file=sys.stderr)
        return 1
    for name, reason in PLATFORM_SKIPS.items():
        print(f"== {name}: SKIPPED here -- {reason} ==", flush=True)
    for marker, args in [*CHECKS, *SWARM_CHECKS]:
        command = _argv(marker, args, python)
        print(f"\n== {marker} ==", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=_ci_env())
        if completed.returncode != 0:
            print(f"\nlocal_ci failed at {marker} (exit {completed.returncode})",
                  file=sys.stderr)
            return completed.returncode
    print("\n== depth-1 checkout (the tree CI is actually handed) ==",
          flush=True)
    parity = shallow_parity(python)
    if parity != 0:
        print("\nlocal_ci failed at the depth-1 checkout "
              f"(exit {parity}): a test leaned on history the runner does not "
              "have", file=sys.stderr)
        return parity
    print("\nlocal_ci passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
