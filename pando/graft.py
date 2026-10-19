"""Capability probe and role assignment for heterogeneous onboarding."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def total_ram_gb() -> float:
    """Best-effort total physical memory in GB."""
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024 ** 3
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


#: A pip-installed member has no checkout to ask. Whoever runs the install
#: leaves this file beside the module it copied, and it is the only honest
#: answer available there. Pinned against ``tools/code_sync.py`` by a test.
CODE_HEAD_STAMP = "_code_head.txt"


def installed_code_head(package_dir: Path | None = None) -> str:
    """The commit recorded beside an installed copy of this package.

    ``code_head()`` probes ``git`` from the directory the module lives in, so
    it cannot answer for a pip-installed member: pip copies the tree and never
    copies ``.git``. Measured 2026-10-19 -- every pi5 stem reported an empty
    ``code_head`` for that reason alone, while its venv was in fact current.
    "Cannot say" and "out of date" then look identical from every peer, which
    is a worse failure than being out of date.
    """
    stamp = (package_dir or Path(__file__).resolve().parent) / CODE_HEAD_STAMP
    try:
        return stamp.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def code_head(root: Path | None = None) -> str:
    """The commit a checkout is on, so a peer can see who is running old code.

    A member that cannot say which code it runs is a member nobody can audit:
    on 2026-10-18 the only way to learn that the Windows node was four commits
    behind was to ask it. Probed from the repo this module lives in unless a
    root is given.

    Never a startup blocker: no git, no checkout, or an unparsable answer all
    yield "". GIT_DIR / GIT_WORK_TREE are stripped first, because a git hook
    (the pre-push gate runs this same tree) exports both and would otherwise
    make an unrelated repository answer for us.

    When there is no checkout to probe *and* the question is about this very
    module, fall back to the installer's stamp. An explicit ``root`` is a
    question about that checkout and is never answered with somebody else's
    commit.
    """
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or ROOT), env=env,
            capture_output=True, text=True, timeout=5)
        head = done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        head = ""
    if head:
        return head
    return "" if root is not None else installed_code_head()


_CODE_HEAD: str | None = None


def running_commit() -> str:
    """The commit this process is running, probed once and then remembered.

    Deliberately memoized: the honest answer is the code that was loaded when
    the process started, not whatever the checkout has been fast-forwarded to
    since. A stem that updated underneath itself still runs the old bytes
    until it restarts, and has to say so.
    """
    global _CODE_HEAD
    if _CODE_HEAD is None:
        _CODE_HEAD = code_head()
    return _CODE_HEAD


def capability() -> dict:
    return {
        "arch": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "ram_gb": round(total_ram_gb(), 2),
        # Top-level fact, not decoration: roots show it per member so an
        # out-of-date node is visible from any peer without asking it.
        "code_head": running_commit(),
    }


def assign_role(cap: dict) -> str:
    """worker runs the decision model; observer only relays/benchmarks."""
    if cap.get("ram_gb", 0) >= 3.5 and cap.get("cpu_count", 0) >= 2:
        return "worker"
    return "observer"


def enumerate_execution_units() -> list[dict]:
    """Enumerate ExecutionUnits (blueprint v0.6): CPU, iGPU, dGPU, NPU.

    Prefers OpenVINO device enumeration when installed; falls back to
    best-effort detection so a bare CPU box still reports one EU.
    Virtual devices (AUTO/HETERO/MULTI) are excluded.
    """
    units: list[dict] = []
    try:
        from openvino import Core  # type: ignore

        core = Core()
        for dev in core.available_devices:
            if dev.split(":")[0] in ("AUTO", "HETERO", "MULTI"):
                continue
            try:
                full_name = core.get_property(dev, "FULL_DEVICE_NAME")
            except Exception:
                full_name = dev
            units.append({"id": dev, "kind": dev.split(":")[0],
                          "name": full_name, "backend": "openvino"})
        if units:
            return units
    except ImportError:
        pass

    units.append({"id": "CPU", "kind": "CPU",
                  "name": platform.processor() or "cpu", "backend": "fallback"})
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=4)
        if out.returncode == 0:
            for i, name in enumerate(out.stdout.strip().splitlines()):
                units.append({"id": f"GPU.{i}", "kind": "GPU",
                              "name": name.strip(), "backend": "nvidia-smi"})
    except (OSError, subprocess.TimeoutExpired):
        pass
    if platform.system() == "Darwin":
        units.append({"id": "GPU.metal", "kind": "GPU",
                      "name": "apple-silicon-gpu", "backend": "metal"})
    return units
