"""Capability probe and role assignment for heterogeneous onboarding."""

from __future__ import annotations

import os
import platform
import subprocess


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


def capability() -> dict:
    return {
        "arch": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "ram_gb": round(total_ram_gb(), 2),
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
