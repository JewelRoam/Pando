"""pandosi doctor: capability report with exact fix commands.

Per ADOPTION.md: every line is [ok] / [miss] / [locked]; a miss always prints
the exact next command. Never elevates, never installs anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .graft import assign_role, capability, enumerate_execution_units


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def checks() -> list[dict]:
    """Return the canonical doctor report without printing anything."""
    rows: list[dict] = []

    def add(status: str, item: str, fix: str = "") -> None:
        rows.append({"status": status, "item": item, "fix": fix})

    ok = sys.version_info >= (3, 11)
    add("ok" if ok else "miss", "python >= 3.11",
        "" if ok else "install python 3.11+")

    for mod, fix in [("nacl", "pip install pynacl"),
                     ("zeroconf", "pip install zeroconf"),
                     ("onnxruntime", "pip install onnxruntime")]:
        add("ok" if _have(mod) else "miss", f"module {mod}",
            "" if _have(mod) else fix)

    have_ov = _have("openvino")
    add("ok" if have_ov else "miss", "openvino runtime",
        "" if have_ov else "pip install openvino (enables NPU/iGPU EUs)")

    if have_ov:
        try:
            eus = enumerate_execution_units()
            add("ok", f"execution units: {', '.join(u['id'] for u in eus)}")
        except Exception as exc:
            add("locked", "execution units", f"EU probe failed: {exc}")

    have_laya = _have("laya")
    add("ok" if have_laya else "miss", "laya package (real decisions)",
        "" if have_laya else "pip install laya")

    from .lab import detect_sandbox
    sand = detect_sandbox()
    add("ok" if sand["sandboxed"] else "locked",
        f"lab sandbox: {sand['kind']}",
        "" if sand["sandboxed"] else
        "run inside Docker/Podman/WSL for spore experiments "
        "(see THREAT-MODEL.md)")

    weights = [p for p in [Path("artifacts/laya-en.int8.onnx"),
                           Path("artifacts/laya-en.onnx")] if p.exists()]
    add("ok" if weights else "miss", "ONNX weights (artifacts/)",
        "" if weights else
        "python tools/vendor/export_onnx.py --model convaiinnovations/laya "
        "--output artifacts/laya-en.onnx")

    pb_exists = Path("assets/playbook.example.toml").exists()
    add("ok" if pb_exists else "miss", "playbook (assets/)",
        "" if pb_exists else "create assets/playbook.example.toml")

    npu_graph = Path("artifacts/laya-en-static-nonisnan.onnx")
    add("ok" if npu_graph.exists() else "miss",
        "NPU-compatible graph (IsNaN-free)",
        "" if npu_graph.exists() else
        "python tools/postprocess_onnx.py --in artifacts/laya-en-static-fixed.onnx "
        "--out artifacts/laya-en-static-nonisnan.onnx")

    cal_dir = Path("assets/calibration")
    cal_files = list(cal_dir.glob("*.json")) if cal_dir.exists() else []
    add("ok" if cal_files else "miss",
        f"calibration data ({len(cal_files)} checkpoints)",
        "" if cal_files else "python tools/generate_calibration.py")

    pb_hist = Path("assets/playbook-history.jsonl")
    add("ok" if pb_hist.exists() else "miss", "playbook history chain",
        "" if pb_hist.exists() else
        "auto-created on first rootd --playbook start")

    cap = capability()
    role = assign_role(cap)
    add("ok", f"role: {role} (ram {cap['ram_gb']}GB, cpus {cap['cpu_count']})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoSI capability report")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="print only the safe onboarding plan as JSON")
    args = parser.parse_args()

    rows = checks()
    from .onboard import build_plan
    plan = build_plan(rows)

    if args.plan or args.json:
        import json
        payload = {"plan": plan} if args.plan else {"rows": rows, "plan": plan}
        print(json.dumps(payload, indent=2))
        return

    icons = {"ok": "[ok]   ", "miss": "[miss] ", "locked": "[locked]"}
    print("pandosi doctor")
    for row in rows:
        print(f"  {icons[row['status']]} {row['item']}")
        if row["fix"]:
            print(f"           -> {row['fix']}")
    blocking = [row for row in rows if row["status"] == "miss"]
    print(f"\n{len(rows) - len(blocking)}/{len(rows)} checks passed. "
          + ("All Ring-0 requirements met." if not blocking
             else "Fix the [miss] items above to unlock more capability."))
    if plan["counts"]["nonprivileged"]:
        print("Next safe step: python -m pandosi.onboard --dry-run")


if __name__ == "__main__":
    main()
