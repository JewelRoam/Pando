"""pandosi doctor: capability report with exact fix commands.

Per ADOPTION.md: every line is [ok] / [miss] / [locked]; a miss always prints
the exact next command. Never elevates, never installs anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from .graft import assign_role, capability, enumerate_execution_units


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoSI capability report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows: list[tuple[str, str, str]] = []  # (status, item, fix)

    ok = sys.version_info >= (3, 11)
    rows.append(("ok" if ok else "miss", "python >= 3.11",
                 "" if ok else "install python 3.11+"))

    for mod, fix in [("nacl", "pip install pynacl"),
                     ("zeroconf", "pip install zeroconf"),
                     ("onnxruntime", "pip install onnxruntime")]:
        have = _have(mod)
        rows.append(("ok" if have else "miss", f"module {mod}",
                     "" if have else fix))

    have_ov = _have("openvino")
    rows.append(("ok" if have_ov else "miss", "openvino runtime",
                 "" if have_ov else "pip install openvino (enables NPU/iGPU EUs)"))

    if have_ov:
        try:
            eus = enumerate_execution_units()
            rows.append(("ok", f"execution units: {', '.join(u['id'] for u in eus)}", ""))
        except Exception as exc:
            rows.append(("locked", "execution units", f"EU probe failed: {exc}"))

    have_laya = _have("laya")
    rows.append(("ok" if have_laya else "miss", "laya package (real decisions)",
                 "" if have_laya else "pip install laya"))

    # Ring 2: sandbox detection for the lab
    from .lab import detect_sandbox
    sand = detect_sandbox()
    rows.append(("ok" if sand["sandboxed"] else "locked",
                 f"lab sandbox: {sand['kind']}",
                 "" if sand["sandboxed"] else
                 "run inside Docker/WSL for spore experiments "
                 "(see THREAT-MODEL.md)"))

    from pathlib import Path
    weights = [p for p in [Path("artifacts/laya-en.int8.onnx"),
                           Path("artifacts/laya-en.onnx")] if p.exists()]
    rows.append(("ok" if weights else "miss",
                 "ONNX weights (artifacts/)",
                 "" if weights else
                 "python tools/vendor/export_onnx.py --model convaiinnovations/laya "
                 "--output artifacts/laya-en.onnx"))

    cap = capability()
    role = assign_role(cap)
    rows.append(("ok", f"role: {role} (ram {cap['ram_gb']}GB, cpus {cap['cpu_count']})", ""))

    if args.json:
        import json
        print(json.dumps({"rows": [{"status": s, "item": i, "fix": f}
                                   for s, i, f in rows]}, indent=2))
        return

    icons = {"ok": "[ok]   ", "miss": "[miss] ", "locked": "[locked]"}
    print("pandosi doctor")
    for status, item, fix in rows:
        print(f"  {icons[status]} {item}")
        if fix:
            print(f"           -> {fix}")
    blocking = [r for r in rows if r[0] == "miss"]
    print(f"\n{len(rows) - len(blocking)}/{len(rows)} checks passed. "
          + ("All Ring-0 requirements met." if not blocking
             else "Fix the [miss] items above to unlock more capability."))


if __name__ == "__main__":
    main()
