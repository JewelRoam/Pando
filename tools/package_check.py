"""Build a wheel in an isolated venv and verify the installed CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def installed_script(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "pando.exe"
    return venv / "bin" / "pando"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="",
                        help="comma-separated extras, e.g. lan,onnx")
    parser.add_argument("--source", choices=["local", "git"], default="local")
    parser.add_argument("--git-ref", default="main")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    temp = Path(tempfile.mkdtemp(prefix="pando-package-check-"))
    venv = temp / "venv"
    report = {"temp": str(temp), "extras": args.extras,
              "source": args.source, "steps": []}
    try:
        code = run([sys.executable, "-m", "venv", str(venv)], temp)
        report["steps"].append({"step": "venv", "code": code.returncode})
        if code.returncode:
            raise RuntimeError(code.stderr)

        python = venv / ("Scripts/python.exe" if os.name == "nt"
                         else "bin/python")
        if args.source == "git":
            requirement = "pando"
            if args.extras:
                requirement += f"[{args.extras}]"
            target = (
                f"{requirement} @ "
                f"git+https://github.com/JewelRoam/PandoSI.git@{args.git_ref}")
        else:
            target = str(ROOT)
            if args.extras:
                target = f"{target}[{args.extras}]"
        report["target"] = target
        install = run(
            [str(python), "-m", "pip", "install", "--no-cache-dir", target],
            temp)
        report["steps"].append({
            "step": "install", "code": install.returncode,
            "tail": (install.stdout + install.stderr)[-2000:],
        })
        if install.returncode:
            raise RuntimeError("pip install failed")

        script = installed_script(venv)
        version = run([str(script), "--version"], temp)
        report["version"] = version.stdout.strip()
        report["steps"].append({
            "step": "version", "code": version.returncode})
        if version.returncode or not report["version"].startswith("pando "):
            raise RuntimeError(f"version failed: {version.stderr}")

        doctor = run([str(script), "doctor", "--json"], temp)
        report["steps"].append({"step": "doctor", "code": doctor.returncode})
        if doctor.returncode:
            raise RuntimeError(f"doctor failed: {doctor.stderr}")
        payload = json.loads(doctor.stdout)
        rows = {row["item"]: row["status"] for row in payload["rows"]}
        report["doctor"] = rows
        if rows.get("playbook (assets/)") != "ok":
            raise RuntimeError("packaged playbook asset was not found")
        calibration = next(
            (status for item, status in rows.items()
             if item.startswith("calibration data (")), None)
        if calibration != "ok":
            raise RuntimeError("packaged calibration assets were not found")
        if args.extras:
            for expected in ("module zeroconf", "module onnxruntime"):
                if rows.get(expected) != "ok":
                    raise RuntimeError(f"{expected} not enabled by extras")
        report["status"] = "OK"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(temp, ignore_errors=True)
            report.pop("temp", None)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
