"""Run an isolated Windows service lifecycle check and always clean up.

The checker creates a temporary expiring grant and token, installs a uniquely
named loopback-only scheduled task, verifies the authenticated root answers,
then stops and deletes the task. It never uses LAN or a production task name.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.grants import append_audit_entry, grant_status, save_grant
from pandosi.service import build_service_plan


def run(command: list[str], timeout: float = 15) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_root(port: int, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/pairing", timeout=1) as resp:
                payload = json.loads(resp.read())
                if payload.get("required") is True:
                    return payload
                last_error = f"unexpected pairing payload: {payload}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise RuntimeError(f"root did not become ready on port {port}: {last_error}")


def wait_stopped(port: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/pairing", timeout=0.25) as resp:
                resp.read()
        except Exception:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0,
                        help="loopback port; 0 selects a free high port")
    parser.add_argument("--yes", action="store_true",
                        help="explicitly approve temporary task installation")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep temporary grants, token, and audit for review")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this installs a temporary Windows task; pass --yes only "
                     "after reviewing tools/service_check.py")
    if sys.platform != "win32":
        raise SystemExit("service_check currently targets native Windows")
    if args.port and (args.port < 1024 or args.port > 65535):
        parser.error("--port must be a high port")

    port = args.port or available_port()
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    task_name = f"PandoSI-ServiceCheck-{stamp}-{secrets.token_hex(3)}"
    summary = {
        "platform": "windows",
        "task_name": task_name,
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "steps": [],
    }

    with tempfile.TemporaryDirectory(prefix="pandosi-servicecheck-") as td:
        work = Path(td)
        grants_path = work / "grants.toml"
        token_path = work / "root-pairing.token"
        token_path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
        expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15)
        save_grant("service_autostart", "user-agent/pandosi-check",
                   "temporary-loopback-check", "service-check",
                   expires.isoformat(), path=grants_path)
        config_dir = work / "service-config"
        plan = build_service_plan(
            "root", platform="auto", lan=False, port=port,
            token_path=str(token_path), name=task_name,
            config_dir=config_dir, working_dir=ROOT,
            grants_path=grants_path, audit=True)
        if not plan.get("install_allowed"):
            summary["status"] = "blocked"
            summary["plan"] = plan
            print(json.dumps(summary, indent=2))
            return 2

        for declared in plan.get("files", []):
            declared_path = Path(declared["path"])
            declared_path.parent.mkdir(parents=True, exist_ok=True)
            declared_path.write_text(declared["content"], encoding="utf-8")

        commands = plan["commands"]
        if args.keep_work:
            keep = ROOT / "artifacts" / "service-checks" / task_name
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "grants.toml").write_bytes(grants_path.read_bytes())
            (keep / "token").write_bytes(token_path.read_bytes())
            (keep / "audit.jsonl").write_bytes(
                grants_path.with_name(
                    grants_path.name + ".audit.jsonl").read_bytes())
            summary["kept_artifacts"] = str(keep)

        try:
            code, output = run(commands["install"])
            summary["steps"].append({"step": "install", "code": code})
            if code:
                raise RuntimeError(f"install failed: {output}")

            code, output = run(commands["start"])
            summary["steps"].append({"step": "start", "code": code})
            if code:
                raise RuntimeError(f"start failed: {output}")
            append_audit_entry("service_autostart", "started",
                               f"temporary check task {task_name}",
                               grants_path, audit_path=None,
                               grant=grant_status(
                                   "service_autostart", grants_path,
                                   include_audit=False)["grant"])

            payload = wait_for_root(port)
            summary["pairing"] = payload
            summary["steps"].append({"step": "verify", "code": 0})
            append_audit_entry("service_autostart", "verified",
                               f"pairing endpoint ready on port {port}",
                               grants_path, audit_path=None,
                               grant=grant_status(
                                   "service_autostart", grants_path,
                                   include_audit=False)["grant"])

            code, output = run(commands["status"])
            summary["status_output"] = output[-2000:]
            summary["steps"].append({"step": "status", "code": code})

            code, output = run(commands["stop"])
            summary["steps"].append({"step": "stop", "code": code})
            if code:
                raise RuntimeError(f"stop failed: {output}")
            if not wait_stopped(port):
                raise RuntimeError("root endpoint remained reachable after stop")
            summary["steps"].append({"step": "stopped_verified", "code": 0})
            append_audit_entry("service_autostart", "stopped",
                               f"temporary check task {task_name} stopped",
                               grants_path, audit_path=None,
                               grant=grant_status(
                                   "service_autostart", grants_path,
                                   include_audit=False)["grant"])

            code, output = run(commands["uninstall"])
            summary["steps"].append({"step": "uninstall", "code": code})
            if code:
                raise RuntimeError(f"uninstall failed: {output}")
            code, output = run(["schtasks", "/Query", "/TN", task_name])
            if code == 0:
                raise RuntimeError("scheduled task still exists after uninstall")
            summary["steps"].append({"step": "uninstall_verified", "code": 0})
            append_audit_entry("service_autostart", "uninstalled",
                               f"temporary check task {task_name} deleted",
                               grants_path, audit_path=None,
                               grant=grant_status(
                                   "service_autostart", grants_path,
                                   include_audit=False)["grant"])
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            # Best-effort cleanup must not mask the original failure.
            for action in ("stop", "uninstall"):
                try:
                    code, output = run(commands[action])
                    summary["steps"].append(
                        {"step": f"cleanup_{action}", "code": code,
                         "output": output[-500:]})
                except Exception as cleanup_exc:
                    summary["steps"].append(
                        {"step": f"cleanup_{action}",
                         "error": str(cleanup_exc)})

    report_dir = ROOT / "metrics" / "service-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"{task_name}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
