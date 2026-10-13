"""Run an isolated, native-host service lifecycle check and always clean up.

The checker creates a temporary expiring grant and token, writes the declared
service files, starts a uniquely named loopback-only root, verifies the
authenticated endpoint, then stops and removes every lifecycle artifact. It
refuses cross-host execution so plans never embed another platform's paths.
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
from pandosi.service import SUPPORTED, build_service_plan, detect_platform


def run(command: list[str], timeout: float = 20) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            timeout=timeout)
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
    raise RuntimeError(
        f"root did not become ready on port {port}: {last_error}")


def wait_stopped(port: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/pairing",
                    timeout=0.25) as resp:
                resp.read()
        except Exception:
            return True
        time.sleep(0.2)
    return False


def uninstall_gone(platform: str, task_name: str) -> bool:
    if platform == "windows":
        code, _ = run(["schtasks", "/Query", "/TN", task_name])
        return code != 0
    if platform == "linux":
        code, _ = run(["systemctl", "--user", "status",
                       f"{task_name}.service"])
        return code != 0
    if platform == "termux":
        service_dir = Path(sys.prefix) / "var/service" / task_name
        code, _ = run(["test", "!", "-d", str(service_dir)])
        return code == 0
    return False


def audit_grant(grants_path: Path, decision: str, reason: str) -> None:
    grant = grant_status("service_autostart", grants_path,
                         include_audit=False)["grant"]
    append_audit_entry("service_autostart", decision, reason, grants_path,
                       audit_path=None, grant=grant)


def check(platform: str, args: argparse.Namespace) -> tuple[dict, Path | None]:
    port = args.port or available_port()
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    task_name = f"pandosi-servicecheck-{stamp}-{secrets.token_hex(3)}"
    summary = {
        "platform": platform,
        "task_name": task_name,
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "steps": [],
    }
    keep_path: Path | None = None

    with tempfile.TemporaryDirectory(
            prefix=f"pandosi-servicecheck-{platform}-") as td:
        work = Path(td)
        grants_path = work / "grants.toml"
        token_path = work / "root-pairing.token"
        config_dir = work / "service-config"
        token_path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
        expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15)
        save_grant("service_autostart", "user-agent/pandosi-check",
                   "temporary-loopback-check", "service-check",
                   expires.isoformat(), path=grants_path)
        plan = build_service_plan(
            "root", platform=platform, lan=False, port=port,
            token_path=str(token_path), name=task_name,
            config_dir=config_dir, working_dir=ROOT,
            grants_path=grants_path, audit=True)
        if not plan.get("install_allowed"):
            summary.update({"status": "blocked", "plan": plan})
            return summary, None

        for declared in plan.get("files", []):
            declared_path = Path(declared["path"])
            declared_path.parent.mkdir(parents=True, exist_ok=True)
            declared_path.write_text(declared["content"],
                                     encoding=declared.get("encoding",
                                                          "utf-8"))
            if declared.get("mode"):
                import os
                os.chmod(declared_path, int(str(declared["mode"]), 8))

        commands = plan["commands"]
        if args.keep_work:
            keep_path = ROOT / "artifacts" / "service-checks" / task_name
            keep_path.mkdir(parents=True, exist_ok=True)
            (keep_path / "grants.toml").write_bytes(
                grants_path.read_bytes())
            (keep_path / "token").write_bytes(token_path.read_bytes())
            (keep_path / "audit.jsonl").write_bytes(
                grants_path.with_name(
                    grants_path.name + ".audit.jsonl").read_bytes())
            for declared in plan.get("files", []):
                source = Path(declared["path"])
                if source.exists():
                    (keep_path / source.name).write_bytes(source.read_bytes())
            summary["kept_artifacts"] = str(keep_path)

        try:
            for action in ("install", "start"):
                code, output = run(commands[action])
                summary["steps"].append({"step": action, "code": code})
                if code:
                    raise RuntimeError(f"{action} failed: {output}")
            audit_grant(grants_path, "started",
                        f"temporary check service {task_name} started")

            try:
                payload = wait_for_root(port)
            except RuntimeError:
                if platform == "linux":
                    code, journal = run(["journalctl", "--user", "-u",
                                         f"{task_name}.service", "--no-pager",
                                         "-n", "80"])
                    summary["service_journal"] = journal[-4000:]
                raise
            summary["pairing"] = payload
            summary["steps"].append({"step": "verify", "code": 0})
            audit_grant(grants_path, "verified",
                        f"pairing endpoint ready on port {port}")

            code, output = run(commands["status"])
            summary["status_output"] = output[-2000:]
            summary["steps"].append({"step": "status", "code": code})

            code, output = run(commands["stop"])
            summary["steps"].append({"step": "stop", "code": code})
            if code:
                raise RuntimeError(f"stop failed: {output}")
            if not wait_stopped(port):
                raise RuntimeError(
                    "root endpoint remained reachable after stop")
            summary["steps"].append({"step": "stopped_verified", "code": 0})
            audit_grant(grants_path, "stopped",
                        f"temporary check service {task_name} stopped")

            code, output = run(commands["uninstall"])
            summary["steps"].append({"step": "uninstall", "code": code})
            if code:
                raise RuntimeError(f"uninstall failed: {output}")
            if not uninstall_gone(platform, task_name):
                raise RuntimeError(
                    "service artifacts remained after uninstall")
            summary["steps"].append({"step": "uninstall_verified", "code": 0})
            audit_grant(grants_path, "uninstalled",
                        f"temporary check service {task_name} removed")
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            for action in ("stop", "uninstall"):
                try:
                    code, output = run(commands[action])
                    summary["steps"].append(
                        {"step": f"cleanup_{action}", "code": code,
                         "output": output[-800:]})
                except Exception as cleanup_exc:
                    summary["steps"].append(
                        {"step": f"cleanup_{action}",
                         "error": str(cleanup_exc)})
    return summary, keep_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="auto",
                        choices=["auto", *SUPPORTED])
    parser.add_argument("--port", type=int, default=0,
                        help="loopback port; 0 selects a free high port")
    parser.add_argument("--yes", action="store_true",
                        help="explicitly approve temporary service installation")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep temporary grants, token, and audit")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this installs a temporary service; pass --yes only "
                     "after reviewing tools/service_check.py")
    if args.port and (args.port < 1024 or args.port > 65535):
        parser.error("--port must be a high port")

    native = detect_platform("auto")
    platform = native if args.platform == "auto" else args.platform
    if platform != native:
        raise SystemExit(
            f"[blocked] requested platform {platform} differs from native "
            f"host {native}; run the checker on the target host")

    summary, _ = check(platform, args)
    report_dir = ROOT / "metrics" / "service-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"{summary['task_name']}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
