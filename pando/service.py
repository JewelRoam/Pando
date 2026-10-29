"""Explicit service autostart plans for heterogeneous operator hosts.

This module generates lifecycle commands; it does not install services. A
plan is safe to inspect before execution and never places a token in argv.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .grants import (append_audit_entry, default_audit_path, grant_status,
                     default_path as default_grants_path)

SUPPORTED = ("windows", "linux", "termux")


def detect_platform(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if sys.platform == "win32":
        return "windows"
    prefix = sys.prefix.lower()
    if "com.termux" in prefix:
        return "termux"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unsupported"


def _token_path(role: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".pando" / f"{role}-pairing.token"


def _process_argv(role: str, platform: str, *, port: int, lan: bool,
                  root_url: str, node_id: str, token: Path,
                  working_dir: Path) -> list[str]:
    if role == "root":
        argv = [sys.executable, "-m", "pando.coordinator",
                "--port", str(port), "--pairing-token-file", str(token)]
        if lan:
            argv.append("--lan")
        return argv
    return [sys.executable, "-m", "pando.worker", "--node-id", node_id,
            "--root", root_url, "--pairing-token-file", str(token)]


def _shell(argv: list[str], platform: str) -> str:
    if platform == "windows":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _windows_paths(name: str, config_dir: Path) -> tuple[Path, Path, Path]:
    config_dir = config_dir.expanduser()
    runner = config_dir / f"{name}.ps1"
    startup_dir = Path(os.environ.get("APPDATA", "")) / \
        "Microsoft/Windows/Start Menu/Programs/Startup"
    startup = startup_dir / f"{name}.cmd"
    return runner, startup, config_dir / f"{name}.pid"


def _powershell_array(values: list[str]) -> str:
    escaped = [value.replace("'", "''") for value in values]
    return "@(" + ",".join(f"'{value}'" for value in escaped) + ")"


def _windows_runner(name: str, argv: list[str], working_dir: Path,
                    config_dir: Path) -> tuple[str, Path, Path, Path]:
    runner, startup, pid_path = _windows_paths(name, config_dir)
    arguments = _powershell_array(argv[1:])
    content = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location -LiteralPath {json.dumps(str(working_dir))}",
        f"$arguments = {arguments}",
        "$process = Start-Process -FilePath "
        f"{json.dumps(sys.executable)} -ArgumentList $arguments "
        "-WindowStyle Hidden -PassThru",
        f"Set-Content -LiteralPath {json.dumps(str(pid_path))} "
        "-Value $process.Id",
        "",
    ])
    startup_content = "\n".join([
        "@echo off",
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
        f"\"{runner}\"",
        "",
    ])
    return content, startup_content, startup, pid_path


def _windows_commands(name: str, argv: list[str], working_dir: Path,
                      config_dir: Path) -> dict[str, list[str]]:
    runner, startup, pid_path = _windows_paths(name, config_dir)
    powershell = ["powershell.exe", "-NoProfile",
                  "-ExecutionPolicy", "Bypass"]
    startup_literal = str(startup).replace("'", "''")
    pid_literal = str(pid_path).replace("'", "''")
    runner_literal = str(runner).replace("'", "''")
    return {
        "install": powershell + [
            "-Command",
            f"Get-Item -LiteralPath '{startup_literal}' -ErrorAction Stop "
            "| Out-Null",
        ],
        "start": powershell + ["-File", str(runner)],
        "status": powershell + [
            "-Command",
            f"$processId = Get-Content -LiteralPath '{pid_literal}'; "
            "Get-Process -Id $processId | "
            "Select-Object Id,ProcessName,StartTime",
        ],
        "stop": powershell + [
            "-Command",
            f"$processId = Get-Content -LiteralPath '{pid_literal}'; "
            "Stop-Process -Id $processId -Force; "
            f"Remove-Item -LiteralPath '{pid_literal}' -Force",
        ],
        "uninstall": powershell + [
            "-Command",
            f"Remove-Item -LiteralPath '{startup_literal}' -Force "
            "-ErrorAction SilentlyContinue; exit 0",
        ],
    }


def _unit_content(platform: str, argv: list[str],
                  working_dir: Path) -> str:
    executable = _shell(argv, platform)
    restart = "on-failure"
    return "\n".join([
        "[Unit]",
        "Description=PandoRSI authenticated swarm worker",
        "After=network-online.target",
        "",
        "[Service]",
        f"WorkingDirectory={working_dir}",
        f"ExecStart={executable}",
        "Environment=PYTHONUNBUFFERED=1",
        f"Restart={restart}",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])


def _posix_commands(name: str, platform: str) -> dict[str, list[str]]:
    if platform == "termux":
        service_dir = Path(sys.prefix) / "var/service" / name
        remove = shlex.quote(str(service_dir))
        return {
            "start": ["sv", "up", name],
            "status": ["sv", "status", name],
            "stop": ["sv", "down", name],
            "uninstall": ["sh", "-c",
                          f"sv down {shlex.quote(name)} 2>/dev/null || true; "
                          f"rm -rf -- {remove}"],
        }
    unit_path = Path.home() / ".config/systemd/user" / f"{name}.service"
    unit = f"{name}.service"
    remove = shlex.quote(str(unit_path))
    enable = shlex.quote(unit)
    return {
        "install": ["sh", "-c",
                    "systemctl --user daemon-reload && "
                    f"systemctl --user enable --now {enable}"],
        "start": ["systemctl", "--user", "start", unit],
        "status": ["systemctl", "--user", "status", unit],
        "stop": ["systemctl", "--user", "stop", unit],
        "uninstall": ["sh", "-c",
                      "systemctl --user disable --now "
                      f"{shlex.quote(unit)} 2>/dev/null || true; "
                      f"rm -f -- {remove}; "
                      "systemctl --user daemon-reload; "
                      "systemctl --user reset-failed 2>/dev/null || true"],
    }


def _token_prepare(platform: str, token: Path) -> list[str]:
    if platform == "windows":
        return [
            "New-Item -ItemType File -Path "
            f"'{token}' -Force",
            f"notepad '{token}'",
            f"icacls '{token}' /inheritance:r "
            "/grant:r \"$env:USERNAME:F\"",
        ]
    return [
        "umask 077",
        f"touch {shlex.quote(str(token))}",
        f"${{EDITOR:-vi}} {shlex.quote(str(token))}",
        f"chmod 600 {shlex.quote(str(token))}",
    ]


def build_service_plan(role: str = "root", *, platform: str = "auto",
                       lan: bool = False, port: int = 8080,
                       root_url: str = "http://127.0.0.1:8080",
                       node_id: str = "service-stem",
                       token_path: str | None = None,
                       name: str | None = None,
                       config_dir: Path | None = None,
                       working_dir: Path | None = None,
                       grants_path: Path | str | None = None,
                       audit: bool = True) -> dict:
    if role not in ("root", "stem"):
        raise ValueError("role must be root or stem")
    detected = detect_platform(platform)
    working = (working_dir or Path.cwd()).resolve()
    token = _token_path(role, token_path)
    service_config_dir = config_dir or Path.home() / ".pando"
    service_name = name or f"pando-{role}"
    autostart = grant_status("service_autostart", grants_path,
                             include_audit=False)
    lan_grant = grant_status("lan_listen", grants_path, include_audit=False)
    native_platform = detect_platform("auto")
    # Cross-host plans could silently embed the wrong Python, home, or PREFIX.
    # They must be generated on the target host after reviewing code there.
    native = platform == "auto" or detected == native_platform

    requirements = [
        {"name": "service_autostart", "required": True,
         "status": autostart["status"],
         "next_command": autostart["next_command"]},
        {"name": "lan_listen", "required": lan,
         "status": "not_required" if not lan else lan_grant["status"],
         "next_command": "" if not lan else lan_grant["next_command"]},
        {"name": "pairing_token_file", "required": True,
         "status": "ok" if token.exists() else "missing",
         "path": str(token),
         "next_command": "" if token.exists()
         else "create and edit the token file, then chmod/icacls it to user-only"},
    ]
    ready = all(item["status"] in ("active", "ok", "not_required")
                for item in requirements)
    ready = ready and native
    audit_path = default_audit_path(grants_path)
    result = {
        "role": role,
        "platform": detected,
        "supported": detected in SUPPORTED and native,
        "service_name": service_name,
        "working_dir": str(working),
        "lan": lan,
        "port": port if role == "root" else None,
        "root_url": None if role == "root" else root_url,
        "pairing_token_file": str(token),
        "token_in_argv": False,
        "requirements": requirements,
        "install_allowed": ready and detected in SUPPORTED,
        "files": [],
        "commands": {},
        "next_commands": [],
        "audit_path": str(audit_path),
    }

    if audit:
        try:
            decision = "planned" if ready and detected in SUPPORTED else "blocked"
            reason = ("service plan passed explicit grants and token checks"
                      if result["install_allowed"] else
                      "service plan is blocked by missing consent or token")
            append_audit_entry("service_autostart", decision, reason,
                               grants_path, audit_path,
                               autostart.get("grant"))
            result["audited"] = True
        except (OSError, PermissionError) as exc:
            result["audited"] = False
            result["audit_error"] = str(exc)
            result["install_allowed"] = False

    if not native:
        result["next_commands"].insert(0,
            f"run this planner on the target {detected} host; cross-host "
            "service plans are refused to avoid embedding wrong paths")
    elif not result["supported"]:
        result["next_commands"].append(
            f"no managed {detected} adapter; use your platform's supported "
            "user-agent mechanism after reviewing the process command")
    for item in requirements:
        if item["status"] not in ("active", "ok", "not_required"):
            result["next_commands"].append(item["next_command"])
    if not result["install_allowed"]:
        return result

    argv = _process_argv(role, detected, port=port, lan=lan,
                         root_url=root_url, node_id=node_id, token=token,
                         working_dir=working)
    result["process_argv"] = argv
    result["process_command"] = _shell(argv, detected)

    if detected == "windows":
        runner_content, startup_content, startup_path, _ = _windows_runner(
            service_name, argv, working, service_config_dir)
        result["files"] = [
            {"path": str(service_config_dir / f"{service_name}.ps1"),
             "content": runner_content},
            {"path": str(startup_path), "content": startup_content},
        ]
        result["commands"] = _windows_commands(
            service_name, argv, working, service_config_dir)
    elif detected in ("linux", "termux"):
        result["commands"] = _posix_commands(service_name, detected)
        if detected == "linux":
            unit_path = Path.home() / ".config/systemd/user" / \
                f"{service_name}.service"
            result["files"] = [{"path": str(unit_path),
                                "content": _unit_content(
                                    detected, argv, working)}]
        else:
            prefix = Path(sys.prefix)
            service_dir = prefix / "var/service" / service_name
            run_path = service_dir / "run"
            run_content = "#!/data/data/com.termux/files/usr/bin/sh\nexec " \
                + _shell(argv, "termux") + "\n"
            result["files"] = [
                {"path": str(run_path), "content": run_content,
                 "mode": "755"},
                {"path": str(service_dir / "down"), "content": "",
                 "mode": "644"},
            ]
            result["commands"]["install"] = [
                "mkdir", "-p", str(service_dir)]
            result["commands"]["install"] = [
                "sh", "-c",
                f"mkdir -p {shlex.quote(str(service_dir))} && "
                f"chmod 755 {shlex.quote(str(run_path))}"]

    # File creation is deliberately declarative. It keeps plan output reviewable.
    result["next_commands"].extend(
        _token_prepare(detected, token) if not token.exists() else [])
    result["next_commands"].append(
        "review files and commands, then run lifecycle commands manually")
    return result


def render_plan(plan: dict) -> str:
    lines = [
        f"PandoRSI {plan['role']} service plan ({plan['platform']})",
        f"Install allowed: {plan['install_allowed']}",
        f"Service: {plan['service_name']}",
        f"Token file: {plan['pairing_token_file']} (not in argv)",
        "",
        "Requirements",
    ]
    for req in plan["requirements"]:
        lines.append(f"  [{req['status']}] {req['name']}")
    if plan.get("process_command"):
        lines.extend(["", "Process", f"  {plan['process_command']}"])
    if plan["files"]:
        lines.extend(["", "Files to review"])
        for item in plan["files"]:
            lines.append(f"  {item['path']}")
    lines.extend(["", "Lifecycle commands"])
    for action, command in plan["commands"].items():
        rendered = _shell(command, plan["platform"]) \
            if plan["platform"] == "windows" else shlex.join(command)
        lines.append(f"  {action}: {rendered}")
    if plan["next_commands"]:
        lines.extend(["", "Next commands"])
        lines.extend(f"  {item}" for item in plan["next_commands"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="root", choices=["root", "stem"])
    parser.add_argument("--platform", default="auto",
                        choices=["auto", *SUPPORTED, "unsupported"])
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--root-url", default="http://127.0.0.1:8080")
    parser.add_argument("--node-id", default="service-stem")
    parser.add_argument("--token-path")
    parser.add_argument("--service-name")
    parser.add_argument("--config-dir")
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--grants-path")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        plan = build_service_plan(
            args.role, platform=args.platform, lan=args.lan, port=args.port,
            root_url=args.root_url, node_id=args.node_id,
            token_path=args.token_path,
            name=args.service_name,
            config_dir=Path(args.config_dir) if args.config_dir else None,
            working_dir=Path(args.working_dir) if args.working_dir else None,
            grants_path=Path(args.grants_path) if args.grants_path else None,
            audit=not args.no_audit)
    except (OSError, PermissionError, ValueError) as exc:
        raise SystemExit(f"[locked] {exc}")
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(render_plan(plan))


if __name__ == "__main__":
    main()
