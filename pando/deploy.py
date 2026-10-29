"""Render per-host service definitions from the untracked host config.

The repository ships no unit files: a checked-in plist would carry one
operator's paths and talk the next host into adopting them. What the repo
owns is the *catalogue* below -- which services exist, what program they
run, and whether the manager should keep them alive -- because that is
safety knowledge (a model server is absent on purpose: restarting one
reloads gigabytes on a small board). What the host owns is the *wiring*:
paths, URLs, node ids and tokens live in ``host.toml`` and never in git.

Config schema::

    [render]
    python = "/host/a/venv/bin/python"    # interpreter for pando services
    repo = "/host/a/Pando"                # checkout path
    state_dir = "/host/a/.pando"
    outdir = "~/.config/systemd/user"        # or ~/Library/LaunchAgents

    [services.think]
    args = "--node-id pi5-proposer --base-url http://127.0.0.1:8085/v1 ..."

    [services.code-sync]
    args = "--repo ... --peer ..."
    interval = 300                            # presence makes it periodic

Rendering is pure text substitution: ``{python}``, ``{repo}`` and
``{state_dir}`` resolve from ``[render]``; anything else the service needs
belongs in that service's ``args``.
"""

from __future__ import annotations

import shlex
import stat
from pathlib import Path

from . import hostconfig


class DeployError(ValueError):
    """The host config cannot produce a complete unit set."""


#: name -> (program, keepalive). ``program`` runs after ``{python}``.
#: A service not listed here cannot be rendered: adding one is a code
#: change, which is what keeps a rendered unit from starting arbitrary
#: software. Model servers (llama-server, cloudflared) are deliberately
#: absent -- no updater may restart them.
SERVICES: dict[str, tuple[str, bool]] = {
    "think": ("-m pando.proposer", True),
    "react": ("-m pando.reactor", True),
    "chat": ("-m pando.messenger", True),
    "root": ("-m pando.coordinator", True),
    "execute": ("{repo}/tools/execute_loop.py 600", True),
    "mailbox": ("{repo}/tools/agent_mailbox.py", True),
    "code-sync": ("{repo}/tools/code_sync.py", False),
    # A oneshot watcher: it must never be kept alive (a resident updater
    # keeps running the updater code it replaced) and never restarted by
    # the fast-forward -- it re-executes itself on the next interval.
    "code-watch": ("{repo}/tools/node_code_watch.py --once", False),
    "seed-pull": ("{repo}/tools/seed_pull.py", False),
    "think-watchdog": ("{repo}/tools/think_watchdog.py", False),
    "stem-text": ("-m pando.worker", True),
    "stem-laya": ("-m pando.worker", True),
    "stem-llm": ("-m pando.worker", True),
}

_LINUX_TEMPLATE = """[Unit]
Description=pando {name}
After=network-online.target

[Service]
ExecStart={exec}
{restart}WorkingDirectory={repo}

[Install]
WantedBy=default.target
"""

_TIMER_TEMPLATE = """[Unit]
Description=pando {name} schedule

[Timer]
OnBootSec=30
OnUnitActiveSec={interval}

[Install]
WantedBy=timers.target
"""

_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
{arguments}
  </array>
  <key>RunAtLoad</key><true/>
  {keepalive}{interval}
  <key>WorkingDirectory</key><string>{repo}</string>
  <key>StandardOutPath</key><string>{state_dir}/logs/{name}.out</string>
  <key>StandardErrorPath</key><string>{state_dir}/logs/{name}.err</string>
</dict>
</plist>
"""


def _platform_id(platform: str) -> str:
    if platform and platform != "auto":
        return platform
    import sys
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def _render_context(config: dict) -> dict:
    render = config.get("render")
    if not isinstance(render, dict):
        raise DeployError("host config has no [render] section")
    context = {}
    for key in ("python", "repo", "state_dir"):
        value = render.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DeployError(f"host config render.{key} must be a non-empty string")
        context[key] = str(Path(value.strip()).expanduser())
    return context


def _service_table(config: dict) -> dict:
    services = config.get("services")
    if not isinstance(services, dict) or not services:
        raise DeployError("host config has no [services.*] entries")
    return services


def render_units(config: dict, platform: str = "auto") -> dict[str, str]:
    """Render every configured service; return ``{filename: content}``."""
    plat = _platform_id(platform)
    if plat == "windows":
        raise DeployError(
            "windows has no unit directory; its roster is the supervisor "
            "script inside the checkout")
    context = _render_context(config)
    units: dict[str, str] = {}
    for name, spec in _service_table(config).items():
        if name not in SERVICES:
            raise DeployError(
                f"unknown service {name!r}; the catalogue in pando.deploy "
                "is the boundary of what may run")
        if not isinstance(spec, dict):
            raise DeployError(f"[services.{name}] must be a table")
        args = spec.get("args", "")
        if not isinstance(args, str):
            raise DeployError(f"services.{name}.args must be a string")
        program, keepalive = SERVICES[name]
        interval = spec.get("interval")
        argv = ([context["python"]]
                + shlex.split(program.format(**context))
                + shlex.split(args))
        if plat == "linux":
            units[f"pando-{name}.service"] = _render_systemd(
                name, argv, context, keepalive)
            if interval is not None:
                units[f"pando-{name}.timer"] = _TIMER_TEMPLATE.format(
                    name=name, interval=int(interval))
        else:
            label = f"com.pando.{name}"
            units[f"{label}.plist"] = _render_plist(
                name, label, argv, context, keepalive, interval)
    return units


def _render_systemd(name: str, argv: list[str], context: dict,
                    keepalive: bool) -> str:
    restart = "Restart=always\nRestartSec=5\n" if keepalive else ""
    return _LINUX_TEMPLATE.format(
        name=name, exec=" ".join(shlex.quote(a) for a in argv),
        restart=restart, repo=context["repo"])


def _render_plist(name: str, label: str, argv: list[str], context: dict,
                  keepalive: bool, interval) -> str:
    arguments = "\n".join(f"    <string>{_xml(a)}</string>" for a in argv)
    keep = "  <key>KeepAlive</key><true/>\n" if keepalive else ""
    timer = (f"  <key>StartInterval</key><integer>{int(interval)}</integer>\n"
             if interval is not None else "")
    return _PLIST_TEMPLATE.format(
        name=name, label=label, arguments=arguments, keepalive=keep,
        interval=timer, repo=context["repo"], state_dir=context["state_dir"])


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def install_units(config: dict, target: str | Path, *,
                  platform: str = "auto") -> dict:
    """Render into ``target``, replacing only files whose content changed.

    A previous copy is left beside the new one as ``<name>.bak``: the one
    artifact that makes a bad render recoverable by hand.
    """
    units = render_units(config, platform)
    directory = Path(target).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    for filename, content in sorted(units.items()):
        path = directory / filename
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if path.exists():
            backup = path.with_name(path.name + ".bak")
            backup.write_text(path.read_text(encoding="utf-8"),
                              encoding="utf-8")
        path.write_text(content, encoding="utf-8")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        changed.append(filename)
    return {"changed": changed, "written": len(changed),
            "total": len(units), "target": str(directory)}


def load_and_render(config_path: str | None = None,
                    platform: str = "auto") -> dict[str, str]:
    """Convenience for the CLI: load the host config, render its units."""
    return render_units(hostconfig.load(config_path), platform)
