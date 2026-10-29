"""Per-host configuration: one untracked TOML file per machine.

The repository carries no machine names, addresses, public keys or user
paths. Each host describes itself in ``~/.pando/host.toml`` (override with
``$PANDO_HOST_CONFIG`` or an explicit path), deployed by the operator and
never committed. A missing default file means "no pins, no overrides": every
consumer keeps working with its explicit arguments. An explicitly named file
that is missing or corrupt raises, because a host that cannot read its own
pins must fail closed.

Schema (every section optional)::

    [host]
    name = "pi5"
    state_dir = "~/.pando"

    [telemetry.publish]        # this host pushes observations as ...
    host = "pi5"
    member = "pi5-root"

    [telemetry.accept]         # this host trusts observations from ...
    host = "pi5"
    member = "pi5-root"
    pubkey = "..."

    [delegation]
    operator_pubkey = "..."    # whose signed grants this host installs

    [profiles."dynamic.threshold.adjustment"]
    node_id = "..."
    target_host = "..."
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = "~/.pando/host.toml"
ENV_VAR = "PANDO_HOST_CONFIG"


class HostConfigError(ValueError):
    """The host config is explicitly named but missing, or unreadable."""


def load(path: str | Path | None = None) -> dict:
    """Parse the host config. ``{}`` only when the default file is absent."""
    explicit = path is not None or bool(os.environ.get(ENV_VAR))
    candidate = path or os.environ.get(ENV_VAR) or DEFAULT_CONFIG_PATH
    target = Path(candidate).expanduser()
    if not target.exists():
        if explicit:
            raise HostConfigError(f"host config not found: {target}")
        return {}
    try:
        with target.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HostConfigError(f"host config {target} unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise HostConfigError(f"host config {target} is not a table")
    return data


def _section(config: dict, *names: str) -> dict:
    node = config
    for name in names:
        node = node.get(name)
        if not isinstance(node, dict):
            return {}
    return node


def _require_text(section: dict, key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HostConfigError(f"host config {where}.{key} must be a non-empty string")
    return value.strip()


def operator_pubkey(config: dict) -> str | None:
    """The pinned delegation signer, or ``None`` when the host pins nobody."""
    value = _section(config, "delegation").get("operator_pubkey")
    if value is None:
        return None
    return _require_text(_section(config, "delegation"),
                         "operator_pubkey", "delegation")


def telemetry_publish(config: dict) -> dict | None:
    """``{host, member}`` this host publishes telemetry as, or ``None``."""
    section = _section(config, "telemetry", "publish")
    if not section:
        return None
    return {"host": _require_text(section, "host", "telemetry.publish"),
            "member": _require_text(section, "member", "telemetry.publish")}


def telemetry_accept(config: dict) -> dict | None:
    """``{host, member, pubkey}`` this host trusts, or ``None``."""
    section = _section(config, "telemetry", "accept")
    if not section:
        return None
    return {"host": _require_text(section, "host", "telemetry.accept"),
            "member": _require_text(section, "member", "telemetry.accept"),
            "pubkey": _require_text(section, "pubkey", "telemetry.accept")}


def profile_params(config: dict, name: str) -> dict:
    """Fleet-supplied parameter overrides for one catalogue profile."""
    section = _section(config, "profiles", name)
    return dict(section)
