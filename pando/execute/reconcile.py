"""What this host knows about its peers, without opening a socket.

``hosts.json`` is the one source of truth for who this host is, which
mailboxes it reads, and which peers pull sealed bundles.  These helpers read
it; nothing here dials a URL.
"""
import importlib.util
from pathlib import Path

def known_hosts(data: dict | None) -> dict[str, str] | None:
    """Host name to root URL. None when there is no table to check against."""
    if not isinstance(data, dict):
        return None
    found: dict[str, str] = {}
    for row in data.get("roots") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        url = str(row.get("url") or "").strip()
        if name and url:
            found[name] = url
    return found


def pull_destinations(data: dict | None) -> set[str]:
    """Hosts that pull a sealed bundle. A pull host needs no root URL."""
    if not isinstance(data, dict):
        return set()
    found = set()
    for row in data.get("roots") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name and str(row.get("seed") or "").strip() == "pull":
            found.add(name)
    return found


def replication_skip(destination: str, hosts: dict[str, str] | None) -> str:
    """Why this named copy is not sent. This function does not open the URL.

    A pull host is handled by ``offer_replication``, which copies a sealed
    bundle into a local outbox. Neither function opens a socket.
    """
    if hosts is not None and destination not in hosts:
        return "unknown-host"
    return "no-transport"


def _seed_module():
    import importlib.util
    tool = Path(__file__).resolve().parents[2] / "tools" / "seed.py"
    spec = importlib.util.spec_from_file_location("pandosi_seed", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("seed tool is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
