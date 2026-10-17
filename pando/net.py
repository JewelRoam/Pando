"""One way to open URLs in PandoSI.

Every component used to carry its own no-proxy opener (12 copies on
2026-10-17) and they drifted - one covered only the LAN IP, another missed
the Tailscale range, which is how a local proxy ended up answering 502 for
tailnet mailboxes. This module is the single implementation:

- non-global literal hosts (LAN, loopback, Tailscale 100.64/10) go direct;
- public hosts keep the system proxy (some networks need it for a tunnel).
"""

from __future__ import annotations

import ipaddress
import urllib.parse
import urllib.request


def is_local_host(host: str) -> bool:
    """True for loopback/LAN/tailnet literals and localhost."""
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return host in ("localhost", "")


def open_url(request: urllib.request.Request, timeout: float):
    """Open `request` with the project's proxy rules."""
    host = urllib.parse.urlsplit(request.full_url).hostname or ""
    if is_local_host(host):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)
