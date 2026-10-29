"""One way to open URLs in PandoSI.

Every component used to carry its own no-proxy opener (12 copies on
2026-10-17) and they drifted - one covered only the LAN IP, another missed
the Tailscale range, which is how a local proxy ended up answering 502 for
tailnet mailboxes. This module is the single implementation:

- non-global literal hosts (LAN, loopback, Tailscale 100.64/10) go direct;
- public hosts keep the system proxy (some networks need it for a tunnel).
"""

from __future__ import annotations

import gzip
import ipaddress
import json
import socket
import urllib.parse
import urllib.request

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def is_local_host(host: str) -> bool:
    """True for loopback/LAN/tailnet literals, localhost and mDNS names."""
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        if host in ("localhost", ""):
            return True
        # mDNS names (*.local) and single-label hostnames resolve on the local
        # segment; a system proxy cannot reach them (seen live: HTTP 502 for
        # pandosi-pi5.local through the Windows proxy).
        return host.endswith(".local") or "." not in host


def open_url(request: urllib.request.Request, timeout: float):
    """Open `request` with the project's proxy rules."""
    host = urllib.parse.urlsplit(request.full_url).hostname or ""
    if is_local_host(host):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def response_bytes(response) -> bytes:
    """Read a response body, undoing the one transfer encoding we request.

    ``urllib`` deliberately does not decompress content encodings for callers.
    Keeping that step next to :func:`open_url` means every mailbox client reads
    gzip listings the same way and the behavior is independent of which helper
    happened to open the socket.
    """
    raw = response.read()
    encoding = str(response.headers.get("Content-Encoding", "")).strip().lower()
    if encoding == "gzip":
        return gzip.decompress(raw)
    return raw


def response_text(response, encoding: str = "utf-8",
                  errors: str = "replace") -> str:
    """Decode a (possibly gzip-compressed) response body as text."""
    return response_bytes(response).decode(encoding, errors)


def response_json(response):
    """Decode a (possibly gzip-compressed) response body as JSON."""
    return json.loads(response_bytes(response))


def local_addresses() -> set[str]:
    """Addresses that mean this machine, never another device."""
    found = set(LOOPBACK)
    try:
        found.add(socket.gethostname())
        for info in socket.getaddrinfo(socket.gethostname(), None):
            found.add(info[4][0])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.168.31.1", 9))
            found.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    return found


def _host_key(host: str) -> str:
    """Normalize a URL hostname or address for local-machine comparison."""
    value = str(host or "").strip().lower().rstrip(".")
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return value


def should_request_gzip(url: str) -> bool:
    """Ask for compression only when the endpoint is not this machine.

    On loopback, decompression costs more than the tiny transfer it saves (the
    2026-10-27 production A/B measured +51 ms per request). LAN and tailnet
    endpoints still benefit from sending large JSON listings compressed.
    """
    host = urllib.parse.urlsplit(str(url)).hostname or ""
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (address.is_loopback or address.is_unspecified):
        return False
    local = {_host_key(item) for item in local_addresses()}
    return _host_key(host) not in local

