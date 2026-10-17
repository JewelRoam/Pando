"""Resolve the group mailbox URL from local candidates - never from GitHub.

Doctrine (2026-10-17, user's rule): in this project GitHub is a code
repository only. Connection information must never be published to
branches, fetched from them, or kept as a resolution fallback. The
`mac-endpoint` branch, `tools/publish_mac_endpoints.sh`, and the
git-reading resolver were retired the same day; anything that recreates
them is a regression.

Resolution is fully local: probe each known host's token-free /health and
return the first that answers. Order prefers the group's home mailbox,
then the resilience mailbox on this machine, then tailnet names.

Usage:
    python tools/check_mac_endpoint.py                # first healthy URL
    python tools/check_mac_endpoint.py --all          # probe everything
    python tools/check_mac_endpoint.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

CANDIDATES = [
    "http://192.168.31.41:8776",   # Mac mailbox on the LAN (fastest at home)
    "http://100.92.184.4:8776",    # Mac mailbox over Tailscale (stable name)
    "http://127.0.0.1:8778",       # win mailbox: local, survives Mac sleep
    "http://100.93.216.90:8778",   # win mailbox over Tailscale (other hosts)
]


def _open(url: str, timeout: float):
    """LAN and tailnet targets bypass any system proxy (see agent_chat)."""
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        direct = not ipaddress.ip_address(host).is_global
    except ValueError:
        direct = host == "localhost"
    if not direct:
        return urllib.request.urlopen(url, timeout=timeout)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def probe(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with _open(url.rstrip("/") + "/health", timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")[:120]
            return resp.status == 200, body
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return False, str(exc)[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="probe every candidate and print each result")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    results = []
    chosen = None
    for url in CANDIDATES:
        ok, detail = probe(url, args.timeout)
        results.append({"url": url, "ok": ok, "detail": detail})
        if ok and chosen is None:
            chosen = url
        if ok and not args.all and not args.json:
            break
    if args.json:
        print(json.dumps({"chosen": chosen, "probes": results}, indent=2))
    elif args.all:
        for row in results:
            mark = "OK " if row["ok"] else ".. "
            print(f"{mark}{row['url']}  {row['detail']}")
    elif chosen:
        print(chosen)
    else:
        print("no mailbox answered /health on any local candidate")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
