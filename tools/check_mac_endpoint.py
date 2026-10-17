"""Resolve the group mailbox URL from local candidates - never from GitHub.

Doctrine (2026-10-17, user's rule): in this project GitHub is a code
repository only. Connection information must never be published to
branches, fetched from them, or kept as a resolution fallback. Code
carries structure (host names, ports, env var names); the addresses
themselves live in per-machine config:

    ~/.pandosi/hosts.json   {"mailboxes": ["<url>", ...]}

overridable with PANDOSI_HOSTS (inline JSON) or PANDOSI_HOSTS_FILE.
Without any config the resolver only tries localhost conventions. See
hosts.example.json for the shape (placeholders only - no real hosts).

Resolution is fully local: probe each configured host's token-free
/health and return the first that answers.

Usage:
    python tools/check_mac_endpoint.py                # first healthy URL
    python tools/check_mac_endpoint.py --all          # probe everything
    python tools/check_mac_endpoint.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.net import open_url  # noqa: E402 - repo root must be importable

CONFIG_PATH = Path(os.environ.get(
    "PANDOSI_HOSTS_FILE", "~/.pandosi/hosts.json")).expanduser()
DEFAULT_LOCAL = ["http://127.0.0.1:8776", "http://127.0.0.1:8778"]


def load_candidates() -> list[str]:
    """Mailbox URLs from per-machine config - never from the repository."""
    inline = os.environ.get("PANDOSI_HOSTS", "").strip()
    if inline:
        try:
            parsed = json.loads(inline)
            hosts = parsed.get("mailboxes") if isinstance(parsed, dict) else parsed
            if isinstance(hosts, list) and hosts:
                return [str(url) for url in hosts if str(url).strip()]
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        hosts = parsed.get("mailboxes") if isinstance(parsed, dict) else parsed
        if isinstance(hosts, list) and hosts:
            return [str(url) for url in hosts if str(url).strip()]
    except (OSError, json.JSONDecodeError):
        pass
    return list(DEFAULT_LOCAL)


def probe(url: str, timeout: float) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(url.rstrip("/") + "/health")
        with open_url(request, timeout=timeout) as resp:
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

    candidates = load_candidates()
    results = []
    chosen = None
    for url in candidates:
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
