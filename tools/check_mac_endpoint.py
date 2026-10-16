"""Read the Mac's published endpoints (control plane) from the repository.

The Mac agent publishes its currently reachable URLs as
``endpoint/mac.json`` on the volatile branch ``mac-endpoint``
(see docs/remote-connectivity.md). This tool fetches that branch and
prints the file; exits non-zero when the branch does not exist yet, so
callers can fall back to the LAN default.

Usage:
    python tools/check_mac_endpoint.py [--no-fetch]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

LAN_DEFAULT = "http://192.168.31.41:8776"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip git fetch; read the local ref only")
    args = parser.parse_args()
    if not args.no_fetch:
        subprocess.run(["git", "fetch", "origin", "mac-endpoint"],
                       capture_output=True)
    show = subprocess.run(
        ["git", "show", "origin/mac-endpoint:endpoint/mac.json"],
        capture_output=True)
    if show.returncode != 0:
        print(f"no mac-endpoint branch yet; LAN default {LAN_DEFAULT}")
        return 1
    try:
        data = json.loads(show.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"mac-endpoint unreadable: {exc}")
        return 1
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
