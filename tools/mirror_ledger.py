"""Copy a remote ledger onto this machine.

The observatory runs under launchd, and launchd on this Mac cannot open
192.168.31.0/24. A process that can ssh writes the file locally. The
observatory then reads a normal path. A failed copy leaves the previous
file in place, so one outage does not blank the row.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def ssh_cat(host: str, remote: str, timeout: float = 8.0) -> str:
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host,
         f"cat {remote}"],
        capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        err = (done.stderr or "").strip() or f"ssh rc={done.returncode}"
        raise RuntimeError(err[:240])
    return done.stdout


def mirror_once(host: str, remote: str, dest: Path, fetch=ssh_cat) -> dict:
    """Write ``dest`` only when the fetch returns a non-empty ledger."""
    try:
        text = fetch(host, remote)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        return {"wrote": False, "error": str(exc)[:240]}
    if not text.strip():
        return {"wrote": False, "error": "empty"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return {"wrote": True, "bytes": len(text.encode("utf-8"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote", required=True,
                        help="path on the remote host; pass an absolute path, a local shell expands ~")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--interval", type=float, default=0,
                        help="seconds between copies; 0 copies once and exits")
    args = parser.parse_args()
    dest = Path(args.dest).expanduser()
    while True:
        report = mirror_once(args.host, args.remote, dest)
        print(time.strftime("%H:%M:%S"), report, flush=True)
        if args.interval <= 0:
            return 0 if report.get("wrote") else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
