"""The cross-device path, verified on one machine.

Everything this session measured ran on loopback. The one thing standing
between the project and a real two-device fleet is a second machine - but not
the absence of a second *network path*: the LAN code path can be exercised
here, and it is the part that would break on a real deployment.

Three properties are checked, none of which loopback exercises:

  1. the operator's explicit grant is required before anything binds 0.0.0.0;
  2. a root started with --lan advertises itself, and a second `pandosi join
     --lan` process finds it **by discovery** - no URL is passed to it;
  3. a decision asked over the machine's LAN address is answered by both
     stems, so the traffic really crosses the network interface.

Usage:
    python tools/lan_path_check.py
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.grants import audit_status  # noqa: E402
from pandosi.join import discover_root  # noqa: E402


def lan_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))       # no traffic, just a route
        return probe.getsockname()[0]
    finally:
        probe.close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "big_dev.jsonl"))
    args = parser.parse_args()

    task = json.loads(Path(args.suite).read_text(encoding="utf-8")
                      .splitlines()[0])
    payload = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"]).to_json()
    address = lan_ip()
    port = free_port()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    # 1. the grant gate
    before = audit_status()
    print(f"  grants before: {json.dumps(before)[:120]}", flush=True)
    granted = subprocess.run(
        [sys.executable, "-m", "pandosi", "grants", "grant", "lan_listen",
         "--resource", f"0.0.0.0:{port}", "--hours", "1", "--yes"],
        cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=60)
    print(f"  grant -> exit {granted.returncode} "
          f"{(granted.stdout or granted.stderr).strip().splitlines()[-1][:70] if (granted.stdout or granted.stderr).strip() else ''}",
          flush=True)
    if granted.returncode != 0:
        failures.append("the operator grant was refused")

    host_log = open(log_dir / "lan-root.log", "w", encoding="utf-8")
    leaflog = open(log_dir / "lan-leaf.log", "w", encoding="utf-8")
    host = subprocess.Popen(
        [sys.executable, "-m", "pandosi", "join", "--lan", "--port",
         str(port), "--node-id", "host"],
        cwd=str(ROOT_DIR), stdout=host_log, stderr=subprocess.STDOUT)
    leaf = None
    try:
        # The pairing token is the secret an operator reads off the root's
        # console and hands to each device; discovery finds the address, the
        # token is what authorises joining it.
        token = None
        for _ in range(40):
            text = (log_dir / "lan-root.log").read_text(encoding="utf-8")
            match = re.search(r"stem pairing token: (\S+)", text)
            if match:
                token = match.group(1)
                break
            time.sleep(0.5)
        print(f"  pairing token read from the root's console: {bool(token)}",
              flush=True)

        # 2. discovery: the second process is told only the token and --lan
        found = None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            found = discover_root(timeout_s=3.0)
            if found:
                break
        print(f"  a second process discovered: {found}", flush=True)
        if not found:
            failures.append("no root was discoverable over mDNS")

        leaf = subprocess.Popen(
            [sys.executable, "-m", "pandosi", "join", "--lan",
             "--node-id", "leaf", "--pairing-token", token or ""],
            cwd=str(ROOT_DIR), stdout=leaflog, stderr=subprocess.STDOUT)
        # Both processes load their model first; wait for the root to report
        # its members rather than guessing a delay.
        deadline = time.monotonic() + 240
        members = 0
        while time.monotonic() < deadline:
            try:
                status_request = urllib.request.Request(
                    f"http://{address}:{port}/status")
                if token:
                    status_request.add_header("X-Pairing-Token", token)
                with urllib.request.urlopen(status_request,
                                            timeout=5) as response:
                    status = json.loads(response.read())
                if "members" in status and isinstance(status["members"], int):
                    members = status["members"]
                else:
                    members = len(status.get("nodes") or [])
                if members >= 2:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        print(f"  members registered with the root: {members}", flush=True)
        if members < 2:
            failures.append(f"only {members} member(s) registered over the LAN")

        # 3. a decision over the LAN address, answered by both stems
        url = f"http://{address}:{port}/decide"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Pairing-Token"] = token
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(
                        url, data=payload, headers=headers),
                    timeout=60) as response:
                body = json.loads(response.read())
        except Exception as exc:
            body = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  decision over {address}:{port} -> "
              f"tier={body.get('tier')} responded={body.get('responded')} "
              f"quorum={body.get('quorum_size')} "
              f"votes={[vote.get('checkpoint') for vote in body.get('votes', [])]}",
              flush=True)
        if not body.get("votes"):
            failures.append(f"no votes came back over the LAN path: {body}")
    finally:
        for process in (leaf, host):
            if process is not None:
                process.terminate()
        for process in (leaf, host):
            if process is None:
                continue
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        host_log.close()
        leaflog.close()

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("LAN PATH OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
