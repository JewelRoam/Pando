"""End to end: a slow member is retired, and stays retired.

Starts a root with one deliberately slow replica and three fast ones, drives
decisions so the latencies are learned, then:

1. tries to retire without the grant -> refused, nothing changes;
2. retires with the grant -> the slow member leaves the registry, and its own
   next heartbeat is refused, which is what makes retirement durable rather
   than a five-second deletion;
3. confirms the swarm keeps serving on the members that remain.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.autonomy import Journal  # noqa: E402
from pando.elastic import Retirer, choose_retirement  # noqa: E402
from pando.grants import save_grant  # noqa: E402

SLOW_MS = 400.0
FAST_MS = 20.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_root(url: str, deadline: float = 30.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=2) as resp:
                if json.loads(resp.read()).get("required"):
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise SystemExit("rootd did not become ready")


def wait_nodes(url: str, token: str, expected: int,
               deadline: float = 60.0) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    end = time.monotonic() + deadline
    seen: dict = {}
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"{url}/nodes", headers=headers),
                    timeout=3) as resp:
                seen = json.loads(resp.read()).get("nodes", {})
            if len(seen) == expected:
                return seen
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.2)
    raise SystemExit(f"wanted {expected} nodes, saw {sorted(seen)}")


def decide(url: str, token: str, index: int) -> dict:
    body = json.dumps({"state": f"retirement probe {index}",
                       "question": "approve?",
                       "candidates": ["approve", "reject"]}).encode()
    request = urllib.request.Request(
        f"{url}/decide", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/stemd processes; pass --yes")

    work = Path(tempfile.mkdtemp(prefix="pando-retire-"))
    token = secrets.token_urlsafe(18)
    token_file = work / "pairing.token"
    token_file.write_text(token, encoding="utf-8")
    grants = work / "grants.toml"
    ledger = Journal(work / "retirement.jsonl")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pando.rootd", "--port", str(port),
             "--pairing-token-file", str(token_file),
             "--rollout-dir", str(work / "rollouts")],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_root(url)
        for index, delay in enumerate([FAST_MS, FAST_MS, FAST_MS, SLOW_MS]):
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "pando.stemd",
                 "--node-id", f"member-{index}",
                 "--root", url,
                 "--pairing-token-file", str(token_file),
                 "--backend", "mock", "--checkpoint", "mock",
                 "--delay-ms", str(delay), "--serialize"],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags))
        wait_nodes(url, token, 4)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: decide(url, token, i), range(6)))
        time.sleep(1.0)

        retirer = Retirer(url, token, grants_path=grants, journal=ledger)
        before = {member.node_id: member.latency_ewma_ms
                  for member in retirer.members()}
        print("members before:", json.dumps(before))

        refused, _ = retirer.sweep(keep=2, max_latency_ms=150.0)
        print(f"1. no grant   -> acted={[o.acted for o in refused]} "
              f"reason={refused[0].reason[:60] if refused else None}")
        if any(outcome.acted for outcome in refused):
            raise SystemExit("retirement proceeded without a grant")
        if len(retirer.members()) != 4:
            raise SystemExit("membership changed without a grant")

        expires = (dt.datetime.now(dt.timezone.utc)
                   + dt.timedelta(hours=1)).isoformat()
        save_grant("retire", "any-member", "operator-host", "check", expires,
                   path=grants)
        retired, _ = retirer.sweep(keep=2, max_latency_ms=150.0)
        if not retired or not retired[0].acted:
            raise SystemExit(f"retirement did not happen: {retired}")
        print(f"2. granted    -> {retired[0].reason}")

        # The stem keeps heartbeating; a durable retirement refuses it.
        after = wait_nodes(url, token, 3, deadline=30)
        if "member-3" in after:
            raise SystemExit("the retired member came back")
        time.sleep(6.0)  # at least one heartbeat interval
        settled = wait_nodes(url, token, 3, deadline=30)
        print(f"3. durable    -> members={sorted(settled)}")

        decision = decide(url, token, 99)
        print(f"4. still serving -> choice={decision.get('choice')} "
              f"responded={decision.get('responded')}")
        if not decision.get("choice"):
            raise SystemExit("the swarm stopped serving after a retirement")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    rows = [json.loads(line) for line in
            (work / "retirement.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    summary = {"members_before": before,
               "refused_without_grant": not any(o.acted for o in refused),
               "retired": retired[0].observation_id if retired else None,
               "members_after": sorted(settled),
               "journal_rows": len(rows)}
    print(json.dumps(summary, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print("RETIREMENT CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
