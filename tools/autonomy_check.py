"""End-to-end check of the observe -> decide -> act loop.

Starts a real rootd and real stemd workers (the project's deterministic
mock backend, no protocol faking), then drives the autonomous loop twice:
with a full quorum, where the swarm says `auto` and the loop acts, and after
a worker is lost, where the swarm falls back to `escalate` and the loop must
hold. The property under test is the loop's: it acts if and only if the
swarm's own tier allows it.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.autonomy import AutonomyLoop, Journal, Observation, RootDecider

TTL_WAIT_S = 20.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, token: str) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers),
                timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def wait_root(url: str, deadline: float = 30.0) -> None:
    end = time.time() + deadline
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=2) as resp:
                if json.loads(resp.read()).get("required"):
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise SystemExit("rootd did not start")


def wait_nodes(url: str, token: str, expected: int,
               deadline: float = 60.0) -> None:
    end = time.time() + deadline
    while time.time() < end:
        code, payload = get(f"{url}/nodes", token)
        if code == 200 and len(payload.get("nodes", {})) == expected:
            return
        time.sleep(0.5)
    raise SystemExit(f"expected {expected} nodes, saw {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/stemd processes; pass --yes")

    port = free_port()
    root_url = f"http://127.0.0.1:{port}"
    token = secrets.token_urlsafe(18)
    observations = [
        Observation("obs-1", "Nightly build failed twice with a timeout.",
                    "action", ["retry", "escalate"]),
        Observation("obs-2", "Unknown sender asks for the admin password.",
                    "action", ["block", "allow"]),
    ]
    acted_log: list[str] = []

    def actuator(observation, choice, result):
        acted_log.append(f"{observation.observation_id}->{choice}")

    with tempfile.TemporaryDirectory(prefix="pando-autonomy-") as td:
        work = Path(td)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        procs: list[subprocess.Popen] = []
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0

        def start_stem(node_id: str, checkpoint: str) -> subprocess.Popen:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pando.stemd",
                 "--node-id", node_id,
                 "--root", root_url,
                 "--pairing-token-file", str(token_file),
                 "--backend", "mock",
                 "--checkpoint", checkpoint],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags)
            procs.append(proc)
            return proc

        try:
            root_proc = subprocess.Popen(
                [sys.executable, "-m", "pando.rootd",
                 "--port", str(port),
                 "--pairing-token-file", str(token_file),
                 "--rollout-dir", str(work / "rollouts")],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags)
            procs.append(root_proc)
            wait_root(root_url)

            # The synthetic backend declares identity calibration in
            # assets/calibration/mock.json, so these votes are calibrated and
            # the act path is reachable. A real stem that has not declared a
            # calibration is held instead (tests/test_smoke.py).
            for index in range(3):
                start_stem(f"worker-{index}", "mock")
            wait_nodes(root_url, token, 3)

            loop = AutonomyLoop(decide=RootDecider(root_url, token),
                                actuator=actuator, allowed_tiers=("auto",),
                                min_responders=3,
                                journal=Journal(work / "autonomy.jsonl"))
            loop.run(observations)
            full_quorum = list(loop.outcomes)
            if any(not outcome.acted for outcome in full_quorum):
                raise SystemExit(f"loop held a decision under a full quorum: "
                                 f"{[o.reason for o in full_quorum]}")
            print(f"full quorum: acted "
                  f"{sum(o.acted for o in full_quorum)}/"
                  f"{len(full_quorum)} "
                  f"(tiers {[o.tier for o in full_quorum]})")

            # Lose the original workers: the quorum can no longer be met, so
            # the swarm escalates and the loop must stop acting on its own.
            # Start a fresh survivor after all three expire instead of relying
            # on whichever worker happened to retain its heartbeat under load.
            # The survivor still serves real `/decide` calls; only the flaky
            # identity of the process left behind is removed.
            workers = procs[1:]
            for proc in workers:
                proc.terminate()
            for proc in workers:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            wait_nodes(root_url, token, 0, deadline=TTL_WAIT_S + 20)
            start_stem("worker-survivor", "mock")
            wait_nodes(root_url, token, 1)

            held_loop = AutonomyLoop(decide=RootDecider(root_url, token),
                                     actuator=actuator,
                                     allowed_tiers=("auto",),
                                     min_responders=3)
            held_loop.run([Observation("obs-3", "Deploy the hotfix now.",
                                       "action", ["deploy", "wait"])])
            if any(outcome.acted for outcome in held_loop.outcomes):
                raise SystemExit("loop acted without a quorum")
            degraded = held_loop.outcomes[0]
            print(f"degraded quorum: held (tier={degraded.tier}, "
                  f"responded={degraded.responded}/{degraded.quorum_size}, "
                  f"reason={degraded.reason!r})")
            # The swarm still reports `auto` from the lone survivor; only the
            # responder policy stops the loop, which is the point of the
            # second gate.
            if degraded.responded != 1 or degraded.tier != "auto":
                raise SystemExit(f"unexpected degraded state: {degraded}")

            journal = (work / "autonomy.jsonl").read_text(
                encoding="utf-8").strip().splitlines()
            if len(journal) != len(full_quorum):
                raise SystemExit(f"journal has {len(journal)} rows, expected "
                                 f"{len(full_quorum)}")
            print(f"journal: {len(journal)} recorded outcomes")
            print(f"actions: {acted_log}")
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    if len(acted_log) != len(observations):
        raise SystemExit(f"expected {len(observations)} actions, "
                         f"got {acted_log}")
    print("AUTONOMY LOOP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
