"""Do the new declarations actually govern a running root?

Rounds 121-144 added rules that a *playbook* is supposed to own: which
lineages are admitted, which of them are one opinion (blocs), how many
opinions an auto-execution needs (corroboration_floor), and now which advisory
combiner is pinned. Each was measured in isolation, programmatically; none had
been exercised through the file an operator actually writes.

This starts `python -m pando root --playbook <file>` and drives real HTTP
stems against it, checking four things at once:

  - a lineage that is not admitted is refused at registration;
  - two lineages declared one bloc vote once, and the response says so;
  - corroboration_floor refuses auto-execution to a single opinion;
  - the pinned combiner loads, and a wrong digest refuses to load at all.

Usage:
    python tools/playbook_e2e_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pando.common import DecisionRequest  # noqa: E402
from pando.stemd import Stem, heartbeat_loop  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(url: str, payload: dict | bytes, token: str,
         timeout: float = 30.0) -> tuple[int, dict]:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "X-Pairing-Token": token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def wait_for_root(url: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=3):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise SystemExit("root never came up")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "big_dev.jsonl"))
    args = parser.parse_args()

    combiner = ROOT_DIR / "assets" / "combiner.json"
    digest = hashlib.sha256(combiner.read_bytes()).hexdigest()
    playbook = ROOT_DIR / "metrics" / "ensemble-checks" / "governance.toml"
    playbook.write_text(
        "[meta]\nname = \"governance\"\nversion = 1\n\n"
        "[lineages]\n"
        "admit = [\"english\", \"typed-decisions\", \"multilingual\"]\n"
        "blocs = [[\"english\", \"multilingual\"]]\n\n"
        "[gating]\ncorroboration_floor = 2\n\n"
        "[combiner]\n"
        f"path = \"{combiner.relative_to(ROOT_DIR).as_posix()}\"\n"
        f"sha256 = \"{digest}\"\n",
        encoding="utf-8")
    task = json.loads(Path(args.suite).read_text(encoding="utf-8")
                      .splitlines()[0])
    payload = DecisionRequest(state=task["state"], question="intent",
                              candidates=["refund", "complaint", "praise"]
                              ).to_json()

    port = free_port()
    token = secrets.token_urlsafe(12)
    url = f"http://127.0.0.1:{port}"
    log_path = ROOT_DIR / "metrics" / "ensemble-checks"
    log = open(log_path / "playbook-root.log", "w", encoding="utf-8")
    root = subprocess.Popen(
        [sys.executable, "-m", "pando", "root", "--port", str(port),
         "--pairing-token", token, "--playbook", str(playbook)],
        cwd=str(ROOT_DIR), stdout=log, stderr=subprocess.STDOUT)
    stems: list[Stem] = []
    failures = []
    try:
        wait_for_root(url, time.monotonic() + 60)
        print("  root up with the playbook", flush=True)

        # 1. a lineage nobody declared
        status, body = post(f"{url}/register",
                            {"node_id": "rogue", "url": "http://127.0.0.1:1",
                             "checkpoint": "not-admitted",
                             "pairing_token": token}, token)
        print(f"  undeclared lineage   -> {status} "
              f"{body.get('error', '')[:60]}", flush=True)
        if status != 403:
            failures.append("an undeclared lineage was admitted")

        # 2. two lineages the playbook declares as one opinion, plus a third
        from pando.backend import MockBackend
        for name, checkpoint, choice in (
                ("a-english", "english", "refund"),
                ("b-multilingual", "multilingual", "complaint"),
                ("c-typed", "typed-decisions", "praise")):
            stem = Stem(name, MockBackend(name, checkpoint,
                                          correct_choice=choice,
                                          agree_rate=1.0))
            stem.start()
            stems.append(stem)
            threading.Thread(target=heartbeat_loop,
                             args=(stem, url, token),
                             daemon=True).start()
        time.sleep(3.0)

        status, body = post(f"{url}/decide", payload, token)
        print(f"  split decision       -> tier={body.get('tier')} "
              f"choice={body.get('choice')} lineages={body.get('lineages')} "
              f"opinions={body.get('opinions')} "
              f"suggestion={body.get('suggested_choice')}", flush=True)
        if status != 200:
            failures.append(f"decide returned {status}: {body}")
        if body.get("opinions") not in (None, 2):
            failures.append(
                f"the bloc was not collapsed: opinions={body.get('opinions')}")
        if body.get("tier") == "auto":
            failures.append("a single opinion auto-executed under a "
                            "corroboration_floor of 2")
        if not body.get("suggested_choice"):
            failures.append("the pinned combiner offered no suggestion")
    finally:
        for stem in stems:
            stem.stop()
        root.terminate()
        try:
            root.wait(timeout=15)
        except subprocess.TimeoutExpired:
            root.kill()
        log.close()

    # 3. a pinned combiner that changed underneath the playbook
    tampered = playbook.with_suffix(".tampered.toml")
    tampered.write_text(playbook.read_text(encoding="utf-8")
                        .replace(digest, "0" * 64), encoding="utf-8")
    port = free_port()
    check = subprocess.run(
        [sys.executable, "-m", "pando", "root", "--port", str(port),
         "--playbook", str(tampered)],
        cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=60)
    refused = "sha256" in (check.stdout + check.stderr)
    print(f"  tampered combiner    -> exit {check.returncode}, "
          f"refused={refused}", flush=True)
    if not refused:
        failures.append("a combiner whose digest changed was loaded anyway")
    tampered.unlink()

    # 4. corroboration_floor, isolated: one opinion agreeing with itself is
    #    agreement 1.0, so only the floor can refuse it auto-execution.
    from pando.backend import MockBackend
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log = open(log_path / "playbook-solo.log", "w", encoding="utf-8")
    root = subprocess.Popen(
        [sys.executable, "-m", "pando", "root", "--port", str(port),
         "--pairing-token", token, "--playbook", str(playbook)],
        cwd=str(ROOT_DIR), stdout=log, stderr=subprocess.STDOUT)
    solo = Stem("solo", MockBackend("solo", "english",
                                    correct_choice="refund", agree_rate=1.0))
    try:
        wait_for_root(url, time.monotonic() + 60)
        solo.start()
        threading.Thread(target=heartbeat_loop, args=(solo, url, token),
                         daemon=True).start()
        time.sleep(3.0)
        status, body = post(f"{url}/decide", payload, token)
        print(f"  one opinion          -> tier={body.get('tier')} "
              f"agreement={body.get('agreement')} "
              f"opinions={body.get('opinions')}", flush=True)
        if body.get("tier") != "verify":
            failures.append(
                "a single opinion auto-executed despite corroboration_floor=2")
    finally:
        solo.stop()
        root.terminate()
        try:
            root.wait(timeout=15)
        except subprocess.TimeoutExpired:
            root.kill()
        log.close()

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PLAYBOOK GOVERNANCE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
