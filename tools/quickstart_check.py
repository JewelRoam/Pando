"""How long from nothing running to the first decision a newcomer serves?

The session's decision work ended with a swarm that beats its best member
(rounds 139-142) and a 19 KB text member that needs no download (round 126).
This measures the on-ramp those enable: two commands, no artifacts to fetch,
no accelerator, no privileges.

    python -m pando root --combiner assets/combiner.json
    python -m pando join --backend text --root <url>

`join` now falls back to the text member when no ONNX weights are present
(before this round it fell back to a mock, which votes from a hash), and it
loads the advisory combiner when the repository carries one. The check times
each step and the first decision that comes back with the newcomer's vote in
it.

Usage:
    python tools/quickstart_check.py
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pando.common import DecisionRequest  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get(url: str, token: str | None = None, timeout: float = 5.0):
    request = urllib.request.Request(url)
    if token:
        request.add_header("X-Pairing-Token", token)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for(url: str, deadline: float, token: str | None = None) -> float:
    started = time.perf_counter()
    while time.perf_counter() < deadline:
        try:
            get(url, token)
            return time.perf_counter() - started
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.2)
    raise SystemExit(f"{url} never answered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "big_dev.jsonl"))
    args = parser.parse_args()

    task = json.loads(Path(args.suite).read_text(encoding="utf-8")
                      .splitlines()[0])
    payload = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"]).to_json()
    port = free_port()
    token = secrets.token_urlsafe(12)
    log = ROOT_DIR / "metrics" / "ensemble-checks"
    log.mkdir(parents=True, exist_ok=True)
    root_log = open(log / "quickstart-root.log", "w", encoding="utf-8")
    join_log = open(log / "quickstart-join.log", "w", encoding="utf-8")

    timeline = {}
    root_url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    root = subprocess.Popen(
        [sys.executable, "-m", "pando", "root", "--port", str(port),
         "--pairing-token", token, "--combiner", "assets/combiner.json"],
        cwd=str(ROOT_DIR), stdout=root_log, stderr=subprocess.STDOUT)
    join = None
    try:
        timeline["root_ready_s"] = round(
            wait_for(f"{root_url}/pairing", time.perf_counter() + 60), 2)
        print(f"  root ready in {timeline['root_ready_s']}s", flush=True)

        joined = time.perf_counter()
        join = subprocess.Popen(
            [sys.executable, "-m", "pando", "join", "--backend", "text",
             "--node-id", "guest", "--root", root_url,
             "--pairing-token", token],
            cwd=str(ROOT_DIR), stdout=join_log, stderr=subprocess.STDOUT)
        deadline = joined + 60
        while time.perf_counter() < deadline:
            try:
                status = get(f"{root_url}/status", token, timeout=5)
                if status.get("members"):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.2)
        timeline["member_registered_s"] = round(
            time.perf_counter() - joined, 2)
        print(f"  member registered {timeline['member_registered_s']}s after "
              f"the join command", flush=True)

        asked = time.perf_counter()
        with urllib.request.urlopen(
                urllib.request.Request(f"{root_url}/decide", data=payload,
                                       headers={
                                           "Content-Type": "application/json",
                                           "X-Pairing-Token": token}),
                timeout=60) as response:
            body = json.loads(response.read())
        timeline["first_decision_s"] = round(time.perf_counter() - asked, 2)
        total = round(time.perf_counter() - started, 2)
        timeline["total_s"] = total
        print(f"  first decision in {timeline['first_decision_s']}s: "
              f"choice={body.get('choice')} tier={body.get('tier')} "
              f"opinions={body.get('opinions')} "
              f"suggestion={body.get('suggested_choice')}", flush=True)
        print(f"  total from launch to a served decision: {total}s")
    finally:
        for process in (join, root):
            if process is not None:
                process.terminate()
        for process in (join, root):
            if process is None:
                continue
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        root_log.close()
        join_log.close()

    out = log / "quickstart.json"
    out.write_text(json.dumps({"timeline_s": timeline,
                               "decision": {key: body.get(key)
                                            for key in ("choice", "tier",
                                                        "opinions",
                                                        "suggested_choice")}},
                              indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
