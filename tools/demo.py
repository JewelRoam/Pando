"""PandoSI end-to-end demo: clone-to-first-vote in one command.

Starts a root + 3 mock stems, sends 3 decisions through the consensus,
renders the dashboard, and tears down. Mock backend = no model weights
needed; swap in real ONNX weights for the full experience (see README).

Usage: python tools/demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = "8090"
TOKEN = "pandosi-demo-token"


def start(args):
    return subprocess.Popen([sys.executable] + args, cwd=str(ROOT),
                            env={**__import__("os").environ,
                                 "PYTHONIOENCODING": "utf-8"})


def main() -> None:
    print("PandoSI demo: root + 3 mock stems + 3 consensus votes", flush=True)
    procs = [start(["-m", "pandosi.rootd", "--port", PORT,
                    "--pairing-token", TOKEN]),
             start(["-m", "pandosi.stemd", "--node-id", "stem0",
                    "--root", f"http://127.0.0.1:{PORT}",
                    "--pairing-token", TOKEN]),
             start(["-m", "pandosi.stemd", "--node-id", "stem1",
                    "--checkpoint", "multilingual",
                    "--root", f"http://127.0.0.1:{PORT}",
                    "--pairing-token", TOKEN]),
             start(["-m", "pandosi.stemd", "--node-id", "stem2",
                    "--checkpoint", "typed-decisions",
                    "--root", f"http://127.0.0.1:{PORT}",
                    "--pairing-token", TOKEN])]
    verdict = "INCOMPLETE"
    try:
        time.sleep(8)
        registered = []
        for _ in range(10):
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/nodes",
                    headers={"Authorization": f"Bearer {TOKEN}"})
                nodes = json.loads(urllib.request.urlopen(
                    request, timeout=5).read())
                registered = sorted(nodes["nodes"].keys())
            except Exception:
                pass
            if len(registered) >= 3:
                break
            time.sleep(2)
        print(f"registered: {registered}", flush=True)

        cases = [
            ("refund request", "intent",
             ["refund", "complaint", "greeting"]),
            ("50 failed logins from one IP", "risk",
             ["negligible", "moderate", "critical"]),
            ("is this message urgent?", "ok",
             ["yes", "no"]),
        ]
        for state, question, candidates in cases:
            body = json.dumps({"state": state, "question": question,
                               "candidates": candidates}).encode()
            t0 = time.perf_counter()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/decide", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {TOKEN}"})
            result = json.loads(urllib.request.urlopen(req, timeout=30).read())
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [{state[:40]}] -> {result['choice']} "
                  f"(tier={result['tier']}, agreement={result['agreement']}, "
                  f"{ms:.0f} ms)", flush=True)

        r = subprocess.run([sys.executable, str(ROOT / "tools/dashboard.py")],
                           capture_output=True, text=True, cwd=str(ROOT))
        print(r.stdout.strip(), flush=True)
        verdict = "OK"
    except Exception as exc:
        print(f"EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
    finally:
        for p in procs:
            p.terminate()
    print(f"DEMO {verdict} - dashboard: docs/dashboard.html", flush=True)


if __name__ == "__main__":
    main()
