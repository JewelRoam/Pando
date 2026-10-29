"""Verify adaptive isolation of a controlled slow HTTP stem.

Starts a real root, three fast mock stems, and one HTTP endpoint that accepts
requests but deliberately exceeds the configured stem timeout. The check
proves that:

1. the first decision times out only the slow stem and still answers;
2. the next immediate decision skips that stem during cooldown;
3. after cooldown the root probes it again.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(url: str, *, method: str = "GET", payload: dict | None = None,
            token: str | None = None, timeout: float = 3) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode()
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers,
                                       method=method),
                timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": str(exc)}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def wait_root(url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=1)
        if code == 200 and payload.get("required") is True:
            return
        time.sleep(0.1)
    raise RuntimeError("root did not become ready")


def wait_nodes(url: str, token: str, expected: set[str],
               timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=2)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return payload
        time.sleep(0.15)
    raise RuntimeError(f"workers not ready: {last}")


def slow_server(delay_s: float) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def log_error(self, fmt, *args):
            pass

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            time.sleep(delay_s)
            body = json.dumps({
                "vote": {"choice": "approve", "probability": 0.9}
            }).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real local processes; pass --yes")

    root_port = free_port()
    root_url = f"http://127.0.0.1:{root_port}"
    token = secrets.token_urlsafe(24)
    summary = {"root_url": root_url, "steps": []}
    processes: list[subprocess.Popen] = []
    server = slow_server(2.0)
    slow_url = f"http://127.0.0.1:{server.server_address[1]}"

    with tempfile.TemporaryDirectory(prefix="pando-slowstem-") as td:
        work = Path(td)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        root_out = work / "coordinator.out"
        root_err = work / "coordinator.err"

        def cleanup() -> None:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            server.shutdown()

        try:
            with root_out.open("wb") as out, root_err.open("wb") as err:
                root_process = subprocess.Popen(
                    [sys.executable, "-m", "pando.coordinator",
                     "--port", str(root_port),
                     "--pairing-token-file", str(token_file),
                     "--initial-stem-timeout", "0.25",
                     "--stem-call-timeout", "1.0"],
                    cwd=ROOT, stdout=out, stderr=err,
                    creationflags=creationflags)
            processes.append(root_process)
            wait_root(root_url)
            for index, checkpoint in enumerate(("en", "ml", "td")):
                processes.append(subprocess.Popen(
                    [sys.executable, "-m", "pando.worker",
                     "--node-id", f"fast-{index}",
                     "--checkpoint", checkpoint,
                     "--root", root_url,
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags))
            code, payload = request(
                f"{root_url}/register", method="POST", token=token,
                payload={"node_id": "slow-0", "url": slow_url,
                         "checkpoint": "slow", "pairing_token": token})
            if code != 200:
                raise RuntimeError(f"slow stem registration failed: {payload}")
            wait_nodes(root_url, token,
                       {"fast-0", "fast-1", "fast-2", "slow-0"})
            summary["steps"].append({"step": "register", "code": 0})

            decision_body = {"state": "controlled slow stem",
                             "question": "approve?",
                             "candidates": ["approve", "reject"]}
            code, first = request(
                f"{root_url}/decide", method="POST", token=token,
                payload=decision_body, timeout=5)
            if code != 200 or first.get("responded") != 3:
                raise RuntimeError(f"first decision failed: {first}")
            summary["first"] = {
                "responded": first["responded"],
                "quorum_size": first["quorum_size"],
                "fanout_ms": first["fanout_ms"],
                "slow_error": next(
                    info["stats"]["last_error"]
                    for node_id, info in request(
                        f"{root_url}/nodes", token=token)[1]["nodes"].items()
                    if node_id == "slow-0"),
            }
            summary["steps"].append({"step": "timeout_slow", "code": 0})

            code, second = request(
                f"{root_url}/decide", method="POST", token=token,
                payload=decision_body, timeout=5)
            if code != 200 or second.get("responded") != 3:
                raise RuntimeError(f"second decision failed: {second}")
            summary["second"] = {
                "responded": second["responded"],
                "quorum_size": second["quorum_size"],
                "fanout_ms": second["fanout_ms"],
            }
            if second["quorum_size"] != 3:
                raise RuntimeError(
                    "slow stem was not skipped during cooldown")
            summary["steps"].append({"step": "skip_during_cooldown",
                                     "code": 0})

            time.sleep(1.15)
            code, third = request(
                f"{root_url}/decide", method="POST", token=token,
                payload=decision_body, timeout=5)
            if code != 200 or third.get("responded") != 3:
                raise RuntimeError(f"third decision failed: {third}")
            summary["third"] = {
                "responded": third["responded"],
                "quorum_size": third["quorum_size"],
                "fanout_ms": third["fanout_ms"],
            }
            if third["quorum_size"] != 4:
                raise RuntimeError(
                    "slow stem was not retried after cooldown")
            summary["steps"].append({"step": "retry_after_cooldown",
                                     "code": 0})
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup()
            summary["root_stdout_tail"] = root_out.read_text(
                encoding="utf-8", errors="replace").splitlines()[-10:]
            summary["root_stderr_tail"] = root_err.read_text(
                encoding="utf-8", errors="replace").splitlines()[-10:]

    report_dir = ROOT / "metrics" / "slow-stem-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"slow-stem-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
