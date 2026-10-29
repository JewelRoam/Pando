"""Exercise quorum behavior under controlled HTTP latency changes.

A local reverse proxy fronts a real mock stem and can change its response
delay at runtime. The check runs three phases:

1. healthy remote-like link;
2. degraded link that exceeds the learned timeout and enters cooldown;
3. recovered link that is selected again after cooldown.
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

from pando.backend import MockBackend
from pando.worker import Stem


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(url: str, *, method: str = "GET", payload: dict | None = None,
            token: str | None = None, timeout: float = 5) -> tuple[int, dict]:
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
        time.sleep(0.1)
    raise RuntimeError(f"nodes not ready: {last}")


def make_proxy(backend_url: str) -> ThreadingHTTPServer:
    class Proxy(ThreadingHTTPServer):
        delay_s = 0.0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def log_error(self, fmt, *args):
            pass

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            delay = float(self.server.delay_s)
            if delay:
                time.sleep(delay)
            upstream = urllib.request.Request(
                backend_url + self.path, data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(upstream, timeout=5) as response:
                    payload = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                status = exc.code
            except urllib.error.URLError as exc:
                payload = json.dumps({"error": str(exc)}).encode()
                status = 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError,
                    ConnectionAbortedError):
                pass

    server = Proxy(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def decision(root_url: str, token: str, state: str,
             timeout: float = 5) -> dict:
    code, result = request(
        f"{root_url}/decide", method="POST", token=token,
        payload={"state": state, "question": "approve?",
                 "candidates": ["approve", "reject"]},
        timeout=timeout)
    if code != 200:
        raise RuntimeError(f"decision failed ({code}): {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real local processes; pass --yes")

    root_port = free_port()
    root_url = f"http://127.0.0.1:{root_port}"
    token = secrets.token_urlsafe(24)
    backend = Stem("remote-backend",
                   MockBackend("remote-backend", "remote"), port=0)
    backend.start()
    proxy = make_proxy(backend.url())
    proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
    summary = {"root_url": root_url, "proxy_url": proxy_url, "steps": []}
    processes: list[subprocess.Popen] = []

    with tempfile.TemporaryDirectory(prefix="pando-remote-latency-") as td:
        token_file = Path(td) / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        root_out = Path(td) / "coordinator.out"
        root_err = Path(td) / "coordinator.err"

        def cleanup() -> None:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            proxy.shutdown()
            backend.stop()

        try:
            with root_out.open("wb") as out, root_err.open("wb") as err:
                processes.append(subprocess.Popen(
                    [sys.executable, "-m", "pando.coordinator",
                     "--port", str(root_port),
                     "--pairing-token-file", str(token_file),
                     "--initial-stem-timeout", "0.5",
                     "--stem-call-timeout", "2.0"],
                    cwd=ROOT, stdout=out, stderr=err,
                    creationflags=creationflags))
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
                payload={"node_id": "remote-like", "url": proxy_url,
                         "checkpoint": "remote", "pairing_token": token})
            if code != 200:
                raise RuntimeError(f"remote registration failed: {payload}")
            wait_nodes(root_url, token,
                       {"fast-0", "fast-1", "fast-2", "remote-like"})

            proxy.delay_s = 0.10
            healthy = decision(root_url, token, "healthy remote")
            if healthy["responded"] != 4:
                raise RuntimeError(f"healthy phase failed: {healthy}")
            summary["healthy"] = {
                "responded": healthy["responded"],
                "quorum_size": healthy["quorum_size"],
                "fanout_ms": healthy["fanout_ms"],
            }
            summary["steps"].append({"step": "healthy", "code": 0})

            proxy.delay_s = 1.5
            degraded = decision(root_url, token, "degraded remote")
            if degraded["responded"] != 3:
                raise RuntimeError(f"degraded phase failed: {degraded}")
            summary["degraded"] = {
                "responded": degraded["responded"],
                "quorum_size": degraded["quorum_size"],
                "fanout_ms": degraded["fanout_ms"],
            }
            summary["steps"].append({"step": "degraded_timeout", "code": 0})

            skipped = decision(root_url, token, "cooldown skip")
            if skipped["quorum_size"] != 3:
                raise RuntimeError(f"cooldown did not skip remote: {skipped}")
            summary["cooldown_skip"] = {
                "responded": skipped["responded"],
                "quorum_size": skipped["quorum_size"],
                "fanout_ms": skipped["fanout_ms"],
            }
            summary["steps"].append({"step": "cooldown_skip", "code": 0})

            time.sleep(1.15)
            proxy.delay_s = 0.05
            recovered = decision(root_url, token, "recovered remote")
            if recovered["responded"] != 4:
                raise RuntimeError(f"recovery phase failed: {recovered}")
            summary["recovered"] = {
                "responded": recovered["responded"],
                "quorum_size": recovered["quorum_size"],
                "fanout_ms": recovered["fanout_ms"],
            }
            summary["steps"].append({"step": "recovered", "code": 0})

            code, nodes = request(
                f"{root_url}/nodes", token=token, timeout=3)
            if code == 200:
                summary["node_stats"] = {
                    node_id: info.get("stats", {})
                    for node_id, info in nodes["nodes"].items()
                }
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

    report_dir = ROOT / "metrics" / "remote-latency-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"remote-latency-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
