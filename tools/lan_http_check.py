"""Compare loopback and LAN-IP HTTP stems inside one local swarm.

The root binds to the LAN interface with an explicit temporary grant. Two
real stem processes register: one advertises loopback, the other advertises
the host's LAN address. Both are selected by the same root quorum, so
``/nodes`` exposes their observed EWMA latency side by side.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.grants import save_grant
from pando.net import open_url
from pando.join import lan_ip


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
        with open_url(
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
    last = None
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=1)
        if code == 200 and payload.get("required") is True:
            return
        last = (code, payload)
        time.sleep(0.15)
    raise RuntimeError(f"root not ready: {last}")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--decisions", type=int, default=8)
    args = parser.parse_args()
    if not args.yes:
        parser.error("this binds a temporary LAN listener; pass --yes")

    lan_address = lan_ip()
    if lan_address == "127.0.0.1":
        raise SystemExit("no non-loopback LAN address detected")
    root_port = free_port()
    root_loopback = f"http://127.0.0.1:{root_port}"
    root_lan = f"http://{lan_address}:{root_port}"
    token = secrets.token_urlsafe(24)
    summary = {
        "lan_ip": lan_address,
        "root_loopback": root_loopback,
        "root_lan": root_lan,
        "steps": [],
    }
    processes: list[subprocess.Popen] = []

    with tempfile.TemporaryDirectory(prefix="pando-lan-http-") as td:
        work = Path(td)
        grants_path = work / "grants.toml"
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15)
        save_grant("lan_listen", f"0.0.0.0:{root_port}",
                   "temporary-local-lan-check", "lan-http-check",
                   expires.isoformat(), path=grants_path)
        host_config = work / "host.toml"
        host_config.write_text(
            f"[paths]\ngrants = \"{grants_path.as_posix()}\"\n",
            encoding="utf-8")
        env = {
            **os.environ,
            "PANDO_HOST_CONFIG": str(host_config),
            "PYTHONIOENCODING": "utf-8",
        }
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
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        try:
            with root_out.open("wb") as out, root_err.open("wb") as err:
                processes.append(subprocess.Popen(
                    [sys.executable, "-m", "pando.coordinator",
                     "--port", str(root_port), "--lan",
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=out, stderr=err, env=env,
                    creationflags=creationflags))
            wait_root(root_loopback)

            loop_process = subprocess.Popen(
                [sys.executable, "-m", "pando.worker",
                 "--node-id", "loopback-stem",
                 "--checkpoint", "loopback",
                 "--root", root_loopback,
                 "--pairing-token-file", str(token_file)],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env,
                creationflags=creationflags)
            processes.append(loop_process)
            lan_process = subprocess.Popen(
                [sys.executable, "-m", "pando.worker",
                 "--node-id", "lan-stem",
                 "--checkpoint", "lan",
                 "--host", "0.0.0.0",
                 "--advertise-host", lan_address,
                 "--root", root_lan,
                 "--pairing-token-file", str(token_file)],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env,
                creationflags=creationflags)
            processes.append(lan_process)
            wait_nodes(root_loopback, token, {"loopback-stem", "lan-stem"})
            summary["steps"].append({"step": "register", "code": 0})

            wire_latencies = []
            for index in range(args.decisions):
                started = time.perf_counter()
                code, result = request(
                    f"{root_loopback}/decide", method="POST", token=token,
                    payload={"state": f"lan comparison {index}",
                             "question": "approve?",
                             "candidates": ["approve", "reject"]},
                    timeout=30)
                wire_latencies.append((time.perf_counter() - started) * 1000)
                if code != 200 or result.get("responded") != 2:
                    raise RuntimeError(f"decision failed: {result}")
            summary["wire_p50_ms"] = round(
                statistics.median(wire_latencies), 3)

            code, nodes = request(
                f"{root_loopback}/nodes", token=token, timeout=3)
            if code != 200:
                raise RuntimeError(f"node stats failed: {nodes}")
            summary["node_stats"] = {
                node_id: info.get("stats", {})
                for node_id, info in nodes["nodes"].items()
            }
            loop_ewma = summary["node_stats"]["loopback-stem"][
                "latency_ewma_ms"]
            lan_ewma = summary["node_stats"]["lan-stem"][
                "latency_ewma_ms"]
            summary["lan_over_loopback_ewma"] = round(
                lan_ewma / loop_ewma, 3) if loop_ewma else None
            summary["steps"].append({"step": "measure", "code": 0})
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

    report_dir = ROOT / "metrics" / "lan-http-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"lan-http-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
