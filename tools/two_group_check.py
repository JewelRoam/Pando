"""Verify hierarchical two-group consensus and partition fallback.

Group A and Group B each run a local root with three mock stems. A bridge
agent exposes Group B's aggregate decision to Group A as one additional
stem. The check then stops Group B's root and verifies that Group A falls
back to its three local workers after the bridge enters cooldown.
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
    raise RuntimeError(f"root not ready: {url}")


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
    raise RuntimeError(f"nodes not ready: {last}")


def decide(url: str, token: str, state: str) -> dict:
    code, payload = request(
        f"{url}/decide", method="POST", token=token,
        payload={"state": state, "question": "approve?",
                 "candidates": ["approve", "reject"]},
        timeout=30)
    if code != 200:
        raise RuntimeError(f"decision failed ({code}): {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts multiple local processes; pass --yes")

    token_a = secrets.token_urlsafe(24)
    token_b = secrets.token_urlsafe(24)
    port_a = free_port()
    port_b = free_port()
    root_a = f"http://127.0.0.1:{port_a}"
    root_b = f"http://127.0.0.1:{port_b}"
    summary = {"group_a": root_a, "group_b": root_b, "steps": []}
    processes: list[subprocess.Popen] = []

    with tempfile.TemporaryDirectory(prefix="pando-two-group-") as td:
        work = Path(td)
        token_a_file = work / "group-a.token"
        token_b_file = work / "group-b.token"
        token_a_file.write_text(token_a, encoding="utf-8")
        token_b_file.write_text(token_b, encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0

        def spawn(arguments: list[str]):
            process = subprocess.Popen(
                arguments, cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags)
            processes.append(process)
            return process

        def cleanup() -> None:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

        try:
            root_a_proc = spawn(
                [sys.executable, "-m", "pando.coordinator",
                 "--port", str(port_a),
                 "--pairing-token-file", str(token_a_file)])
            root_b_proc = spawn(
                [sys.executable, "-m", "pando.coordinator",
                 "--port", str(port_b),
                 "--pairing-token-file", str(token_b_file)])
            wait_root(root_a)
            wait_root(root_b)

            group_a_ids = []
            group_b_ids = []
            for group, root_url, token_file, ids in (
                    ("a", root_a, token_a_file, group_a_ids),
                    ("b", root_b, token_b_file, group_b_ids)):
                for index, checkpoint in enumerate(
                        (f"{group}-en", f"{group}-ml", f"{group}-td")):
                    node_id = f"group-{group}-stem-{index}"
                    ids.append(node_id)
                    spawn([
                        sys.executable, "-m", "pando.worker",
                        "--node-id", node_id,
                        "--checkpoint", checkpoint,
                        "--root", root_url,
                        "--pairing-token-file", str(token_file)])
            wait_nodes(root_a, token_a, set(group_a_ids))
            wait_nodes(root_b, token_b, set(group_b_ids))
            summary["steps"].append({"step": "local_quorums", "code": 0})

            bridge = spawn([
                sys.executable, "-m", "pando.bridge",
                "--node-id", "group-b-bridge",
                "--local-root", root_a,
                "--local-pairing-token-file", str(token_a_file),
                "--upstream-root", root_b,
                "--upstream-pairing-token-file", str(token_b_file),
                "--checkpoint", "group-b",
                "--upstream-timeout", "1.0"])
            wait_nodes(root_a, token_a,
                       set(group_a_ids) | {"group-b-bridge"})
            summary["steps"].append({"step": "bridge_registered", "code": 0})

            joined = decide(root_a, token_a, "two-group joined")
            summary["joined"] = {
                "responded": joined.get("responded"),
                "quorum_size": joined.get("quorum_size"),
                "fanout_ms": joined.get("fanout_ms"),
                "checkpoints": sorted(
                    vote["checkpoint"] for vote in joined.get("votes", [])),
            }
            if joined.get("responded") != 4:
                raise RuntimeError(f"joined decision failed: {joined}")
            if "group-b" not in summary["joined"]["checkpoints"]:
                raise RuntimeError(
                    f"bridge checkpoint missing: {summary['joined']}")
            summary["steps"].append({"step": "joined_decision", "code": 0})

            root_b_proc.terminate()
            root_b_proc.wait(timeout=5)
            partitioned = decide(root_a, token_a, "after group-b partition")
            summary["partition_first"] = {
                "responded": partitioned.get("responded"),
                "quorum_size": partitioned.get("quorum_size"),
                "fanout_ms": partitioned.get("fanout_ms"),
                "checkpoints": sorted(
                    vote["checkpoint"]
                    for vote in partitioned.get("votes", [])),
            }
            if partitioned.get("responded") != 3:
                raise RuntimeError(
                    f"partition first decision failed: {partitioned}")
            summary["steps"].append({"step": "partition_first", "code": 0})

            fallback = decide(root_a, token_a, "bridge cooldown fallback")
            summary["partition_cooldown"] = {
                "responded": fallback.get("responded"),
                "quorum_size": fallback.get("quorum_size"),
                "fanout_ms": fallback.get("fanout_ms"),
            }
            if fallback.get("quorum_size") != 3:
                raise RuntimeError(
                    f"partition fallback did not skip bridge: {fallback}")
            summary["steps"].append({"step": "partition_cooldown", "code": 0})
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup()

    report_dir = ROOT / "metrics" / "two-group-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"two-group-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
