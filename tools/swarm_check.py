"""Run an isolated multi-process PandoSI swarm and verify elastic shrinkage.

This exercises the production CLI path: independent rootd and stemd OS
processes, token-file pairing, protected root endpoints, real HTTP votes,
heartbeat expiry, and quorum shrinkage after one worker dies.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
            token: str | None = None, timeout: float = 3) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
        except Exception:
            payload = {"error": str(exc)}
        return exc.code, payload
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def wait_root(url: str, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    last = "not attempted"
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=1)
        if code == 200 and payload.get("required") is True:
            return payload
        last = f"{code}: {payload}"
        time.sleep(0.2)
    raise RuntimeError(f"root not ready: {last}")


def wait_nodes(url: str, token: str, expected: set[str],
               timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=1)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return payload
        time.sleep(0.25)
    raise RuntimeError(f"nodes not registered; last={last}")


def wait_absent(url: str, token: str, absent: set[str],
                timeout: float = 25) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=1)
        if code == 200:
            last = payload
            if not (absent & set(payload.get("nodes", {}))):
                return payload
        time.sleep(0.5)
    raise RuntimeError(f"dead stem still registered; last={last}")


def decide(url: str, token: str, state: str) -> dict:
    code, payload = request(
        f"{url}/decide", method="POST",
        payload={"state": state, "question": "approve?",
                 "candidates": ["approve", "reject"]},
        token=token, timeout=35)
    if code != 200:
        raise RuntimeError(f"decide failed ({code}): {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-port", type=int, default=0,
                        help="root loopback port; 0 selects a free port")
    parser.add_argument("--stems", type=int, default=5)
    parser.add_argument("--yes", action="store_true",
                        help="approve local isolated OS process creation")
    args = parser.parse_args()
    if not args.yes:
        parser.error("create local root/stem processes only with --yes")
    if not 3 <= args.stems <= 16:
        parser.error("--stems must be between 3 and 16")
    if args.root_port and not 1024 <= args.root_port <= 65535:
        parser.error("--root-port must be a high port")

    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        creationflags = 0
    python = sys.executable
    checkpoints = ["en", "ml", "td", "en", "ml", "td", "en", "ml",
                   "td", "en", "ml", "td", "en", "ml", "td", "en"]
    stem_ids = [f"swarm-stem-{index}" for index in range(args.stems)]
    summary: dict = {"root_url": None, "stems": stem_ids, "steps": []}

    with tempfile.TemporaryDirectory(prefix="pandosi-swarmcheck-") as td:
        work = Path(td)
        token = secrets.token_urlsafe(24)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        root_port = args.root_port or free_port()
        root_url = f"http://127.0.0.1:{root_port}"
        summary["root_url"] = root_url
        root_proc: subprocess.Popen | None = None
        stem_procs: list[subprocess.Popen] = []

        def cleanup() -> None:
            for proc in stem_procs:
                if proc.poll() is None:
                    proc.terminate()
            if root_proc and root_proc.poll() is None:
                root_proc.terminate()
            for proc in [*stem_procs, root_proc]:
                if not proc:
                    continue
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

        try:
            root_out = work / "rootd.out"
            root_err = work / "rootd.err"
            with root_out.open("wb") as out, root_err.open("wb") as err:
                root_proc = subprocess.Popen(
                    [python, "-m", "pandosi.rootd", "--port",
                     str(root_port), "--pairing-token-file",
                     str(token_file)],
                    cwd=ROOT, stdout=out, stderr=err,
                    creationflags=creationflags)
            wait_root(root_url)
            code, payload = request(f"{root_url}/register", method="POST",
                                    payload={"node_id": "unauthorized",
                                             "url": "http://127.0.0.1:1",
                                             "checkpoint": "mock"})
            summary["steps"].append({"step": "reject_missing_token",
                                     "code": code})
            if code != 403:
                raise RuntimeError(
                    f"unauthenticated registration was not rejected: {payload}")

            for stem_id, checkpoint in zip(stem_ids, checkpoints):
                stem_procs.append(subprocess.Popen(
                    [python, "-m", "pandosi.stemd", "--node-id", stem_id,
                     "--checkpoint", checkpoint, "--root", root_url,
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags))
                time.sleep(0.05)

            nodes = wait_nodes(root_url, token, set(stem_ids))
            summary["registered"] = sorted(nodes.get("nodes", {}))
            summary["steps"].append({"step": "register", "code": 0,
                                     "count": len(summary["registered"])})

            first = decide(root_url, token, "swarm-check-before-shrink")
            summary["decision_full"] = {
                "choice": first.get("choice"),
                "tier": first.get("tier"),
                "responded": first.get("responded"),
                "quorum_size": first.get("quorum_size"),
                "latency_note": "independent CLI root and stems"}
            summary["steps"].append({"step": "decision_full", "code": 0})
            if first.get("responded") != args.stems:
                raise RuntimeError(
                    f"full quorum responded {first.get('responded')}, "
                    f"expected {args.stems}")

            dead = stem_ids[-1]
            dead_proc = stem_procs.pop()
            dead_proc.terminate()
            dead_proc.wait(timeout=5)
            nodes_after = wait_absent(root_url, token, {dead})
            summary["registered_after_shrink"] = sorted(
                nodes_after.get("nodes", {}))
            summary["steps"].append({"step": "shrink", "code": 0,
                                     "removed": dead,
                                     "count": len(nodes_after["nodes"])})

            second = decide(root_url, token, "swarm-check-after-shrink")
            summary["decision_shrunk"] = {
                "choice": second.get("choice"),
                "tier": second.get("tier"),
                "responded": second.get("responded"),
                "quorum_size": second.get("quorum_size")}
            summary["steps"].append({"step": "decision_shrunk", "code": 0})
            if second.get("responded") != args.stems - 1:
                raise RuntimeError(
                    "shrunk quorum responded "
                    f"{second.get('responded')}, expected {args.stems - 1}")
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup()
            if root_proc and root_out.exists():
                summary["root_stdout_tail"] = root_out.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-8:]
                summary["root_stderr_tail"] = root_err.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-8:]

    report_dir = ROOT / "metrics" / "swarm-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = report_dir / f"swarm-{stamp}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
