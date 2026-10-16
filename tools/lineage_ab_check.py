"""Run a live HTTP A/B between the baseline and a trained lineage."""

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
            token: str | None = None, timeout: float = 120) -> tuple[int, dict]:
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
               timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=3)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return payload
        time.sleep(0.2)
    raise RuntimeError(f"stems not ready: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_holdout.jsonl"))
    parser.add_argument("--baseline-onnx",
                        default=str(ROOT / "artifacts" / "laya-en.onnx"))
    parser.add_argument("--candidate-onnx",
                        default=str(ROOT / "artifacts" / "laya-rlcd-v7.onnx"))
    parser.add_argument("--candidate-repo",
                        default=str(ROOT / "artifacts" / "rlcd-cuda-v7"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this loads real model workers; pass --yes")

    tasks = [
        json.loads(line)
        for line in Path(args.suite).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for path in (args.baseline_onnx, args.candidate_onnx, args.candidate_repo):
        if not Path(path).exists():
            raise SystemExit(f"missing path: {path}")

    root_port = free_port()
    root_url = f"http://127.0.0.1:{root_port}"
    token = secrets.token_urlsafe(24)
    summary = {"root_url": root_url, "tasks": len(tasks), "steps": []}
    processes: list[subprocess.Popen] = []

    with tempfile.TemporaryDirectory(prefix="pando-lineage-ab-") as td:
        token_file = Path(td) / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0

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
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "pando.rootd",
                 "--port", str(root_port),
                 "--pairing-token-file", str(token_file)],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags))
            wait_root(root_url)

            processes.append(subprocess.Popen(
                [sys.executable, "-m", "pando.stemd",
                 "--node-id", "baseline-english",
                 "--checkpoint", "english",
                 "--backend", "onnx",
                 "--onnx-path", args.baseline_onnx,
                 "--device", "CPU",
                 "--root", root_url,
                 "--pairing-token-file", str(token_file)],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags))
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "pando.stemd",
                 "--node-id", "candidate-rlcd-v7",
                 "--checkpoint", "rlcd-v7",
                 "--backend", "onnx",
                 "--onnx-path", args.candidate_onnx,
                 "--repo", args.candidate_repo,
                 "--device", "CPU",
                 "--root", root_url,
                 "--pairing-token-file", str(token_file)],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags))
            wait_nodes(root_url, token,
                       {"baseline-english", "candidate-rlcd-v7"})
            summary["steps"].append({"step": "register", "code": 0})

            details = []
            correct = {"english": 0, "rlcd-v7": 0}
            for task in tasks:
                code, result = request(
                    f"{root_url}/decide", method="POST", token=token,
                    payload={"state": task["state"],
                             "question": task["question"],
                             "candidates": task.get(
                                 "candidates", ["true", "false"])})
                if code != 200:
                    raise RuntimeError(f"decision failed: {result}")
                votes = {
                    vote["checkpoint"]: vote
                    for vote in result.get("votes", [])
                }
                row = {"task_id": task["task_id"],
                       "expected": task["expected"],
                       "votes": votes}
                for checkpoint in correct:
                    if votes.get(checkpoint, {}).get("choice") == task["expected"]:
                        correct[checkpoint] += 1
                details.append(row)
            summary["details"] = details
            summary["accuracy"] = {
                checkpoint: round(hits / len(tasks), 4)
                for checkpoint, hits in correct.items()
            }
            summary["agreement"] = round(
                sum(1 for row in details
                    if row["votes"].get("english", {}).get("choice")
                    == row["votes"].get("rlcd-v7", {}).get("choice"))
                / len(tasks),
                4)
            summary["steps"].append({"step": "ab_decisions", "code": 0,
                                     "count": len(tasks)})
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup()

    report_dir = ROOT / "metrics" / "lineage-ab-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"lineage-ab-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
