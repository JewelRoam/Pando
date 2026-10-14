"""Run real Laya lineages as HTTP stems and measure checkpoint diversity.

This uses three CPU ONNX graphs (english, multilingual, typed-decisions)
with explicit lineage identities. It validates that real ONNX stems no
longer collapse to device-only checkpoint names such as ``onnx``.
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
               timeout: float = 150) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=3)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return payload
        time.sleep(0.4)
    raise RuntimeError(f"lineages not ready: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--tasks", type=int, default=12)
    args = parser.parse_args()
    if not args.yes:
        parser.error("this loads real model workers; pass --yes")

    graphs = [
        ("english", ROOT / "artifacts" / "laya-en.onnx"),
        ("multilingual", ROOT / "artifacts" / "laya-ml.onnx"),
        ("typed-decisions", ROOT / "artifacts" / "laya-td.onnx"),
    ]
    for _, graph in graphs:
        if not graph.exists():
            raise SystemExit(f"missing graph: {graph}")

    suite = [
        json.loads(line)
        for line in (ROOT / "assets" / "graft_suite.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [task for task in suite if task["qtype"] == "choice"][:args.tasks]
    if not tasks:
        raise SystemExit("no choice tasks in graft suite")

    root_port = free_port()
    root_url = f"http://127.0.0.1:{root_port}"
    token = secrets.token_urlsafe(24)
    summary = {"root_url": root_url, "lineages": [name for name, _ in graphs],
               "tasks": len(tasks), "steps": []}
    processes: list[subprocess.Popen] = []

    with tempfile.TemporaryDirectory(prefix="pandosi-lineage-http-") as td:
        token_file = Path(td) / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        root_out = Path(td) / "rootd.out"
        root_err = Path(td) / "rootd.err"

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
                    [sys.executable, "-m", "pandosi.rootd",
                     "--port", str(root_port),
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=out, stderr=err,
                    creationflags=creationflags))
            wait_root(root_url)

            for checkpoint, graph in graphs:
                processes.append(subprocess.Popen(
                    [sys.executable, "-m", "pandosi.stemd",
                     "--node-id", f"lineage-{checkpoint}",
                     "--checkpoint", checkpoint,
                     "--backend", "onnx",
                     "--onnx-path", str(graph),
                     "--device", "CPU",
                     "--root", root_url,
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags))
            nodes = wait_nodes(root_url, token,
                               {f"lineage-{name}" for name, _ in graphs})
            summary["registered"] = sorted(nodes["nodes"])
            summary["steps"].append({"step": "register", "code": 0})

            details = []
            divergences = 0
            for task in tasks:
                code, result = request(
                    f"{root_url}/decide", method="POST", token=token,
                    payload={"state": task["state"],
                             "question": task["question"],
                             "candidates": task["candidates"]})
                if code != 200:
                    raise RuntimeError(f"decision failed: {result}")
                votes = {
                    vote["checkpoint"]: vote["choice"]
                    for vote in result.get("votes", [])
                }
                split = len(set(votes.values())) > 1
                divergences += int(split)
                details.append({
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "choices": votes,
                    "split": split,
                    "tier": result.get("tier"),
                })
            summary["divergence"] = {
                "split_cases": divergences,
                "tasks": len(tasks),
                "rate": round(divergences / len(tasks), 4),
            }
            summary["details"] = details
            summary["steps"].append({"step": "decide", "code": 0,
                                     "count": len(tasks)})
            expected = {"english", "multilingual", "typed-decisions"}
            if set(details[-1]["choices"]) != expected:
                raise RuntimeError(
                    f"lineage identity mismatch: {details[-1]['choices']}")
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

    report_dir = ROOT / "metrics" / "lineage-http-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"lineage-http-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
