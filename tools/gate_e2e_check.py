"""Confirm the gate offline numbers on the live swarm.

tools/gating_check.py computes the risk-coverage curve locally with the
production calibration; this runs the same suite through three real ONNX
lineage stems over HTTP and drives the round-98 autonomy loop over it, so
the coverage and accuracy the loop actually achieves can be compared with
what was predicted offline.

Disagreement between the two would mean the wire path (checkpoint identity,
calibration lookup, aggregation) does not match the local model.
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

from pando.autonomy import AutonomyLoop, Journal, Observation, RootDecider

LINEAGES = [
    ("english", "convaiinnovations/laya", "artifacts/laya-en.onnx"),
    ("multilingual", "convaiinnovations/laya-multilingual",
     "artifacts/laya-ml.onnx"),
    ("typed-decisions", "convaiinnovations/laya-typed-decisions",
     "artifacts/laya-td.onnx"),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url: str, *, method: str = "GET", payload: dict | None = None,
            token: str | None = None, timeout: float = 180) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers,
                                       method=method), timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": str(exc)}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def wait_root(url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=2)
        if code == 200 and payload.get("required"):
            return
        time.sleep(0.2)
    raise SystemExit("coordinator did not become ready")


def wait_nodes(url: str, token: str, expected: set[str],
               timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=5)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return
        time.sleep(0.5)
    raise SystemExit(f"lineages did not register: {last}")


def offline_prediction() -> dict | None:
    reports = sorted((ROOT / "metrics" / "gating-checks").glob("gating-*.json"),
                     key=lambda path: path.stat().st_mtime)
    if not reports:
        return None
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    return {"coverage": report["auto_coverage"],
            "accuracy": report["auto_accuracy"],
            "report": str(reports[-1])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--tolerance", type=float, default=0.08,
                        help="allowed |live - offline| coverage difference")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this loads three real model workers; pass --yes")

    tasks = [json.loads(line) for line in
             Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    tasks = [task for task in tasks if task.get("qtype") == "choice"]
    for name, _, onnx in LINEAGES:
        if not (ROOT / onnx).exists():
            raise SystemExit(f"missing graph for {name}: {ROOT / onnx}")

    expected = {task["task_id"]: task["expected"] for task in tasks}
    observations = [Observation(task["task_id"], task["state"],
                                task["question"], task["candidates"])
                    for task in tasks]
    acted_choices: dict[str, str] = {}

    def actuator(observation, choice, result):
        acted_choices[observation.observation_id] = choice

    port = free_port()
    root_url = f"http://127.0.0.1:{port}"
    token = secrets.token_urlsafe(24)
    procs: list[subprocess.Popen] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(prefix="pando-gate-e2e-") as td:
        work = Path(td)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        try:
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "pando.coordinator",
                 "--port", str(port),
                 "--pairing-token-file", str(token_file),
                 "--rollout-dir", str(work / "rollouts")],
                cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creationflags))
            wait_root(root_url)
            for name, repo, onnx in LINEAGES:
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "pando.worker",
                     "--node-id", name,
                     "--root", root_url,
                     "--pairing-token-file", str(token_file),
                     "--backend", "onnx",
                     "--onnx-path", str(ROOT / onnx),
                     "--repo", repo,
                     "--checkpoint", name,
                     "--device", "CPU"],
                    cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=creationflags))
            wait_nodes(root_url, token, {name for name, _, _ in LINEAGES})
            print(f"3 real lineage stems registered at {root_url}")

            loop = AutonomyLoop(decide=RootDecider(root_url, token),
                                actuator=actuator, allowed_tiers=("auto",),
                                min_responders=3,
                                journal=Journal(work / "autonomy.jsonl"))
            loop.run(observations)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    outcomes = loop.outcomes
    acted = [outcome for outcome in outcomes if outcome.acted]
    held = [outcome for outcome in outcomes if not outcome.acted]
    acted_correct = sum(
        1 for outcome in acted
        if acted_choices.get(outcome.observation_id)
        == expected[outcome.observation_id])
    held_correct = sum(
        1 for outcome in held
        if outcome.choice == expected[outcome.observation_id])

    summary = {
        "tasks": len(outcomes),
        "acted": len(acted),
        "held": len(held),
        "live_coverage": round(len(acted) / len(outcomes), 4),
        "live_accuracy_on_acted": round(acted_correct / len(acted), 4)
        if acted else None,
        "held_accuracy": round(held_correct / len(held), 4) if held else None,
        "tiers": {tier: sum(1 for outcome in outcomes if outcome.tier == tier)
                  for tier in {outcome.tier for outcome in outcomes}},
        "offline": offline_prediction(),
    }
    if summary["offline"]:
        summary["coverage_delta"] = round(
            summary["live_coverage"] - summary["offline"]["coverage"], 4)
        summary["accuracy_delta"] = round(
            (summary["live_accuracy_on_acted"] or 0.0)
            - (summary["offline"]["accuracy"] or 0.0), 4)

    out_dir = ROOT / "metrics" / "gating-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"gate-e2e-{int(time.time())}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    print(json.dumps(summary, indent=2))

    if summary["offline"] and abs(summary["coverage_delta"]) > args.tolerance:
        raise SystemExit(
            f"live coverage {summary['live_coverage']} differs from offline "
            f"{summary['offline']['coverage']} by more than "
            f"{args.tolerance}")
    if (summary["live_accuracy_on_acted"] or 0.0) < 0.85:
        raise SystemExit(
            f"acted accuracy {summary['live_accuracy_on_acted']} below the "
            f"offline promise")
    print("GATE E2E OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
