"""Churn arena: run a live fleet under random membership churn.

Bet A from docs/evidence/reflections.md: the elastic fabric's unique claim must be shown
under device join/leave/death with live traffic, not only in scripted one-shot
scenarios. This launches one real coordinator plus N real worker OS processes over
loopback HTTP, then runs a decision workload on a fixed cadence while randomly
killing and restarting stems. Every event lands in a JSONL trace; the summary
reports availability, latency, and membership evolution.

Honesty note: vote CONTENT comes from mock checkpoints (en/ml/td), so this
exercises lifecycle, quorum, and transport under churn - not model accuracy.
Accuracy-under-churn with a real backend is the v2 hook.

Usage:
    python tools/churn_arena.py --yes --stems 5 --decisions 24
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
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

CHECKPOINTS = ["en", "ml", "td"]


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
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read())
        except Exception:
            parsed = {"error": str(exc)}
        return exc.code, parsed
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def wait_root(url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    last = "not attempted"
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=1)
        if code == 200 and payload.get("required") is True:
            return
        last = f"{code}: {payload}"
        time.sleep(0.2)
    raise RuntimeError(f"root not ready: {last}")


def nodes_now(url: str, token: str, timeout: float = 2) -> set[str]:
    code, payload = request(f"{url}/nodes", token=token, timeout=timeout)
    if code != 200:
        return set()
    return set(payload.get("nodes", {}))


def decide(url: str, token: str, state: str,
           timeout: float) -> tuple[int, dict, float]:
    started = time.monotonic()
    code, payload = request(
        f"{url}/decide", method="POST",
        payload={"state": state, "question": "approve?",
                 "candidates": ["approve", "reject"]},
        token=token, timeout=timeout)
    return code, payload, (time.monotonic() - started) * 1000.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="approve local isolated OS process creation")
    parser.add_argument("--stems", type=int, default=5,
                        help="total fleet size (3..24)")
    parser.add_argument("--initial", type=int, default=3,
                        help="stems started before the workload (2..stems)")
    parser.add_argument("--decisions", type=int, default=24,
                        help="decision attempts on the fixed cadence (8..500)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between decision ticks")
    parser.add_argument("--churn-prob", type=float, default=0.35,
                        help="probability of one kill/spawn event per tick")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-availability", type=float, default=0.90,
                        help="required ok/attempts ratio for a PASS")
    parser.add_argument("--decide-timeout", type=float, default=15.0)
    parser.add_argument("--out", default="",
                        help="JSONL trace path (default metrics/churn-arena-<utc>.jsonl)")
    parser.add_argument("--root-port", type=int, default=0)
    args = parser.parse_args()

    if not args.yes:
        parser.error("create local root/stem processes only with --yes")
    if not 3 <= args.stems <= 24:
        parser.error("--stems must be between 3 and 24")
    if not 2 <= args.initial <= args.stems:
        parser.error("--initial must be between 2 and --stems")
    if not 8 <= args.decisions <= 500:
        parser.error("--decisions must be between 8 and 500")
    if not 0.0 <= args.churn_prob <= 0.9:
        parser.error("--churn-prob must be between 0 and 0.9")
    if args.root_port and not 1024 <= args.root_port <= 65535:
        parser.error("--root-port must be a high port")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = Path(args.out) if args.out else Path("metrics") / f"churn-arena-{stamp}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    python = sys.executable

    trace_lines: list[str] = []
    started = time.monotonic()

    def emit(event: dict) -> None:
        event = {"t": round(time.monotonic() - started, 3), **event}
        trace_lines.append(json.dumps(event, ensure_ascii=False))

    stem_ids = [f"arena-stem-{i}" for i in range(args.stems)]
    alive: dict[str, subprocess.Popen] = {}
    dead: list[str] = list(stem_ids)
    kills = 0
    spawns = 0
    ok = 0
    latencies: list[float] = []
    member_counts: list[int] = []
    root_proc: subprocess.Popen | None = None

    def spawn_stem(stem_id: str) -> None:
        checkpoint = CHECKPOINTS[stem_ids.index(stem_id) % len(CHECKPOINTS)]
        alive[stem_id] = subprocess.Popen(
            [python, "-m", "pando.worker", "--node-id", stem_id,
             "--checkpoint", checkpoint, "--root", root_url,
             "--pairing-token-file", str(token_file)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=creationflags)

    try:
        work_ctx = tempfile.TemporaryDirectory(prefix="pando-arena-")
        work = Path(work_ctx.name)
        token = secrets.token_urlsafe(24)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        root_port = args.root_port or free_port()
        root_url = f"http://127.0.0.1:{root_port}"

        root_proc = subprocess.Popen(
            [python, "-m", "pando.coordinator", "--port", str(root_port),
             "--pairing-token-file", str(token_file)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=creationflags)
        wait_root(root_url)
        emit({"event": "fleet_start", "stems": args.stems,
              "initial": args.initial, "interval": args.interval,
              "churn_prob": args.churn_prob, "seed": args.seed})

        for stem_id in stem_ids[:args.initial]:
            dead.remove(stem_id)
            spawn_stem(stem_id)
            spawns += 1
            emit({"event": "spawn", "stem": stem_id})
            time.sleep(0.05)

        for tick in range(args.decisions):
            tick_start = time.monotonic()

            if rng.random() < args.churn_prob and alive:
                if dead and rng.random() < 0.45:
                    stem_id = rng.choice(dead)
                    dead.remove(stem_id)
                    spawn_stem(stem_id)
                    spawns += 1
                    emit({"event": "churn_spawn", "stem": stem_id,
                          "alive": len(alive)})
                else:
                    stem_id = rng.choice(list(alive))
                    proc = alive.pop(stem_id)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    dead.append(stem_id)
                    kills += 1
                    emit({"event": "churn_kill", "stem": stem_id,
                          "alive": len(alive)})

            members = nodes_now(root_url, token)
            member_counts.append(len(members))
            code, payload, latency_ms = decide(
                root_url, token, f"churn-arena-{tick}", args.decide_timeout)
            if code == 200:
                ok += 1
                latencies.append(latency_ms)
            emit({"event": "decide", "tick": tick, "code": code,
                  "ok": code == 200, "choice": payload.get("choice"),
                  "tier": payload.get("tier"),
                  "responded": payload.get("responded"),
                  "quorum_size": payload.get("quorum_size"),
                  "members_registered": len(members),
                  "latency_ms": round(latency_ms, 1),
                  "error": None if code == 200 else payload.get("error")})

            elapsed = time.monotonic() - tick_start
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    finally:
        for proc in list(alive.values()):
            if proc.poll() is None:
                proc.terminate()
        if root_proc and root_proc.poll() is None:
            root_proc.terminate()
        for proc in [*alive.values(), root_proc]:
            if not proc:
                continue
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if "work_ctx" in locals():
            work_ctx.cleanup()

    availability = ok / args.decisions if args.decisions else 0.0
    summary = {
        "attempts": args.decisions,
        "ok": ok,
        "availability": round(availability, 4),
        "min_availability": args.min_availability,
        "kills": kills,
        "spawns": spawns,
        "mean_members_registered": round(
            sum(member_counts) / len(member_counts), 2) if member_counts else 0,
        "final_members_registered": member_counts[-1] if member_counts else 0,
        "latency_ms_p50": round(percentile(latencies, 0.5) or 0, 1),
        "latency_ms_p95": round(percentile(latencies, 0.95) or 0, 1),
        "trace": str(trace_path),
        "verdict": "PASS" if availability >= args.min_availability else "FAIL",
    }
    emit({"event": "summary", **summary})
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
