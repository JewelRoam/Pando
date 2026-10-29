"""The Mac joins the Windows group: one remote group, one vote.

The first member from outside the Laya family, outside Intel, outside
Windows: a MacBook on the LAN running the repository's naive-Bayes text
model, attached through ``pando.bridge`` so the Windows root sees the
whole Mac group as one HTTP stem with checkpoint ``text-nb-v1``. The Mac
keeps its own root - we do not start one there, and the bridge direction is
Windows -> Mac, never the reverse.

What this verifies, in order:

  1. the Mac root answers /pairing and an authenticated /decide;
  2. a Windows root whose playbook admits ``english`` + ``text-nb-v1``
     accepts the bridge's registration of the remote group;
  3. decisions through the Windows root carry the Mac group's vote in a
     real quorum (k=2 with one local iGPU english stem);
  4. the Mac's solo answers and confidence scale on the same rows, so the
     combination question is measured rather than assumed: a text model
     out of its scope answers with probability 0.0 (round 157's live case),
     and a parent gate reading that number must decide what it means.

Usage:
    PANDOSI_MAC_TOKEN=<mac pairing token> python tools/mac_group_check.py
        --yes [--rows 24]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

MAC_ROOT = "http://192.168.31.41:8765"
# Never hardcode pairing tokens: the old literal is still in git history and
# must be rotated (see docs/operations/GO-PUBLIC-PLAN.md). Export PANDOSI_MAC_TOKEN or
# pass --mac-token instead.
MAC_TOKEN = os.environ.get("PANDOSI_MAC_TOKEN", "")
SAMPLE = {"state": "the build failed and the tests are red",
          "question": "next", "candidates": ["fix", "ship", "ignore"]}


def request(url: str, *, method: str = "GET", payload: dict | None = None,
            token: str | None = None, timeout: float = 10) -> tuple[int, dict]:
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


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_until(check, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = check()
        if last:
            return last
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {what}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--mac-root", default=MAC_ROOT)
    parser.add_argument("--mac-token", default=MAC_TOKEN)
    parser.add_argument("--suite", default=str(ROOT_DIR / "assets"
                                               / "heldout3_test.jsonl"))
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts local root/stem/bridge processes")
    if not args.mac_token:
        parser.error("missing Mac pairing token: export PANDOSI_MAC_TOKEN "
                     "or pass --mac-token")

    work = ROOT_DIR / "metrics" / "lan-check"
    work.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()][:args.rows]

    print(f"1. the Mac group at {args.mac_root}", flush=True)
    code, pairing = request(f"{args.mac_root}/pairing", timeout=8)
    print(f"   /pairing -> {code} {pairing}")
    if code != 200:
        raise SystemExit("Mac root unreachable")
    code, sample = request(f"{args.mac_root}/decide", method="POST",
                           payload=SAMPLE, token=args.mac_token, timeout=30)
    print(f"   sample question -> {code} choice={sample.get('choice')} "
          f"tier={sample.get('tier')} "
          f"mean_probability={sample.get('mean_probability')} "
          f"fanout={sample.get('fanout_ms')} ms")

    token = secrets.token_urlsafe(24)
    token_file = work / "win.token"
    token_file.write_text(token, encoding="utf-8")
    playbook = work / "win-playbook.toml"
    playbook.write_text(
        '[meta]\nname = "mac-group-check"\nversion = 1\n\n'
        '[lineages]\nadmit = ["english", "text-nb-v1"]\n',
        encoding="utf-8")

    port = free_port()
    local_root = f"http://127.0.0.1:{port}"
    logs, processes = {}, {}

    def spawn(name: str, command: list[str]) -> None:
        log = open(work / f"{name}.log", "w", encoding="utf-8")
        logs[name] = log
        processes[name] = subprocess.Popen(
            command, cwd=str(ROOT_DIR), stdout=log, stderr=subprocess.STDOUT,
            creationflags=0x08000000)

    print("2. the Windows root and its local member", flush=True)
    spawn("win-root", [
        sys.executable, "-m", "pando.coordinator", "--port", str(port),
        "--pairing-token-file", str(token_file),
        "--playbook", str(playbook),
        "--quorum-k", "2",
        "--combiner", "assets/combiner.json"])
    wait_until(lambda: request(f"{local_root}/pairing", timeout=2)[0] == 200,
               30, "the Windows root")
    spawn("win-igpu", [
        sys.executable, "-m", "pando.worker",
        "--node-id", "win-igpu", "--checkpoint", "english",
        "--backend", "onnx",
        "--onnx-path", "artifacts/laya-en-static-fixed.onnx",
        "--repo", "convaiinnovations/laya", "--device", "GPU",
        "--pad-to", "256", "--host", "127.0.0.1",
        "--root", local_root, "--pairing-token-file", str(token_file)])
    time.sleep(1.0)

    print("3. the bridge (Windows asks the Mac; never the reverse)", flush=True)
    spawn("mac-bridge", [
        sys.executable, "-m", "pando.bridge",
        "--node-id", "mac-group",
        "--upstream-root", args.mac_root,
        "--upstream-pairing-token", args.mac_token,
        "--local-root", local_root,
        "--local-pairing-token-file", str(token_file),
        "--checkpoint", "text-nb-v1",
        "--host", "127.0.0.1"])

    def nodes():
        code, payload = request(f"{local_root}/nodes", token=token, timeout=3)
        if code == 200 and {"win-igpu", "mac-group"} <= set(
                payload.get("nodes", {})):
            return payload["nodes"]
        return None

    try:
        live = wait_until(nodes, 240, "both Windows members")
        for name, info in sorted(live.items()):
            print(f"   registered: {name} checkpoint={info.get('checkpoint')} "
                  f"host={info.get('host')} url={info.get('url')}")
        bridge_url = live["mac-group"]["url"]
        started = time.perf_counter()
        code, direct = request(f"{bridge_url}/decide", method="POST",
                               payload=SAMPLE, timeout=30)
        hop_ms = (time.perf_counter() - started) * 1000
        print(f"   bridge end-to-end on the sample: {code} "
              f"{hop_ms:.1f} ms -> {json.dumps(direct.get('vote', direct))}")

        print(f"4. {len(tasks)} rows through the Windows root (k=2: "
              f"english + mac-group)", flush=True)
        rows = []
        for task in tasks:
            payload = {"state": task["state"], "question": task["question"],
                       "candidates": task["candidates"]}
            started = time.perf_counter()
            code, body = request(f"{local_root}/decide", method="POST",
                                 payload=payload, token=token, timeout=120)
            latency = (time.perf_counter() - started) * 1000
            rows.append({"task": task, "code": code, "body": body,
                         "ms": round(latency, 1)})
        mac_direct = []
        for task in tasks:
            payload = {"state": task["state"], "question": task["question"],
                       "candidates": task["candidates"]}
            code, body = request(f"{args.mac_root}/decide", method="POST",
                                 payload=payload, token=args.mac_token,
                                 timeout=30)
            mac_direct.append({"task": task, "body": body})
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except Exception:
                process.kill()
        for log in logs.values():
            log.close()

    mac_votes = [row for row in rows
                 if any(vote.get("node_id") == "mac-group"
                        for vote in row["body"].get("votes", []))]
    tiers = {}
    for row in rows:
        tiers[row["body"].get("tier")] = tiers.get(row["body"].get("tier"),
                                                   0) + 1
    decided = 0
    for row in rows:
        body = row["body"]
        expected = row["task"]["expected"]
        if body.get("tier") == "auto" and body.get("choice") == expected:
            decided += 1
        elif body.get("tier") != "auto" \
                and body.get("suggested_choice") == expected:
            decided += 1
    mac_probs = [vote["probability"] for row in rows
                 for vote in row["body"].get("votes", [])
                 if vote.get("node_id") == "mac-group"]
    mac_solo_hits = sum(1 for entry in mac_direct
                        if entry["body"].get("choice")
                        == entry["task"]["expected"])
    mac_solo_probs = [entry["body"].get("mean_probability", 0.0)
                      for entry in mac_direct]
    latencies = sorted(row["ms"] for row in rows)

    print(f"\n{len(rows)} rows through the Windows root with the Mac "
          f"attached as one group")
    print(f"  mac-group votes present in {len(mac_votes)}/{len(rows)} rows")
    print(f"  tiers: {tiers}; decided {decided}/{len(rows)} = "
          f"{decided / len(rows):.4f}")
    if mac_probs:
        print(f"  mac vote probability: min {min(mac_probs):.4f} "
              f"median {statistics.median(mac_probs):.4f} "
              f"max {max(mac_probs):.4f}")
    print(f"  mac solo on the same rows: {mac_solo_hits}/{len(rows)} = "
          f"{mac_solo_hits / len(rows):.4f} at mean probability "
          f"{statistics.mean(mac_solo_probs):.4f}")
    print(f"  p50 {statistics.median(latencies):.0f} ms end to end")

    report = {"mac_root": args.mac_root, "rows": len(rows),
              "mac_votes_present": len(mac_votes), "tiers": tiers,
              "decided": round(decided / len(rows), 4),
              "mac_vote_probabilities": {
                  "min": min(mac_probs) if mac_probs else None,
                  "median": round(statistics.median(mac_probs), 4)
                  if mac_probs else None,
                  "max": max(mac_probs) if mac_probs else None},
              "mac_solo_accuracy": round(mac_solo_hits / len(rows), 4),
              "mac_solo_mean_probability": round(
                  statistics.mean(mac_solo_probs), 4),
              "p50_ms": round(statistics.median(latencies), 1)}
    out = work / "mac-group.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
