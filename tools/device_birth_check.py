"""What does one more device cost, and what does it buy?

The objective's third clause is reproduction, and the project has measured its
parts: a delta transfer of 1610 MB becomes 105 MB (round 107), a replica is
one opinion rather than a second one (round 121), and units of one lineage on
one host add nothing beyond the second (rounds 132-133). What has never been
measured end to end is *time to first vote*: how long from "a device decides
to join" until the fleet's quorum hears from it - the artifact over the wire,
the model load, the registration, the first decision it serves.

That is the birth latency of a swarm member, and it is the number a
reproduction budget needs. This measures it against a running root, with the
transfer, the load, the registration and the first vote timed separately, and
pairs it with what the fleet gains: the same load drive before and after the
new member appears.

Usage:
    python tools/device_birth_check.py [--clients 2] [--decisions 15]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pandosi.backend import OnnxLayaBackend  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402
from pandosi.stemd import Stem, heartbeat_loop  # noqa: E402

REPO = "convaiinnovations/laya"
ARTIFACT = "artifacts/laya-en-static-fixed.onnx"
CHECKPOINT = "english"


def drive(root: Root, payloads: list[bytes], clients: int,
          total: int) -> dict:
    latencies: list[float] = []
    servers: dict[str, int] = {}
    lock = threading.Lock()
    counter = {"sent": 0}

    def worker() -> None:
        while True:
            with lock:
                if counter["sent"] >= total:
                    return
                index = counter["sent"]
                counter["sent"] += 1
            started = time.perf_counter()
            body = None
            for _ in range(20):
                try:
                    with urllib.request.urlopen(
                            urllib.request.Request(
                                f"{root.url()}/decide",
                                data=payloads[index % len(payloads)],
                                headers={"Content-Type":
                                         "application/json"}),
                            timeout=300) as response:
                        body = json.loads(response.read())
                    break
                except Exception:
                    time.sleep(0.02)
            if body is None:
                continue
            served = (body.get("votes") or [{}])[0].get("node_id", "none")
            with lock:
                latencies.append((time.perf_counter() - started) * 1000)
                servers[served] = servers.get(served, 0) + 1

    threads = [threading.Thread(target=worker) for _ in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    latencies.sort()
    return {"completed": len(latencies),
            "throughput_per_s": round(len(latencies) / elapsed, 2),
            "p50_ms": round(statistics.median(latencies), 1) if latencies
            else None,
            "served_by": servers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--decisions", type=int, default=15)
    parser.add_argument("--suite",
                        default=str(ROOT_DIR / "assets" / "heldout_test.jsonl"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    payloads = [DecisionRequest(state=task["state"],
                                question=task["question"],
                                candidates=task["candidates"]).to_json()
                for task in tasks]
    total = args.clients * args.decisions
    artifact = ROOT_DIR / ARTIFACT

    root = Root(spore_files={"laya-en-static-fixed.onnx": artifact},
                coordinator=Coordinator(
                    quorum_k=1, call_timeout=300.0,
                    initial_stem_timeout=240.0, timeout_multiplier=6.0,
                    min_stem_timeout=30.0))
    root.start()
    born_dir = ROOT_DIR / "metrics" / "ensemble-checks" / "born"
    incubent = None
    newborn = None
    timeline: dict[str, float | None] = {}
    try:
        # The device that is already there.
        print("  starting the incumbent (iGPU) ...", flush=True)
        incubent = Stem("incumbent",
                        OnnxLayaBackend("incumbent", str(artifact),
                                        repo=REPO, device="GPU", pad_to=256,
                                        checkpoint=CHECKPOINT))
        incubent.start()
        threading.Thread(target=heartbeat_loop, args=(incubent, root.url()),
                         daemon=True).start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and not root.coordinator.live_nodes():
            time.sleep(1)
        drive(root, payloads, 1, 3)          # warm the incumbent
        before = drive(root, payloads, args.clients, total)
        print(f"  fleet of one: {before['throughput_per_s']}/s, "
              f"p50 {before['p50_ms']} ms, served by {before['served_by']}",
              flush=True)

        # A new device, from the artifact it already holds. The *transfer* leg
        # is deliberately not re-measured here: pulling a spore requires
        # sandbox_confirmed, a threat-model gate this check will not assert on
        # its own authority. Rounds 106-108 measured that leg under their own
        # tools; the numbers are cited in the report, and this measures the
        # legs that follow it.
        started = time.perf_counter()
        backend = OnnxLayaBackend("newborn", str(artifact), repo=REPO,
                                  device="GPU", pad_to=256,
                                  checkpoint=CHECKPOINT)
        timeline["load_s"] = round(time.perf_counter() - started, 2)

        started = time.perf_counter()
        newborn = Stem("newborn", backend)
        newborn.start()
        threading.Thread(target=heartbeat_loop, args=(newborn, root.url()),
                         daemon=True).start()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and "newborn" not in \
                root.coordinator.live_nodes():
            time.sleep(0.2)
        timeline["register_s"] = round(time.perf_counter() - started, 2)

        root.coordinator.quorum_k = 2       # force the newcomer into the quorum
        started = time.perf_counter()
        seen = None
        for _ in range(60):
            body = drive(root, payloads, 1, 1)
            seen = body["served_by"]
            if any("newborn" in name for name in seen):
                break
        timeline["first_vote_s"] = round(time.perf_counter() - started, 2)
        print(f"  born (transfer excluded, see rounds 106-108): load "
              f"{timeline['load_s']}s, register {timeline['register_s']}s, "
              f"first vote {timeline['first_vote_s']}s  (saw {seen})",
              flush=True)

        after = drive(root, payloads, args.clients, total)
        print(f"  fleet of two: {after['throughput_per_s']}/s, "
              f"p50 {after['p50_ms']} ms, served by {after['served_by']}",
              flush=True)
    finally:
        for stem in (incubent, newborn):
            if stem is not None:
                stem.stop()
        root.stop()

    summary = {"artifact_mb": round(artifact.stat().st_size / 1e6, 1),
               "timeline_s": timeline,
               "before": before, "after": after}
    out = ROOT_DIR / "metrics" / "ensemble-checks" / "device-birth.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
