"""Whom does a latency target wake?

The adaptive quorum shrinks k when decisions run over the target (round 110)
and the picker round-robins across opinions, so pressure sheds a replica
rather than an opinion (rounds 121-155). But the seat *within* a lineage went
to the cheapest member by host cost (round 134) - and on this machine the
cheapest english member is the NPU (5.9 ms CPU per decision, ~110 ms per
decision) ahead of the iGPU (56 ms CPU, ~40 ms per decision).

So a 100 ms target can be unreachable while the fast twin sits idle: the root
settles at the floor with the slow member seated, over target, with nothing
left to spend. This runs the live fleet - iGPU english, NPU english, 3070
ft-v3 - one sequential client, adaptive quorum, target 100 ms, over the 240
rows of split D, and reports which seats the target actually bought, what
the answers were, and what the reachable configuration was.

Run before and after the pick_quorum change to see the difference.

Usage:
    python tools/latency_pick_check.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "tools"))

from cpu_free_fleet_check import (STEMS, process_tree_cpu_seconds,  # noqa: E402
                                  stem_command)
from pando.combiner import Combiner  # noqa: E402
from pando.common import DecisionRequest  # noqa: E402
from pando.rootd import Coordinator, Root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT_DIR / "assets"
                                               / "heldout3_test.jsonl"))
    parser.add_argument("--combiner", default=str(ROOT_DIR / "assets"
                                                  / "combiner.json"))
    parser.add_argument("--target", type=float, default=100.0)
    parser.add_argument("--quorum", type=int, default=3)
    parser.add_argument("--floor", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"prediction for the settled configuration (from the round-155 "
          f"ladder): decided 0.9750 either way;"
          f" seats bought by the target decide the latency -"
          f" the cost-ranked picker seats the NPU (~110 ms),"
          f" a latency-ranked picker the iGPU (~62 ms for the pair)", flush=True)

    coordinator = Coordinator(
        quorum_k=args.quorum, adaptive_quorum=True, quorum_floor=args.floor,
        latency_target_ms=args.target, call_timeout=300.0,
        initial_stem_timeout=args.timeout, timeout_multiplier=6.0,
        min_stem_timeout=30.0, combiner=Combiner.load(args.combiner),
        count_lineages=True)
    root = Root(coordinator=coordinator)
    root.start()
    log_dir = ROOT_DIR / "metrics" / "ensemble-checks"
    logs, processes = {}, {}
    try:
        for name in STEMS:
            log = open(log_dir / f"latency-pick-{name}.log", "w",
                       encoding="utf-8")
            logs[name] = log
            processes[name] = subprocess.Popen(
                stem_command(name, STEMS[name], root.url()),
                cwd=str(ROOT_DIR), stdout=log, stderr=subprocess.STDOUT,
                creationflags=0x08000000)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and \
                len(coordinator.live_nodes()) < len(STEMS):
            time.sleep(1.0)
        if len(coordinator.live_nodes()) < len(STEMS):
            raise SystemExit("stems did not all register in time")
        print(f"  {len(coordinator.live_nodes())} stems live; driving "
              f"{len(tasks)} rows with target {args.target:.0f} ms",
              flush=True)

        before = {name: process_tree_cpu_seconds(process.pid)
                  for name, process in processes.items()}
        subsets = collections.Counter()
        k_seen = collections.Counter()
        served = collections.Counter()
        auto = auto_hits = decided = 0
        suggestion_total = suggestion_hits = 0
        latencies = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"])
            row_started = time.perf_counter()
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{root.url()}/decide", data=request.to_json(),
                        headers={"Content-Type": "application/json"}),
                    timeout=300) as response:
                body = json.loads(response.read())
            latencies.append((time.perf_counter() - row_started) * 1000)
            k_seen[coordinator.quorum_k] += 1
            voters = frozenset(vote.get("node_id")
                               for vote in body.get("votes", []))
            subsets[tuple(sorted(voters))] += 1
            served.update(voters)
            if body.get("tier") == "auto":
                auto += 1
                auto_hits += body.get("choice") == task["expected"]
            if body.get("suggested_choice"):
                suggestion_total += 1
                suggestion_hits += body["suggested_choice"] \
                    == task["expected"]
            if body.get("tier") == "auto":
                decided += body.get("choice") == task["expected"]
            else:
                decided += body.get("suggested_choice") \
                    == task["expected"]
            if index % 60 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s, "
                      f"k={coordinator.quorum_k})", flush=True)
        elapsed = time.perf_counter() - started
        cpu_seconds = {name: process_tree_cpu_seconds(process.pid)
                       - before[name] for name, process in processes.items()}
        ewma = coordinator._decision_ewma_ms
        health = {}
        for name in STEMS:
            url = coordinator.nodes.get(name, {}).get("url")
            if not url:
                continue
            try:
                with urllib.request.urlopen(f"{url}/health",
                                            timeout=5) as response:
                    health[name] = json.loads(response.read())
            except Exception as exc:
                health[name] = {"error": str(exc)}
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
        root.stop()

    print(f"\n{len(tasks)} rows through the adaptive fleet, "
          f"target {args.target:.0f} ms, floor {args.floor}")
    print(f"  decided {decided}/{len(tasks)} = {decided / len(tasks):.4f}")
    print(f"  auto {auto} at {auto_hits / max(1, auto):.4f}; "
          f"suggestions {suggestion_total} at "
          f"{suggestion_hits / max(1, suggestion_total):.4f}")
    print(f"  p50 {statistics.median(latencies):.0f} ms, "
          f"{len(tasks) / elapsed:.2f} rows/s, final EWMA "
          f"{ewma:.0f} ms, final k {coordinator.quorum_k}")
    print(f"  k over the run:      {dict(k_seen)}")
    print(f"  opinions in the responses: "
          f"{ {','.join(k): v for k, v in subsets.items()} }")
    stem_stats = {}
    for name in STEMS:
        stats = coordinator.nodes.get(name, {}).get("stats", {})
        asked = health.get(name, {}).get("decisions")
        stem_stats[name] = {
            "asked": asked,
            "failures": stats.get("failures"),
            "consecutive_failures": stats.get("consecutive_failures"),
            "last_error": stats.get("last_error"),
            "latency_ewma_ms": stats.get("latency_ewma_ms"),
            "unhealthy_in_s": round(max(
                0.0, stats.get("unhealthy_until", 0.0) - time.monotonic()), 1),
        }
        print(f"  {name:4s} asked {asked if asked is not None else '?':>4} "
              f"times, {1000 * cpu_seconds[name] / max(1, asked or 0):8.2f} "
              f"ms CPU/decision, failures {stats.get('failures')}, "
              f"last_error {stats.get('last_error')}")

    report = {"target_ms": args.target, "floor": args.floor,
              "decided": round(decided / len(tasks), 4),
              "auto": auto,
              "auto_accuracy": round(auto_hits / max(1, auto), 4),
              "suggestions": suggestion_total,
              "suggestion_accuracy": round(suggestion_hits
                                           / max(1, suggestion_total), 4),
              "p50_ms": round(statistics.median(latencies), 1),
              "rows_per_s": round(len(tasks) / elapsed, 2),
              "final_ewma_ms": round(ewma or 0.0, 1),
              "final_k": coordinator.quorum_k,
              "k_seen": {str(key): value for key, value in k_seen.items()},
              "seat_subsets": {",".join(key): value
                               for key, value in subsets.items()},
              "served": dict(served), "stem_stats": stem_stats}
    out = log_dir / "latency-pick.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
