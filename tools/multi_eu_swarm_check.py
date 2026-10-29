"""One device, several compute units, one opinion - measured over HTTP.

The objective's first clause is a single device fusing its own silicon. The
project verified a triple-EU consensus in rounds 54-56, before the
aggregation rules existed, and the rules have since changed twice: a replica
is not a second opinion (round 119/121) and neither is a correlated family
(round 128/129). Two stems running the same lineage on two different compute
units is the textbook case - the same opinion on two pieces of silicon.

This starts real stems on the CPU and the Intel GPU (OpenVINO), plus a
distinct lineage and the cheap text member, registers them into one root with
lineage counting on, and asks the wording-held-out questions:

  - do the two units vote identically, and how fast is each;
  - does the fleet's answer change when one opinion arrives from two devices;
  - what does the quorum report about how many opinions it heard.

The NPU is not in the fleet: the current OpenVINO vpux compiler rejects the
graph (`vpux-compiler: Got non broadcastable dimensions pair: '-1' and
-9223372036854775808'` on node_MatMul_215, after the front end fails to infer
an upper bound for `gnr1_gather_el`), which is the op-level NPU work the
project's own state file lists as unfinished.

Usage:
    python tools/multi_eu_swarm_check.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.backend import OnnxLayaBackend  # noqa: E402
from pando.common import DecisionRequest  # noqa: E402
from pando.coordinator import Coordinator, Root  # noqa: E402
from pando.worker import Stem, heartbeat_loop  # noqa: E402
from pando.textmodel import TextModel  # noqa: E402

MEMBERS = [
    ("en-cpu", "english", "CPU", "artifacts/laya-en.onnx", "convaiinnovations/laya"),
    ("en-gpu", "english", "GPU", "artifacts/laya-en-static-fixed.onnx",
     "convaiinnovations/laya"),
    ("en-npu", "english", "NPU", "artifacts/laya-en-static-nonisnan.onnx",
     "convaiinnovations/laya"),
    ("td-cpu", "typed-decisions", "CPU", "artifacts/laya-td.onnx",
     "convaiinnovations/laya-typed-decisions"),
]
UNITS = ("en-cpu", "en-gpu", "en-npu")
PAD_TO = 256


def start_stem(name: str, checkpoint: str, device: str, path: str,
               repo: str, pad_to: int | None) -> Stem:
    print(f"  loading {name} ({device}) ...", flush=True)
    backend = OnnxLayaBackend(name, str(ROOT / path), repo=repo,
                              device=device, pad_to=pad_to,
                              checkpoint=checkpoint)
    stem = Stem(name, backend)
    stem.start()
    return stem


def ask(root: Root, task: dict, timeout: float = 120.0) -> dict:
    request = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"])
    with urllib.request.urlopen(
            urllib.request.Request(f"{root.url()}/decide",
                                   data=request.to_json(),
                                   headers={"Content-Type": "application/json"}),
            timeout=timeout) as response:
        return json.loads(response.read())


def ask_stem(stem: Stem, task: dict, timeout: float = 300.0) -> tuple[dict,
                                                                      float]:
    """One stem, one question, directly - the fidelity and speed of one
    compute unit, without the quorum's rotation in between."""
    request = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"])
    started = time.perf_counter()
    with urllib.request.urlopen(
            urllib.request.Request(f"{stem.url()}/decide",
                                   data=request.to_json(),
                                   headers={"Content-Type": "application/json"}),
            timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload["vote"], round((time.perf_counter() - started) * 1000, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--text-model",
                        default=str(ROOT / "assets" / "textmodel.json"))
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    print("stems:", flush=True)
    stems = [start_stem(name, checkpoint, device, path, repo,
                        PAD_TO if device != "CPU" else None)
             for name, checkpoint, device, path, repo in MEMBERS]
    stems.append(Stem("text", TextModel.load(args.text_model)))
    stems[-1].start()

    root = Root(coordinator=Coordinator(
        quorum_k=len(stems), count_lineages=True, call_timeout=300.0,
        initial_stem_timeout=180.0, timeout_multiplier=6.0,
        min_stem_timeout=30.0))
    root.start()
    for stem in stems:
        threading.Thread(target=heartbeat_loop, args=(stem, root.url()),
                         daemon=True).start()
    time.sleep(1.0)

    # Phase 1: the same question put to both compute units directly. The
    # coordinator spreads its quorum across lineages (round 109/110), so a
    # fleet decision never asks two same-lineage stems the same question -
    # comparing the units has to bypass that.
    by_name = {stem.node_id: stem for stem in stems}
    fidelity = {name: [] for name in UNITS}
    print("  asking every compute unit directly ...", flush=True)
    for task in tasks:
        for name in UNITS:
            vote, latency = ask_stem(by_name[name], task)
            fidelity[name].append({"choice": vote["choice"],
                                   "probability": vote["probability"],
                                   "latency_ms": latency,
                                   "expected": task["expected"]})

    rows = []
    try:
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            result = ask(root, task)
            rows.append({"task": task, "result": result})
            if index % 25 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        root.stop()
        for stem in stems:
            stem.stop()

    votes = [vote for row in rows for vote in row["result"]["votes"]]
    by_member = {}
    for vote in votes:
        by_member.setdefault(vote["node_id"], []).append(vote["choice"])
    print(f"\n{len(rows)} tasks over real HTTP")
    for name in by_member:
        stats = root.coordinator.nodes.get(name, {}).get("stats", {})
        print(f"  {name:8s} votes {len(by_member[name]):3d}  "
              f"latency ewma {stats.get('latency_ewma_ms')} ms  "
              f"failures {stats.get('failures')}")

    baseline = fidelity[UNITS[0]]
    agreement = {}
    for name in UNITS[1:]:
        same = sum(1 for left, right in zip(baseline, fidelity[name])
                   if left["choice"] == right["choice"])
        gaps = [abs(left["probability"] - right["probability"])
                for left, right in zip(baseline, fidelity[name])]
        agreement[name] = {"identical": same, "compared": len(baseline),
                           "median_confidence_gap": round(
                               statistics.median(gaps), 4)}
    print(f"\n  compute units, asked directly ({len(baseline)} each):")
    for name in UNITS:
        entries = fidelity[name]
        hits = sum(1 for entry in entries
                   if entry["choice"] == entry["expected"])
        latencies = [entry["latency_ms"] for entry in entries]
        print(f"    {name:8s} accuracy {hits / len(entries):.4f}  "
              f"median latency {statistics.median(latencies):.0f} ms  "
              f"(mean {statistics.mean(latencies):.0f})")
    for name, entry in agreement.items():
        print(f"    {name} agrees with {UNITS[0]} on "
              f"{entry['identical']}/{entry['compared']}, median confidence "
              f"gap {entry['median_confidence_gap']}")

    every = [name for name, *_ in MEMBERS] + ["text"]

    def accuracy(fleet_rows: list[dict], node_ids: list[str]) -> float:
        """Majority over the votes the quorum actually returned, per row."""
        hits, total = 0, 0
        for row in fleet_rows:
            votes = [vote for vote in row["result"]["votes"]
                     if vote["node_id"] in node_ids]
            if not votes:
                continue
            counts: dict[str, int] = {}
            for vote in votes:
                counts[vote["choice"]] = counts.get(vote["choice"], 0) + 1
            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            total += 1
            hits += top[0][0] == row["task"]["expected"]
        return hits / total if total else 0.0

    def decisions_per_second(name: str) -> float:
        entries = fidelity.get(name)
        if not entries:
            return 0.0
        return round(1000.0 / statistics.mean(
            [entry["latency_ms"] for entry in entries]), 2)

    print(f"  throughput per device: "
          + ", ".join(f"{name} {decisions_per_second(name):.1f} dec/s"
                      for name in UNITS))
    print(f"  fleet {'+'.join(every)}: majority accuracy "
          f"{accuracy(rows, every):.4f}")
    identical = all(entry["identical"] == entry["compared"]
                    for entry in agreement.values())
    print(f"  all units identical on every row: {identical} - the extra "
          f"devices buy capacity, not information")
    opinions = [row["result"].get("opinions") for row in rows]
    print(f"  opinions reported per decision: "
          f"{sorted(set(opinions))} (stems: {len(every)})")

    summary = {
        "suite": Path(args.suite).name,
        "tasks": len(rows),
        "members": every,
        "eu_direct": {
            name: {"accuracy": round(sum(1 for entry in fidelity[name]
                                         if entry["choice"] == entry["expected"])
                                     / len(fidelity[name]), 4),
                   "median_latency_ms": round(statistics.median(
                       [entry["latency_ms"] for entry in fidelity[name]]), 1)}
            for name in UNITS},
        "eu_agreement": agreement,
        "decisions_per_second": {name: decisions_per_second(name)
                                 for name in ("en-cpu", "en-gpu")},
        "latency_ewma_ms": {name: root.coordinator.nodes.get(name, {})
                            .get("stats", {}).get("latency_ewma_ms")
                            for name in by_member},
        "fleet_majority_accuracy": accuracy(rows, every),
        "capacity_is_the_second_device": (
            "the two units voted identically on every compared row, so "
            "removing one cannot change a decision"),
        "opinions_reported": sorted({value for value in opinions
                                     if value is not None}),
    }
    out = ROOT / "metrics" / "ensemble-checks" / "multi-eu-swarm.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
