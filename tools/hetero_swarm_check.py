"""Does a cheap, independent member make a real swarm better, over the wire?

Rounds 122-125 measured offline that error independence - not accuracy - is
what a swarm buys, and that nothing derived from the neural family supplies
it: a fine-tuned descendant and a reversed ballot agreed on 100% of answers,
a quantised sibling on 80%, all with positive error correlation. The one
member that paid was a naive Bayes over word tokens, at 57.5% accuracy.

This runs the claim through the live path. Two roots are started on loopback
over the same set of real HTTP stems - one holding the three neural lineages,
one holding those plus the text model - and both are asked the same questions
from the wording-held-out split, so the members are answering phrasing none
of them has seen. The comparison is paired: the same stems, the same tasks,
and the only difference is whether the fourth vote exists.

Usage:
    python tools/hetero_swarm_check.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import OnnxLayaBackend  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402
from pandosi.stemd import Stem, heartbeat_loop  # noqa: E402
from pandosi.textmodel import TextModel  # noqa: E402

NEURAL = [
    ("english", "convaiinnovations/laya", "artifacts/laya-en.onnx"),
    ("multilingual", "convaiinnovations/laya-multilingual",
     "artifacts/laya-ml.onnx"),
    ("typed-decisions", "convaiinnovations/laya-typed-decisions",
     "artifacts/laya-td.onnx"),
]
TEXT_CHECKPOINT = "text-nb-v1"
TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def start_root(quorum_k: int) -> Root:
    # Three ONNX graphs answering concurrently saturate a laptop CPU, so the
    # shipped 5s initial stem budget (right for one device serving one small
    # model) would time out the first calls and then skip the "unhealthy"
    # stems for the rest of the run - which reads as an accuracy collapse
    # rather than a timeout. This check measures the swarm's decisions, not
    # its latency policy, so the budget is generous and the degraded quorums
    # are reported below.
    root = Root(coordinator=Coordinator(
        quorum_k=quorum_k, call_timeout=180.0, initial_stem_timeout=120.0,
        timeout_multiplier=6.0))
    root.start()
    return root


def register(root: Root, stems: list[Stem]) -> list[threading.Thread]:
    """Register the stems the way a deployment does.

    `HEARTBEAT_TTL` is 15 seconds: a stem that registers once and goes quiet
    is dropped, the quorum becomes empty, and every decision escalates -
    silently, because the response reports a quorum of zero as neatly as a
    real one. The production path (`stemd --root`) runs this loop; a check
    that skips it measures the TTL, not the swarm.
    """
    threads = []
    for stem in stems:
        thread = threading.Thread(target=heartbeat_loop,
                                  args=(stem, root.url()),
                                  daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def ask(root: Root, task: dict) -> dict:
    request = DecisionRequest(state=task["state"], question=task["question"],
                              candidates=task["candidates"])
    with urllib.request.urlopen(
            urllib.request.Request(f"{root.url()}/decide",
                                   data=request.to_json(),
                                   headers={"Content-Type": "application/json"}),
            timeout=60) as response:
        return json.loads(response.read())


def accepted(rows: list[dict]) -> tuple[int, float | None]:
    auto = [row for row in rows if row["result"]["tier"] == "auto"]
    if not auto:
        return 0, None
    hits = sum(1 for row in auto
               if row["result"]["choice"] == row["task"]["expected"])
    return len(auto), hits / len(auto)


def full_accuracy(rows: list[dict]) -> float:
    hits = sum(1 for row in rows
               if row["result"]["choice"] == row["task"]["expected"])
    return hits / len(rows)


def at_matched_coverage(rows: list[dict], count: int) -> float:
    """The same number of rows accepted, ranked by the swarm's own gate."""
    ordered = sorted(
        rows, key=lambda row: (TIER_ORDER[row["result"]["tier"]],
                               -row["result"]["mean_probability"]))
    window = ordered[:count]
    hits = sum(1 for row in window
               if row["result"]["choice"] == row["task"]["expected"])
    return hits / len(window)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout_test.jsonl"),
                        help="wording the members have not seen")
    parser.add_argument("--text-model",
                        default=str(ROOT / "assets" / "textmodel.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    stems = []
    for name, repo, path in NEURAL:
        print(f"loading {name} ...", flush=True)
        stems.append(Stem(name, OnnxLayaBackend(
            name, str(ROOT / path), repo=repo, device="CPU",
            checkpoint=name)))
    stems.append(Stem("text", TextModel.load(args.text_model)))
    for stem in stems:
        stem.start()
    print("stems:", ", ".join(f"{stem.node_id}={stem.backend.checkpoint}"
                              for stem in stems), flush=True)

    root_all = start_root(len(stems))
    root_neural = start_root(len(NEURAL))
    try:
        register(root_all, stems)
        register(root_neural, stems[:len(NEURAL)])
        time.sleep(1.0)      # let the first heartbeat land before asking
        started = time.perf_counter()
        rows = []
        for index, task in enumerate(tasks, start=1):
            rows.append({"task": task, "result": ask(root_all, task),
                         "baseline": ask(root_neural, task)})
            if index % 20 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        root_all.stop()
        root_neural.stop()
        for stem in stems:
            stem.stop()

    # A dropped stem shortens one root's quorum and not the other's, so the
    # headline comparison uses only the rows where both roots heard from
    # every member; the rest are reported rather than quietly averaged in.
    complete = [row for row in rows
                if row["result"]["responded"] == row["result"]["quorum_size"]
                and row["baseline"]["responded"]
                == row["baseline"]["quorum_size"]]
    with_neural = [{"task": row["task"], "result": row["baseline"]}
                   for row in complete]
    with_text = [{"task": row["task"], "result": row["result"]}
                 for row in complete]
    neural_coverage, neural_accuracy = accepted(with_neural)
    text_coverage, text_accuracy = accepted(with_text)
    print(f"\n{len(rows)} tasks over real HTTP; {len(complete)} with both "
          f"quorums complete")
    print(f"  three neural lineages   full {full_accuracy(with_neural):.4f}  "
          f"accepted {neural_coverage} at {neural_accuracy:.4f}")
    print(f"  plus the text member    full {full_accuracy(with_text):.4f}  "
          f"accepted {text_coverage} at {text_accuracy:.4f}")
    print(f"  matched at {neural_coverage} rows: neural "
          f"{at_matched_coverage(with_neural, neural_coverage):.4f} -> "
          f"with text "
          f"{at_matched_coverage(with_text, neural_coverage):.4f}")
    flips = sum(1 for row in rows
                if row["result"]["choice"] != row["baseline"]["choice"])
    print(f"  the text vote changed the swarm's answer on {flips} rows")
    for label, data in (("neural", with_neural), ("with text", with_text)):
        short = [row for row in data
                 if row["result"]["responded"] < row["result"]["quorum_size"]]
        print(f"  {label:10s} quorums short of their size: {len(short)}"
              f"/{len(data)}")
        empty = sum(1 for row in data if row["result"]["quorum_size"] == 0)
        print(f"  {label:10s} decisions with no quorum at all: {empty}"
              f"/{len(data)}")

    summary = {
        "suite": Path(args.suite).name,
        "tasks": len(rows),
        "tasks_with_complete_quorums": len(complete),
        "members": [stem.backend.checkpoint for stem in stems],
        "neural": {"full": full_accuracy(with_neural),
                   "accepted": neural_coverage,
                   "accuracy_where_accepted": neural_accuracy},
        "with_text": {"full": full_accuracy(with_text),
                      "accepted": text_coverage,
                      "accuracy_where_accepted": text_accuracy},
        "matched_at_baseline_coverage": {
            "neural": at_matched_coverage(with_neural, neural_coverage),
            "with_text": at_matched_coverage(with_text, neural_coverage)},
        "answer_changed_on": flips,
    }
    out = ROOT / "metrics" / "ensemble-checks" / "hetero-swarm.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
