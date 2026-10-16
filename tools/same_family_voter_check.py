"""Can a vote drawn from the same family as an admitted lineage be independent?

Round 123 measured what the ensemble actually buys: a member whose errors are
independent is worth several points even when it is 19 points *less* accurate
than the rival candidate for its seat. Every member this project has produced
sits in one family, and round 119 caught the degenerate case - a fine-tuned
descendant that decided identically to its parent on 240 of 240 tasks.

The elastic premise needs a diversity source a device can actually supply, so
this scores the three cheapest candidates in one harness. Each re-runs one
lineage over a recorded report and asks the two questions that decide whether
the candidate is worth a seat: how correlated is it with the base's original
vote, and does the ensemble improve when it joins.

  --reverse-ballot          the same graph, the candidates presented in the
                            opposite order (a device that holds one
                            checkpoint can still render two ballots)
  --candidate <onnx> --name the quantised sibling the project would ship to
                            an NPU or a set-top box
  (default)                 whatever graph --candidate names, on the
                            canonical ballot

Usage:
    python tools/same_family_voter_check.py --report <ensemble-*.json> \
        --base english --candidate artifacts/laya-en.int8.onnx --name en-int8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ensemble_value_check import LINEAGES, load_backends  # noqa: E402
from foreign_voter_check import curve, divergence, row_for  # noqa: E402
from pando.backend import OnnxLayaBackend  # noqa: E402
from pando.common import DecisionRequest  # noqa: E402
from pando.calibrate import Calibrator  # noqa: E402


def candidate_vote(backend, calibrator, task: dict,
                   reverse_ballot: bool) -> dict:
    candidates = task["candidates"]
    request = DecisionRequest(
        state=task["state"], question=task["question"],
        candidates=list(reversed(candidates)) if reverse_ballot
        else list(candidates))
    vote = backend.decide(request)
    probability, _ = calibrator.calibrate(vote.checkpoint, vote.probability)
    return {"choice": vote.choice, "probability": round(probability, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "big_dev.jsonl"))
    parser.add_argument("--base", default="typed-decisions",
                        help="the lineage whose recorded votes are compared")
    parser.add_argument("--candidate", default=None,
                        help="a graph to score as the candidate voter "
                             "(default: the base lineage's own graph)")
    parser.add_argument("--name", default=None,
                        help="the candidate's name (default: derived)")
    parser.add_argument("--reverse-ballot", action="store_true",
                        help="present the candidates in the opposite order")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N tasks (smoke runs only)")
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    recorded = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if args.limit:
        recorded = dict(recorded, rows=recorded["rows"][:args.limit])
    by_id = {task["task_id"]: task for task in tasks}
    neural = [name for name in recorded["rows"][0]["per_lineage"]]
    name = args.name or (f"{args.base}-reversed" if args.reverse_ballot
                         else f"{args.base}-candidate")

    if args.candidate:
        repo = LINEAGES[args.base][0]
        print(f"loading {args.candidate} as {name} ...", flush=True)
        backend = OnnxLayaBackend(name, str(ROOT / args.candidate), repo=repo,
                                  device="CPU", checkpoint=args.base)
    else:
        backend = load_backends([args.base])[0]
    calibrator = Calibrator()
    started = time.perf_counter()
    foreign, rows = {}, []
    for index, record in enumerate(recorded["rows"], start=1):
        task = by_id[record["task_id"]]
        foreign = candidate_vote(backend, calibrator, task,
                                 args.reverse_ballot)
        rows.append(row_for(task, record, foreign, name))
        print(f"  {task['task_id']} [{index}/{len(recorded['rows'])}] "
              f"{record['per_lineage'][args.base]} -> "
              f"{foreign['choice']}", flush=True)
    elapsed = time.perf_counter() - started

    disagree = sum(1 for record, row in zip(recorded["rows"], rows)
                   if row["per_lineage"][name]
                   != record["per_lineage"][args.base])
    accuracy = sum(1 for row in rows
                   if row["per_lineage"][name] == row["expected"])
    print(f"\n{len(rows)} tasks in {elapsed:.1f}s; "
          f"{name} accuracy {accuracy / len(rows):.4f}")
    print(f"  flipped the {args.base} answer on {disagree} rows "
          f"({disagree / len(rows):.1%})")
    for lineage in neural:
        entry = divergence(rows, lineage, name)
        print(f"  divergence from {lineage:18s} "
              f"disagreement {entry['disagreement']:.3f}  "
              f"phi {entry['phi']:+.3f}")

    members = neural + [name]
    coverage = recorded.get("swarm_auto_count", len(rows))
    print(f"  ensemble, matched at the {coverage} rows the shipped gate "
          f"accepts:")
    curves = {}
    for rule in ["majority", "margin", "gate"]:
        before = curve(recorded["rows"], rule, neural, coverage)
        after = curve(rows, rule, members, coverage)
        curves[rule] = {"before": before, "after": after}
        print(f"    {rule:10s} @{coverage} {before['at_coverage']:.4f} -> "
              f"{after['at_coverage']:.4f}   "
              f"full {before['full']:.4f} -> {after['full']:.4f}")

    summary = {
        "report": str(args.report),
        "lineage": args.base,
        "task_count": len(rows),
        "accuracy": round(accuracy / len(rows), 4),
        "flipped": disagree,
        "divergence": {lineage: divergence(rows, lineage, name)
                       for lineage in neural},
        "curves": {rule: {side: {k: v for k, v in entry.items()
                                 if k != "hits"}
                          for side, entry in sides.items()}
                   for rule, sides in curves.items()},
    }
    out = ROOT / "metrics" / "ensemble-checks" / "ballot-order.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
