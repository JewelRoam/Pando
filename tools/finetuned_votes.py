"""Vote with a fine-tuned checkpoint on a labelled split (runs on the GPU).

Round 150 fine-tuned the `english` lineage on the 720 labelled rows this
session produced, with the encoder unfrozen, and moved 62 of 240 decisions on
a wording split it had never seen - from 0.7333 to 0.9125. That is the first
member in this project that is both *strong* and *different from its parent*,
which is exactly what rounds 122-137 said the swarm needed and could not get.

This records its votes beside the trio's so the ensemble question can be asked
of it. It has to run under the CUDA interpreter:

    .venv-cuda/Scripts/python tools/finetuned_votes.py \
        --model artifacts/rlcd-fi-v3 --suite assets/heldout2_test.jsonl

Usage:
    .venv-cuda/Scripts/python tools/finetuned_votes.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import TorchLayaBackend  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "artifacts"
                                              / "rlcd-fi-v3"))
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "heldout2_test.jsonl"))
    parser.add_argument("--name", default="ft-v3")
    parser.add_argument("--out", default=str(ROOT / "metrics"
                                             / "ensemble-checks"
                                             / "finetuned-votes.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = [json.loads(line) for line
             in Path(args.suite).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"loading {args.model} ...", flush=True)
    backend = TorchLayaBackend(args.name, model=args.model, device="cuda",
                               checkpoint=args.name)
    print(f"  loaded in {time.perf_counter():.1f}s", flush=True)

    rows, started = [], time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        vote = backend.decide(DecisionRequest(
            state=task["state"], question=task["question"],
            candidates=task["candidates"]))
        rows.append({"task_id": task["task_id"], "expected": task["expected"],
                     "choice": vote.choice, "probability": vote.probability})
        if index % 60 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    hits = sum(1 for row in rows if row["choice"] == row["expected"])
    print(f"accuracy {hits}/{len(rows)} = {hits / len(rows):.4f}")
    confidences = sorted(row["probability"] for row in rows)
    print(f"confidence: min {confidences[0]:.3f} "
          f"median {confidences[len(confidences) // 2]:.3f} "
          f"max {confidences[-1]:.3f}  "
          f"at 1.0: {sum(1 for value in confidences if value >= 0.999)}")

    out = Path(args.out)
    out.write_text(json.dumps({"model": args.model, "suite":
                               Path(args.suite).name, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
