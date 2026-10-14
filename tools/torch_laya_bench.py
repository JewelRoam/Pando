"""Exercise the native PyTorch Laya backend (the non-OpenVINO ecosystem).

Loads the real checkpoint through ``TorchLayaBackend`` and answers the same
typed decision the HTTP stems serve, reporting load time and warm latency.
Skips cleanly when torch is not installed so it is safe on CPU-only CI.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from pandosi.common import DecisionRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="CUDA")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not installed; skipping torch backend bench")
        return 0

    from pandosi.backend import TorchLayaBackend

    if args.device.strip().upper() == "CUDA" and not torch.cuda.is_available():
        print("CUDA unavailable; skipping torch backend bench")
        return 0

    started = time.perf_counter()
    backend = TorchLayaBackend("torch-bench", model=args.model,
                               device=args.device)
    load_s = time.perf_counter() - started
    print(f"loaded {backend.checkpoint} in {load_s:.1f}s "
          f"device={args.device}", flush=True)

    request = DecisionRequest(
        state="The nightly build failed twice with a network timeout.",
        question="choose the best next action",
        candidates=["retry", "escalate", "abort"])
    votes = []
    latencies = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        vote = backend.decide(request)
        latencies.append((time.perf_counter() - started) * 1000)
        votes.append(vote)

    choices = {vote.choice for vote in votes}
    if len(choices) != 1:
        raise SystemExit(f"torch backend is non-deterministic: {choices}")
    if not {vote.choice for vote in votes} <= set(request.candidates):
        raise SystemExit("torch backend returned a choice outside the options")
    print(f"choice={votes[0].choice} p={votes[0].probability} "
          f"p50={statistics.median(latencies):.1f}ms "
          f"min={min(latencies):.1f}ms over {args.repeats} runs")
    print("TORCH LAYA BACKEND OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
