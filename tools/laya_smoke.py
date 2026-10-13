"""One-shot real-model smoke: load Laya, run one typed decision.

Usage: python tools/laya_smoke.py [--checkpoint english|multilingual]
Downloads weights on first run (~800MB). Run with proxy env if needed.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="english")
    args = parser.parse_args()

    repo = "convaiinnovations/laya" if args.checkpoint == "english" \
        else "convaiinnovations/laya-multilingual"

    print(f"loading {repo} ...", flush=True)
    t0 = time.perf_counter()
    import laya
    model = laya.load(repo)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    questions = {
        "intent": {"type": "choice",
                   "instructions": "Classify the intent of the message.",
                   "criteria": ["refund", "complaint", "greeting"]},
        "urgent": {"type": "noul",
                   "instructions": "Is this message urgent?"},
    }
    result = model.predict(
        "User message: I want a refund, the order never arrived",
        questions,
    )
    dt_ms = (time.perf_counter() - t1) * 1000
    print(f"decision: {result} in {dt_ms:.0f} ms", flush=True)
    assert isinstance(result, dict) and result, "empty prediction"
    assert result is not None
    print("LAYA SMOKE OK")


if __name__ == "__main__":
    main()
