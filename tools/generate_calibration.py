"""Generate per-checkpoint calibration data from the graft suite.

Runs each lineage on the suite, compares predicted vs expected, and
computes the optimal temperature that minimizes ECE. Outputs a
calibration.json per checkpoint.

Usage: python tools/generate_calibration.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from laya.onnx_agent import ONNXAgent

SUITES = {
    "english": ("convaiinnovations/laya", PROJECT_ROOT / "artifacts/laya-en.onnx"),
    "multilingual": ("convaiinnovations/laya-multilingual", PROJECT_ROOT / "artifacts/laya-ml.onnx"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions", PROJECT_ROOT / "artifacts/laya-td.onnx"),
}


def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ece(confidences: list[float], corrects: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, k) for c, k in zip(confidences, corrects)
                  if (lo <= c < hi) or (b == bins - 1 and lo <= c <= hi)]
        if bucket:
            acc = sum(k for _, k in bucket) / len(bucket)
            conf = sum(c for c, _ in bucket) / len(bucket)
            e += (len(bucket) / total) * abs(acc - conf)
    return e


def optimal_temperature(confidences: list[float], corrects: list[int]) -> float:
    best_t, best_e = 1.0, float("inf")
    for t_int in range(1, 51):
        t = t_int * 0.1
        calibrated = [min(max(sigmoid(logit(c) / t), 1e-7), 1 - 1e-7)
                      for c in confidences]
        e = ece(calibrated, corrects)
        if e < best_e:
            best_t, best_e = t, e
    return round(best_t, 2)


def main() -> None:
    suite_path = PROJECT_ROOT / "assets" / "graft_suite.jsonl"
    tasks = [json.loads(l) for l in
             suite_path.read_text(encoding="utf-8").splitlines()]

    out_dir = PROJECT_ROOT / "assets" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt, (repo, onnx) in SUITES.items():
        ckpt_tasks = [t for t in tasks if t["checkpoint"] == ckpt]
        print(f"[{ckpt}] running {len(ckpt_tasks)} tasks ...", flush=True)

        agent = ONNXAgent(repo, onnx_path=str(onnx))
        confidences, corrects = [], []
        for t in ckpt_tasks:
            q = {t["question"]: {"type": t["qtype"],
                                 "instructions": t["question"],
                                 **({"criteria": t.get("candidates") or t.get("criteria", [])}
                                    if t["qtype"] in ("choice", "score") else {})}}
            result = agent.predict(t["state"], q)
            ans = result["answers"][t["question"]]
            conf = float(ans.get("answer_confidence") or ans.get("confidence") or 0.5)
            pred = str(ans.get("choice", ans.get("noul", "")))
            corrects.append(int(pred == t["expected"]))
            confidences.append(conf)

        raw_ece = ece(confidences, corrects)
        optimal_t = optimal_temperature(confidences, corrects)
        calibrated = [min(max(sigmoid(logit(c) / optimal_t), 1e-7), 1 - 1e-7)
                      for c in confidences]
        calibrated_ece = ece(calibrated, corrects)

        cal_data = {
            "checkpoint": ckpt,
            "temperature": optimal_t,
            "raw_ece": round(raw_ece, 4),
            "calibrated_ece": round(calibrated_ece, 4),
            "n_tasks": len(ckpt_tasks),
            "accuracy": round(sum(corrects) / len(corrects), 4),
        }
        out_path = out_dir / f"{ckpt}.json"
        out_path.write_text(json.dumps(cal_data, indent=2), encoding="utf-8")
        print(f"  T={optimal_t} | raw ECE={raw_ece:.4f} -> "
              f"calibrated ECE={calibrated_ece:.4f} | "
              f"accuracy={cal_data['accuracy']}", flush=True)


if __name__ == "__main__":
    main()
