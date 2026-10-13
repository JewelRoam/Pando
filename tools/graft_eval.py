"""Run the graft acceptance suite against every exported Laya lineage.

Produces accuracy per checkpoint, per-question-type breakdown, decision
latency, and a 10-bin ECE over answer confidences. Results are written to
docs/research/graft-report.json and printed as Markdown.

Usage: python tools/graft_eval.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "english": ("convaiinnovations/laya", ROOT / "artifacts/laya-en.onnx"),
    "multilingual": ("convaiinnovations/laya-multilingual", ROOT / "artifacts/laya-ml.onnx"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions", ROOT / "artifacts/laya-td.onnx"),
}


def build_question(task: dict) -> dict:
    qtype = task["qtype"]
    q = {"type": qtype, "instructions": task["question"]}
    if qtype == "choice":
        q["criteria"] = task["candidates"]
    elif qtype == "score":
        q["criteria"] = task["criteria"]
    elif qtype == "noul":
        pass  # noul takes no positional criteria; omit entirely
    return q


def predicted_answer(task: dict, answer: dict) -> str:
    if task["qtype"] == "noul":
        return "true" if float(answer.get("noul", 0.0)) > 0.5 else "false"
    if task["qtype"] == "score":
        # The float "score" is an expected value between levels; the bucket
        # distribution is the actual decision. Use its argmax + legend.
        probs = answer.get("probabilities") or {}
        if probs:
            top = max(probs, key=lambda k: probs[k])
            legend = answer.get("legend") or {}
            return str(legend.get(top, top))
        crit = task["criteria"]
        idx = int(round(float(answer.get("score", -1))))
        return crit[idx] if 0 <= idx < len(crit) else ""
    return str(answer.get("choice", ""))


def confidence_of(task: dict, answer: dict) -> float:
    return float(answer.get("answer_confidence") or answer.get("confidence") or 0.5)


def ece(confidences: list[float], corrects: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, k) for c, k in zip(confidences, corrects) if lo <= c < hi] \
            if b < bins - 1 else [(c, k) for c, k in zip(confidences, corrects) if lo <= c <= hi]
        if bucket:
            acc = sum(k for _, k in bucket) / len(bucket)
            conf = sum(c for c, _ in bucket) / len(bucket)
            e += (len(bucket) / total) * abs(acc - conf)
    return round(e, 4)


def main() -> None:
    suite_path = ROOT / "assets" / "graft_suite.jsonl"
    tasks = [json.loads(l) for l in suite_path.read_text(encoding="utf-8").splitlines()]

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {}}
    for ckpt, (repo, onnx) in SUITES.items():
        ckpt_tasks = [t for t in tasks if t["checkpoint"] == ckpt]
        print(f"[{ckpt}] loading {onnx.name} ...", flush=True)
        from laya.onnx_agent import ONNXAgent
        agent = ONNXAgent(repo, onnx_path=str(onnx))

        correct = 0
        confidences, corrects, latencies = [], [], []
        per_type: dict[str, list[int]] = {}
        details = []
        for task in ckpt_tasks:
            questions = {task["question"]: build_question(task)}
            t0 = time.perf_counter()
            result = agent.predict(task["state"], questions)
            lat_ms = (time.perf_counter() - t0) * 1000
            answer = result["answers"][task["question"]]
            pred = predicted_answer(task, answer)
            ok = int(pred == task["expected"])
            correct += ok
            conf = confidence_of(task, answer)
            confidences.append(conf)
            corrects.append(ok)
            per_type.setdefault(task["qtype"], []).append(ok)
            latencies.append(lat_ms)
            details.append({"task_id": task["task_id"], "pred": pred,
                            "expected": task["expected"], "ok": bool(ok),
                            "confidence": conf, "latency_ms": round(lat_ms, 1)})

        entry = {
            "accuracy": round(correct / len(ckpt_tasks), 4),
            "ece": ece(confidences, corrects),
            "latency_p50_ms": round(statistics.median(latencies), 1),
            "latency_p95_ms": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 1),
            "per_type": {k: f"{sum(v)}/{len(v)}" for k, v in per_type.items()},
            "details": details,
        }
        report["suites"][ckpt] = entry
        print(f"[{ckpt}] accuracy {entry['accuracy']} | ece {entry['ece']} | "
              f"p50 {entry['latency_p50_ms']} ms | {entry['per_type']}", flush=True)

    out = ROOT / "docs" / "research" / "graft-report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
