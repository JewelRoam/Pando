"""Benchmark one real Laya ONNX graph with CUDA and CPU providers.

Run inside a Python environment that has ``onnxruntime-gpu`` installed:

    .venv-cuda/Scripts/python tools/cuda_bench.py

The script captures the exact tensors produced by Laya's ONNXAgent, then
executes the same graph with CUDAExecutionProvider and CPUExecutionProvider.
This isolates the provider/hardware difference; it does not compare different
graphs or different model lineages.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE = "User message: I want a refund, the order never arrived"
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Classify the intent of the message.",
        "criteria": ["refund", "complaint", "greeting"],
    },
    "urgent": {
        "type": "noul",
        "instructions": "Is this message urgent?",
    },
}


def build_question(task: dict) -> dict:
    qtype = task["qtype"]
    question = {"type": qtype, "instructions": task["question"]}
    if qtype in ("choice", "score"):
        question["criteria"] = task.get("criteria", task["candidates"])
    return question


def predicted_answer(task: dict, answer: dict) -> str:
    if task["qtype"] == "noul":
        return "true" if float(answer.get("noul", 0.0)) > 0.5 else "false"
    if task["qtype"] == "score":
        probabilities = answer.get("probabilities") or {}
        if probabilities:
            top = max(probabilities, key=lambda key: probabilities[key])
            legend = answer.get("legend") or {}
            return str(legend.get(top, top))
        criteria = task["criteria"]
        index = int(round(float(answer.get("score", -1))))
        return criteria[index] if 0 <= index < len(criteria) else ""
    return str(answer.get("choice", ""))


def answer_confidence(answer: dict) -> float:
    return float(answer.get("answer_confidence")
                 or answer.get("confidence") or 0.5)


def decision_parity(agent, cpu_session: ort.InferenceSession) -> dict:
    suite_path = ROOT / "assets" / "graft_suite.jsonl"
    tasks = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [task for task in tasks if task["checkpoint"] == "english"]
    gpu_agent = agent
    details = []
    gpu_correct = cpu_correct = matches = 0
    max_confidence_delta = 0.0
    for task in tasks:
        questions = {task["question"]: build_question(task)}
        gpu_result = gpu_agent.predict(task["state"], questions)
        original_session = gpu_agent.session
        try:
            gpu_agent.session = cpu_session
            cpu_result = gpu_agent.predict(task["state"], questions)
        finally:
            gpu_agent.session = original_session

        gpu_answer = gpu_result["answers"][task["question"]]
        cpu_answer = cpu_result["answers"][task["question"]]
        gpu_pred = predicted_answer(task, gpu_answer)
        cpu_pred = predicted_answer(task, cpu_answer)
        matched = gpu_pred == cpu_pred
        matches += int(matched)
        gpu_correct += int(gpu_pred == task["expected"])
        cpu_correct += int(cpu_pred == task["expected"])
        confidence_delta = abs(answer_confidence(gpu_answer)
                               - answer_confidence(cpu_answer))
        max_confidence_delta = max(max_confidence_delta, confidence_delta)
        details.append({
            "task_id": task["task_id"],
            "expected": task["expected"],
            "gpu": gpu_pred,
            "cpu": cpu_pred,
            "match": matched,
            "gpu_confidence": round(answer_confidence(gpu_answer), 4),
            "cpu_confidence": round(answer_confidence(cpu_answer), 4),
            "confidence_delta": round(confidence_delta, 4),
        })
    total = len(tasks)
    return {
        "tasks": total,
        "matches": matches,
        "match_rate": round(matches / total, 4) if total else 0.0,
        "gpu_accuracy": round(gpu_correct / total, 4) if total else 0.0,
        "cpu_accuracy": round(cpu_correct / total, 4) if total else 0.0,
        "max_confidence_delta": round(max_confidence_delta, 4),
        "details": details,
    }


def capture_real_inputs(onnx_path: str) -> tuple[dict[str, np.ndarray], object]:
    """Return the real tensors and the loaded ONNXAgent."""
    from laya.onnx_agent import ONNXAgent

    captured: dict[str, np.ndarray] = {}
    original_run = ort.InferenceSession.run

    def spy_run(self, output_names, input_feed, *args, **kwargs):
        for name, value in input_feed.items():
            captured[name] = np.array(value)
        return original_run(self, output_names, input_feed, *args, **kwargs)

    ort.InferenceSession.run = spy_run
    try:
        agent = ONNXAgent("convaiinnovations/laya", onnx_path=onnx_path)
        result = agent.predict(STATE, QUESTIONS)
        if not result["answers"]:
            raise RuntimeError("ONNXAgent returned no answers")
    finally:
        ort.InferenceSession.run = original_run
    return captured, agent


def benchmark_session(session: ort.InferenceSession,
                      feed: dict[str, np.ndarray], output_names: list[str],
                      warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        outputs = session.run(output_names, feed)
    times_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        outputs = session.run(output_names, feed)
        times_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "providers": session.get_providers(),
        "p50_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(sorted(times_ms)[
            max(0, int(0.95 * len(times_ms)) - 1)], 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
        "outputs": [np.asarray(value) for value in outputs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default=str(ROOT / "artifacts" /
                                               "laya-en-fixed.onnx"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "CUDAExecutionProvider unavailable; install onnxruntime-gpu "
            "in an isolated environment before running this benchmark")

    started = time.perf_counter()
    feed, agent = capture_real_inputs(args.onnx)
    agent_load_s = time.perf_counter() - started
    output_names = ["logits", "act_logits"]

    cuda = benchmark_session(agent.session, feed, output_names,
                             args.warmup, args.runs)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_feed = dict(feed)
    started = time.perf_counter()
    cpu_session = ort.InferenceSession(
        args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    cpu_load_s = time.perf_counter() - started
    cpu = benchmark_session(cpu_session, cpu_feed, output_names,
                            args.warmup, args.runs)

    max_abs_diff = max(
        float(np.max(np.abs(cuda["outputs"][index] - cpu["outputs"][index])))
        for index in range(len(output_names)))
    speedup = round(cpu["p50_ms"] / cuda["p50_ms"], 2) \
        if cuda["p50_ms"] else None

    result = {
        "onnx": str(Path(args.onnx).resolve()),
        "cuda_available_providers": providers,
        "agent_load_s": round(agent_load_s, 3),
        "cpu_session_load_s": round(cpu_load_s, 3),
        "runs": args.runs,
        "warmup": args.warmup,
        "cuda": {k: v for k, v in cuda.items() if k != "outputs"},
        "cpu": {k: v for k, v in cpu.items() if k != "outputs"},
        "max_abs_logit_diff": max_abs_diff,
        "p50_speedup_cpu_over_cuda": speedup,
    }
    if not args.skip_parity:
        result["decision_parity"] = decision_parity(agent, cpu_session)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"onnx: {result['onnx']}")
        print(f"CUDA providers: {cuda['providers']}")
        print(f"CUDA p50/p95: {cuda['p50_ms']} / {cuda['p95_ms']} ms")
        print(f"CPU  p50/p95: {cpu['p50_ms']} / {cpu['p95_ms']} ms")
        print(f"CPU/CUDA p50 speedup: {speedup}x")
        print(f"max abs logit diff: {max_abs_diff:.6f}")
        if "decision_parity" in result:
            parity = result["decision_parity"]
            print(f"decision parity: {parity['matches']}/{parity['tasks']} "
                  f"({parity['match_rate']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
