"""Benchmark Laya ONNX across OpenVINO execution units with REAL inputs.

Captures one real input batch by running the ONNXAgent (onnxruntime) once on a
fixed state+questions, then feeds the captured tensors to OpenVINO-compiled
models on every EU. Honest per-EU results: compile failures are reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import openvino as ov

STATE = "User message: I want a refund, the order never arrived"
QUESTIONS = {
    "intent": {"type": "choice",
               "instructions": "Classify the intent of the message.",
               "criteria": ["refund", "complaint", "greeting"]},
    "urgent": {"type": "noul",
               "instructions": "Is this message urgent?"},
}


def capture_real_inputs(onnx_path: str) -> dict[str, np.ndarray]:
    """Run ONNXAgent once; capture the tensors it feeds to onnxruntime."""
    import onnxruntime as ort
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
        assert result["answers"], "no answers from ONNXAgent"
    finally:
        ort.InferenceSession.run = original_run
    print("captured inputs:", {k: v.shape for k, v in captured.items()}, flush=True)
    return captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="artifacts/laya-en.int8.onnx")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    feed = capture_real_inputs(args.onnx)

    core = ov.Core()
    print("available devices:", core.available_devices, flush=True)
    ov_model = core.read_model(args.onnx)
    input_names = [i.any_name for i in ov_model.inputs]
    ov_feed = {name: feed[name] for name in input_names if name in feed}

    # NPU/GPU need static shapes; pin every dynamic dim to the captured batch.
    reshape_map = {}
    for inp in ov_model.inputs:
        shape = list(feed[inp.any_name].shape)
        pshape = inp.get_partial_shape()
        concrete = [ov.Dimension(shape[i]) if not d.is_static else d.get_length()
                    for i, d in enumerate(pshape)]
        reshape_map[inp.any_name] = ov.PartialShape(concrete)
    ov_model.reshape(reshape_map)
    print("static shapes applied from captured batch", flush=True)

    results = {}
    for device in ["CPU", "GPU", "NPU"]:
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(ov_model, device)
            compile_s = time.perf_counter() - t0
        except Exception as exc:
            results[device] = {"status": "compile-failed", "error": str(exc)[:200]}
            print(f"[{device}] COMPILE FAILED: {exc}", flush=True)
            continue

        try:
            infer = compiled.create_infer_request()
            for _ in range(3):
                infer.infer(ov_feed)
            times = []
            for _ in range(args.runs):
                t = time.perf_counter()
                infer.infer(ov_feed)
                times.append((time.perf_counter() - t) * 1000)
        except Exception as exc:
            results[device] = {"status": "infer-failed", "error": str(exc)[:200]}
            print(f"[{device}] INFER FAILED: {exc}", flush=True)
            continue

        results[device] = {
            "status": "ok",
            "compile_s": round(compile_s, 1),
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(sorted(times)[max(0, int(0.95 * len(times)) - 1)], 1),
        }
        print(f"[{device}] compile {compile_s:.1f}s | "
              f"p50 {results[device]['p50_ms']} ms | "
              f"p95 {results[device]['p95_ms']} ms", flush=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
