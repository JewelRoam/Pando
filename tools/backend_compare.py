"""Compare Laya decisions across backends: ORT-CPU vs OpenVINO-iGPU.

Same lineage, same weights, different numerics. Quantifies whether backend
numerics produce divergent decisions (an additional diversity axis for
consensus, on top of checkpoint diversity - round 2/14).

Usage: python tools/backend_compare.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.onnx_agent import ONNXAgent  # noqa: E402

from pando.backend import _make_openvino_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "assets" / "graft_suite.jsonl"
ONNX_CPU = ROOT / "artifacts/laya-en.onnx"
ONNX_FIXED = ROOT / "artifacts/laya-en-fixed.onnx"
REPO = "convaiinnovations/laya"


def build_question(task: dict) -> dict:
    q = {"type": task["qtype"], "instructions": task["question"]}
    if task["qtype"] in ("choice", "score"):
        q["criteria"] = task.get("candidates") or task.get("criteria")
    return q


def extract(task: dict, answer: dict) -> tuple[str, float]:
    if task["qtype"] == "noul":
        return ("true" if float(answer.get("noul", 0)) > 0.5 else "false",
                float(answer.get("noul", 0)))
    if task["qtype"] == "score":
        probs = answer.get("probabilities") or {}
        top = max(probs, key=lambda k: probs[k])
        label = (answer.get("legend") or {}).get(top, top)
        return (label, float(answer.get("answer_confidence") or 0))
    return (str(answer.get("choice", "")),
            float(answer.get("answer_confidence") or 0))


def run_suite(agent, label: str) -> dict:
    tasks = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines()]
    out = {}
    for t in tasks:
        q = {t["question"]: build_question(t)}
        t0 = time.perf_counter()
        result = agent.predict(t["state"], q)
        ms = (time.perf_counter() - t0) * 1000
        ans = result["answers"][t["question"]]
        choice, conf = extract(t, ans)
        out[t["task_id"]] = {"choice": choice, "conf": round(conf, 4),
                             "ms": round(ms, 1)}
    lat = [v["ms"] for v in out.values()]
    print(f"[{label}] p50 {statistics.median(lat):.1f} ms | "
          f"{len(out)} tasks", flush=True)
    return out


def main() -> None:
    print("loading ORT-CPU agent (fp32) ...", flush=True)
    cpu_agent = ONNXAgent(REPO, onnx_path=str(ONNX_CPU))
    cpu = run_suite(cpu_agent, "ORT-CPU")

    print("loading OV-iGPU agent (fp32, surgery graph) ...", flush=True)
    gpu_agent = ONNXAgent(REPO, onnx_path=str(ONNX_FIXED))
    gpu_agent.session = _make_openvino_session(str(ONNX_FIXED), "GPU",
                                           cpu_agent.session)
    gpu = run_suite(gpu_agent, "OV-iGPU")

    divergent = []
    for tid in cpu:
        if cpu[tid]["choice"] != gpu[tid]["choice"]:
            divergent.append(tid)
    conf_diffs = [abs(cpu[tid]["conf"] - gpu[tid]["conf"]) for tid in cpu]
    mean_conf_diff = round(sum(conf_diffs) / len(conf_diffs), 4)

    report = {
        "tasks": len(cpu),
        "divergent_decisions": divergent,
        "divergence_rate": round(len(divergent) / len(cpu), 4),
        "mean_abs_confidence_diff": mean_conf_diff,
        "cpu_p50_ms": round(statistics.median([v["ms"] for v in cpu.values()]), 1),
        "gpu_p50_ms": round(statistics.median([v["ms"] for v in gpu.values()]), 1),
    }
    print(json.dumps(report, indent=2))
    out_path = ROOT / "docs/archive/research/backend-compare.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
