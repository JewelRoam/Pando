"""Do two ecosystems running the same model make the same decision?

Loads the Laya English checkpoint twice on this host - once through the
ONNX Runtime path and once through native PyTorch - and answers the same
40-task graft suite with both. Reports per-ecosystem accuracy, choice
agreement between the ecosystems, and the confidence gap.

This is the decision-level counterpart to the protocol-level result in
round 92: a heterogeneous quorum can only be called one intelligence if
its members agree, or if their disagreement is at least known.

Run with the CUDA environment, which carries both runtimes:
    .venv-cuda/Scripts/python tools/cross_ecosystem_check.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from pando.backend import OnnxLayaBackend, TorchLayaBackend  # noqa: E402
from pando.common import DecisionRequest, aggregate  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--onnx",
                        default=str(ROOT / "artifacts" / "laya-en.onnx"))
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--torch-device", default="CUDA")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tasks = [task for task in load_tasks(Path(args.suite))
             if task.get("qtype") == "choice"]
    print(f"suite: {len(tasks)} choice tasks", flush=True)

    onnx = OnnxLayaBackend("onnx-cpu", args.onnx, device="CPU",
                           checkpoint="onnx")
    torch_backend = TorchLayaBackend("torch-cuda", model=args.model,
                                     device=args.torch_device)
    print(f"loaded {onnx.checkpoint} and {torch_backend.checkpoint}",
          flush=True)

    rows = []
    for task in tasks:
        request = DecisionRequest(state=task["state"],
                                  question=task["question"],
                                  candidates=task["candidates"])
        started = time.perf_counter()
        a = onnx.decide(request)
        onnx_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        b = torch_backend.decide(request)
        torch_ms = (time.perf_counter() - started) * 1000

        expected = task["expected"]
        rows.append({
            "task_id": task["task_id"],
            "expected": expected,
            "onnx": a.choice, "onnx_conf": a.probability, "onnx_ms": onnx_ms,
            "torch": b.choice, "torch_conf": b.probability,
            "torch_ms": torch_ms,
            "agree": a.choice == b.choice,
            "onnx_correct": a.choice == expected,
            "torch_correct": b.choice == expected,
        })
        # Same-state quorum: the two ecosystems vote together.
        quorum = aggregate([a, b], min_quorum=2)
        rows[-1]["quorum_choice"] = quorum.choice
        rows[-1]["quorum_agree"] = quorum.agreement
        rows[-1]["quorum_tier"] = quorum.tier

    total = len(rows)
    agrees = sum(row["agree"] for row in rows)
    both_right = sum(row["onnx_correct"] and row["torch_correct"]
                     for row in rows)
    both_wrong = sum(not row["onnx_correct"] and not row["torch_correct"]
                     for row in rows)
    only_onnx = sum(row["onnx_correct"] and not row["torch_correct"]
                    for row in rows)
    only_torch = sum(row["torch_correct"] and not row["onnx_correct"]
                     for row in rows)
    conf_gap = [abs(row["onnx_conf"] - row["torch_conf"]) for row in rows]
    summary = {
        "suite": str(args.suite),
        "tasks": total,
        "checkpoints": [onnx.checkpoint, torch_backend.checkpoint],
        "choice_agreement": round(agrees / total, 4),
        "agree_count": agrees,
        "onnx_accuracy": round(sum(row["onnx_correct"] for row in rows) / total,
                               4),
        "torch_accuracy": round(
            sum(row["torch_correct"] for row in rows) / total, 4),
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "only_onnx_correct": only_onnx,
        "only_torch_correct": only_torch,
        "mean_confidence_gap": round(statistics.mean(conf_gap), 4),
        "max_confidence_gap": round(max(conf_gap), 4),
        "onnx_ms_p50": round(statistics.median(row["onnx_ms"] for row in rows),
                             2),
        "torch_ms_p50": round(statistics.median(row["torch_ms"]
                                                for row in rows), 2),
        "quorum_auto": sum(row["quorum_tier"] == "auto" for row in rows),
        "quorum_verify": sum(row["quorum_tier"] == "verify" for row in rows),
        "disagreements": [row for row in rows if not row["agree"]],
    }

    out_dir = ROOT / "metrics" / "cross-ecosystem"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"cross-{int(time.time())}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"agreement      : {summary['agree_count']}/{total} "
              f"({summary['choice_agreement']:.1%})")
        print(f"accuracy       : onnx {summary['onnx_accuracy']:.1%} | "
              f"torch {summary['torch_accuracy']:.1%}")
        print(f"both correct {both_right} | both wrong {both_wrong} | "
              f"only-onnx {only_onnx} | only-torch {only_torch}")
        print(f"confidence gap : mean {summary['mean_confidence_gap']} "
              f"max {summary['max_confidence_gap']}")
        print(f"latency p50    : onnx {summary['onnx_ms_p50']} ms | "
              f"torch {summary['torch_ms_p50']} ms")
        print(f"2-voter quorum : auto {summary['quorum_auto']} | "
              f"verify {summary['quorum_verify']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
