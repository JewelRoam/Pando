"""Reproducible CUDA RLCD fine-tune plus before/after decision evaluation.

Run with the CUDA environment:

    .venv-cuda/Scripts/python tools/rlcd_cuda_train.py

The script resolves the base Laya checkpoint, trains on the accumulated RLCD
contrast rows, then evaluates the original and trained checkpoints on the
configured graft slice.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_model(model: str) -> str:
    if Path(model).exists():
        return model
    cache_root = Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_cache = cache_root / "hub" / ("models--" +
                                       model.replace("/", "--"))
    snapshots = repo_cache / "snapshots"
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            required = (
                snapshot / "model.safetensors",
                snapshot / "rl_agent_config.json",
                snapshot / "tokenizer",
            )
            if all(path.exists() for path in required):
                return str(snapshot)
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        return snapshot_download(model)


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


def confidence_of(answer: dict) -> float:
    return float(answer.get("answer_confidence")
                 or answer.get("confidence") or 0.5)


def evaluate_checkpoint(model_dir: str, suite_path: Path,
                        checkpoint: str,
                        exclude_states: set[str] | None = None) -> dict:
    import laya

    tasks = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [
        task for task in tasks
        if checkpoint == "all" or task["checkpoint"] == checkpoint
    ]
    excluded_states = exclude_states or set()
    tasks = [
        task for task in candidates
        if task["state"] not in excluded_states
    ]
    if not tasks:
        raise RuntimeError("held-out evaluation selected zero tasks")
    started = time.perf_counter()
    agent = laya.load(model_dir, device="cuda")
    load_s = time.perf_counter() - started
    details = []
    correct = 0
    latencies = []
    for task in tasks:
        started = time.perf_counter()
        result = agent.predict(task["state"],
                               {task["question"]: build_question(task)})
        latencies.append((time.perf_counter() - started) * 1000)
        answer = result["answers"][task["question"]]
        prediction = predicted_answer(task, answer)
        ok = prediction == task["expected"]
        correct += int(ok)
        details.append({
            "task_id": task["task_id"],
            "expected": task["expected"],
            "predicted": prediction,
            "ok": ok,
            "confidence": round(confidence_of(answer), 4),
        })
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "load_s": round(load_s, 3),
        "candidates": len(candidates),
        "excluded_training_overlap": len(candidates) - len(tasks),
        "evaluated": len(tasks),
        "accuracy": round(correct / len(tasks), 4) if tasks else 0.0,
        "p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "details": details,
    }


def compare(before: dict, after: dict) -> dict:
    before_by_id = {item["task_id"]: item for item in before["details"]}
    matches = 0
    max_confidence_delta = 0.0
    changes = []
    for item in after["details"]:
        prior = before_by_id[item["task_id"]]
        matched = prior["predicted"] == item["predicted"]
        matches += int(matched)
        delta = abs(item["confidence"] - prior["confidence"])
        max_confidence_delta = max(max_confidence_delta, delta)
        if not matched:
            changes.append({
                "task_id": item["task_id"],
                "before": prior["predicted"],
                "after": item["predicted"],
                "expected": item["expected"],
            })
    total = len(after["details"])
    return {
        "tasks": total,
        "matches": matches,
        "match_rate": round(matches / total, 4) if total else 0.0,
        "max_confidence_delta": round(max_confidence_delta, 4),
        "label_changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="convaiinnovations/laya")
    parser.add_argument("--data", default=str(ROOT / "rollouts" /
                                               "train_rlcd.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" /
                                                     "rlcd-cuda-v1"))
    parser.add_argument("--eval-suite",
                        default=str(ROOT / "assets" / "graft_suite.jsonl"))
    parser.add_argument("--eval-checkpoint", default="english")
    parser.add_argument("--include-training-eval", action="store_true",
                        help="allow training states in before/after evaluation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--head-max-len", type=int, default=32)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed. Two runs of the same recipe can "
                             "differ by more than ten points of accuracy "
                             "(round 152), so a single run is one sample of "
                             "the loop, not a measurement of it")
    parser.add_argument("--unfreeze", action="store_true",
                        help="train the encoder too. The default freezes it, "
                             "which is what the published notebook does; "
                             "round 150 measured that a frozen encoder moves "
                             "1 decision in 240 and an unfrozen one needs "
                             "~6.4 GB of optimiser state for 400M parameters")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA PyTorch unavailable; install torch+cu130 in .venv-cuda")

    base_model = resolve_model(args.base_model)
    output_dir = Path(args.output_dir)
    report = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    if not args.skip_train:
        from laya.train import TrainConfig, finetune

        torch.cuda.reset_peak_memory_stats()
        config = TrainConfig(
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            freeze_encoder=not args.unfreeze,
            seed=args.seed,
            max_len=args.max_len,
            head_max_len=args.head_max_len,
            log_every=1,
            amp=True)
        started = time.perf_counter()
        report["training"] = finetune(
            args.data, base_model, str(output_dir), config, device="cuda")
        report["train_wall_s"] = round(time.perf_counter() - started, 3)
        report["train_peak_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        gc.collect()
        torch.cuda.empty_cache()

    suite_path = Path(args.eval_suite)
    exclude_states: set[str] = set()
    if not args.include_training_eval:
        exclude_states = {
            row["state"]
            for row in (
                json.loads(line)
                for line in Path(args.data).read_text(
                    encoding="utf-8").splitlines()
                if line.strip()
            )
            if row.get("state")
        }
    report["before"] = evaluate_checkpoint(
        base_model, suite_path, args.eval_checkpoint, exclude_states)
    report["after"] = evaluate_checkpoint(
        str(output_dir), suite_path, args.eval_checkpoint, exclude_states)
    report["comparison"] = compare(report["before"], report["after"])

    report_path = output_dir / "rlcd-cuda-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"device: {report['device']}")
        print(f"train peak memory: {report.get('train_peak_memory_mb')} MB")
        print(f"before accuracy: {report['before']['accuracy']}")
        print(f"after accuracy:  {report['after']['accuracy']}")
        print(f"decision match:  {report['comparison']['matches']}/"
              f"{report['comparison']['tasks']}")
        print(f"max confidence delta: "
              f"{report['comparison']['max_confidence_delta']}")
        print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
