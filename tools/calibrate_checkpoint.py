"""Fit temperature calibration for a local Laya checkpoint.

Calibration uses a labeled suite disjoint from the holdout suite. The script
saves a new checkpoint with updated temperature fields and reports holdout
accuracy, ECE, and confidence before/after.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.common import canonical_question


def labeled_choice_rows(suite_path: Path) -> list[dict]:
    rows = []
    for line in suite_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if task.get("qtype") != "choice":
            continue
        qid, options = canonical_question(task["question"], task["candidates"])
        rows.append({
            "state": task["state"],
            "questions": {
                qid: {
                    "type": "choice",
                    "instructions": f"Classify: {task['question']}.",
                    "criteria": options,
                },
            },
            "expected": {qid: task["expected"]},
        })
    return rows


def evaluate(model_dir: str, suite_path: Path) -> dict:
    import laya

    tasks = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("qtype") == "choice"
    ]
    agent = laya.load(model_dir, device="cuda")
    details = []
    correct = 0
    confidences = []
    correctness = []
    for task in tasks:
        qid, options = canonical_question(task["question"], task["candidates"])
        question = {qid: {
            "type": "choice",
            "instructions": f"Classify: {task['question']}.",
            "criteria": options,
        }}
        answer = agent.predict(task["state"], question)["answers"][qid]
        choice = answer.get("choice")
        confidence = float(answer.get("answer_confidence")
                           or answer.get("confidence") or 0.5)
        ok = choice == task["expected"]
        correct += int(ok)
        confidences.append(confidence)
        correctness.append(int(ok))
        details.append({
            "task_id": task["task_id"],
            "expected": task["expected"],
            "predicted": choice,
            "ok": ok,
            "confidence": round(confidence, 4),
        })

    def ece(bins: int = 5) -> float:
        total = len(confidences)
        if not total:
            return 0.0
        error = 0.0
        for index in range(bins):
            low = index / bins
            high = (index + 1) / bins
            bucket = [
                (confidence, correct)
                for confidence, correct in zip(confidences, correctness)
                if low <= confidence < high
                or (index == bins - 1 and confidence == high)
            ]
            if bucket:
                mean_confidence = sum(item[0] for item in bucket) / len(bucket)
                accuracy = sum(item[1] for item in bucket) / len(bucket)
                error += len(bucket) / total * abs(mean_confidence - accuracy)
        return round(error, 4)

    return {
        "tasks": len(tasks),
        "accuracy": round(correct / len(tasks), 4) if tasks else 0.0,
        "ece": ece(),
        "mean_confidence": round(
            sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration",
                        default=str(ROOT / "assets" / "graft_train.jsonl"))
    parser.add_argument("--holdout",
                        default=str(ROOT / "assets" / "graft_holdout.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")

    from laya.train import (calibration_records, fit_temperature_map,
                            items_from_rows, load_checkpoint, save_checkpoint)

    rows = labeled_choice_rows(Path(args.calibration))
    model, tokenizer, config = load_checkpoint(args.checkpoint)
    model.to("cuda")
    max_len = int(config.get("max_len", 512))
    head_max_len = int(config.get("head_max_len", 192))
    items, skipped = items_from_rows(
        tokenizer, rows, max_len, head_max_len, label_smoothing=0.0)
    if len(items) < 10:
        raise SystemExit(f"only {len(items)} calibration items; need >= 10")

    records = calibration_records(
        model, tokenizer, items, torch.device("cuda"),
        max_len, head_max_len, batch_size=8,
        parallel=config.get("option_layout", "sequential") == "parallel")
    fitted = fit_temperature_map(records, compute_ece=True)
    config = dict(config)
    config["temperature"] = fitted["temperature"]
    config["temperature_by_options"] = fitted["temperature_by_options"]
    save_checkpoint(model, tokenizer, config, args.output)

    report = {
        "checkpoint": args.checkpoint,
        "output": args.output,
        "calibration_items": len(items),
        "skipped": skipped,
        "temperature": fitted["temperature"],
        "temperature_by_options": fitted["temperature_by_options"],
        "n_by_bucket": fitted["n_by_bucket"],
        "fit_report": fitted.get("report"),
        "before": evaluate(args.checkpoint, Path(args.holdout)),
        "after": evaluate(args.output, Path(args.holdout)),
    }
    report["holdout_accuracy_delta"] = round(
        report["after"]["accuracy"] - report["before"]["accuracy"], 4)
    report_path = Path(args.output) / "calibration_eval.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"calibration items: {len(items)}")
        print(f"temperature: {fitted['temperature']}")
        print(f"holdout accuracy: "
              f"{report['before']['accuracy']} -> "
              f"{report['after']['accuracy']}")
        print(f"holdout ECE:      "
              f"{report['before']['ece']} -> {report['after']['ece']}")
        print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
