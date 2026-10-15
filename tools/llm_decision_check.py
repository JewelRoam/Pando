"""Evaluate an OpenAI-compatible LLM as a decision worker.

This is the Mac-side follow-up to the task-mailbox round: a local LLM is
asked to choose one candidate for every row of a held-out decision suite.  It
keeps the output deliberately simple so the resulting vote file can be merged
with the project's existing swarm decisions.

Typical use from the repository root::

    .venv/bin/python tools/llm_decision_check.py \
        --base-url http://127.0.0.1:8080/v1 \
        --api-key-file ~/.pandosi/llm-api.token \
        --model qwen3-4b-instruct-2507

The script writes ``metrics/llm-votes-d.json`` and prints a compact summary
that includes accuracy, parse failures, latency, tokens/s, and 5-bin ECE.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "assets" / "heldout3_test.jsonl"
DEFAULT_OUT = ROOT / "metrics" / "llm-votes-d.json"
SYSTEM_PROMPT = (
    "你是决策工人。只输出 JSON，形如 {\"choice\": \"<candidates 之一>\", "
    "\"probability\": <0~1 小数>}，不得输出任何其他文字。"
)

# Task 004 found a candidate-only prompt ("variant B") beats the JSON prompt
# ("variant A") on held-out wording, so both are kept selectable here.
PROMPT_VARIANTS = {
    "a": SYSTEM_PROMPT,
    "b": ("只输出候选词本身，不要 JSON、不要置信度，不要任何其他文字。"),
}

# Variant B carries no confidence, so its votes are recorded at chance.
VARIANT_B_PROBABILITY = 0.5
VARIANT_B_PROBABILITY_NOTE = (
    "variant b (candidate-only prompt) emits no confidence; "
    "probability recorded as 0.5 for every row"
)


def load_tasks(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_json_object(text: str) -> dict | None:
    """Return the first decodable JSON object in *text*, if any."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def candidate_match(text: str, candidates: list[str]) -> str | None:
    """Find an exact candidate token, preferring longer names."""
    for candidate in sorted(candidates, key=len, reverse=True):
        pattern = (r"(?<![A-Za-z0-9_])" + re.escape(candidate)
                   + r"(?![A-Za-z0-9_])")
        if re.search(pattern, text, flags=re.IGNORECASE):
            return candidate
    return None


def parse_answer(content: str, candidates: list[str]) -> dict:
    """Parse the model's response using JSON, then a candidate fallback."""
    parsed = first_json_object(content)
    choice = None
    probability = 0.0

    if parsed is not None:
        raw_choice = parsed.get("choice")
        if isinstance(raw_choice, str):
            exact = {candidate.lower(): candidate for candidate in candidates}
            choice = exact.get(raw_choice.strip().lower())
        raw_probability = parsed.get("probability")
        if isinstance(raw_probability, (int, float)) and not isinstance(
                raw_probability, bool):
            probability = float(raw_probability)
        elif isinstance(raw_probability, str):
            try:
                probability = float(raw_probability)
            except ValueError:
                probability = 0.0
        if math.isnan(probability) or math.isinf(probability):
            probability = 0.0
        probability = min(1.0, max(0.0, probability))
        if choice is not None:
            return {"choice": choice, "probability": probability,
                    "parse_mode": "json"}

    fallback = candidate_match(content, candidates)
    if fallback is not None:
        return {"choice": fallback, "probability": probability,
                "parse_mode": "regex"}

    return {"choice": None, "probability": 0.0,
            "parse_mode": "parse_error"}


def load_api_key(path: Path | None) -> str:
    if path is None:
        return ""
    return path.expanduser().read_text(encoding="utf-8").strip()


def post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"unreachable: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {body[:500]!r}") from exc


def completion_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def run_one(args, task: dict, api_key: str) -> dict:
    candidates = list(task["candidates"])
    user_prompt = (
        f"状态：{task['state']}\n"
        f"问题：{task['question']}\n"
        "候选：\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
        + "\n只从以上候选中选择一个。"
    )
    variant = getattr(args, "variant", "a")
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": PROMPT_VARIANTS[variant]},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
    }

    started = time.perf_counter()
    response = post_json(args.base_url.rstrip("/") + "/chat/completions",
                         payload, api_key, args.timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0
    content = completion_text(response)
    if variant == "b":
        choice = candidate_match(content, candidates)
        answer = {
            "choice": choice,
            "probability": VARIANT_B_PROBABILITY,
            "parse_mode": "regex" if choice is not None else "parse_error",
        }
    else:
        answer = parse_answer(content, candidates)

    usage = response.get("usage") or {}
    completion_tokens = number(usage.get("completion_tokens"))
    timings = response.get("timings") or {}
    predicted = number(timings.get("predicted_per_second"))
    if predicted is not None:
        tokens_per_second = predicted
    elif completion_tokens is not None and latency_ms > 0:
        tokens_per_second = completion_tokens / (latency_ms / 1000.0)
    else:
        tokens_per_second = None

    return {
        "task_id": task["task_id"],
        "expected": task["expected"],
        "choice": answer["choice"],
        "probability": answer["probability"],
        "parse_mode": answer["parse_mode"],
        "content": content,
        "latency_ms": latency_ms,
        "prompt_tokens": number(usage.get("prompt_tokens")),
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
        "response_model": response.get("model"),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["choice"] == row["expected"])
    errors = sum(1 for row in rows if row["choice"] is None)
    json_parses = sum(1 for row in rows if row["parse_mode"] == "json")
    regex_fallbacks = sum(1 for row in rows
                          if row["parse_mode"] == "regex")
    latencies = [row["latency_ms"] for row in rows
                 if row.get("latency_ms") is not None]
    throughputs = [row["tokens_per_second"] for row in rows
                   if row.get("tokens_per_second") is not None]
    probabilities = [float(row["probability"]) for row in rows
                     if row.get("probability") is not None]
    bins = []
    ece = 0.0
    for index in range(5):
        low = index / 5.0
        high = (index + 1) / 5.0
        if index == 4:
            members = [row for row in rows
                       if low <= float(row["probability"]) <= high]
        else:
            members = [row for row in rows
                       if low <= float(row["probability"]) < high]
        count = len(members)
        if count:
            accuracy = sum(1 for row in members
                           if row["choice"] == row["expected"]) / count
            confidence = sum(float(row["probability"])
                             for row in members) / count
            ece += (count / total) * abs(accuracy - confidence)
            bins.append({"low": low, "high": high, "count": count,
                         "accuracy": accuracy, "confidence": confidence})
        else:
            bins.append({"low": low, "high": high, "count": 0,
                         "accuracy": None, "confidence": None})
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_errors": errors,
        "regex_fallbacks": regex_fallbacks,
        "json_parse_rate": (json_parses / total if total else 0.0),
        "avg_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_tokens_per_second": (statistics.fmean(throughputs)
                                  if throughputs else 0.0),
        "probability_mean": (statistics.fmean(probabilities)
                             if probabilities else 0.0),
        "ece_5bin": ece,
        "bins": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-file", type=Path,
                        default=Path("~/.pandosi/llm-api.token"))
    parser.add_argument("--model", default="qwen3-4b-instruct-2507")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N rows for a smoke test")
    parser.add_argument("--summary-out", type=Path, default=None,
                        help="optionally save the detailed summary as JSON")
    parser.add_argument("--variant", choices=sorted(PROMPT_VARIANTS),
                        default="a",
                        help="a: JSON choice+probability prompt (default); "
                             "b: candidate-only prompt (no confidence; "
                             "probability recorded as 0.5)")
    args = parser.parse_args()

    tasks = load_tasks(args.suite)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("no tasks to run")

    api_key = load_api_key(args.api_key_file)

    print(f"suite={args.suite} rows={len(tasks)} model={args.model} "
          f"variant={getattr(args, 'variant', 'a')}", flush=True)
    rows: list[dict] = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        row = run_one(args, task, api_key)
        rows.append(row)
        if index == 1 or index % 24 == 0 or index == len(tasks):
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(tasks)} elapsed={elapsed:.1f}s "
                  f"choice={row['choice']!r} "
                  f"parse={row['parse_mode']}", flush=True)

    summary = summarize(rows)
    summary["model"] = args.model
    summary["suite"] = args.suite.name
    summary["prompt_variant"] = getattr(args, "variant", "a")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True),
          flush=True)

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    vote_rows = [
        {
            "task_id": row["task_id"],
            "expected": row["expected"],
            "choice": row["choice"],
            "probability": row["probability"],
        }
        for row in rows
    ]
    variant = getattr(args, "variant", "a")
    record = {
        "model": args.model,
        "suite": args.suite.name,
        "prompt": {"variant": variant,
                   "system": PROMPT_VARIANTS[variant]},
        "rows": vote_rows,
    }
    if variant == "b":
        record["probability_note"] = VARIANT_B_PROBABILITY_NOTE
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)

    if args.summary_out:
        summary_out = args.summary_out.expanduser()
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        detailed = dict(summary)
        detailed["rows"] = rows
        summary_out.write_text(json.dumps(detailed, ensure_ascii=False,
                                          indent=2) + "\n",
                               encoding="utf-8")
        print(f"wrote {summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
