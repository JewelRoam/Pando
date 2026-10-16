"""Distil the cluster's strongest zero-shot member into training data.

Task 020.  The fine-tuned lineage (ft-v3: D 0.9417 / E 0.9917) was fitted on
720 rows a previous round labelled by hand.  The next generation of training
data should not need a human: the cluster already contains a stronger
zero-shot layer (8B: E 0.9250 / F 0.9833), and a second independent member
(4B) can filter its answers by agreement.  This tool turns that into a corpus:

  1. the caller first builds set G (``tools/build_large_suite.py --cues g``),
     an inventory of 48 phrasings for the eight original intents that appear
     verbatim in no earlier inventory;
  2. every G row is put to the 8B teacher with the candidate-only prompt
     ("variant B", the prompt that won on held-out wording in task 004);
  3. the same row is put to the 4B filter member;
  4. only rows where the two agree survive, with the agreed label as
     ``expected`` and ``teacher_agreement: true``.

The wording-overlap audit is part of the tool, not a claim in a report: it
recomputes the exact-string overlap of set G against A/A2/B/C/D/E (the cue
tables in ``tools/build_large_suite.py``) and F (the phrasings behind
``assets/heldout5_test.jsonl``), and refuses to label anything unless every
number is zero.

Rows are streamed to a raw log after each row so a long run can be resumed
with ``--resume`` without re-asking the teachers.

Usage:
    python tools/distill_corpus.py --suite assets/big8_g.jsonl \
        --out-train rollouts/distill_train.jsonl \
        --out-stats rollouts/distill_stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_large_suite as bls  # noqa: E402

# The candidate-only prompt, verbatim from tools/llm_decision_check.py's
# variant "b" - the one task 004 found wins on held-out wording.
VARIANT_B_SYSTEM = "只输出候选词本身，不要 JSON、不要置信度，不要任何其他文字。"

F_SUITE = ROOT / "assets" / "heldout5_test.jsonl"


def load_tasks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def candidate_match(text: str, candidates: list[str]) -> str | None:
    for candidate in sorted(candidates, key=len, reverse=True):
        pattern = (r"(?<![A-Za-z0-9_])" + re.escape(candidate)
                   + r"(?![A-Za-z0-9_])")
        if re.search(pattern, text, flags=re.IGNORECASE):
            return candidate
    return None


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
        raise RuntimeError(f"HTTP {exc.code}: "
                           f"{exc.read().decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"unreachable: {exc}") from exc
    return json.loads(body)


def completion_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    return content if isinstance(content, str) else str(content or "")


def ask_teacher(base_url: str, model: str, api_key: str, task: dict,
                timeout: float) -> dict:
    candidates = list(task["candidates"])
    user_prompt = (
        f"状态：{task['state']}\n"
        f"问题：{task['question']}\n"
        "候选：\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
        + "\n只从以上候选中选择一个。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VARIANT_B_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(base_url.rstrip("/") + "/chat/completions",
                         payload, api_key, timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0
    content = completion_text(response)
    return {
        "choice": candidate_match(content, candidates),
        "content": content,
        "latency_ms": latency_ms,
    }


def load_api_key(path: Path | None) -> str:
    if path is None:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def cue_strings(table: dict) -> set:
    return {cue.lower() for group in table.values() for cue in group}


def f_cue_strings() -> set:
    """Recover F's cue phrasings from the states it produced.

    ``assets/heldout5_test.jsonl`` is the only F artifact on main (the builder
    lives on the ``split-f`` branch), but every F state is
    ``<prefix><cue>`` optionally followed by ``. <filler>``, so the cue is the
    text between the prefix and the first sentence break.
    """
    cues: set[str] = set()
    for row in load_tasks(F_SUITE):
        state = row["state"]
        for prefix in ("User says: ", "Message: "):
            if state.startswith(prefix):
                state = state[len(prefix):]
                break
        cue = state.split(". ")[0].strip().rstrip(".")
        if cue:
            cues.add(cue.lower())
    return cues


def audit_overlap(g_name: str = "g") -> dict:
    """Exact-string overlap of set G against every earlier inventory."""
    target = cue_strings(bls.CUE_SETS[g_name])
    tables = {name: cue_strings(table)
              for name, table in bls.CUE_SETS.items() if name != g_name}
    # A2 is not a separate inventory: big2 reuses cue set A, so the exact
    # string set is identical and needs no separate comparison.
    tables["f"] = f_cue_strings()
    overlaps = {name: sorted(target & strings)
                for name, strings in sorted(tables.items())}
    report = {
        "target": g_name,
        "cues": len(target),
        "per_set_overlap": {name: len(hits)
                            for name, hits in overlaps.items()},
        "total_overlap": sum(len(hits) for hits in overlaps.values()),
        "examples": {name: hits[:5] for name, hits in overlaps.items()
                     if hits},
    }
    for name in ("a", "b", "c", "d", "e", "f"):
        report.setdefault("per_set_overlap", {})[name] = \
            report["per_set_overlap"].get(name, 0)
    return report


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def build_stats(rows: list[dict]) -> dict:
    total = len(rows)
    agreed = [row for row in rows
              if row["teacher_8b"] and row["teacher_4b"]
              and row["teacher_8b"] == row["teacher_4b"]]
    latencies = [row["latency_8b_ms"] for row in rows
                 if row.get("latency_8b_ms") is not None]
    latencies_4b = [row["latency_4b_ms"] for row in rows
                    if row.get("latency_4b_ms") is not None]
    kept_by_label = collections.Counter(row["teacher_8b"] for row in agreed)
    generated_by_label = collections.Counter(row["cue_label"] for row in rows)
    parse_failures_8b = sum(1 for row in rows if not row["teacher_8b"])
    parse_failures_4b = sum(1 for row in rows if not row["teacher_4b"])
    return {
        "total_generated": total,
        "kept": len(agreed),
        "discarded": total - len(agreed),
        "teacher_agreement_rate": (len(agreed) / total) if total else 0.0,
        "by_agreed_label": dict(sorted(kept_by_label.items())),
        "by_cue_label": dict(sorted(generated_by_label.items())),
        "parse_failures_8b": parse_failures_8b,
        "parse_failures_4b": parse_failures_4b,
        "latency_8b_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "latency_4b_ms": {
            "mean": statistics.fmean(latencies_4b) if latencies_4b else 0.0,
            "p50": percentile(latencies_4b, 0.50),
            "p95": percentile(latencies_4b, 0.95),
        },
    }


def discarded_examples(rows: list[dict], limit: int = 5) -> list[dict]:
    """Disagreements, biased towards the interesting ones.

    A parse failure is a mechanical miss, so real choice disagreements are
    reported first and parse failures only fill any remaining slots.
    """
    disagreements = [row for row in rows
                     if row["teacher_8b"] and row["teacher_4b"]
                     and row["teacher_8b"] != row["teacher_4b"]]
    failures = [row for row in rows
                if not row["teacher_8b"] or not row["teacher_4b"]]
    picked = disagreements[:limit]
    if len(picked) < limit:
        picked = picked + failures[:limit - len(picked)]
    return [{
        "task_id": row["task_id"],
        "state": row["state"],
        "cue_label": row["cue_label"],
        "teacher_8b": row["teacher_8b"],
        "teacher_4b": row["teacher_4b"],
    } for row in picked]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path,
                        default=ROOT / "assets" / "big8_g.jsonl")
    parser.add_argument("--out-train", type=Path,
                        default=ROOT / "rollouts" / "distill_train.jsonl")
    parser.add_argument("--out-stats", type=Path,
                        default=ROOT / "rollouts" / "distill_stats.json")
    parser.add_argument("--raw-out", type=Path,
                        default=ROOT / "rollouts" / "distill_raw.jsonl",
                        help="per-row teacher votes, streamed for resume")
    parser.add_argument("--base-url-8b", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--base-url-4b", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--key-file-8b", type=Path,
                        default=Path("~/.pandosi/llm-api-8b.token"))
    parser.add_argument("--key-file-4b", type=Path,
                        default=Path("~/.pandosi/llm-api.token"))
    parser.add_argument("--model-8b", default="qwen3-8b-instruct")
    parser.add_argument("--model-4b", default="qwen3-4b-instruct-2507")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="skip rows already present in --raw-out")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true",
                        help="run the overlap audit and stop")
    args = parser.parse_args()

    report = audit_overlap("g")
    print("overlap audit: " + json.dumps(report, ensure_ascii=False),
          flush=True)
    if report["total_overlap"] != 0:
        raise SystemExit("cue set G overlaps an earlier inventory; refusing "
                         "to label")
    if args.audit_only:
        return 0

    tasks = load_tasks(args.suite)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("no tasks to label")

    done: dict[str, dict] = {}
    if args.resume and args.raw_out.exists():
        for row in load_tasks(args.raw_out):
            done[row["task_id"]] = row
        print(f"resume: {len(done)} rows already labelled", flush=True)

    key_8b = load_api_key(args.key_file_8b)
    key_4b = load_api_key(args.key_file_4b)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    started = time.perf_counter()
    with args.raw_out.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            if task["task_id"] in done:
                rows.append(done[task["task_id"]])
                continue
            vote_8b = ask_teacher(args.base_url_8b, args.model_8b, key_8b,
                                  task, args.timeout)
            vote_4b = ask_teacher(args.base_url_4b, args.model_4b, key_4b,
                                  task, args.timeout)
            row = {
                "task_id": task["task_id"],
                "state": task["state"],
                "question": task["question"],
                "candidates": task["candidates"],
                "cue_label": task["expected"],
                "teacher_8b": vote_8b["choice"],
                "teacher_4b": vote_4b["choice"],
                "latency_8b_ms": vote_8b["latency_ms"],
                "latency_4b_ms": vote_4b["latency_ms"],
                "content_8b": vote_8b["content"].strip()[:200],
                "content_4b": vote_4b["content"].strip()[:200],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            if index == 1 or index % args.progress_every == 0 \
                    or index == len(tasks):
                elapsed = time.perf_counter() - started
                agree = sum(1 for r in rows
                            if r["teacher_8b"] and r["teacher_4b"]
                            and r["teacher_8b"] == r["teacher_4b"])
                print(f"  {index}/{len(tasks)} elapsed={elapsed:.0f}s "
                      f"agree={agree}/{len(rows)}", flush=True)

    stats = build_stats(rows)
    stats["suite"] = args.suite.name
    stats["teachers"] = {
        "labeler": {"model": args.model_8b, "base_url": args.base_url_8b,
                    "prompt": "variant b (candidate only)"},
        "filter": {"model": args.model_4b, "base_url": args.base_url_4b,
                   "prompt": "variant b (candidate only)"},
    }
    stats["overlap_audit"] = report
    stats["discarded_examples"] = discarded_examples(rows, 5)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["teacher_8b"] and row["teacher_4b"] \
                    and row["teacher_8b"] == row["teacher_4b"]:
                handle.write(json.dumps({
                    "task_id": row["task_id"],
                    "checkpoint": "english",
                    "qtype": "choice",
                    "state": row["state"],
                    "question": row["question"],
                    "candidates": row["candidates"],
                    "expected": row["teacher_8b"],
                    "teacher_agreement": True,
                }, ensure_ascii=False) + "\n")
    args.out_stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("STATS " + json.dumps({k: stats[k] for k in
                                 ("total_generated", "kept", "discarded",
                                  "teacher_agreement_rate")},
                                ensure_ascii=False), flush=True)
    print(f"wrote {args.out_train} and {args.out_stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
