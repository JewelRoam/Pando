#!/usr/bin/env python3
"""Paired memory-ablation check for the resident proposer.

The colony can count action shares and repeated proposals, but those numbers do
not answer whether the proposer's own history changes what it proposes. This
tool runs four same-seed arms against one local model:

* ``memory``: history rows, action counts, and the last action;
* ``last_action``: only the last action and its anti-repetition instruction;
* ``counts``: only the aggregate action counts;
* ``control``: none of those memory-derived sections.

Group proposals and outcome activity are fixed empty in every arm. This turns
the original all-or-nothing comparison into one-factor comparisons and keeps
the request order balanced across arms.

This is a check, not a claim that the swarm evolves. It reports action-share
uncertainty, paired label-permutation p-values, and Holm-adjusted p-values over
the six planned pairwise contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from pando.thinkd import (Ledger, OpenAIProposer, SYSTEM_PROMPT, extract_json,
                          host_names_from, replicate_streak)

INVALID = "(invalid)"
ARMS = ("memory", "last_action", "counts", "control")
ARM_ORDERS = (
    ("memory", "last_action", "counts", "control"),
    ("control", "counts", "last_action", "memory"),
    ("counts", "memory", "control", "last_action"),
    ("last_action", "control", "memory", "counts"),
)
PAIRWISE = (
    ("memory", "control", "full_memory_vs_none"),
    ("last_action", "control", "last_action_vs_none"),
    ("counts", "control", "counts_vs_none"),
    ("memory", "last_action", "full_memory_vs_last_action"),
    ("memory", "counts", "full_memory_vs_counts"),
    ("last_action", "counts", "last_action_vs_counts"),
)


def _history_lines(rows: list[dict]) -> list[str]:
    out = []
    for row in rows:
        status = "已发布" if row.get("published") else "未发布"
        if row.get("error"):
            status += f"，失败原因：{row['error']}"
        out.append(
            f"- r{row.get('round')}: {row.get('action') or '(无)'} "
            f"目标「{row.get('goal') or ''}」 -> {status}")
    return out


def build_prompts(ledger: Ledger, round_no: int, history_window: int,
                  host_names: list[str], action_space) -> dict:
    """Build four prompts that differ only in memory-derived sections.

    All arms retain the same round number, fixed empty group/outcome context,
    action-space destination hint, and final output instruction. The ablated
    arms keep an empty-history placeholder so the section structure remains
    recognizable; only the supplied memory content changes.
    """
    own = ledger.tail(history_window)
    counts = ledger.summary()
    last_action = str(own[-1].get("action") or "") if own else ""

    def prompt(include_history: bool, include_counts: bool,
               include_last_action: bool) -> str:
        lines = [f"这是你的第 {round_no} 轮思考。", "", "【你自己的历史】"]
        if include_history:
            history = _history_lines(own)
            lines.extend(history or ["（还没有。这是你的第一轮。）"])
        else:
            lines.append("（记忆对照：本轮不提供你自己的历史明细。）")
        lines.extend([
            "",
            "【群体里其他人的提案】",
            "（固定为空：实验控制，各臂均不提供。）",
            "",
            "【群体最近实际做的事（真实发生过的，不是提案）】",
            "（固定为空：实验控制，各臂均不提供。）",
        ])
        if include_counts and counts:
            lines.extend(["", "【你已经提过的动作次数】"])
            for action, stats in counts.items():
                lines.append(f"- {action}: {stats['proposed']} 次")
        if include_last_action and last_action:
            lines.extend([
                "",
                f"【上一轮你提的是 {last_action}】",
                "如果这轮还是同一个动作，你必须在 goal 或 "
                "expected_effect 里写出这次和上次的差别；写不出来就换一个动作。",
            ])
        if include_history and replicate_streak(own):
            lines.extend([
                "",
                "最近的提案里 seed.replicate 已经太多。这一轮不要再选它。",
            ])
        if host_names:
            from pando.thinkd import destination_hint
            lines.extend([
                "",
                "【可以写进 destination 的主机】",
                destination_hint(host_names),
            ])
        lines.extend(["", "请提出下一步。只输出一行 JSON。"])
        return "\n".join(lines)

    return {
        "memory": prompt(True, True, True),
        "last_action": prompt(False, False, True),
        "counts": prompt(False, True, False),
        "control": prompt(False, False, False),
        "last_action_value": last_action,
        "action_space": sorted(action_space),
    }


def parse_action(text: str, action_space) -> str:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return INVALID
    action = str(obj.get("action") or "")
    return action if action in set(action_space) else INVALID


def total_variation(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    n_left = sum(left.values())
    n_right = sum(right.values())
    if n_left <= 0 or n_right <= 0:
        return 1.0
    return 0.5 * sum(abs(left[k] / n_left - right[k] / n_right)
                     for k in keys)


def _swapped_counts(pairs, mask: int):
    left = Counter()
    right = Counter()
    for i, (original_left, original_right) in enumerate(pairs):
        if (mask >> i) & 1:
            original_left, original_right = original_right, original_left
        left[original_left] += 1
        right[original_right] += 1
    return left, right


def paired_permutation(pairs, observed: float, trials: int = 20000,
                       seed: int = 0) -> float:
    """Two-sided paired permutation for total-variation distance."""
    if not pairs:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        mask = rng.getrandbits(len(pairs))
        left, right = _swapped_counts(pairs, mask)
        if total_variation(left, right) >= observed - 1e-15:
            hits += 1
    return (hits + 1) / (trials + 1)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in the original input order."""
    count = len(raw_p_values)
    order = sorted(range(count), key=lambda i: raw_p_values[i])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, raw_p_values[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _normalise_pairs(pairs) -> list[dict[str, str]]:
    if not pairs:
        return []
    if isinstance(pairs[0], dict):
        return pairs
    # Backward-compatible form for the original two-arm experiment/tests.
    return [{"memory": left, "control": right} for left, right in pairs]


def summarise(pairs, actions: list[str], last_action: str, trials: int) -> dict:
    rows = _normalise_pairs(pairs)
    n = len(rows)
    observed_actions = sorted(set(actions) | {
        action for row in rows for action in row.values()
    })
    arm_stats = {}
    for arm in ARMS:
        counter = Counter(row.get(arm, INVALID) for row in rows)
        same = sum(row.get(arm) == last_action for row in rows) if last_action else 0
        arm_stats[arm] = {
            "actions": [{
                "action": action,
                "count": counter[action],
                "share": counter[action] / n if n else 0.0,
                "wilson95": wilson_interval(counter[action], n),
            } for action in observed_actions],
            "same_as_last": {
                "count": same,
                "share": same / n if n else 0.0,
                "wilson95": wilson_interval(same, n),
            },
        }

    comparisons = []
    for index, (left_arm, right_arm, label) in enumerate(PAIRWISE):
        paired = [(row.get(left_arm, INVALID), row.get(right_arm, INVALID))
                  for row in rows]
        left = Counter(a for a, _b in paired)
        right = Counter(b for _a, b in paired)
        tvd = total_variation(left, right)
        comparisons.append({
            "label": label,
            "left": left_arm,
            "right": right_arm,
            "metric": "total_variation_distance",
            "observed": tvd,
            "paired_permutation_p": paired_permutation(
                paired, tvd, trials=trials, seed=1000 + index),
        })
    adjusted = holm_adjust([
        row["paired_permutation_p"] for row in comparisons
    ])
    for row, adjusted_p in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = adjusted_p

    return {
        "n_pairs": n,
        "primary": {
            "comparisons": comparisons,
            "permutations": trials,
            "multiplicity": "holm",
            "family_size": len(comparisons),
            "family_wise_alpha": 0.05,
        },
        "secondary": {
            "last_action": last_action,
        },
        "arms": arm_stats,
        "valid_actions": actions,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_secret(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def sample_once(proposer: OpenAIProposer, system: str, user: str,
                seed: int, temperature: float, max_tokens: int) -> tuple[str, dict, float]:
    payload = {
        "model": proposer.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    started = time.time()
    data = proposer._post(payload)
    latency_ms = (time.time() - started) * 1000.0
    choices = data.get("choices") or []
    text = ""
    if choices:
        text = str((choices[0].get("message") or {}).get("content") or "")
    return text, data.get("usage") or {}, latency_ms


def run(args) -> dict:
    ledger_path = Path(args.ledger).expanduser()
    ledger = Ledger(ledger_path)
    round_no = len(ledger.tail(100_000)) + 1
    host_names = host_names_from(Path(args.hosts).expanduser())
    from pando.thinkd import effective_action_space
    action_space = effective_action_space()
    prompts = build_prompts(ledger, round_no, args.history_window,
                            host_names, action_space)
    proposer = OpenAIProposer(
        base_url=args.base_url, model=args.model,
        api_key=_read_secret(args.api_key_file),
        max_tokens=args.max_tokens, temperature=args.temperature,
        timeout=args.timeout,
    )
    pairs = []
    rows = []
    # Warm every prompt shape once and discard the samples. This avoids making
    # the first scored call also pay for connection/model warm-up.
    for arm in ARMS:
        sample_once(proposer, SYSTEM_PROMPT, prompts[arm], args.seed_base - 1,
                    args.temperature, args.max_tokens)
    for i in range(args.pairs):
        seed = args.seed_base + i
        order = ARM_ORDERS[i % len(ARM_ORDERS)]
        got = {}
        for position, arm in enumerate(order):
            text, usage, latency = sample_once(
                proposer, SYSTEM_PROMPT, prompts[arm], seed,
                args.temperature, args.max_tokens)
            action = parse_action(text, action_space)
            got[arm] = action
            rows.append({
                "pair": i,
                "arm": arm,
                "position": position,
                "seed": seed,
                "action": action,
                "latency_ms": latency,
                "usage": usage,
                "raw": text,
            })
        pairs.append(got)
    summary = summarise(pairs, sorted(action_space),
                        prompts["last_action_value"], args.permutations)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "n_pairs": args.pairs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed_base": args.seed_base,
            "history_window": args.history_window,
            "ledger": str(ledger_path),
            "arms": list(ARMS),
            "arm_orders": [list(order) for order in ARM_ORDERS],
            "fixed_group_context": "empty",
            "fixed_activity_context": "empty",
        },
        "prompts": {
            "sha256": {arm: _sha256(prompts[arm]) for arm in ARMS},
            "chars": {arm: len(prompts[arm]) for arm in ARMS},
            "memory_sha256": _sha256(prompts["memory"]),
            "control_sha256": _sha256(prompts["control"]),
            "memory_chars": len(prompts["memory"]),
            "control_chars": len(prompts["control"]),
        },
        "summary": summary,
        "rows": rows,
    }


def _format_p(value: float) -> str:
    return f"{value:.4g}" if value >= 0.0001 else f"{value:.2e}"


def print_report(result: dict) -> None:
    cfg = result["config"]
    summary = result["summary"]
    n = summary["n_pairs"]
    print(f"model={cfg['model']} endpoint={cfg['base_url']} "
          f"n_pairs={n} seed_base={cfg['seed_base']}")
    print("prompt chars: " + ", ".join(
        f"{arm}={result['prompts']['chars'][arm]}" for arm in ARMS))
    print("fixed context: group proposals=empty, outcome activity=empty")
    for arm in ARMS:
        parts = []
        for row in summary["arms"][arm]["actions"]:
            if row["count"]:
                parts.append(f"{row['action']}={row['count']}")
        print(f"{arm:<12} " + ", ".join(parts))

    print("\nplanned paired contrasts (TVD; raw p; Holm p)")
    for row in summary["primary"]["comparisons"]:
        print(f"{row['left']:<12} vs {row['right']:<12} "
              f"TVD={row['observed']:.3f}  p={_format_p(row['paired_permutation_p'])}  "
              f"Holm={_format_p(row['holm_adjusted_p'])}")

    last = summary["secondary"]["last_action"]
    print(f"\nsame as last action {last!r}")
    for arm in ARMS:
        same = summary["arms"][arm]["same_as_last"]
        low, high = same["wilson95"]
        print(f"{arm:<12} {same['count']}/{n} ({same['share']:.1%}; "
              f"Wilson95 {low:.1%}-{high:.1%})")
    print("caveat: one model, one fixed no-group/no-outcome context; these are "
          "single-factor output-distribution checks, not a group-intelligence result.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="~/.pandosi/think-ledger.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="qwen3-8b-q5km-instruct")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--hosts", default="~/.pandosi/hosts.json")
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=3520)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = run(args)
    print_report(result)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
