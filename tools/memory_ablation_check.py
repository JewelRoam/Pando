#!/usr/bin/env python3
"""Paired memory-ablation check for the resident proposer.

The colony can count action shares and repeated proposals, but those numbers do
not answer whether the proposer's own history changes what it proposes. This
tool runs the smallest missing control: the same local model, system prompt,
round number, host list, and paired random seed, with and without the proposer's
own history/counts/last-action feedback.

It deliberately fixes group proposals and outcome activity to an empty snapshot
for both arms. That removes mailbox drift from the comparison; the controlled
factor is the proposer's own memory only.

This is a check, not a claim that the swarm evolves. It prints effect sizes,
per-arm sampling uncertainty, and a paired label-permutation p-value.
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
                  host_names: list[str], action_space) -> dict[str, str]:
    """Build matched prompts with memory present vs ablated.

    Group proposals and outcome activity are fixed empty in both arms. The
    control keeps the same round number and all common instructions, so it does
    not introduce the contradictory "round 3520 / this is your first round"
    wording that an empty production ledger would produce.
    """
    own = ledger.tail(history_window)
    counts = ledger.summary()
    last_action = str(own[-1].get("action") or "") if own else ""

    def prompt(include_memory: bool) -> str:
        lines = [f"这是你的第 {round_no} 轮思考。", "", "【你自己的历史】"]
        if include_memory:
            history = _history_lines(own)
            lines.extend(history or ["（还没有。这是你的第一轮。）"])
        else:
            lines.append("（记忆对照：本轮不提供你自己的历史、动作计数或上一动作。）")
        lines.extend([
            "",
            "【群体里其他人的提案】",
            "（固定为空：实验控制，两臂均不提供。）",
            "",
            "【群体最近实际做的事（真实发生过的，不是提案）】",
            "（固定为空：实验控制，两臂均不提供。）",
        ])
        if include_memory and counts:
            lines.extend(["", "【你已经提过的动作次数】"])
            for action, stats in counts.items():
                lines.append(f"- {action}: {stats['proposed']} 次")
        if include_memory and last_action:
            lines.extend([
                "",
                f"【上一轮你提的是 {last_action}】",
                "如果这轮还是同一个动作，你必须在 goal 或 "
                "expected_effect 里写出这次和上次的差别；写不出来就换一个动作。",
            ])
        if include_memory and replicate_streak(own):
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

    memory_prompt = prompt(True)
    control_prompt = prompt(False)
    return {
        "memory": memory_prompt,
        "control": control_prompt,
        "last_action": last_action,
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
    memory = Counter()
    control = Counter()
    for i, (m, c) in enumerate(pairs):
        if (mask >> i) & 1:
            m, c = c, m
        memory[m] += 1
        control[c] += 1
    return memory, control


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


def summarise(pairs, actions: list[str], last_action: str, trials: int) -> dict:
    memory = Counter(m for m, _c in pairs)
    control = Counter(c for _m, c in pairs)
    n = len(pairs)
    tvd = total_variation(memory, control)
    p_value = paired_permutation(pairs, tvd, trials=trials)
    action_rows = []
    for action in sorted(set(memory) | set(control)):
        mc = memory[action]
        cc = control[action]
        action_rows.append({
            "action": action,
            "memory": mc,
            "control": cc,
            "memory_share": mc / n if n else 0.0,
            "control_share": cc / n if n else 0.0,
            "memory_wilson95": wilson_interval(mc, n),
            "control_wilson95": wilson_interval(cc, n),
        })
    same_memory = sum(m == last_action for m, _c in pairs) if last_action else 0
    same_control = sum(c == last_action for _m, c in pairs) if last_action else 0
    return {
        "n_pairs": n,
        "primary": {
            "metric": "total_variation_distance",
            "observed": tvd,
            "paired_permutation_p": p_value,
            "permutations": trials,
        },
        "secondary": {
            "last_action": last_action,
            "same_as_last": {
                "memory": same_memory,
                "control": same_control,
                "difference": (same_memory - same_control) / n if n else 0.0,
            },
        },
        "actions": action_rows,
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
    # Warm both prompt shapes once and discard the samples. This avoids making
    # the first scored call also pay for connection/model warm-up.
    for arm in ("memory", "control"):
        sample_once(proposer, SYSTEM_PROMPT, prompts[arm], args.seed_base - 1,
                    args.temperature, args.max_tokens)
    for i in range(args.pairs):
        seed = args.seed_base + i
        order = ("memory", "control") if i % 2 == 0 else ("control", "memory")
        got = {}
        for arm in order:
            text, usage, latency = sample_once(
                proposer, SYSTEM_PROMPT, prompts[arm], seed,
                args.temperature, args.max_tokens)
            action = parse_action(text, action_space)
            got[arm] = action
            rows.append({
                "pair": i,
                "arm": arm,
                "seed": seed,
                "action": action,
                "latency_ms": latency,
                "usage": usage,
                "raw": text,
            })
        pairs.append((got["memory"], got["control"]))
    summary = summarise(pairs, sorted(action_space), prompts["last_action"],
                        args.permutations)
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
            "fixed_group_context": "empty",
            "fixed_activity_context": "empty",
        },
        "prompts": {
            "memory_sha256": _sha256(prompts["memory"]),
            "control_sha256": _sha256(prompts["control"]),
            "memory_chars": len(prompts["memory"]),
            "control_chars": len(prompts["control"]),
        },
        "summary": summary,
        "rows": rows,
    }


def print_report(result: dict) -> None:
    cfg = result["config"]
    summary = result["summary"]
    print(f"model={cfg['model']} endpoint={cfg['base_url']} "
          f"n_pairs={summary['n_pairs']} seed_base={cfg['seed_base']}")
    print(f"prompt_sha256 memory={result['prompts']['memory_sha256']} "
          f"control={result['prompts']['control_sha256']}")
    print("fixed context: group proposals=empty, outcome activity=empty")
    print("\naction                    memory       control      delta")
    for row in summary["actions"]:
        delta = row["memory_share"] - row["control_share"]
        print(f"{row['action']:<25} {row['memory']:>3}/{summary['n_pairs']:<3} "
              f"({row['memory_share']:6.1%}) {row['control']:>3}/{summary['n_pairs']:<3} "
              f"({row['control_share']:6.1%}) {delta:+7.1%}")
    primary = summary["primary"]
    print(f"\nprimary TVD={primary['observed']:.3f} "
          f"paired-permutation p={primary['paired_permutation_p']:.4f} "
          f"({primary['permutations']} shuffles)")
    same = summary["secondary"]["same_as_last"]
    print(f"same as last action: memory={same['memory']}/{summary['n_pairs']} "
          f"control={same['control']}/{summary['n_pairs']} "
          f"delta={same['difference']:+.1%}")
    print("caveat: this tests one model on one fixed no-group/no-outcome context; "
          "it is not a group-intelligence result.")


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
