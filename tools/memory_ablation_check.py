#!/usr/bin/env python3
"""Paired memory-ablation check for the resident proposer.

The colony can count action shares and repeated proposals, but those numbers do
not answer whether the proposer's own history changes what it proposes. This
tool runs seven same-seed arms against one local model:

* ``memory``: history rows, action counts, and the last action;
* ``last_action``: only the last action and its anti-repetition instruction;
* ``counts``: only the aggregate action counts;
* ``neutral``: a length-matched, colony-neutral filler prompt;
* ``pseudo_history``: history-shaped text with the same actions, counts, and
  last action, but false round numbers, goals, and publication states;
* ``coherent_fiction``: the same actions, counts, rounds, last action, and real
  publication states, with only the host fact in each goal changed to a
  plausible host where that action did not happen. The rewrite preserves both
  character count and, for the frozen replay ledger, tokenizer count;
* ``control``: none of those memory-derived sections.

Group proposals and outcome activity are fixed empty in every arm. This turns
the original all-or-nothing comparison into one-factor comparisons and keeps
the request order balanced across arms.

This is a check, not a claim that the swarm evolves. It reports action-share
uncertainty, paired label-permutation p-values, and Holm-adjusted p-values over
the planned pairwise contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pando.thinkd import (Ledger, OpenAIProposer, SYSTEM_PROMPT, extract_json,
                          host_names_from, replicate_streak)

INVALID = "(invalid)"
ARMS = ("memory", "coherent_fiction", "pseudo_history", "last_action",
        "counts", "neutral", "control")
ARM_ORDERS = (
    ("memory", "coherent_fiction", "pseudo_history", "last_action", "counts",
     "neutral", "control"),
    ("coherent_fiction", "pseudo_history", "last_action", "counts", "neutral",
     "control", "memory"),
    ("pseudo_history", "last_action", "counts", "neutral", "control", "memory",
     "coherent_fiction"),
    ("last_action", "counts", "neutral", "control", "memory",
     "coherent_fiction", "pseudo_history"),
    ("counts", "neutral", "control", "memory", "coherent_fiction",
     "pseudo_history", "last_action"),
    ("neutral", "control", "memory", "coherent_fiction", "pseudo_history",
     "last_action", "counts"),
    ("control", "memory", "coherent_fiction", "pseudo_history", "last_action",
     "counts", "neutral"),
)
PAIRWISE = (
    ("memory", "control", "full_memory_vs_none"),
    ("last_action", "control", "last_action_vs_none"),
    ("counts", "control", "counts_vs_none"),
    ("neutral", "control", "length_neutral_vs_none"),
    ("memory", "neutral", "full_memory_vs_length_neutral"),
    ("memory", "last_action", "full_memory_vs_last_action"),
    ("memory", "counts", "full_memory_vs_counts"),
    ("last_action", "counts", "last_action_vs_counts"),
    ("pseudo_history", "control", "pseudo_history_vs_none"),
    ("coherent_fiction", "control", "coherent_fiction_vs_none"),
    ("memory", "pseudo_history", "full_memory_vs_pseudo_history"),
    ("memory", "coherent_fiction", "full_memory_vs_coherent_fiction"),
    ("coherent_fiction", "pseudo_history", "coherent_fiction_vs_pseudo_history"),
)
FINAL_INSTRUCTION = "\n\n请提出下一步。只输出一行 JSON。"
NEUTRAL_FILLER_MARKER = "\n\n【中性长度填充】\n"
NEUTRAL_FILLER_UNIT = (
    "中性填充只用于匹配长度，不包含殖民地记忆、提案、动作、目标或结果。"
    "neutral padding only."
)
PSEUDO_GOAL_UNITS = (
    "虚构内容", "不存在的目标", "内容不可用", "内容不可用",
    "内容不可用", "内容不可用", "内容不可用", "内容不可用",
)
COHERENT_GOAL_UNITS = {
    "seed.export": (
        "整理本机保存的种子包，核对摘要和文件清单，把可复现的副本路径写入"
        "交接记录，并安排下一位成员独立抽查。"
    ),
    "seed.replicate": (
        "把已经校验的种子包同步到备用主机，确认两端摘要一致，记录传输结果，"
        "并保留失败时可重新执行的步骤。"
    ),
    "adaptive.governance": (
        "根据近期运行记录核对各主机上的策略偏差，更新治理规则，保留可回滚"
        "的变更说明、审批痕迹和下一轮验证条件。"
    ),
    "dynamic.threshold.adjustment": (
        "根据最近一批观测误差调整判定阈值，同时保留旧值、新值和适用范围，"
        "再用下一批独立样本复测调整效果。"
    ),
    "resource.rebalance": (
        "把部分计算任务迁往负载较低的节点，记录迁移前后的队列长度、内存"
        "占用和失败率，并设置自动回退条件。"
    ),
    "policy.set": (
        "更新节点策略并保存可回滚的版本编号，补充适用范围、冲突处理规则"
        "和用于验收的测试样例。"
    ),
    "meta.propose": (
        "整理新的治理动作提案，写明输入输出、允许范围、失败方式和回滚办法，"
        "再交给一个独立成员复核。"
    ),
    "culture.refine": (
        "修订现有操作手册，补上可复现的验收步骤、已知误报和人工接管条件，"
        "让下一位成员无需口头解释即可执行。"
    ),
}
COHERENT_GOAL_FALLBACK = (
    "核对当前运行状态，执行一项范围明确且可回滚的维护操作，把结果和验证"
    "证据写入交接记录。"
)
COHERENT_HOST_REPLACEMENTS = (("pi5", "nuc"), ("mac", "win"))


def _false_digits(value) -> str:
    """Change every digit while preserving the original number of characters."""
    text = str(value)
    changed = "".join(
        str(9 - int(char)) if char.isdigit() else char
        for char in text)
    return changed if changed != text else "0" * len(text)


def _false_text(value, unit: str = "内容不可用") -> str:
    """Return a visibly false value with the same character length."""
    text = str(value or "")
    repeats = (len(text) + len(unit) - 1) // len(unit)
    return (unit * repeats)[:len(text)]


def _fit_text(value, unit: str) -> str:
    """Return plausible prose of the exact requested character length."""
    text = str(value or "")
    if not text:
        return ""
    repeats = (len(text) + len(unit) - 1) // len(unit)
    return (unit * repeats)[:len(text)]


def _coherent_text(value, action: str = "") -> str:
    """Change a host fact while preserving the goal skeleton and length."""
    text = str(value or "")
    changed = text
    for source, target in COHERENT_HOST_REPLACEMENTS:
        changed = changed.replace(source, target)
    if changed != text:
        return changed
    unit = COHERENT_GOAL_UNITS.get(action, COHERENT_GOAL_FALLBACK)
    return _fit_text(changed, unit)


def _history_lines(rows: list[dict], pseudo: bool = False,
                   coherent_fiction: bool = False) -> list[str]:
    out = []
    for index, row in enumerate(rows):
        if pseudo:
            round_no = _false_digits(row.get("round"))
        else:
            round_no = row.get("round")
        goal_unit = PSEUDO_GOAL_UNITS[index % len(PSEUDO_GOAL_UNITS)]
        if pseudo:
            goal = _false_text(row.get("goal"), goal_unit)
        elif coherent_fiction:
            goal = _coherent_text(row.get("goal"), str(row.get("action") or ""))
        else:
            goal = row.get("goal") or ""
        published = bool(row.get("published"))
        status = "已发布" if published else "未发布"
        if pseudo:
            status = "未发布" if published else "已发布"
        if row.get("error"):
            if pseudo:
                error = _false_text(row["error"], goal_unit)
            else:
                error = row["error"]
            status += f"，失败原因：{error}"
        out.append(
            f"- r{round_no}: {row.get('action') or '(无)'} "
            f"目标「{goal}」 -> {status}")
    return out


def _length_match_neutral(control: str, target_chars: int) -> str:
    """Add colony-neutral filler of a fixed character count.

    Character equality is deterministic and model-independent. The filler is
    real prose rather than repeated punctuation so its token scale remains in
    the same band as the Chinese memory prompt.
    """
    if not control.endswith(FINAL_INSTRUCTION):
        raise ValueError("control prompt is missing its final instruction")
    gap = target_chars - len(control)
    available = gap - len(NEUTRAL_FILLER_MARKER)
    if available < 0:
        raise ValueError("memory prompt is too short for the neutral filler")
    repeats = (available + len(NEUTRAL_FILLER_UNIT) - 1) // len(NEUTRAL_FILLER_UNIT)
    body = (NEUTRAL_FILLER_UNIT * repeats)[:available]
    neutral = (control[:-len(FINAL_INSTRUCTION)] + NEUTRAL_FILLER_MARKER +
               body + FINAL_INSTRUCTION)
    if len(neutral) != target_chars:
        raise AssertionError("neutral prompt length did not match memory")
    return neutral


def build_prompts(ledger: Ledger, round_no: int, history_window: int,
                  host_names: list[str], action_space) -> dict:
    """Build seven prompts that differ only in memory-derived sections.

    All arms retain the same round number, fixed empty group/outcome context,
    action-space destination hint, and final output instruction. The ablated
    arms keep an empty-history placeholder so the section structure remains
    recognizable; only the supplied memory content changes. The neutral arm
    starts from the control prompt and adds colony-neutral filler until it has
    the same character count as the memory prompt. The pseudo-history arm keeps
    the exact action sequence, aggregate counts, and last action, but replaces
    round numbers, goals, and publication states with false values of equal
    length. The coherent-fiction arm also keeps the action sequence, counts,
    rounds, last action, and real publication states, but changes only the host
    fact in each goal. For this frozen ledger that rewrite is tokenizer-exact.
    It isolates token scale and visible damage from coherent but false facts.
    """
    own = ledger.tail(history_window)
    counts = ledger.summary()
    last_action = str(own[-1].get("action") or "") if own else ""

    def prompt(include_history: bool, include_counts: bool,
               include_last_action: bool, pseudo: bool = False,
               coherent_fiction: bool = False) -> str:
        lines = [f"这是你的第 {round_no} 轮思考。", "", "【你自己的历史】"]
        if include_history:
            history = _history_lines(
                own, pseudo=pseudo, coherent_fiction=coherent_fiction)
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

    control = prompt(False, False, False)
    memory = prompt(True, True, True)
    pseudo_history = prompt(True, True, True, pseudo=True)
    coherent_fiction = prompt(True, True, True, coherent_fiction=True)
    if len(pseudo_history) != len(memory):
        raise AssertionError("pseudo-history prompt length did not match memory")
    if len(coherent_fiction) != len(memory):
        raise AssertionError(
            "coherent-fiction prompt length did not match memory")
    return {
        "memory": memory,
        "coherent_fiction": coherent_fiction,
        "pseudo_history": pseudo_history,
        "last_action": prompt(False, False, True),
        "counts": prompt(False, True, False),
        "neutral": _length_match_neutral(control, len(memory)),
        "control": control,
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


def _file_identity(path: Path) -> dict:
    data = path.read_bytes()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "lines": data.count(b"\n"),
        "mtime_ns": stat.st_mtime_ns,
    }


def prompt_artifact(prompts: dict,
                    token_counts: dict[str, int] | None = None) -> dict:
    """Return replayable prompt text plus derived metadata."""
    artifact = {
        "system": SYSTEM_PROMPT,
        "system_sha256": _sha256(SYSTEM_PROMPT),
        "text": {arm: prompts[arm] for arm in ARMS},
        "sha256": {arm: _sha256(prompts[arm]) for arm in ARMS},
        "chars": {arm: len(prompts[arm]) for arm in ARMS},
        "memory_sha256": _sha256(prompts["memory"]),
        "control_sha256": _sha256(prompts["control"]),
        "neutral_sha256": _sha256(prompts["neutral"]),
        "memory_chars": len(prompts["memory"]),
        "control_chars": len(prompts["control"]),
        "neutral_chars": len(prompts["neutral"]),
        "neutral_char_matched_to_memory": (
            len(prompts["neutral"]) == len(prompts["memory"])
        ),
        "pseudo_history_sha256": _sha256(prompts["pseudo_history"]),
        "pseudo_history_chars": len(prompts["pseudo_history"]),
        "pseudo_history_char_matched_to_memory": (
            len(prompts["pseudo_history"]) == len(prompts["memory"])
        ),
        "coherent_fiction_sha256": _sha256(prompts["coherent_fiction"]),
        "coherent_fiction_chars": len(prompts["coherent_fiction"]),
        "coherent_fiction_char_matched_to_memory": (
            len(prompts["coherent_fiction"]) == len(prompts["memory"])
        ),
    }
    if token_counts is not None:
        artifact["tokenizer"] = {
            "counts": token_counts,
            "memory_vs_coherent_fiction_delta": (
                token_counts["coherent_fiction"] - token_counts["memory"]
            ),
            "memory_vs_coherent_fiction_exact": (
                token_counts["coherent_fiction"] == token_counts["memory"]
            ),
        }
    return artifact


def _read_secret(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def _default_tokenize_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/tokenize", "", ""))


def tokenize_text(text: str, url: str, api_key: str = "") -> int:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps({"content": text}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"tokenizer returned no token list from {url}")
    return len(tokens)


def tokenize_prompts(prompts: dict, url: str,
                     api_key: str = "") -> dict[str, int]:
    return {arm: tokenize_text(prompts[arm], url, api_key) for arm in ARMS}


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
    ledger_before = _file_identity(ledger_path)
    ledger = Ledger(ledger_path)
    round_no = len(ledger.tail(100_000)) + 1
    host_names = host_names_from(Path(args.hosts).expanduser())
    from pando.thinkd import effective_action_space
    action_space = effective_action_space()
    prompts = build_prompts(ledger, round_no, args.history_window,
                            host_names, action_space)
    ledger_after = _file_identity(ledger_path)
    if ledger_before != ledger_after:
        raise RuntimeError(
            "ledger changed while prompts were being built; pass a frozen "
            "snapshot via --ledger"
        )
    api_key = _read_secret(args.api_key_file)
    tokenize_url = args.tokenize_url or _default_tokenize_url(args.base_url)
    token_counts = tokenize_prompts(prompts, tokenize_url, api_key)
    if token_counts["memory"] != token_counts["coherent_fiction"]:
        raise RuntimeError(
            "coherent-fiction prompt is not tokenizer-exact: "
            f"memory={token_counts['memory']} "
            f"coherent_fiction={token_counts['coherent_fiction']}"
        )
    proposer = OpenAIProposer(
        base_url=args.base_url, model=args.model,
        api_key=api_key,
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
            "tokenize_url": tokenize_url,
            "model": args.model,
            "n_pairs": args.pairs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed_base": args.seed_base,
            "history_window": args.history_window,
            "ledger": str(ledger_path),
            "round_no": round_no,
            "host_names": host_names,
            "arms": list(ARMS),
            "arm_orders": [list(order) for order in ARM_ORDERS],
            "fixed_group_context": "empty",
            "fixed_activity_context": "empty",
        },
        "ledger_snapshot": ledger_before,
        "prompts": prompt_artifact(prompts, token_counts),
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
    parser.add_argument("--tokenize-url", default="")
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
