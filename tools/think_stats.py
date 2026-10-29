#!/usr/bin/env python3
"""think_stats: read a proposer ledger and report what it can *actually* support.

The proposer now runs on more than one device, so the obvious question is
"is it evolving, or just pacing?" Both look identical in a log tail. This
tool answers only the part that is answerable from a ledger, and it refuses
to answer the part that is not.

What it reports, per ledger:
  * n rounds, wall-clock span, rounds/hour - the sample size, first of all;
  * action distribution over the 8-action space, and how many of the 8 were
    ever used (an action space you never leave is not a direction you chose);
  * repeat / novelty_forced / retried / publish-failure rates, with the
    1-sigma band on every share. At n=11 a share carries ~15pp of error:
    a difference smaller than that is not a difference;
  * Shannon entropy in bits against log2(8)=3.0, plus a bootstrap SE. High
    entropy means varied, NOT better - a fair coin is maximally entropic;
  * latency p50/p90/max, because a proposer that only thinks when nobody
    is looking is not a resident proposer.

What it deliberately does NOT report: whether any of this is progress. The
ledger cannot see outcomes, and the same eight actions drawn with the same
temperature from the same model produce the same spread with or without
memory. The control that would separate "the model is responding to its own
history" from "we are sampling the model's prior" - same model, same prompt,
ledger memory switched off - has not been run on any device. Until it is,
every diversity number here is a description, not evidence of evolution.

Usage:
  python tools/think_stats.py ~/.pando/think-ledger.jsonl
  python tools/think_stats.py --json mac.jsonl pi5.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

# The action space is enumerable today (pando/genesis.py ACTION_SPACE).
SPACE_SIZE = 8


def load_ledger(path: str | Path) -> list[dict]:
    """Read a jsonl ledger, skipping unparseable lines rather than dying."""
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def share_with_se(successes: int, n: int) -> tuple[float, float]:
    """Share and its 1-sigma standard error. p=0 gives se=0, not nan."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    return p, math.sqrt(max(0.0, p * (1.0 - p)) / n)


def shannon_bits(counts: Counter) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    total = 0.0
    for c in counts.values():
        if c:
            p = c / n
            total -= p * math.log2(p)
    return total


def entropy_se(counts: Counter, draws: int = 2000,
               seed: int = 20261017) -> float:
    """Bootstrap SE of the entropy. The ledger is small; do not pretend."""
    values = [a for a, c in counts.items() for _ in range(c)]
    n = len(values)
    if n < 2 or len(counts) < 2:
        return 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        resample = Counter(rng.choice(values) for _ in range(n))
        samples.append(shannon_bits(resample))
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
    return math.sqrt(var)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarise(rows: list[dict]) -> dict:
    """Pure reduction of ledger rows into the numbers above."""
    n = len(rows)
    # A round with no action is a recorded failure, not a distinct action.
    actions = [str(r.get("action") or "").strip() for r in rows]
    counts = Counter(a for a in actions if a)
    published = sum(1 for r in rows if r.get("published"))
    failed = sum(1 for r in rows if str(r.get("error") or "").strip())
    # A round that produced nothing is not a round that thought.
    produced = sum(1 for a in actions if a)

    latencies = sorted(
        float(r.get("latency_ms") or 0.0) / 1000.0 for r in rows
        if r.get("latency_ms"))
    served = Counter(str(r.get("served_by") or "?") for r in rows)

    starts = [float(r["started_at"]) for r in rows if r.get("started_at")]
    span_s = (max(starts) - min(starts)) if len(starts) >= 2 else 0.0

    rep_p, rep_se = share_with_se(
        sum(1 for r in rows if r.get("repeated")), n)
    forced_p, forced_se = share_with_se(
        sum(1 for r in rows if r.get("novelty_forced")), n)
    retry_p, retry_se = share_with_se(
        sum(1 for r in rows if r.get("retried")), n)
    fail_p, fail_se = share_with_se(failed, n)
    pub_p, pub_se = share_with_se(published, n)

    return {
        "n": n,
        "distinct_actions": len(counts),
        "space_size": SPACE_SIZE,
        "unused_actions": max(0, SPACE_SIZE - len(counts)),
        "action_counts": dict(counts.most_common()),
        "action_shares": {a: c / n for a, c in counts.items()} if n else {},
        "entropy_bits": shannon_bits(counts),
        "entropy_se_bits": entropy_se(counts),
        "entropy_ceiling_bits": math.log2(SPACE_SIZE),
        "repeat": {"share": rep_p, "se": rep_se,
                   "count": sum(1 for r in rows if r.get("repeated"))},
        "novelty_forced": {"share": forced_p, "se": forced_se,
                           "count": sum(1 for r in rows if r.get("novelty_forced"))},
        "retried": {"share": retry_p, "se": retry_se,
                    "count": sum(1 for r in rows if r.get("retried"))},
        "errored": {"share": fail_p, "se": fail_se, "count": failed},
        "published": {"share": pub_p, "se": pub_se, "count": published},
        "silent_rounds": n - produced,
        "served_by": dict(served),
        "latency_s": {
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "max": latencies[-1] if latencies else 0.0,
        },
        "span_s": span_s,
        "rounds_per_hour": (n / (span_s / 3600.0)) if span_s > 0 else 0.0,
    }


def format_report(name: str, s: dict) -> str:
    if s["n"] == 0:
        return f"== {name}: empty ledger (0 rounds) ==\n"
    out = []
    out.append(f"== {name} ==")
    out.append(
        f"n={s['n']} rounds over {s['span_s'] / 3600.0:.1f}h "
        f"({s['rounds_per_hour']:.1f}/h)  "
        f"served_by={s['served_by']}")
    out.append(
        f"actions: {s['distinct_actions']}/{s['space_size']} of the space used "
        f"({s['unused_actions']} never proposed)")
    for action, count in s["action_counts"].items():
        bar = "#" * count
        out.append(f"    {count:3d}  {count / s['n'] * 100:5.1f}%  {action:<20} {bar}")
    e, se = s["entropy_bits"], s["entropy_se_bits"]
    out.append(
        f"entropy: {e:.2f} bits +- {se:.2f} (ceiling {s['entropy_ceiling_bits']:.2f} "
        f"for {s['space_size']} actions)")
    def band(label, d):
        return (f"    {label:<10} {d['share'] * 100:5.1f}% +- {d['se'] * 100:4.1f}pp "
                f"({d['count']}/{s['n']})")
    out.append("rates (1-sigma; a gap smaller than the band is not a gap):")
    out.append(band("repeated", s["repeat"]))
    out.append(band("forced", s["novelty_forced"]))
    out.append(band("retried", s["retried"]))
    out.append(band("errored", s["errored"]))
    out.append(band("published", s["published"]))
    lat = s["latency_s"]
    out.append(
        f"latency: p50 {lat['p50']:.1f}s  p90 {lat['p90']:.1f}s  max {lat['max']:.1f}s")
    if s["silent_rounds"]:
        out.append(f"silent rounds (no action produced): {s['silent_rounds']}")
    return "\n".join(out) + "\n"


def caveats(stats: list[dict]) -> str:
    lines = ["-- 反思与质疑 --"]
    worst = min((s["n"] for s in stats if s["n"]), default=0)
    if worst:
        lines.append(
            f"1) 统计功效: 最小的账本 n={worst}。n={worst} 上 50% 的份额带约 "
            f"{math.sqrt(0.25 / worst) * 100:.0f}pp 的 1 SE，"
            f"任何小于约 {math.sqrt(0.25 / worst) * 100:.0f}pp 的份额差异都在噪声里，"
            f"不得称为提升。熵的 bootstrap SE 已随行给出。")
    lines.append(
        "2) 缺对照: 没有'关掉账本记忆、同模型同提示'的对照组，因此无法区分"
        "'模型在响应自己的历史'与'我们在采样它的先验'。这是能证伪'进化'的最小实验，"
        "至今未跑。等内存预算下'多台小模型 vs 一台大模型'的关键对照同样不存在。")
    lines.append(
        "3) 外部锚点: 账本只记录提案者自己发的动作，动作空间(pando/genesis.py "
        "ACTION_SPACE)由本仓库定义。整套评价是内部自洽、外部未锚定——"
        "熵高只等于'多样'，不等于'更好'；公平硬币也是最大熵。")
    lines.append(
        "4) 运维代价: 提案者本身只占几十 MB，但它依赖的常驻模型(Mac 8B+4B / pi5 1.7B) "
        "吃内存与 swap，延迟尾部由上面的 p90/max 给出。人类审批次数、被换页时"
        "'served_by' 落到备用模型(算力下降)都要计入分母，不能当零成本。")
    lines.append(
        "5) 可证伪的替代解释: 观测到的动作分布可能只是该模型在 temperature 下的"
        "固定点，与群体历史无关。证伪方式: 固定 seed 与提示、清空历史窗口跑 N 轮，"
        "若分布与当前无显著差异，则测到的是模型先验而非演化。")
    lines.append(
        "另: 本工具看不见提案是否被执行、是否带来收益——账本里没有结果的字段。"
        "'提出了什么'与'群体因此变强了'是两件事，这里只回答前者。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ledgers", nargs="+", help="one or more jsonl ledgers")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args(argv)

    stats = []
    for path in args.ledgers:
        rows = load_ledger(path)
        s = summarise(rows)
        s["path"] = str(path)
        s["name"] = Path(path).name if len(args.ledgers) > 1 else str(path)
        stats.append(s)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    first = True
    for s in stats:
        if not first:
            print()
        first = False
        print(format_report(s["name"], s))
    print(caveats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
