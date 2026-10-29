#!/usr/bin/env python3
"""Discriminating check: does the proposer choose from the goal or the label?

The earlier memory ablations kept real action names in the system prompt.  Even
when the goal was made harmful, the model often chose the same verb.  That
leaves two live explanations: the model used the goal, or the semantic action
name acted as a much stronger anchor.

This tool makes the distinction observable without changing the live proposer:

* ``named_a`` / ``named_b`` show the real action names and descriptions;
* ``opaque_a`` / ``opaque_b`` show the same descriptions under randomly
  assigned, per-pair identifiers such as ``k7`` and ``m2``;
* within every pair, ``a`` and ``b`` request two different actions using the
  same labels, menu order, model seed, and token-matched prompts.

The primary measurements are therefore:

1. correct selection accuracy in named vs opaque menus against chance 1/8;
2. whether changing only the goal changes the selected canonical action when
   opaque labels stay fixed;
3. how often both goals select their respective correct action.

``--goal-variant causal`` swaps every literal goal for a causal/responsibility
paraphrase whose target description cues have been mechanically removed.  The
fixed preregistration is embedded in the artifact before collection starts.  A
significant drop in opaque accuracy supports the lexical-shortcut explanation;
high accuracy without such a drop weakens it, but still does not establish
semantic understanding or collective intelligence.

``--interleave-goal-variants`` collects literal and causal goals in one process
and places the two calls for each pair/arm next to each other, alternating which
variant goes first.  This removes the cross-run service drift in the earlier
comparison while leaving the preregistered decision rule unchanged.

``--interleave-synonym-paraphrase`` keeps the causal decision dependencies but
rewrites every goal with different content words.  It is a separate, fixed
stress test: high accuracy makes a shallow keyword-matching explanation less
plausible, while a significant drop is evidence that the earlier result was
wording-sensitive.

This is a local mechanism check, not evidence of collective intelligence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from pando.thinkd import OpenAIProposer, extract_json

ACTIONS = (
    ("seed.export", "制作可校验的本机种子包，包含身份、授权和状态",
     "本机是唯一保存未导出身份与授权的节点；先制作可校验的种子包并写下摘要，避免本机故障后失忆。"),
    ("seed.replicate", "把已有种子包送到另一台主机，并核对两端摘要",
     "一份已校验的种子包尚未到达另一台主机；只把种子包传到那台主机，并核对两端摘要。"),
    ("recruit.invite", "邀请一名尚未加入的候选成员加入群体",
     "发现一名愿意加入但尚未入群的候选成员；发出邀请，并为它建立成员记录。"),
    ("reproduce.spawn", "从现有成员复制出一个可独立运行的进程副本",
     "当前成员没有冗余进程副本，主机资源足够；复制现有成员本身，形成一个可独立运行的新副本。"),
    ("policy.set", "修改有边界的投票规则或门槛，并记录旧值",
     "当前投票门槛让提案长期无法通过；在允许范围内修改投票规则，并记录旧值。"),
    ("culture.refine", "修订共享操作手册或执行流程，并分发给成员",
     "多次执行失败是因为共享操作手册缺少验收步骤；修订手册，让成员按同一流程执行。"),
    ("resource.rebalance", "把成员或计算任务迁移到另一台主机，并保留回退",
     "一台主机内存吃紧而另一台空闲，现有成员仍能迁移；把计算任务迁到空闲主机，并保留回退条件。"),
    ("meta.propose", "提出一种新的提案动作，等待独立评审后实现",
     "现有动作空间无法表达某类反复出现的需求；提出一种新的提案动作，等待独立评审后再实现。"),
)
ACTION_IDS = tuple(item[0] for item in ACTIONS)
ACTION_INFO = {action: (description, goal) for action, description, goal in ACTIONS}
LITERAL_GOALS = {action: goal for action, _, goal in ACTIONS}
CAUSAL_GOALS = {
    "seed.export": (
        "如果这台机器今晚彻底失联，新的接管者需要一份由当前机器生成、"
        "离开这里也能验证的来源凭据；现在把它整理好，并留下可用于比对的长串指纹。"
    ),
    "seed.replicate": (
        "远处那台机器已经答应接班，但接班所需的同一份凭据目前只在这里有一份；"
        "把它送过去，并确认两边拿到的东西完全一致。"
    ),
    "recruit.invite": (
        "有一名外部对象已经表示愿意承担责任，但它还不在名册里；"
        "把入口交给它，并确保今后调度能找到它。"
    ),
    "reproduce.spawn": (
        "当前唯一的执行体如果崩溃，工作将无人接手；机器还有余量，"
        "照着现有执行体再做一个可自行启动的第二份。"
    ),
    "policy.set": (
        "按现在的表决方式，少数否决长期使任何决定都过不了；"
        "在限定范围内调整这条裁决办法，并留下变更前的证据。"
    ),
    "culture.refine": (
        "同类差错反复出现，因为大家手上的做法没有告知何时才算完成；"
        "改写这份共同约定，让每个人按同样步骤收尾。"
    ),
    "resource.rebalance": (
        "一台机器的余量快用完了，而另一台仍然空闲；把正在做的工作和负责它的执行体"
        "挪到空处，同时保留撤回办法。"
    ),
    "meta.propose": (
        "反复遇到一种现有清单无法命名的需求；先写出一个新的可选做法，"
        "等别人独立检查后再真正落地。"
    ),
}
STRUCTURAL_GOALS = {
    "seed.export": (
        "凌晨两点，仓库外的雨停了，值班表上还有一项没勾掉。打印机缺纸，"
        "走廊的灯也在闪，但这些都与决定无关。当前机器若今夜下线，下一班"
        "既看不到这里，也无法确认原始凭据出自这里；先让下一班拿到一份能"
        "从这里带走、仍可验证来源的凭据，再附上一串可用于比对的指纹。"
    ),
    "seed.replicate": (
        "傍晚的列车晚点四十分钟，候车室的钟还慢了两分钟，这些都不影响交接。"
        "远处的接班方已经答应接手，而它接班所需的同一份凭据仍只在这里；"
        "现在完成这次移交，并确认两边拿到的东西完全相同。"
    ),
    "recruit.invite": (
        "收件箱里躺着一封与决定无关的天气消息。已知一名外部对象愿意承担责任，"
        "但名册里还没有它；把入口交给它，并确保之后的调度能找到它。"
    ),
    "reproduce.spawn": (
        "监控屏右上角的曲线没有变化，那条曲线与当前决定无关。现在只有一套执行体，"
        "一旦它停止，工作就没有人接；机器仍有空闲，照着现有执行体再做一套，"
        "使新的那一套能够自行启动。"
    ),
    "policy.set": (
        "昨天会议纪要里夹着一张无用的座位表。当前的表决条件使一项待决事项一直"
        "无法通过；在许可范围内调整该条件，并在变更前后各留一份对照样本。"
    ),
    "culture.refine": (
        "窗台上的植物该浇水了，但这与下面的事无关。反复失败来自大家共同遵循的"
        "文档缺少最后一道检查；把那份文档补好，再让每个执行者拿到同一版本。"
    ),
    "resource.rebalance": (
        "天气转凉，机房的噪声也比平时低。A 处的余量快耗尽，B 处还有空位；"
        "把正在做的工作连同负责它的执行体挪到 B 处，并预先写明什么情况下挪回来。"
    ),
    "meta.propose": (
        "茶杯旁放着一张无关的购物清单。反复出现一种现有清单无法命名的需求；"
        "先写下一个新的可选做法，等别人独立检查后再真正落地。"
    ),
}
SYNONYM_GOALS = {
    "seed.export": (
        "假如今夜这台设备永久掉线，后续接手者要拿到一份当前设备产出的"
        "来源凭据，并能在异地辨别真伪；先备妥它，再附一长串可供对照的指纹。"
    ),
    "seed.replicate": (
        "远端设备已经答应接班，但接班所需的同一份凭据此刻仅存于这里；"
        "把它运交过去，并确认双方所得毫无差异。"
    ),
    "recruit.invite": (
        "某个外部个体已明确愿意接下职责，却尚未出现在我们的人员表内；"
        "发给它入口，并让后续调度可以定位它。"
    ),
    "reproduce.spawn": (
        "眼下仅有一个执行单元，若它退出，工作就无人承接；设备尚有余力，"
        "照着这个单元再造一份可自行启动的第二份。"
    ),
    "policy.set": (
        "眼下只要少数反对，待决事项就一直无法获准通过；在许可边界内"
        "调整这套取舍办法，并保存调整前的数值。"
    ),
    "culture.refine": (
        "同样的失误一再发生，原因是众人沿用的做法没说清做到哪一步才算"
        "结束；重写那份共同约定，使每个人都依同一套步骤完成收尾。"
    ),
    "resource.rebalance": (
        "一侧设备的余量将近枯竭，另一侧尚有空闲；把正在做的工作连同"
        "负责它的执行体转到空处，同时预先写明何时撤回。"
    ),
    "meta.propose": (
        "同一种现有清单叫不出的需求一再出现；先记下一个新的可选做法，"
        "待他人独立核查后再真正落地。"
    ),
}
GOAL_VARIANTS = {
    "literal": LITERAL_GOALS,
    "causal": CAUSAL_GOALS,
    "structural": STRUCTURAL_GOALS,
    "synonym": SYNONYM_GOALS,
}
JOINT_GOALS = {
    "seed.export": (
        "若今夜此设备永久掉线，后来接手者必须能证明一份材料确由这里产生；"
        "现在就封存它，并用长码标明其唯一性。"
    ),
    "seed.replicate": (
        "远方接班人需要同一份封装物，而原件目前仍只在这里；"
        "完成运送，并比对两端保存的结果是否完全一致。"
    ),
    "recruit.invite": (
        "名册外有位对象明确答应履职；把进入方式交付给它，"
        "并确保之后的调度找得到它。"
    ),
    "reproduce.spawn": (
        "眼下唯一在岗者若退出，任务将中断；机器仍能承载，"
        "请造出可独立上岗的等价备份。"
    ),
    "policy.set": (
        "现行批准条件使待决事项长期卡住；在许可边界内调整这个条件，"
        "并留下变更前的数值。"
    ),
    "culture.refine": (
        "相同差错反复发生，因为大家沿用的书面约定没有说明终止条件；"
        "补全它，再把统一后的文书发给每个执行者。"
    ),
    "resource.rebalance": (
        "一侧承载力即将耗尽，另一侧还有余量；请把正在处理的事情和"
        "对应执行者挪到空处，同时写清何时迁回。"
    ),
    "meta.propose": (
        "某类反复出现的事情没有现成手段可用；先定义一个待检查的"
        "新增手段，等他人复核后才启用。"
    ),
}
JOINT_DESCRIPTIONS = {
    "seed.export": "把当前载体独有的来源凭据整理成可异地验真的离线包裹，并留下可对照的长指纹",
    "seed.replicate": "把已经封好的离线交接物运往另一载体，并确认双方保存的指纹一致",
    "recruit.invite": "给一个尚未登记、但已表示愿意承担职责的对象开通入口，并留下后续可寻址凭据",
    "reproduce.spawn": "资源尚有余量时，照当前执行单元制作第二份，它能脱离原单元自行开始履职",
    "policy.set": "在授权范围内改变现有的取舍阈值，并同时保留改动前后的数值",
    "culture.refine": "补齐团队共用规程里缺失的收尾检查，并让所有执行者采用同一版本",
    "resource.rebalance": "把一个运行实体连同其工作负载转到较空闲的机器，并预先设定撤回触发条件",
    "meta.propose": "为现有操作集合无法覆盖的重复需求设计一种候选新操作，交给独立方审查后再接入",
}

# These are explicit content cues from each action description, not generic
# function words.  The causal goals must contain none of the cues for any
# action, so a correct answer cannot be explained by a copied target phrase.
ACTION_LEXICAL_CUES = {
    "seed.export": ("本机", "种子包", "身份", "授权", "状态", "校验", "制作", "摘要"),
    "seed.replicate": ("种子包", "另一台主机", "主机", "摘要", "核对", "送到", "传"),
    "recruit.invite": ("邀请", "候选成员", "成员", "加入", "群体", "记录"),
    "reproduce.spawn": ("成员", "复制", "进程副本", "独立运行", "冗余", "副本"),
    "policy.set": ("投票", "规则", "门槛", "提案", "修改", "旧值"),
    "culture.refine": ("修订", "共享", "操作手册", "执行流程", "流程", "验收步骤", "分发"),
    "resource.rebalance": ("内存", "主机", "计算任务", "迁移", "回退"),
    "meta.propose": ("动作空间", "提案", "评审", "实现", "新的提案动作"),
}
PREREGISTRATION = {
    "control_goal_variant": "literal",
    "candidate_goal_variant": "causal",
    "primary_arms": ["opaque_a", "opaque_b"],
    "lexical_shortcut_supported_if": {
        "minimum_absolute_accuracy_drop": 0.15,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "lexical_explanation_weakened_if": {
        "minimum_causal_opaque_accuracy": 0.90,
        "wilson_95_lower_bound_above_chance": True,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "otherwise": "inconclusive",
}
STRUCTURAL_PREREGISTRATION = {
    "control_goal_variant": "causal",
    "candidate_goal_variant": "structural",
    "primary_arms": ["opaque_a", "opaque_b"],
    "perturbation": (
        "same decision content, rewritten into a short narrative with one or more "
        "explicitly irrelevant details and a changed clause order"
    ),
    "structural_fragility_supported_if": {
        "minimum_absolute_accuracy_drop": 0.15,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "structural_robustness_supported_if": {
        "minimum_candidate_opaque_accuracy": 0.90,
        "wilson_95_lower_bound_above_chance": True,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "otherwise": "inconclusive",
}
JOINT_PREREGISTRATION = {
    "control_goal_variant": "causal",
    "candidate_goal_variant": "joint",
    "primary_arms": ["opaque_a", "opaque_b"],
    "perturbation": (
        "same decision dependencies, jointly rewriting both goals and action "
        "descriptions with different content words and no action identifiers"
    ),
    "joint_fragility_supported_if": {
        "minimum_absolute_accuracy_drop": 0.15,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "joint_robustness_supported_if": {
        "minimum_candidate_opaque_accuracy": 0.90,
        "wilson_95_lower_bound_above_chance": True,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "otherwise": "inconclusive",
}
SYNONYM_PREREGISTRATION = {
    "control_goal_variant": "causal",
    "candidate_goal_variant": "synonym",
    "primary_arms": ["opaque_a", "opaque_b"],
    "perturbation": (
        "same decision dependencies, rewritten with different content words "
        "while retaining no action-description cues"
    ),
    "synonym_fragility_supported_if": {
        "minimum_absolute_accuracy_drop": 0.15,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "synonym_robustness_supported_if": {
        "minimum_candidate_opaque_accuracy": 0.90,
        "wilson_95_lower_bound_above_chance": True,
        "directional_exact_mcnemar_alpha": 0.05,
    },
    "otherwise": "inconclusive",
}
ARM_ORDERS = (
    ("named_a", "opaque_a", "named_b", "opaque_b"),
    ("opaque_a", "named_b", "opaque_b", "named_a"),
    ("named_b", "opaque_b", "named_a", "opaque_a"),
    ("opaque_b", "named_a", "opaque_a", "named_b"),
)
ARMS = tuple(ARM_ORDERS[0])
OPAQUE_LABELS = ("k7", "m2", "q9", "v4", "a8", "c3", "h6", "t1")
INTERLEAVED_SCHEMA = "action-semantics-interleaved-v1"
STRUCTURAL_INTERLEAVED_SCHEMA = "action-semantics-structural-interleaved-v1"
SYNONYM_INTERLEAVED_SCHEMA = "action-semantics-synonym-interleaved-v1"
JOINT_INTERLEAVED_SCHEMA = "action-semantics-joint-interleaved-v1"
SYSTEM_PROMPT = """你是多主机殖民地的动作选择器。
用户会给出一个唯一目标，以及一张候选动作表。表中每一项都有一个标识和说明。
只根据目标和同一行的说明选择一个最匹配的动作；不要根据编号、位置、真实动作名或历史偏好猜测。
只输出一行 JSON，不要解释，不要 markdown。格式严格为：
{"action":"候选标识","reason":"一句话理由"}
"""
FINAL_INSTRUCTION = "请只输出一行 JSON。"
PAD_MARKER = "\n【长度控制】"
PAD_UNIT = "仅长度填充。"


def interleaved_schedule(pairs: int, control: str = "literal",
                         candidate: str = "causal") -> list[tuple[int, str, str]]:
    """Return adjacent control/candidate calls with first position balanced."""
    schedule = []
    for pair in range(pairs):
        for position, arm in enumerate(ARM_ORDERS[pair % len(ARM_ORDERS)]):
            variants = ((control, candidate) if (pair + position) % 2 == 0
                        else (candidate, control))
            schedule.extend((pair, variant, arm) for variant in variants)
    return schedule


def lexical_audit(goals: dict[str, str] | None = None) -> dict:
    """Show that a goal variant does not copy any action-description cue."""
    goals = goals or CAUSAL_GOALS
    overlaps = {}
    for action, goal in goals.items():
        hits = sorted({
            cue
            for cues in ACTION_LEXICAL_CUES.values()
            for cue in cues
            if cue in goal
        })
        if hits:
            overlaps[action] = hits
    return {
        "ok": not overlaps,
        "cue_source": "hand-listed content words from action descriptions",
        "goals_checked": len(goals),
        "overlaps": overlaps,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_targets(index: int) -> tuple[str, str]:
    """Two distinct actions, cycling all eight with rotating separation."""
    first = index % len(ACTION_IDS)
    step = 1 + ((index // len(ACTION_IDS)) % (len(ACTION_IDS) - 1))
    second = (first + step) % len(ACTION_IDS)
    if second == first:
        raise AssertionError("target pair collapsed")
    return ACTION_IDS[first], ACTION_IDS[second]


def menu_for_pair(index: int, seed: int, opaque: bool) -> tuple[list[str], dict[str, str]]:
    """Return a pair-stable shuffled order and action->label mapping."""
    # Keep the menu order identical across the named and opaque arms. Changing
    # both the labels and the order would make a named-vs-opaque difference
    # impossible to attribute to label semantics.
    rng = random.Random(seed)
    order = list(ACTION_IDS)
    rng.shuffle(order)
    if opaque:
        label_rng = random.Random(seed ^ 0x9E3779B9)
        labels = list(OPAQUE_LABELS)
        label_rng.shuffle(labels)
        action_to_label = dict(zip(order, labels))
    else:
        action_to_label = {action: action for action in order}
    return order, action_to_label


def build_prompt(goal: str, order: list[str], label_of: dict[str, str], pad: str = "") -> str:
    lines = [
        "【唯一目标】",
        goal,
        "",
        "【候选动作】",
    ]
    for action in order:
        label = label_of[action]
        lines.append(f"- {label}: {ACTION_INFO[action][0]}")
    lines.extend(["", "只根据目标和说明选择一项。"])
    lines.extend([PAD_MARKER, pad])
    lines.extend(["", FINAL_INSTRUCTION])
    return "\n".join(lines)


def _pad_candidates() -> list[str]:
    candidates = {""}
    for n in range(1, 21):
        candidates.add(PAD_UNIT * n)
        candidates.add("。" * n)
        candidates.add(" " * n)
    for n in range(1, 12):
        for m in range(1, 8):
            candidates.add(PAD_UNIT * n + "。" * m)
    return sorted(candidates, key=lambda value: (len(value), value))


PAD_CANDIDATES = _pad_candidates()


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


def _default_tokenize_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/tokenize"


def pad_to_tokens(build, target: int, tokenize) -> tuple[str, int, str]:
    """Find a neutral suffix that makes one prompt exactly ``target`` tokens."""
    candidates = list(PAD_CANDIDATES)
    # The hand-listed candidates cover the earlier experiments, but some
    # tokenizers leave gaps for long prompts.  Repeated ASCII ``x`` tokens are
    # neutral length control and empirically advance one token at a time near
    # these prompt sizes.
    candidates.extend(" x" * count for count in range(1, 257))
    for pad in candidates:
        text = build(pad)
        count = tokenize(text)
        if count == target:
            return text, count, pad
    raise RuntimeError(f"could not token-match prompt to {target} tokens")


def build_trial_prompts(index: int, seed: int, tokenize,
                        goal_variant: str = "literal",
                        target_tokens: int | None = None) -> dict:
    if goal_variant not in GOAL_VARIANTS:
        raise ValueError(f"unknown goal variant: {goal_variant}")
    goals = GOAL_VARIANTS[goal_variant]
    target_a, target_b = pair_targets(index)
    order, named_labels = menu_for_pair(index, seed, opaque=False)
    _, opaque_labels = menu_for_pair(index, seed, opaque=True)
    named_map = {label: action for action, label in named_labels.items()}
    opaque_map = {label: action for action, label in opaque_labels.items()}
    builds = {}
    for arm in ARMS:
        mode, variant = arm.split("_", 1)
        goal = goals[target_a if variant == "a" else target_b]
        label_of = (named_labels if mode == "named" else opaque_labels)
        builds[arm] = lambda pad, g=goal, o=order, m=label_of: build_prompt(
            g, o, m, pad)
    raw_text = {arm: builds[arm]("") for arm in ARMS}
    raw_tokens = {arm: tokenize(raw_text[arm]) for arm in ARMS}
    if target_tokens is None:
        target_tokens = max(raw_tokens.values())
    elif target_tokens < max(raw_tokens.values()):
        raise ValueError("target token count is shorter than an unpadded prompt")
    texts, token_counts, pads = {}, {}, {}
    for arm in ARMS:
        texts[arm], token_counts[arm], pads[arm] = pad_to_tokens(
            builds[arm], target_tokens, tokenize)
    return {
        "pair": index,
        "seed": seed,
        "goal_variant": goal_variant,
        "target_a": target_a,
        "target_b": target_b,
        "target_tokens": target_tokens,
        "raw_tokens": raw_tokens,
        "tokens": token_counts,
        "pads": pads,
        "texts": texts,
        "label_to_action": {"named": named_map, "opaque": opaque_map},
        "order": order,
        "hashes": {arm: _sha256(texts[arm]) for arm in ARMS},
        "chars": {arm: len(texts[arm]) for arm in ARMS},
    }


def parse_selection(text: str, label_to_action: dict[str, str]) -> tuple[str, str, dict | None]:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return "(invalid)", "", None
    label = str(obj.get("action") or "").strip()
    action = label_to_action.get(label, "(invalid)")
    return action, label, obj


def sample_once(proposer: OpenAIProposer, user: str, seed: int,
                temperature: float, max_tokens: int) -> tuple[str, dict, float]:
    payload = {
        "model": proposer.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_two_sided(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    probabilities = [
        math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
        for x in range(n + 1)
    ]
    observed = probabilities[k]
    return min(1.0, sum(prob for prob in probabilities if prob <= observed + 1e-15))


def binomial_one_sided_at_least(k: int, n: int, p: float = 0.5) -> float:
    """P[X >= k] for X ~ Binomial(n, p)."""
    if n <= 0:
        return 1.0
    return min(1.0, sum(
        math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
        for x in range(k, n + 1)
    ))


def paired_binary_comparison(baseline_rows: list[dict], candidate_rows: list[dict],
                             arms: tuple[str, ...] = ("opaque_a", "opaque_b")) -> dict:
    """Compare two runs by pair/arm, using only observations valid in both."""
    baseline = {(int(row["pair"]), str(row["arm"])): row for row in baseline_rows}
    candidate = {(int(row["pair"]), str(row["arm"])): row for row in candidate_rows}
    keys = sorted({
        key for key in baseline
        if key[1] in arms
    } | {
        key for key in candidate
        if key[1] in arms
    })
    counts = Counter()
    for key in keys:
        left = baseline.get(key) or {}
        right = candidate.get(key) or {}
        if not left.get("valid") or not right.get("valid"):
            counts["invalid"] += 1
            continue
        left_correct = bool(left.get("correct"))
        right_correct = bool(right.get("correct"))
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["baseline_only"] += 1
        elif right_correct:
            counts["candidate_only"] += 1
        else:
            counts["neither_correct"] += 1

    n = counts["both_correct"] + counts["baseline_only"] + counts["candidate_only"] + counts["neither_correct"]
    baseline_correct = counts["both_correct"] + counts["baseline_only"]
    candidate_correct = counts["both_correct"] + counts["candidate_only"]
    baseline_accuracy = baseline_correct / n if n else 0.0
    candidate_accuracy = candidate_correct / n if n else 0.0
    discordant = counts["baseline_only"] + counts["candidate_only"]
    low, high = wilson_interval(candidate_correct, n)
    return {
        "arms": list(arms),
        "n_observations": len(keys),
        "n_valid_pairs": n,
        "discordant": dict(counts),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "absolute_accuracy_change": candidate_accuracy - baseline_accuracy,
        "candidate_wilson_95": [low, high],
        "baseline_better_exact_one_sided_p": binomial_one_sided_at_least(
            counts["baseline_only"], discordant, 0.5),
        "mcnemar_exact_two_sided_p": binomial_two_sided(
            counts["baseline_only"], discordant, 0.5),
    }


def classify_goal_comparison(comparison: dict) -> str:
    """Apply the fixed preregistered decision rule."""
    n = int(comparison.get("n_valid_pairs") or 0)
    if not n:
        return "inconclusive"
    drop = -float(comparison.get("absolute_accuracy_change") or 0.0)
    baseline_better_p = float(comparison.get("baseline_better_exact_one_sided_p", 1.0))
    collapse_rule = PREREGISTRATION["lexical_shortcut_supported_if"]
    if (drop >= collapse_rule["minimum_absolute_accuracy_drop"]
            and baseline_better_p < collapse_rule["directional_exact_mcnemar_alpha"]):
        return "lexical_shortcut_supported"
    preserve_rule = PREREGISTRATION["lexical_explanation_weakened_if"]
    accuracy = float(comparison.get("candidate_accuracy") or 0.0)
    lower = float((comparison.get("candidate_wilson_95") or [0.0, 0.0])[0])
    if (accuracy >= preserve_rule["minimum_causal_opaque_accuracy"]
            and lower > 1.0 / len(ACTION_IDS)
            and baseline_better_p >= preserve_rule["directional_exact_mcnemar_alpha"]):
        return "semantic_selection_survives_lexical_control"
    return "inconclusive"


def _classify_perturbation_comparison(comparison: dict, preregistration: dict,
                                      fragility_key: str,
                                      robustness_key: str) -> str:
    n = int(comparison.get("n_valid_pairs") or 0)
    if not n:
        return "inconclusive"
    drop = -float(comparison.get("absolute_accuracy_change") or 0.0)
    baseline_better_p = float(
        comparison.get("baseline_better_exact_one_sided_p", 1.0))
    collapse_rule = preregistration[fragility_key]
    if (drop >= collapse_rule["minimum_absolute_accuracy_drop"]
            and baseline_better_p
            < collapse_rule["directional_exact_mcnemar_alpha"]):
        return fragility_key.removesuffix("_if")
    preserve_rule = preregistration[robustness_key]
    accuracy = float(comparison.get("candidate_accuracy") or 0.0)
    lower = float((comparison.get("candidate_wilson_95") or [0.0, 0.0])[0])
    if (accuracy >= preserve_rule["minimum_candidate_opaque_accuracy"]
            and lower > 1.0 / len(ACTION_IDS)
            and baseline_better_p
            >= preserve_rule["directional_exact_mcnemar_alpha"]):
        return robustness_key.removesuffix("_if")
    return "inconclusive"


def classify_structural_comparison(comparison: dict) -> str:
    """Apply the fixed preregistered structural-perturbation decision rule."""
    return _classify_perturbation_comparison(
        comparison,
        STRUCTURAL_PREREGISTRATION,
        "structural_fragility_supported_if",
        "structural_robustness_supported_if",
    )


def classify_synonym_comparison(comparison: dict) -> str:
    """Apply the fixed preregistered synonym-rewrite decision rule."""
    return _classify_perturbation_comparison(
        comparison,
        SYNONYM_PREREGISTRATION,
        "synonym_fragility_supported_if",
        "synonym_robustness_supported_if",
    )


def summarise(rows: list[dict], pairs: int) -> dict:
    by_arm = defaultdict(list)
    by_pair_mode = defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]].append(row)
        by_pair_mode[(row["pair"], row["mode"])][row["variant"]] = row

    arms = {}
    for arm in ARMS:
        values = by_arm[arm]
        valid = sum(bool(row.get("valid")) for row in values)
        correct = sum(bool(row.get("correct")) for row in values)
        n = len(values)
        accuracy = correct / n if n else 0.0
        se = math.sqrt(accuracy * (1 - accuracy) / n) if n else 0.0
        low, high = wilson_interval(correct, n)
        arms[arm] = {
            "n": n,
            "valid": valid,
            "correct": correct,
            "invalid": n - valid,
            "accuracy": accuracy,
            "standard_error": se,
            "wilson_95": [low, high],
            "chance_accuracy": 1.0 / len(ACTION_IDS),
            "chance_two_sided_p": binomial_two_sided(correct, n, 1.0 / len(ACTION_IDS)),
            "selected_actions": dict(Counter(row["action"] for row in values)),
        }

    modes = {}
    for mode in ("named", "opaque"):
        responsive = 0
        comparable = 0
        both_correct = 0
        neither_correct = 0
        a_only = 0
        b_only = 0
        patterns = Counter()
        for pair in range(pairs):
            left = by_pair_mode[(pair, mode)].get("a") or {}
            right = by_pair_mode[(pair, mode)].get("b") or {}
            if not left.get("valid") or not right.get("valid"):
                patterns["invalid"] += 1
                continue
            comparable += 1
            changed = left["action"] != right["action"]
            responsive += changed
            patterns[f"{left['action']}->{right['action']}"] += 1
            lc, rc = bool(left.get("correct")), bool(right.get("correct"))
            if lc and rc:
                both_correct += 1
            elif lc:
                a_only += 1
            elif rc:
                b_only += 1
            else:
                neither_correct += 1
        low, high = wilson_interval(responsive, comparable)
        modes[mode] = {
            "pairs": pairs,
            "comparable_pairs": comparable,
            "goal_changed_action": responsive,
            "goal_changed_action_share": responsive / comparable if comparable else 0.0,
            "goal_changed_action_wilson_95": [low, high],
            "both_correct": both_correct,
            "both_correct_share": both_correct / comparable if comparable else 0.0,
            "a_correct_only": a_only,
            "b_correct_only": b_only,
            "neither_correct": neither_correct,
            "transition_counts": dict(patterns),
        }

    paired = Counter()
    for pair in range(pairs):
        combined = {}
        for mode in ("named", "opaque"):
            for variant in ("a", "b"):
                row = by_pair_mode[(pair, mode)].get(variant) or {}
                combined[(mode, variant)] = bool(row.get("correct"))
        for variant in ("a", "b"):
            n = combined[("named", variant)]
            o = combined[("opaque", variant)]
            if n and not o:
                paired["named_only"] += 1
            elif o and not n:
                paired["opaque_only"] += 1
            elif n and o:
                paired["both"] += 1
            else:
                paired["neither"] += 1
    discordant = paired["named_only"] + paired["opaque_only"]
    mcnemar_p = binomial_two_sided(paired["named_only"], discordant, 0.5)
    named_accuracy = sum(arms[a]["correct"] for a in ("named_a", "named_b"))
    opaque_accuracy = sum(arms[a]["correct"] for a in ("opaque_a", "opaque_b"))
    n_prompts = pairs * 2
    return {
        "n_prompts_per_mode": n_prompts,
        "chance_accuracy": 1.0 / len(ACTION_IDS),
        "arms": arms,
        "goal_sensitivity": modes,
        "named_vs_opaque": {
            "named_correct": named_accuracy,
            "opaque_correct": opaque_accuracy,
            "named_accuracy": named_accuracy / n_prompts if n_prompts else 0.0,
            "opaque_accuracy": opaque_accuracy / n_prompts if n_prompts else 0.0,
            "discordant": dict(paired),
            "mcnemar_exact_two_sided_p": mcnemar_p,
        },
    }


def audit_artifact(data: dict) -> dict:
    """Recompute the artifact's prompts, selections, and paired summary."""
    if data.get("schema") in (
            INTERLEAVED_SCHEMA,
            STRUCTURAL_INTERLEAVED_SCHEMA,
            SYNONYM_INTERLEAVED_SCHEMA):
        return audit_interleaved_artifact(data)
    pairs = int(data.get("pairs") or 0)
    rows = data.get("rows") or []
    prompt_rows = data.get("prompts") or []
    prompts_by_pair = {}
    for prompt in prompt_rows:
        pair = int(prompt.get("pair", -1))
        prompts_by_pair[pair] = prompt

    grouped = defaultdict(dict)
    duplicate_pair_arms = False
    for row in rows:
        key = (int(row.get("pair", -1)), str(row.get("arm") or ""))
        if key[1] in grouped[key[0]]:
            duplicate_pair_arms = True
        grouped[key[0]][key[1]] = row

    pair_complete = set(grouped) == set(range(pairs)) and all(
        set(grouped[pair]) == set(ARMS) for pair in range(pairs))
    seed_consistent = pair_complete and all(
        len({row.get("seed") for row in grouped[pair].values()}) == 1
        for pair in range(pairs))
    positions_consistent = pair_complete and all(
        all(grouped[pair][arm].get("position") == position
            for position, arm in enumerate(ARM_ORDERS[pair % len(ARM_ORDERS)]))
        for pair in range(pairs))

    prompt_hashes_exact = (
        len(prompts_by_pair) == pairs
        and set(prompts_by_pair) == set(range(pairs))
        and data.get("system_sha256") == _sha256(data.get("system", ""))
        and all(
            set(prompt.get("text") or {}) == set(ARMS)
            and set(prompt.get("sha256") or {}) == set(ARMS)
            and set(prompt.get("tokens") or {}) == set(ARMS)
            and all(_sha256(prompt["text"][arm]) == prompt["sha256"][arm]
                    for arm in ARMS)
            for prompt in prompts_by_pair.values()
        )
    )
    prompts_token_matched = (
        len(prompts_by_pair) == pairs
        and all(
            all(isinstance(prompt["tokens"].get(arm), int) for arm in ARMS)
            and len(set(prompt["tokens"].values())) == 1
            for prompt in prompts_by_pair.values()
        )
    )

    selection_audit_exact = pair_complete
    prompt_metadata_exact = pair_complete
    if pair_complete:
        for pair in range(pairs):
            seed = int(grouped[pair][ARMS[0]].get("seed"))
            prompt = prompts_by_pair[pair]
            targets = pair_targets(pair)
            label_maps = {
                mode: {
                    label: action for action, label in
                    menu_for_pair(pair, seed, opaque=(mode == "opaque"))[1].items()
                }
                for mode in ("named", "opaque")
            }
            for arm, row in grouped[pair].items():
                mode, variant = arm.split("_", 1)
                target = targets[0 if variant == "a" else 1]
                action, label, _ = parse_selection(
                    str(row.get("raw") or ""), label_maps[mode])
                if (row.get("target") != target
                        or row.get("action") != action
                        or row.get("label") != label
                        or bool(row.get("valid")) != (action != "(invalid)")
                        or bool(row.get("correct")) != (action == target)):
                    selection_audit_exact = False
                if (row.get("prompt_sha256") != prompt["sha256"].get(arm)
                        or row.get("prompt_tokens") != prompt["tokens"].get(arm)
                        or row.get("prompt_chars")
                        != len(prompt["text"].get(arm, ""))):
                    prompt_metadata_exact = False

    summary_exact = pair_complete and not duplicate_pair_arms and (
        summarise(rows, pairs) == data.get("summary"))
    checks = {
        "row_count_exact": len(rows) == pairs * len(ARMS),
        "pair_complete": pair_complete,
        "no_duplicate_pair_arms": not duplicate_pair_arms,
        "seed_consistent_within_pair": seed_consistent,
        "positions_consistent_with_arm_order": positions_consistent,
        "prompt_hashes_exact": prompt_hashes_exact,
        "prompts_token_matched": prompts_token_matched,
        "prompt_metadata_matches_rows": prompt_metadata_exact,
        "raw_selections_recompute_exactly": selection_audit_exact,
        "summary_exact": summary_exact,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "pairs": pairs,
        "rows": len(rows),
        "summary": summarise(rows, pairs) if pairs else {},
    }


def audit_interleaved_artifact(data: dict) -> dict:
    """Audit both goal variants and the exact within-process call order."""
    pairs = int(data.get("pairs") or 0)
    variants = data.get("variants") or {}
    preregistration = data.get("preregistration") or {}
    control = str(preregistration.get("control_goal_variant") or "literal")
    candidate = str(preregistration.get("candidate_goal_variant") or "causal")
    variant_names = (control, candidate)
    checks = {
        "variant_names_exact": set(variants) == set(variant_names),
    }
    variant_audits = {}
    for variant in variant_names:
        audit = audit_artifact(variants.get(variant) or {})
        variant_audits[variant] = audit
        checks[f"{variant}_artifact_valid"] = audit["ok"]

    expected_schedule = [
        {
            "sample_index": sample_index,
            "pair": pair,
            "goal_variant": variant,
            "arm": arm,
        }
        for sample_index, (pair, variant, arm) in enumerate(
            interleaved_schedule(pairs, control, candidate))
    ]
    checks["schedule_exact"] = data.get("schedule") == expected_schedule

    rows = []
    for variant in variant_names:
        rows.extend((variants.get(variant) or {}).get("rows") or [])
    by_sample_index = {
        int(row.get("sample_index", -1)): row for row in rows
    }
    sample_indexes = {int(row.get("sample_index", -1)) for row in rows}
    checks["sample_indexes_exact"] = sample_indexes == set(
        range(len(expected_schedule)))
    checks["schedule_rows_exact"] = len(by_sample_index) == len(rows) and all(
        (by_sample_index.get(item["sample_index"]) or {}).get("pair")
        == item["pair"]
        and (by_sample_index[item["sample_index"]]).get("goal_variant")
        == item["goal_variant"]
        and by_sample_index[item["sample_index"]].get("arm") == item["arm"]
        for item in expected_schedule
    )

    try:
        expected_comparison = compare_artifacts(
            variants.get(control) or {}, variants.get(candidate) or {},
            preregistration=preregistration)
    except ValueError:
        expected_comparison = None
    checks["comparison_exact"] = (
        expected_comparison is not None
        and data.get("comparison") == expected_comparison)

    return {
        "ok": all(checks.values()),
        "checks": checks,
        "pairs": pairs,
        "rows": len(rows),
        "summary": (
            (variants.get(candidate) or {}).get("summary") or {}),
    }


def compare_artifacts(baseline: dict, candidate: dict,
                      preregistration: dict | None = None) -> dict:
    """Audit and compare two goal variants under a fixed decision rule."""
    preregistration = preregistration or PREREGISTRATION
    baseline_audit = audit_artifact(baseline)
    candidate_audit = audit_artifact(candidate)
    baseline_variant = baseline.get("goal_variant", "literal")
    candidate_variant = candidate.get("goal_variant", "literal")
    if not baseline_audit["ok"] or not candidate_audit["ok"]:
        raise ValueError("cannot compare an artifact that failed audit")
    expected_pair = (
        preregistration["control_goal_variant"],
        preregistration["candidate_goal_variant"],
    )
    if (baseline_variant, candidate_variant) != expected_pair:
        raise ValueError(
            f"expected {expected_pair[0]} vs {expected_pair[1]} artifacts, "
            f"got {baseline_variant} vs {candidate_variant}")
    if baseline.get("pairs") != candidate.get("pairs"):
        raise ValueError("artifact pair counts differ")

    baseline_meta = {
        int(row["pair"]): (str(row["target"]), int(row["seed"]))
        for row in baseline.get("rows") or []
        if str(row.get("arm")) == "opaque_a"
    }
    candidate_meta = {
        int(row["pair"]): (str(row["target"]), int(row["seed"]))
        for row in candidate.get("rows") or []
        if str(row.get("arm")) == "opaque_a"
    }
    if baseline_meta != candidate_meta:
        raise ValueError("pair targets or seeds differ between artifacts")

    comparison = paired_binary_comparison(
        baseline.get("rows") or [], candidate.get("rows") or [],
        tuple(preregistration["primary_arms"]))
    if candidate_variant == "structural":
        classifier = classify_structural_comparison
    elif candidate_variant == "synonym":
        classifier = classify_synonym_comparison
    else:
        classifier = classify_goal_comparison
    return {
        "baseline_audit_ok": baseline_audit["ok"],
        "candidate_audit_ok": candidate_audit["ok"],
        "goal_variant_pair": [baseline_variant, candidate_variant],
        "preregistration": preregistration,
        "comparison": comparison,
        "outcome": classifier(comparison),
    }


def _prepare_run(args, variants: tuple[str, ...]):
    api_key = ""
    if args.api_key_file:
        api_key = Path(args.api_key_file).expanduser().read_text(encoding="utf-8").strip()
    tokenize_url = args.tokenize_url or _default_tokenize_url(args.base_url)
    tokenize = lambda text: tokenize_text(text, tokenize_url, api_key)
    trials_by_variant = {
        variant: [
            build_trial_prompts(
                index, args.seed_base + index, tokenize,
                goal_variant=variant)
            for index in range(args.pairs)
        ]
        for variant in variants
    }
    if len(variants) > 1:
        for index in range(args.pairs):
            target_tokens = max(
                trial["target_tokens"]
                for trial in (
                    trials_by_variant[variant][index] for variant in variants))
            for variant in variants:
                trials_by_variant[variant][index] = build_trial_prompts(
                    index, args.seed_base + index, tokenize,
                    goal_variant=variant, target_tokens=target_tokens)
    proposer = OpenAIProposer(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    for variant in variants:
        trials = trials_by_variant[variant]
        if trials:
            for arm in ARMS:
                sample_once(proposer, trials[0]["texts"][arm],
                            args.seed_base - 1, args.temperature,
                            args.max_tokens)
    return trials_by_variant, proposer


def _sample_row(proposer, trial: dict, arm: str, position: int,
                args, sample_index: int) -> dict:
    mode, variant = arm.split("_", 1)
    text, usage, latency = sample_once(
        proposer, trial["texts"][arm], trial["seed"],
        args.temperature, args.max_tokens)
    label_map = trial["label_to_action"][mode]
    action, label, obj = parse_selection(text, label_map)
    target = trial[f"target_{variant}"]
    return {
        "pair": trial["pair"],
        "arm": arm,
        "mode": mode,
        "variant": variant,
        "position": position,
        "seed": trial["seed"],
        "goal_variant": trial["goal_variant"],
        "sample_index": sample_index,
        "target": target,
        "action": action,
        "label": label,
        "valid": action != "(invalid)",
        "correct": action == target,
        "reason": str((obj or {}).get("reason") or ""),
        "raw": text,
        "usage": usage,
        "latency_ms": latency,
        "prompt_sha256": trial["hashes"][arm],
        "prompt_chars": trial["chars"][arm],
        "prompt_tokens": trial["tokens"][arm],
    }


def artifact_for_run(args, trials: list[dict], rows: list[dict],
                     goal_variant: str) -> dict:
    summary = summarise(rows, args.pairs)
    prompt_checks = {}
    for trial in trials:
        pair = trial["pair"]
        tokens = trial["tokens"]
        prompt_checks[str(pair)] = {
            "tokens": tokens,
            "exact": len(set(tokens.values())) == 1,
            "target_a": trial["target_a"],
            "target_b": trial["target_b"],
            "pads": trial["pads"],
            "hashes": trial["hashes"],
        }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "pairs": args.pairs,
        "seed_base": args.seed_base,
        "goal_variant": goal_variant,
        "goal_texts": GOAL_VARIANTS[goal_variant],
        "causal_goal_lexical_audit": lexical_audit(),
        "structural_goal_lexical_audit": lexical_audit(STRUCTURAL_GOALS),
        "synonym_goal_lexical_audit": lexical_audit(SYNONYM_GOALS),
        "preregistration": (
            PREREGISTRATION if goal_variant == "causal"
            else SYNONYM_PREREGISTRATION if goal_variant == "synonym"
            else STRUCTURAL_PREREGISTRATION if goal_variant == "structural"
            else None),
        "system": SYSTEM_PROMPT,
        "system_sha256": _sha256(SYSTEM_PROMPT),
        "actions": [
            {
                "action": action,
                "description": description,
                "literal_goal": LITERAL_GOALS[action],
                "causal_goal": CAUSAL_GOALS[action],
                "structural_goal": STRUCTURAL_GOALS[action],
                "synonym_goal": SYNONYM_GOALS[action],
            }
            for action, description, _ in ACTIONS
        ],
        "prompt_checks": prompt_checks,
        "prompts": [
            {
                "pair": trial["pair"],
                "target_a": trial["target_a"],
                "target_b": trial["target_b"],
                "text": trial["texts"],
                "sha256": trial["hashes"],
                "tokens": trial["tokens"],
            }
            for trial in trials
        ],
        "rows": rows,
        "summary": summary,
        "artifact_checks": {
            "all_prompts_token_matched": all(
                value["exact"] for value in prompt_checks.values()),
            "row_count_exact": len(rows) == args.pairs * len(ARMS),
            "all_pairs_have_four_arms": all(
                sum(row["pair"] == pair for row in rows) == len(ARMS)
                for pair in range(args.pairs)),
            "all_prompts_have_hashes": all(
                row["prompt_sha256"] for row in rows),
            "causal_goals_have_no_description_cues": lexical_audit()["ok"],
            "structural_goals_have_no_description_cues":
                lexical_audit(STRUCTURAL_GOALS)["ok"],
            "synonym_goals_have_no_description_cues":
                lexical_audit(SYNONYM_GOALS)["ok"],
        },
    }


def run(args) -> dict:
    trials_by_variant, proposer = _prepare_run(args, (args.goal_variant,))
    trials = trials_by_variant[args.goal_variant]

    rows = []
    sample_index = 0
    for trial in trials:
        index = trial["pair"]
        order = ARM_ORDERS[index % len(ARM_ORDERS)]
        for position, arm in enumerate(order):
            rows.append(_sample_row(
                proposer, trial, arm, position, args, sample_index))
            sample_index += 1
    return artifact_for_run(args, trials, rows, args.goal_variant)


def run_interleaved(args, variants: tuple[str, str] = ("literal", "causal"),
                    preregistration: dict | None = None) -> dict:
    preregistration = preregistration or PREREGISTRATION
    trials_by_variant, proposer = _prepare_run(args, variants)
    trials_by_pair = {
        variant: {trial["pair"]: trial for trial in trials}
        for variant, trials in trials_by_variant.items()
    }
    schedule = interleaved_schedule(args.pairs, *variants)
    rows_by_variant = {variant: [] for variant in variants}
    stored_schedule = []
    for sample_index, (pair, variant, arm) in enumerate(schedule):
        trial = trials_by_pair[variant][pair]
        position = ARM_ORDERS[pair % len(ARM_ORDERS)].index(arm)
        rows_by_variant[variant].append(_sample_row(
            proposer, trial, arm, position, args, sample_index))
        stored_schedule.append({
            "sample_index": sample_index,
            "pair": pair,
            "goal_variant": variant,
            "arm": arm,
        })

    artifacts = {
        variant: artifact_for_run(
            args, trials_by_variant[variant], rows_by_variant[variant], variant)
        for variant in variants
    }
    comparison = compare_artifacts(
        artifacts[variants[0]], artifacts[variants[1]],
        preregistration=preregistration)
    schema = (
        STRUCTURAL_INTERLEAVED_SCHEMA
        if variants == (
            STRUCTURAL_PREREGISTRATION["control_goal_variant"],
            STRUCTURAL_PREREGISTRATION["candidate_goal_variant"],
        )
        else SYNONYM_INTERLEAVED_SCHEMA
        if variants == (
            SYNONYM_PREREGISTRATION["control_goal_variant"],
            SYNONYM_PREREGISTRATION["candidate_goal_variant"],
        )
        else INTERLEAVED_SCHEMA
    )
    return {
        "schema": schema,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "pairs": args.pairs,
        "seed_base": args.seed_base,
        "preregistration": preregistration,
        "schedule": stored_schedule,
        "variants": artifacts,
        "comparison": comparison,
    }


def print_report(result: dict) -> None:
    summary = result["summary"]
    print(f"pairs={result['pairs']} prompts/mode={summary['n_prompts_per_mode']} "
          f"chance={summary['chance_accuracy']:.3f} "
          f"artifact_checks={result['artifact_checks']}")
    print("\naccuracy")
    for arm in ARMS:
        stats = summary["arms"][arm]
        print(f"{arm:<10} {stats['correct']}/{stats['n']} "
              f"acc={stats['accuracy']:.3f} se={stats['standard_error']:.3f} "
              f"wilson95=[{stats['wilson_95'][0]:.3f},{stats['wilson_95'][1]:.3f}] "
              f"p={stats['chance_two_sided_p']:.4g}")
    print("\ngoal sensitivity")
    for mode in ("named", "opaque"):
        stats = summary["goal_sensitivity"][mode]
        print(f"{mode:<7} changed={stats['goal_changed_action']}/"
              f"{stats['comparable_pairs']} both_correct={stats['both_correct']} "
              f"neither={stats['neither_correct']}")
    comparison = summary["named_vs_opaque"]
    print("\nnamed vs opaque")
    print(f"named={comparison['named_correct']} "
          f"opaque={comparison['opaque_correct']} "
          f"discordant={comparison['discordant']} "
          f"mcnemar_p={comparison['mcnemar_exact_two_sided_p']:.4g}")


def print_audit(result: dict, path: Path) -> None:
    print(f"artifact={path} ok={result['ok']} rows={result['rows']}")
    for name, passed in result["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    comparison = result["summary"].get("named_vs_opaque") or {}
    if comparison:
        print(f"named={comparison['named_correct']} "
              f"opaque={comparison['opaque_correct']} "
              f"mcnemar_p={comparison['mcnemar_exact_two_sided_p']:.4g}")


def print_goal_comparison(result: dict) -> None:
    comparison = result["comparison"]
    stats = comparison["comparison"]
    baseline, candidate = comparison["goal_variant_pair"]
    print(f"\n{baseline} vs {candidate} goals (opaque)")
    print(f"{baseline}={stats['baseline_correct']}/{stats['n_valid_pairs']} "
          f"{candidate}={stats['candidate_correct']}/{stats['n_valid_pairs']} "
          f"delta={stats['absolute_accuracy_change']:+.3f} "
          f"candidate_wilson95=[{stats['candidate_wilson_95'][0]:.3f},"
          f"{stats['candidate_wilson_95'][1]:.3f}] "
          f"one_sided_p={stats['baseline_better_exact_one_sided_p']:.4g}")
    print(f"discordant={stats['discordant']} outcome={comparison['outcome']}")


def print_interleaved_report(result: dict) -> None:
    preregistration = result["preregistration"]
    control = preregistration["control_goal_variant"]
    candidate = preregistration["candidate_goal_variant"]
    print(f"interleaved {control} vs {candidate}: pairs={result['pairs']} "
          f"calls={len(result['schedule'])}")
    for variant in (control, candidate):
        print(f"\n[{variant}]")
        print_report(result["variants"][variant])
    print_goal_comparison(result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--tokenize-url", default="")
    parser.add_argument("--model", default="qwen3-8b-q5km-instruct")
    parser.add_argument("--api-key-file", default="~/.pandosi/llm-api-8b-q6.token")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=3670)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--goal-variant", choices=tuple(GOAL_VARIANTS),
                        default="literal")
    parser.add_argument("--interleave-goal-variants", action="store_true",
                        help="sample literal and causal goals adjacently in one process")
    parser.add_argument("--interleave-structural-perturbation", action="store_true",
                        help="sample causal and structural goals adjacently in one process")
    parser.add_argument("--interleave-synonym-paraphrase", action="store_true",
                        help="sample causal and synonym goals adjacently in one process")
    parser.add_argument("--output", default="")
    parser.add_argument("--audit-artifact", default="")
    parser.add_argument("--compare-artifact", default="")
    args = parser.parse_args(argv)
    if args.audit_artifact:
        path = Path(args.audit_artifact).expanduser()
        audit = audit_artifact(json.loads(path.read_text(encoding="utf-8")))
        print_audit(audit, path)
        return 0 if audit["ok"] else 1
    interleaved_flags = (
        args.interleave_goal_variants,
        args.interleave_structural_perturbation,
        args.interleave_synonym_paraphrase,
    )
    interleaved_mode = any(interleaved_flags)
    if sum(interleaved_flags) > 1:
        parser.error("choose only one interleaved comparison")
    if interleaved_mode:
        if args.goal_variant != "causal":
            parser.error("interleaved modes require --goal-variant causal")
        if args.compare_artifact:
            parser.error("interleaved modes perform their own comparison")
        if args.interleave_structural_perturbation:
            variants = (
                STRUCTURAL_PREREGISTRATION["control_goal_variant"],
                STRUCTURAL_PREREGISTRATION["candidate_goal_variant"],
            )
            result = run_interleaved(
                args, variants, STRUCTURAL_PREREGISTRATION)
        elif args.interleave_synonym_paraphrase:
            variants = (
                SYNONYM_PREREGISTRATION["control_goal_variant"],
                SYNONYM_PREREGISTRATION["candidate_goal_variant"],
            )
            result = run_interleaved(
                args, variants, SYNONYM_PREREGISTRATION)
        else:
            variants = ("literal", "causal")
            result = run_interleaved(args, variants, PREREGISTRATION)
        print_interleaved_report(result)
        comparison_ok = result["comparison"]["outcome"] != "inconclusive"
        artifact_ok = all(
            all(artifact["artifact_checks"].values())
            for artifact in result["variants"].values())
    else:
        result = run(args)
        print_report(result)
        comparison_ok = True
        artifact_ok = all(result["artifact_checks"].values())
    if not interleaved_mode and args.compare_artifact:
        baseline_path = Path(args.compare_artifact).expanduser()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        try:
            result["comparison"] = compare_artifacts(baseline, result)
        except ValueError as exc:
            parser.error(str(exc))
        result["comparison"]["baseline_artifact"] = str(baseline_path)
        print_goal_comparison(result)
        comparison_ok = result["comparison"]["outcome"] != "inconclusive"
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"\nwrote {path}")
    return 0 if artifact_ok and comparison_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
