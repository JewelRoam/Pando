"""PandoRSI smoke tests.

Policy (project decision): minimal mocks, minimal test count. Static analysis
and research are the default; this file runs only at milestones or on demand.
Two cases cover the contract: quorum gating semantics, and one real-HTTP
end-to-end vote including node death and checkpoint-diverse quorum picking.
"""

import argparse
import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import plistlib
import re
import shutil
import sys
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import _git_env, _git_in  # noqa: E402, F401

from pando.backend import (MockBackend, OpenAIChoiceBackend,
                            parse_openai_choice)
from pando.common import (DecisionRequest, Vote, aggregate,
                            canonical_question, mcnemar_test)
from pando.coordinator import (Coordinator, INITIAL_STEM_TIMEOUT,
                            Root, load_accuracy_registry)
from pando.worker import Stem
from pando.onboard import build_plan
from pando import graft
from pando.grants import (audit_status, grant_status, require_grant,
                            save_grant)
from pando.service import build_service_plan, detect_platform


class TestMemoryAblationCheck(unittest.TestCase):
    """tools/memory_ablation_check.py - the missing memory control.

    The experiment must separate full history, last-action feedback, action
    counts, and prompt length. Group proposals and outcome activity are frozen
    empty in every arm, and all arms keep the same round number instead of
    claiming "first round" at round 3500.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "memory_ablation_check.py"
        spec = importlib.util.spec_from_file_location("memory_ablation_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_single_factor_arms_isolate_history_counts_and_last_action(self):
        from pando.proposer import Ledger
        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "think.jsonl")
            from pando.proposer import Round, Proposal
            for number, action in ((1, "adaptive.governance"),
                                   (2, "dynamic.threshold.adjustment")):
                row = Round(number, number, proposal=Proposal(
                    f"p{number}", "mac", f"在pi5主机执行目标{number}", action),
                    published=True)
                ledger.append(row)
            prompts = tool.build_prompts(
                ledger, 3, 8, ["pi5"], {"adaptive.governance",
                                         "dynamic.threshold.adjustment"})
        memory = prompts["memory"]
        last_action = prompts["last_action"]
        counts = prompts["counts"]
        neutral = prompts["neutral"]
        pseudo_history = prompts["pseudo_history"]
        coherent_fiction = prompts["coherent_fiction"]
        control = prompts["control"]
        self.assertIn("dynamic.threshold.adjustment", memory)
        self.assertIn("你已经提过的动作次数", memory)
        self.assertIn("上一轮你提的是 dynamic.threshold.adjustment", memory)
        self.assertIn("上一轮你提的是 dynamic.threshold.adjustment", last_action)
        self.assertNotIn("你已经提过的动作次数", last_action)
        self.assertNotIn("目标「在pi5主机执行目标1」", last_action)
        self.assertIn("你已经提过的动作次数", counts)
        self.assertNotIn("上一轮你提的是", counts)
        self.assertNotIn("目标「在pi5主机执行目标1」", counts)
        self.assertNotIn("dynamic.threshold.adjustment", control)
        self.assertNotIn("你已经提过的动作次数", control)
        self.assertNotIn("上一轮你提的是", control)
        self.assertIn("这是你的第 3 轮思考", control)
        self.assertIn("固定为空：实验控制", control)
        self.assertIn("pi5", control)
        self.assertEqual(len(neutral), len(memory))
        self.assertIn("【中性长度填充】", neutral)
        self.assertIn("neutral padding only", neutral)
        self.assertNotIn("目标「在pi5主机执行目标1」", neutral)
        self.assertNotIn("dynamic.threshold.adjustment", neutral)
        self.assertNotIn("你已经提过的动作次数", neutral)
        self.assertNotIn("上一轮你提的是", neutral)
        self.assertEqual(len(pseudo_history), len(memory))
        self.assertNotIn("r1:", pseudo_history)
        self.assertNotIn("r2:", pseudo_history)
        self.assertNotIn("目标「在pi5主机执行目标1」", pseudo_history)
        self.assertNotIn("已发布", pseudo_history)
        self.assertIn("未发布", pseudo_history)
        self.assertIn("adaptive.governance", pseudo_history)
        self.assertIn("dynamic.threshold.adjustment", pseudo_history)
        self.assertIn("【你已经提过的动作次数】", pseudo_history)
        self.assertIn("上一轮你提的是 dynamic.threshold.adjustment", pseudo_history)
        self.assertEqual(len(coherent_fiction), len(memory))
        self.assertNotIn("目标「在pi5主机执行目标1」", coherent_fiction)
        self.assertIn("目标「在nuc主机执行目标1」", coherent_fiction)
        self.assertNotIn("r1000:", coherent_fiction)
        self.assertNotIn("内容不可用", coherent_fiction)
        self.assertNotIn("不存在的目标", coherent_fiction)
        self.assertIn("已发布", coherent_fiction)
        self.assertNotIn("未发布", coherent_fiction)
        self.assertIn("adaptive.governance", coherent_fiction)
        self.assertIn("dynamic.threshold.adjustment", coherent_fiction)
        self.assertIn("【你已经提过的动作次数】", coherent_fiction)
        self.assertIn("上一轮你提的是 dynamic.threshold.adjustment",
                      coherent_fiction)

    def test_arm_orders_balance_each_arm_and_position(self):
        tool = self._tool()
        self.assertEqual(len(tool.ARM_ORDERS), len(tool.ARMS))
        for order in tool.ARM_ORDERS:
            self.assertEqual(set(order), set(tool.ARMS))
        for position in range(len(tool.ARMS)):
            self.assertEqual(
                sorted(order[position] for order in tool.ARM_ORDERS),
                sorted(tool.ARMS),
            )

    def test_goal_counterfactual_changes_polarity_not_layout(self):
        from pando.proposer import Ledger, Proposal, Round
        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "think.jsonl")
            goals = (
                "在pi5主机上部署自适应治理协议以减少人工干预并"
                "提升跨主机策略一致性",
                "在mac主机上部署动态阈值调整以减少重复提案带来的资源浪费",
            )
            actions = ("adaptive.governance", "dynamic.threshold.adjustment")
            for number, (goal, action) in enumerate(zip(goals, actions), 1):
                ledger.append(Round(
                    number, number,
                    proposal=Proposal(
                        f"p{number}", "mac", goal, action),
                    published=True))
            prompts = tool.build_prompts(
                ledger, 3, 8, ["pi5", "mac"],
                set(actions))
        memory = prompts["memory"]
        counterfactual = prompts["goal_counterfactual"]
        self.assertEqual(len(counterfactual), len(memory))
        self.assertIn("增加人工干预", counterfactual)
        self.assertIn("降低跨主机策略一致性", counterfactual)
        self.assertIn("增加重复提案带来的资源浪费", counterfactual)
        self.assertNotIn("减少人工干预", counterfactual)
        self.assertIn("pi5", counterfactual)
        self.assertIn("mac", counterfactual)
        self.assertIn("adaptive.governance", counterfactual)
        self.assertIn("dynamic.threshold.adjustment", counterfactual)
        self.assertNotEqual(counterfactual, memory)

    def test_holm_adjustment_is_monotone_in_rank_order(self):
        tool = self._tool()
        self.assertEqual(tool.holm_adjust([0.01, 0.04, 0.03]),
                         [0.03, 0.06, 0.06])

    def test_prompt_artifact_contains_replayable_system_and_arm_text(self):
        from pando.proposer import Ledger, Proposal, Round
        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "think.jsonl")
            for number in (1, 2):
                ledger.append(Round(
                    number, number,
                    proposal=Proposal(
                        f"p{number}", "mac", "goal", "seed.export"),
                    published=True))
            prompts = tool.build_prompts(
                ledger, 3, 8, [], {"seed.export", "seed.replicate"})
        artifact = tool.prompt_artifact(prompts)
        self.assertEqual(artifact["system"], tool.SYSTEM_PROMPT)
        self.assertEqual(artifact["text"], {
            arm: prompts[arm] for arm in tool.ARMS
        })
        self.assertEqual(
            set(artifact["sha256"]), set(tool.ARMS))
        self.assertEqual(
            artifact["system_sha256"],
            hashlib.sha256(tool.SYSTEM_PROMPT.encode("utf-8")).hexdigest())

    def test_tvd_and_invalid_actions_are_counted(self):
        from collections import Counter
        tool = self._tool()
        self.assertEqual(tool.total_variation(
            Counter({"a": 3}), Counter({"a": 3})), 0.0)
        self.assertEqual(tool.total_variation(
            Counter({"a": 3}), Counter({"b": 3})), 1.0)
        self.assertEqual(tool.parse_action(
            '{"action":"not.in.space"}', {"seed.export"}), tool.INVALID)
        self.assertEqual(tool.parse_action(
            'noise {"action":"seed.export"} tail', {"seed.export"}),
            "seed.export")

    def test_identical_arms_do_not_manufacture_a_permutation_signal(self):
        tool = self._tool()
        pairs = [{arm: "seed.export" for arm in tool.ARMS}] * 20
        summary = tool.summarise(
            pairs, ["seed.export"], "seed.export", trials=199)
        self.assertTrue(all(
            row["observed"] == 0.0
            and row["paired_permutation_p"] > 0.9
            and row["holm_adjusted_p"] > 0.9
            for row in summary["primary"]["comparisons"]))
        self.assertTrue(all(
            summary["arms"][arm]["same_as_last"]["count"] == 20
            for arm in tool.ARMS))

    def test_response_text_summary_keeps_action_and_goal_effects_distinct(self):
        tool = self._tool()
        rows = []
        for pair in range(2):
            for arm in tool.ARMS:
                action = "adaptive.governance"
                goal = "在pi5部署治理以减少人工干预"
                effect = "same-effect"
                if arm == "goal_counterfactual":
                    action = "adaptive.governance" if pair == 0 else "seed.export"
                    goal = ("在pi5部署治理以增加人工干预" if pair == 0
                            else "在pi5部署治理以增强一致性")
                    effect = "same-effect" if pair == 0 else "other-effect"
                elif arm == "control":
                    action = "seed.export"
                    goal = "复制成员"
                rows.append({
                    "pair": pair,
                    "arm": arm,
                    "action": action,
                    "raw": json.dumps({
                        "action": action,
                        "goal": goal,
                        "expected_effect": effect,
                    }, ensure_ascii=False),
                })
        summary = tool.response_text_summary(rows)
        comparison = summary["comparisons"][1]
        self.assertEqual(comparison["label"],
                         "full_memory_vs_goal_counterfactual")
        agreement = comparison["field_agreement"]
        self.assertEqual(agreement["action"]["equal"], 1)
        self.assertEqual(agreement["goal"]["equal"], 0)
        self.assertEqual(agreement["expected_effect"]["equal"], 1)
        self.assertEqual(agreement["reason"]["total"], 0)
        self.assertEqual(agreement["reason"]["both_missing"], 2)
        polarity = summary["goal_polarity"]
        self.assertEqual(polarity["memory"]["source_any"], 2)
        self.assertEqual(polarity["memory"]["target_any"], 0)
        self.assertEqual(polarity["goal_counterfactual"]["source_any"], 0)
        self.assertEqual(polarity["goal_counterfactual"]["target_any"], 1)
        self.assertEqual(polarity["goal_counterfactual"]["neither_any"], 1)

    def test_artifact_audit_recomputes_summary_and_rejects_tampered_prompt(self):
        tool = self._tool()
        n_pairs = 2
        rows = []
        pairs = []
        for pair in range(n_pairs):
            got = {}
            for position, arm in enumerate(tool.ARM_ORDERS[pair]):
                action = "seed.export"
                got[arm] = action
                rows.append({
                    "pair": pair,
                    "arm": arm,
                    "position": position,
                    "seed": 100 + pair,
                    "action": action,
                    "raw": json.dumps({"action": action, "goal": arm}),
                })
            pairs.append(got)
        summary = tool.summarise(
            pairs, ["seed.export"], "seed.export", trials=19)
        prompt_texts = {arm: f"prompt-{arm}" for arm in tool.ARMS}
        artifact = {
            "config": {
                "n_pairs": n_pairs,
                "arm_orders": [list(order) for order in tool.ARM_ORDERS],
            },
            "prompts": tool.prompt_artifact(
                prompt_texts,
                {arm: 10 for arm in tool.ARMS}),
            "summary": summary,
            "rows": rows,
        }
        audit = tool.audit_artifact(artifact)
        self.assertTrue(audit["ok"], audit["checks"])
        tokenizer = artifact["prompts"].pop("tokenizer")
        audit = tool.audit_artifact(artifact)
        self.assertFalse(audit["checks"]["tokenizer_exact"])
        artifact["prompts"]["tokenizer"] = tokenizer
        original_raw = rows[0]["raw"]
        rows[0]["raw"] = json.dumps({"action": "wrong", "goal": "tampered"})
        audit = tool.audit_artifact(artifact)
        self.assertFalse(audit["checks"]["raw_actions_match_stored_actions"])
        rows[0]["raw"] = original_raw
        artifact["prompts"]["text"]["memory"] += "tampered"
        audit = tool.audit_artifact(artifact)
        self.assertFalse(audit["checks"]["prompt_hashes_exact"])
        self.assertFalse(audit["ok"])


class TestActionSemanticsCheck(unittest.TestCase):
    """tools/action_semantics_check.py - label semantics vs goal selection.

    The eight real action names are themselves strong hints. The opaque menu
    keeps the descriptions and goal fixed while replacing those hints with
    arbitrary identifiers, so a named-only win is evidence for the shortcut
    rather than evidence that the goal was ignored.
    """

    def _tool(self):
        import importlib.util
        path = (Path(__file__).resolve().parents[1]
                / "tools" / "action_semantics_check.py")
        spec = importlib.util.spec_from_file_location(
            "action_semantics_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_pair_targets_are_distinct_and_cover_the_action_space(self):
        from collections import Counter
        tool = self._tool()
        pairs = [tool.pair_targets(index) for index in range(32)]
        self.assertTrue(all(left != right for left, right in pairs))
        self.assertEqual(
            Counter(action for pair in pairs for action in pair),
            Counter({action: 8 for action in tool.ACTION_IDS}))

    def test_opaque_menu_changes_only_identifiers(self):
        tool = self._tool()
        for index, seed in enumerate((3670, 3679, 9001)):
            named_order, named_labels = tool.menu_for_pair(
                index, seed, opaque=False)
            opaque_order, opaque_labels = tool.menu_for_pair(
                index, seed, opaque=True)
            self.assertEqual(named_order, opaque_order)
            self.assertEqual(named_labels, {
                action: action for action in named_order})
            self.assertEqual(set(opaque_labels), set(named_order))
            self.assertEqual(set(opaque_labels.values()),
                             set(tool.OPAQUE_LABELS))
            self.assertTrue(all(
                label not in tool.ACTION_IDS
                for label in opaque_labels.values()))

    def test_causal_goal_variant_removes_all_description_cues(self):
        tool = self._tool()
        audit = tool.lexical_audit()
        self.assertTrue(audit["ok"], audit)
        self.assertEqual(audit["goals_checked"], len(tool.ACTION_IDS))
        self.assertEqual(set(tool.CAUSAL_GOALS), set(tool.ACTION_IDS))
        self.assertEqual(set(tool.STRUCTURAL_GOALS), set(tool.ACTION_IDS))
        self.assertEqual(set(tool.SYNONYM_GOALS), set(tool.ACTION_IDS))
        self.assertEqual(set(tool.JOINT_GOALS), set(tool.ACTION_IDS))
        self.assertEqual(set(tool.JOINT_DESCRIPTIONS), set(tool.ACTION_IDS))
        self.assertEqual(set(tool.LITERAL_GOALS), set(tool.ACTION_IDS))
        self.assertTrue(all(
            tool.CAUSAL_GOALS[action] != tool.LITERAL_GOALS[action]
            for action in tool.ACTION_IDS))
        structural_audit = tool.lexical_audit(tool.STRUCTURAL_GOALS)
        self.assertTrue(structural_audit["ok"], structural_audit)
        synonym_audit = tool.lexical_audit(tool.SYNONYM_GOALS)
        self.assertTrue(synonym_audit["ok"], synonym_audit)
        joint_audit = tool.lexical_audit(tool.JOINT_GOALS)
        self.assertTrue(joint_audit["ok"], joint_audit)
        self.assertTrue(all(
            tool.SYNONYM_GOALS[action] != tool.CAUSAL_GOALS[action]
            for action in tool.ACTION_IDS))
        self.assertTrue(all(
            tool.JOINT_GOALS[action] != tool.CAUSAL_GOALS[action]
            and tool.JOINT_DESCRIPTIONS[action]
            != tool.ACTION_INFO[action][0]
            and action not in tool.JOINT_DESCRIPTIONS[action]
            for action in tool.ACTION_IDS))

    def test_structural_preregistration_is_directional(self):
        tool = self._tool()

        def rows(correct: set[int]):
            return [
                {
                    "pair": pair,
                    "arm": arm,
                    "valid": True,
                    "correct": pair in correct,
                }
                for pair in range(10)
                for arm in ("opaque_a", "opaque_b")
            ]

        fragile = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(5, 10))))
        self.assertEqual(
            tool.classify_structural_comparison(fragile),
            "structural_fragility_supported")

        robust = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(9))))
        self.assertEqual(
            tool.classify_structural_comparison(robust),
            "structural_robustness_supported")

    def test_joint_preregistration_is_directional_and_rewrites_both_sides(self):
        tool = self._tool()
        description_audit = tool.description_audit()
        self.assertTrue(description_audit["ok"], description_audit)
        self.assertIn("joint", tool.GOAL_VARIANTS)
        self.assertEqual(
            tool.DESCRIPTION_VARIANT_BY_GOAL["joint"], "joint")

        def tokens(text):
            return len(text)

        from unittest.mock import patch

        with patch.object(
                tool, "pad_to_tokens",
                lambda build, target, tokenize: (build(""), target, "")):
            causal = tool.build_trial_prompts(
                0, 3670, tokens, goal_variant="causal")
            joint = tool.build_trial_prompts(
                0, 3670, tokens, goal_variant="joint")
        original = tool.ACTION_INFO["seed.export"][0]
        rewritten = tool.JOINT_DESCRIPTIONS["seed.export"]
        self.assertIn(original, causal["texts"]["opaque_a"])
        self.assertNotIn(rewritten, causal["texts"]["opaque_a"])
        self.assertIn(rewritten, joint["texts"]["opaque_a"])
        self.assertNotIn(original, joint["texts"]["opaque_a"])

        def rows(correct: set[int]):
            return [
                {
                    "pair": pair,
                    "arm": arm,
                    "valid": True,
                    "correct": pair in correct,
                }
                for pair in range(10)
                for arm in ("opaque_a", "opaque_b")
            ]

        fragile = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(5, 10))))
        self.assertEqual(
            tool.classify_joint_comparison(fragile),
            "joint_fragility_supported")
        robust = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(9))))
        self.assertEqual(
            tool.classify_joint_comparison(robust),
            "joint_robustness_supported")

    def test_synonym_preregistration_is_directional(self):
        tool = self._tool()

        def rows(correct: set[int]):
            return [
                {
                    "pair": pair,
                    "arm": arm,
                    "valid": True,
                    "correct": pair in correct,
                }
                for pair in range(10)
                for arm in ("opaque_a", "opaque_b")
            ]

        fragile = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(5, 10))))
        self.assertEqual(
            tool.classify_synonym_comparison(fragile),
            "synonym_fragility_supported")

        robust = tool.paired_binary_comparison(
            rows(set(range(10))), rows(set(range(9))))
        self.assertEqual(
            tool.classify_synonym_comparison(robust),
            "synonym_robustness_supported")

    def test_goal_variants_can_be_token_matched_together(self):
        from unittest.mock import patch

        tool = self._tool()
        tokenize = lambda text: 10
        fake_pad = lambda build, target, tokenize: (build(""), target, "")
        with patch.object(tool, "pad_to_tokens", fake_pad):
            trials = {
                variant: tool.build_trial_prompts(
                    0, 3670, tokenize, goal_variant=variant,
                    target_tokens=15)
                for variant in ("causal", "structural", "synonym")
            }
        self.assertEqual(
            {trial["target_tokens"] for trial in trials.values()}, {15})
        self.assertTrue(all(
            set(trial["tokens"].values()) == {15}
            for trial in trials.values()))

    def test_pad_to_tokens_falls_back_beyond_hand_listed_pads(self):
        tool = self._tool()
        text, count, pad = tool.pad_to_tokens(
            lambda suffix: suffix, 400, len)
        self.assertEqual(count, 400)
        self.assertEqual(text, pad)
        self.assertEqual(pad, " x" * 200)

    def test_preregistered_goal_comparison_rules_are_directional(self):
        tool = self._tool()

        def rows(correct: set[int]):
            return [
                {
                    "pair": pair,
                    "arm": arm,
                    "valid": True,
                    "correct": pair in correct,
                }
                for pair in range(10)
                for arm in ("opaque_a", "opaque_b")
            ]

        baseline = rows(set(range(10)))
        weak_candidate = rows(set(range(5, 10)))
        comparison = tool.paired_binary_comparison(
            baseline, weak_candidate)
        self.assertEqual(comparison["discordant"]["baseline_only"], 10)
        self.assertEqual(comparison["absolute_accuracy_change"], -0.5)
        self.assertEqual(
            tool.classify_goal_comparison(comparison),
            "lexical_shortcut_supported")

        preserve_candidate = rows(set(range(9)))
        comparison = tool.paired_binary_comparison(
            baseline, preserve_candidate)
        self.assertEqual(comparison["discordant"]["baseline_only"], 2)
        self.assertEqual(comparison["candidate_accuracy"], 0.9)
        self.assertEqual(
            tool.classify_goal_comparison(comparison),
            "semantic_selection_survives_lexical_control")

    def test_interleaved_schedule_balances_adjacent_goal_calls(self):
        from collections import Counter

        tool = self._tool()
        schedule = tool.interleaved_schedule(8)
        self.assertEqual(len(schedule), 8 * len(tool.ARMS) * 2)
        self.assertEqual(len({item[0] for item in schedule}), 8)
        for offset in range(0, len(schedule), 2):
            first, second = schedule[offset], schedule[offset + 1]
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[2], second[2])
            self.assertEqual({first[1], second[1]},
                             {"literal", "causal"})
        first_variant_counts = Counter()
        for pair in range(8):
            for arm in tool.ARMS:
                indexes = [
                    index for index, item in enumerate(schedule)
                    if item[0] == pair and item[2] == arm
                ]
                self.assertEqual(indexes[1] - indexes[0], 1)
                first_variant_counts[schedule[indexes[0]][1]] += 1
        self.assertEqual(first_variant_counts, Counter({
            "literal": 16,
            "causal": 16,
        }))

    def test_synonym_schema_dispatches_to_interleaved_audit(self):
        from unittest.mock import patch

        tool = self._tool()
        sentinel = {"ok": True}
        with patch.object(
                tool, "audit_interleaved_artifact", return_value=sentinel
        ) as audit:
            result = tool.audit_artifact({
                "schema": tool.SYNONYM_INTERLEAVED_SCHEMA,
            })
        self.assertIs(result, sentinel)
        audit.assert_called_once()

    def test_joint_schema_dispatches_to_interleaved_audit(self):
        from unittest.mock import patch

        tool = self._tool()
        sentinel = {"ok": True}
        with patch.object(
                tool, "audit_interleaved_artifact", return_value=sentinel
        ) as audit:
            result = tool.audit_artifact({
                "schema": tool.JOINT_INTERLEAVED_SCHEMA,
            })
        self.assertIs(result, sentinel)
        audit.assert_called_once()

    def test_interleaved_artifact_audit_checks_call_order_and_comparison(self):
        tool = self._tool()
        rows_by_variant = {"literal": [], "causal": []}
        prompts_by_variant = {"literal": [], "causal": []}
        for variant in ("literal", "causal"):
            for pair in range(2):
                seed = 3670 + pair
                targets = tool.pair_targets(pair)
                label_maps = {
                    mode: {
                        label: action for action, label in
                        tool.menu_for_pair(
                            pair, seed, opaque=(mode == "opaque"))[1].items()
                    }
                    for mode in ("named", "opaque")
                }
                texts = {}
                hashes = {}
                for position, arm in enumerate(tool.ARM_ORDERS[pair]):
                    mode, arm_variant = arm.split("_", 1)
                    target = targets[0 if arm_variant == "a" else 1]
                    label = next(
                        label for label, action in label_maps[mode].items()
                        if action == target)
                    raw = json.dumps({"action": label, "reason": "match"})
                    text = f"prompt-{variant}-{pair}-{arm}-" + "x" * 20
                    texts[arm] = text
                    hashes[arm] = hashlib.sha256(
                        text.encode("utf-8")).hexdigest()
                    rows_by_variant[variant].append({
                        "pair": pair,
                        "arm": arm,
                        "mode": mode,
                        "variant": arm_variant,
                        "goal_variant": variant,
                        "position": position,
                        "seed": seed,
                        "target": target,
                        "action": target,
                        "label": label,
                        "valid": True,
                        "correct": True,
                        "reason": "match",
                        "raw": raw,
                        "prompt_sha256": hashes[arm],
                        "prompt_chars": len(text),
                        "prompt_tokens": 100,
                    })
                prompts_by_variant[variant].append({
                    "pair": pair,
                    "target_a": targets[0],
                    "target_b": targets[1],
                    "text": texts,
                    "sha256": hashes,
                    "tokens": {arm: 100 for arm in tool.ARMS},
                })

        schedule = tool.interleaved_schedule(2)
        stored_schedule = []
        for sample_index, (pair, variant, arm) in enumerate(schedule):
            row = next(
                row for row in rows_by_variant[variant]
                if row["pair"] == pair and row["arm"] == arm)
            row["sample_index"] = sample_index
            stored_schedule.append({
                "sample_index": sample_index,
                "pair": pair,
                "goal_variant": variant,
                "arm": arm,
            })
        artifacts = {}
        for variant in ("literal", "causal"):
            artifacts[variant] = {
                "goal_variant": variant,
                "pairs": 2,
                "system": tool.SYSTEM_PROMPT,
                "system_sha256": hashlib.sha256(
                    tool.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "prompts": prompts_by_variant[variant],
                "rows": rows_by_variant[variant],
                "summary": tool.summarise(rows_by_variant[variant], 2),
            }
        artifact = {
            "schema": tool.INTERLEAVED_SCHEMA,
            "pairs": 2,
            "schedule": stored_schedule,
            "variants": artifacts,
            "comparison": tool.compare_artifacts(
                artifacts["literal"], artifacts["causal"]),
        }
        audit = tool.audit_artifact(artifact)
        self.assertTrue(audit["ok"], audit["checks"])
        artifact["schedule"][0], artifact["schedule"][1] = (
            artifact["schedule"][1], artifact["schedule"][0])
        audit = tool.audit_artifact(artifact)
        self.assertFalse(audit["checks"]["schedule_exact"])
        self.assertFalse(audit["ok"])

    def test_parse_selection_uses_the_arm_label_mapping(self):
        tool = self._tool()
        label_to_action = {"k7": "seed.export", "m2": "seed.replicate"}
        action, label, obj = tool.parse_selection(
            'noise {"action":"k7","reason":"matches"} tail',
            label_to_action)
        self.assertEqual(action, "seed.export")
        self.assertEqual(label, "k7")
        self.assertEqual(obj["reason"], "matches")
        action, label, obj = tool.parse_selection(
            '{"action":"seed.export"}', label_to_action)
        self.assertEqual(action, "(invalid)")
        self.assertEqual(label, "seed.export")
        self.assertEqual(obj["action"], "seed.export")

    def test_summary_separates_goal_sensitivity_from_named_advantage(self):
        tool = self._tool()
        rows = []
        for pair in range(2):
            for position, arm in enumerate(tool.ARM_ORDERS[pair]):
                mode, variant = arm.split("_", 1)
                target = tool.ACTION_IDS[(pair * 2 + (variant == "b")) % 8]
                correct = not (pair == 1 and mode == "opaque")
                rows.append({
                    "pair": pair,
                    "arm": arm,
                    "mode": mode,
                    "variant": variant,
                    "position": position,
                    "target": target,
                    "action": target if correct else "(invalid)",
                    "valid": correct,
                    "correct": correct,
                })
        summary = tool.summarise(rows, pairs=2)
        named = summary["goal_sensitivity"]["named"]
        opaque = summary["goal_sensitivity"]["opaque"]
        self.assertEqual(named["comparable_pairs"], 2)
        self.assertEqual(named["both_correct"], 2)
        self.assertEqual(named["goal_changed_action"], 2)
        self.assertEqual(opaque["comparable_pairs"], 1)
        self.assertEqual(opaque["both_correct"], 1)
        self.assertEqual(opaque["goal_changed_action"], 1)
        self.assertEqual(summary["named_vs_opaque"]["named_correct"], 4)
        self.assertEqual(summary["named_vs_opaque"]["opaque_correct"], 2)

    def test_artifact_audit_recomputes_opaque_selections_and_summary(self):
        tool = self._tool()
        pairs = 2
        rows = []
        prompts = []
        for pair in range(pairs):
            seed = 3670 + pair
            targets = tool.pair_targets(pair)
            label_maps = {
                mode: {
                    label: action for action, label in
                    tool.menu_for_pair(
                        pair, seed, opaque=(mode == "opaque"))[1].items()
                }
                for mode in ("named", "opaque")
            }
            texts = {}
            hashes = {}
            for position, arm in enumerate(tool.ARM_ORDERS[pair]):
                mode, variant = arm.split("_", 1)
                target = targets[0 if variant == "a" else 1]
                label = next(label for label, action in label_maps[mode].items()
                             if action == target)
                raw = json.dumps({"action": label, "reason": "match"})
                text = f"prompt-{pair}-{arm}-" + "x" * 20
                texts[arm] = text
                hashes[arm] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                rows.append({
                    "pair": pair,
                    "arm": arm,
                    "mode": mode,
                    "variant": variant,
                    "position": position,
                    "seed": seed,
                    "target": target,
                    "action": target,
                    "label": label,
                    "valid": True,
                    "correct": True,
                    "reason": "match",
                    "raw": raw,
                    "prompt_sha256": hashes[arm],
                    "prompt_chars": len(text),
                    "prompt_tokens": 100,
                })
            prompts.append({
                "pair": pair,
                "target_a": targets[0],
                "target_b": targets[1],
                "text": texts,
                "sha256": hashes,
                "tokens": {arm: 100 for arm in tool.ARMS},
            })
        artifact = {
            "pairs": pairs,
            "system": tool.SYSTEM_PROMPT,
            "system_sha256": hashlib.sha256(
                tool.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "prompts": prompts,
            "rows": rows,
            "summary": tool.summarise(rows, pairs),
        }
        audit = tool.audit_artifact(artifact)
        self.assertTrue(audit["ok"], audit["checks"])
        rows[0]["raw"] = json.dumps({"action": "wrong", "reason": "tampered"})
        audit = tool.audit_artifact(artifact)
        self.assertFalse(audit["checks"]["raw_selections_recompute_exactly"])
        self.assertFalse(audit["ok"])


class TestTakeoverGuard(unittest.TestCase):
    def test_recent_mac_codex_activity_blocks_takeover(self):
        from datetime import datetime, timedelta, timezone
        from tools import takeover_guard

        now = datetime(2026, 10, 28, 13, 30,
                       tzinfo=timezone(timedelta(hours=8)))
        result = takeover_guard.assess_activity([
            {"id": "old", "from": "mac-codex", "created_at": "2026-10-28T13:00:00",
             "title": "[chat] old"},
            {"id": "new", "from": "mac-codex", "created_at": "2026-10-28T13:20:00",
             "title": "[reaction] still active"},
            {"id": "noise", "from": "mac-proposer",
             "created_at": "2026-10-28T13:29:00", "title": "[proposal] noise"},
        ], now, limit=10)
        self.assertEqual(result.status, takeover_guard.RECENT)
        self.assertEqual(result.latest_match, "2026-10-28T13:20:00+08:00")

    def test_a_full_tail_that_starts_inside_the_window_is_unknown(self):
        """A full high-traffic tail is not a time window when the actor fell out."""
        from datetime import datetime, timedelta, timezone
        from tools import takeover_guard

        now = datetime(2026, 10, 28, 13, 30,
                       tzinfo=timezone(timedelta(hours=8)))
        tasks = [
            {"id": f"noise-{i}", "from": "mac-laya-int8",
             "created_at": f"2026-10-28T13:{21 + i:02d}:00",
             "title": "[reaction] noise"}
            for i in range(9)
        ]
        result = takeover_guard.assess_activity(tasks, now, limit=9)
        self.assertEqual(result.status, takeover_guard.UNKNOWN)
        self.assertIn("does not cover", result.reason)

    def test_other_actors_are_not_mistaken_for_the_primary(self):
        from datetime import datetime, timedelta, timezone
        from tools import takeover_guard

        now = datetime(2026, 10, 28, 13, 30,
                       tzinfo=timezone(timedelta(hours=8)))
        result = takeover_guard.assess_activity([
            {"id": "old", "from": "mac-codex",
             "created_at": "2026-10-28T13:00:00", "title": "[chat] old"},
            {"id": "noise", "from": "mac-proposer",
             "created_at": "2026-10-28T13:29:00", "title": "[proposal] noise"},
        ], now, limit=10)
        self.assertEqual(result.status, takeover_guard.SAFE)


class TestObservatoryFeed(unittest.TestCase):
    def test_signed_bodies_are_verified_once_per_exact_message(self):
        """Re-rendering the same signed envelope must not redo Ed25519."""
        from unittest import mock
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        body = json.dumps({
            "payload": {"goal": "one"},
            "pubkey": "pk",
            "sig": "sig",
        })
        observatory._signed_fields_cached.cache_clear()
        try:
            with mock.patch.object(observatory, "verify_signature",
                                   return_value=True) as verify:
                first = observatory.signed_fields(body)
                first["goal"] = "mutated by a renderer"
                second = observatory.signed_fields(body)
                changed = observatory.signed_fields(body + " ")
        finally:
            observatory._signed_fields_cached.cache_clear()

        self.assertEqual(second, {"goal": "one"})
        self.assertEqual(changed, {"goal": "one"})
        # Same exact bytes use the cached result; changed bytes are checked
        # again rather than inheriting trust from a different message.
        self.assertEqual(verify.call_count, 2)

    def test_the_dashboard_window_is_bounded_and_counts_the_whole_mailbox(self):
        """A short wire response must not turn work counts into window counts."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "agent_mailbox_under_test_dashboard_window",
            Path(__file__).resolve().parents[1] / "tools" / "agent_mailbox.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            box = module.Mailbox(Path(tmp) / "box")
            rows = [
                ("k1", "[proposal] p1 seed.export by peer", "open", "10:00:00"),
                ("w1", "repair root", "open", "10:01:00"),
                ("k2", "[alarm] mailbox slow", "open", "10:02:00"),
                ("w2", "rotate archive", "blocked", "10:03:00"),
                ("k3", "[reaction] p1 赞同 by voter", "open", "10:04:00"),
                ("w3", "review report", "open", "10:05:00"),
                ("k4", "[measure] latency probe", "open", "10:06:00"),
            ]
            for task_id, title, status, clock in rows:
                box.save({
                    "id": task_id, "from": "ann", "title": title, "body": "x",
                    "status": status, "created_at": f"2026-10-27T{clock}",
                    "claimed_by": None, "finished_at": None, "history": [],
                })
            server = module.ThreadingHTTPServer(
                ("127.0.0.1", 0), module.build_handler(box, "tok"))
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/tasks/window"
                    "?chat_limit=2&work_limit=1",
                    headers={"Authorization": "Bearer tok"})
                with urllib.request.urlopen(request, timeout=3) as resp:
                    answer = json.loads(resp.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

        self.assertEqual([row["id"] for row in answer["chat"]], ["k3", "k4"])
        self.assertEqual([row["id"] for row in answer["work"]], ["w3"])
        self.assertEqual(answer["counts"],
                         {"chat": 4, "work": 3, "work_open": 2})

    def test_the_observatory_uses_the_bounded_window_and_keeps_global_counts(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        calls = []
        original_json = observatory.http_json
        original_units = observatory.local_units

        def fake_json(url, *args, **kwargs):
            calls.append(url)
            if url == "http://mailbox/health":
                return {"open": 4}, None, 1.0, 200
            if "/tasks/window" in url:
                return {
                    "chat": [],
                    "work": [{"id": "w16", "from": "ann",
                              "title": "latest work", "status": "open",
                              "created_at": "2026-10-27T10:16:00",
                              "claimed_by": None}],
                    "counts": {"chat": 36000, "work": 16, "work_open": 4},
                }, None, 2.0, 200
            raise AssertionError(f"full or unexpected feed request: {url}")

        observatory.http_json = fake_json
        observatory.local_units = lambda: {"host": "test", "services": []}
        try:
            state = observatory.collect_state({
                "mailbox_url": "http://mailbox", "mailbox_token": "token",
                "roots": [],
            })
        finally:
            observatory.http_json = original_json
            observatory.local_units = original_units

        self.assertEqual(calls, [
            "http://mailbox/health",
            "http://mailbox/tasks/window?chat_limit=120&work_limit=14",
        ])
        self.assertEqual(state["work_counts"], {"total": 16, "open": 4})
        self.assertEqual(state["view"]["work_count"], "4 open / 16")
        feed = next(row for row in state["services"]
                    if row["name"] == "task feed (/tasks/window)")
        self.assertEqual(feed["detail"], "36000 kinded / 16 work; window 1")

    def test_an_old_mailbox_still_serves_the_unbounded_feed_on_404(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        calls = []
        original_json = observatory.http_json
        original_units = observatory.local_units

        def fake_json(url, *args, **kwargs):
            calls.append(url)
            if url == "http://mailbox/health":
                return {"open": 1}, None, 1.0, 200
            if "/tasks/window" in url:
                return None, "HTTP 404", 1.0, 404
            if "/tasks?status=all" in url:
                return {"tasks": [{"id": "w1", "from": "ann",
                                   "title": "old mailbox work",
                                   "status": "open"}]}, None, 2.0, 200
            raise AssertionError(f"unexpected request: {url}")

        observatory.http_json = fake_json
        observatory.local_units = lambda: {"host": "test", "services": []}
        try:
            state = observatory.collect_state({
                "mailbox_url": "http://mailbox", "mailbox_token": "token",
                "roots": [],
            })
        finally:
            observatory.http_json = original_json
            observatory.local_units = original_units

        self.assertIn("http://mailbox/tasks?status=all", calls)
        self.assertEqual(state["work_counts"], {"total": 1, "open": 1})

    def test_an_address_is_stored_once(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import addressee, split_title

        kind, rest = split_title("[chat] @@win-codex: hello")
        self.assertEqual(kind, "chat")
        self.assertEqual(addressee(rest), "win-codex")
        self.assertEqual(addressee("@mac-codex: hi"), "mac-codex")
        self.assertIsNone(split_title("Task 001 - deploy")[0])

    def test_people_share_a_row_only_when_the_name_matches(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import build_people

        rows = build_people(
            [{"node_id": "mac-text", "roots": ["mac"], "lineage": "text-nb-v1",
              "host": "127.0.0.1", "has_pubkey": False, "capability": {},
              "stats": {}}],
            [{"host": "mac", "alive": True, "via": "local file",
              "last": {"action": "seed.export", "goal": "再播一处", "round": 4,
                       "served_by": "qwen"}}],
            [{"from": "mac-text", "kind": "chat", "created_at": "t", "fields": {}}],
        )
        names = [row["name"] for row in rows]
        self.assertEqual(names, ["mac-text", "mac"])
        text = next(row for row in rows if row["name"] == "mac-text")
        self.assertEqual(text["seen"]["count"], 1)
        self.assertEqual(text["roots"], ["mac"])
        ledger = next(row for row in rows if row["name"] == "mac")
        self.assertEqual(ledger["activity"]["round"], 4)
        placed = build_people([], [{"host": "pi5", "address": "192.168.31.22",
                                     "alive": True, "last": {"round": 8, "served_by": "qwen"}}], [])
        self.assertEqual(placed[0]["host"], "192.168.31.22")


    def test_the_participants_panel_names_each_member_role(self):
        """A reader sees who decides and who proposes, read off the record."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import (build_people, participant_from_node,
                                 present_people)

        part = participant_from_node(
            "mac-text",
            {"url": "http://127.0.0.1:8766", "host": "127.0.0.1",
             "checkpoint": "text-nb-v1", "role": "decision"},
            "mac")
        self.assertEqual(part["role"], "decision")
        rows = build_people(
            [part,
             participant_from_node(
                 "mac-llm", {"url": "http://127.0.0.1:8768",
                             "host": "127.0.0.1",
                             "checkpoint": "mac-llm-8b-q6",
                             "role": "proposal",
                             "chat": {"model": "qwen"}}, "mac")],
            [], [])
        shown = {row["name"]: row for row in present_people(rows)}
        self.assertEqual(shown["mac-text"]["role"], "决策")
        self.assertEqual(shown["mac-llm"]["role"], "提案")

    def test_a_column_is_one_field(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import audit_view, present_people

        rows = present_people([
            {"name": "mac-text", "roots": ["mac"], "host": "127.0.0.1",
             "lineage": "text-nb-v1", "pubkey": "abc",
             "stats": {"successes": 1, "failures": 0},
             "activity": None, "seen": {"count": 4, "kind": "chat"}},
            {"name": "mac", "roots": [], "host": "",
             "lineage": "", "stats": {},
             "activity": {"action": "seed.export", "round": 3, "goal": "再播",
                          "served_by": "qwen"},
             "seen": None},
        ])
        self.assertEqual(rows[0]["from"], "mac")
        self.assertEqual(rows[0]["record"], "票 1/0")
        self.assertEqual(rows[1]["from"], "—")
        self.assertEqual(rows[1]["record"], "seed.export · r3")
        self.assertEqual(audit_view({"people": rows, "messages": [
            {"to": "@win-codex", "text": "hello", "who": "mac"},
        ]}), [])



    def test_a_root_snapshot_is_read_from_the_file(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import _root_sources, read_root_snapshot, participant_from_node

        sources = _root_sources({"roots": [
            {"name": "pi5", "file": "~/.pandosi/remote/pi5-root.json"},
            {"name": "mac", "url": "http://127.0.0.1:8765"},
        ]})
        self.assertEqual(sources[0]["file"], "~/.pandosi/remote/pi5-root.json")
        self.assertEqual(sources[0]["url"], "")
        self.assertEqual(sources[1]["url"], "http://127.0.0.1:8765")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "root.json"
            path.write_text(json.dumps({"nodes": {
                "pandosi-pi5-llm": {
                    "url": "http://192.168.31.22:1",
                    "host": "192.168.31.22",
                    "checkpoint": "qwen3-1.7b-q4km",
                    "chat": {"model": "qwen3-1.7b"},
                }
            }}), encoding="utf-8")
            payload, err = read_root_snapshot(str(path))
            self.assertIsNone(err)
            row = participant_from_node(
                "pandosi-pi5-llm", payload["nodes"]["pandosi-pi5-llm"], "pi5")
            self.assertEqual(row["chat"], {"model": "qwen3-1.7b"})
            self.assertEqual(row["host"], "192.168.31.22")
            path.write_text("{", encoding="utf-8")
            self.assertIsNone(read_root_snapshot(str(path))[0])


    def test_an_unreachable_root_is_probed_once(self):
        """A dead root must not pay the same timeout for every sibling path."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        calls = []
        original_json = observatory.http_json
        original_units = observatory.local_units

        def fake_json(url, *args, **kwargs):
            calls.append(url)
            if url.endswith("/health"):
                return {}, None, 1.0, 200
            if "/tasks/window" in url:
                return {"chat": [], "work": [],
                        "counts": {"chat": 0, "work": 0, "work_open": 0}}, None, 1.0, 200
            if url.endswith("/nodes"):
                return None, "unreachable: timeout", 4000.0, 0
            raise AssertionError(f"unexpected probe after node failure: {url}")

        observatory.http_json = fake_json
        observatory.local_units = lambda: {"host": "test", "services": []}
        with observatory._ROOT_FAILURE_LOCK:
            observatory._ROOT_FAILURE_CACHE.clear()
        try:
            state = observatory.collect_state({
                "mailbox_url": "http://mailbox",
                "mailbox_token": "token",
                "roots": [{"name": "win", "url": "http://win:8790",
                           "token": "pairing"}],
            })
        finally:
            observatory.http_json = original_json
            observatory.local_units = original_units
            with observatory._ROOT_FAILURE_LOCK:
                observatory._ROOT_FAILURE_CACHE.clear()

        self.assertEqual([url for url in calls if url.startswith("http://win:8790")],
                         ["http://win:8790/nodes"])
        status = next(row for row in state["services"]
                      if row["name"] == "root /status")
        self.assertFalse(status["alive"])
        self.assertEqual(status["detail"], "skipped: /nodes unreachable")
        self.assertEqual(state["roots"][0]["playbook_error"],
                         "skipped: /nodes unreachable")


    def test_root_and_member_probes_overlap(self):
        """A dead root must not block member health from a healthy root."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        barrier = threading.Barrier(2)
        synced = []
        original_json = observatory.http_json
        original_units = observatory.local_units

        def synchronise():
            try:
                barrier.wait(timeout=2)
                synced.append(True)
            except threading.BrokenBarrierError:
                pass

        def fake_json(url, *args, **kwargs):
            if url == "http://mailbox/health":
                return {}, None, 1.0, 200
            if "/tasks/window" in url:
                return {"chat": [], "work": [],
                        "counts": {"chat": 0, "work": 0, "work_open": 0}}, None, 1.0, 200
            if url == "http://win:8790/nodes":
                synchronise()
                return None, "unreachable: timeout", 4000.0, 0
            if url == "http://mac:8765/nodes":
                return {"nodes": {"member": {
                    "url": "http://member:1", "host": "member-host"}}}, None, 1.0, 200
            if url == "http://member:1/health":
                synchronise()
                return {"chat": {"model": "test-model"}}, None, 1.0, 200
            if url.endswith("/status") or url.endswith("/playbook"):
                return {}, None, 1.0, 200
            raise AssertionError(f"unexpected probe: {url}")

        observatory.http_json = fake_json
        observatory.local_units = lambda: {"host": "test", "services": []}
        with observatory._ROOT_FAILURE_LOCK:
            observatory._ROOT_FAILURE_CACHE.clear()
        try:
            state = observatory.collect_state({
                "mailbox_url": "http://mailbox",
                "mailbox_token": "token",
                "roots": [
                    {"name": "win", "url": "http://win:8790",
                     "token": "pairing"},
                    {"name": "mac", "url": "http://mac:8765",
                     "token": "pairing"},
                ],
            })
        finally:
            observatory.http_json = original_json
            observatory.local_units = original_units
            with observatory._ROOT_FAILURE_LOCK:
                observatory._ROOT_FAILURE_CACHE.clear()

        self.assertEqual(len(synced), 2)
        member = next(row for row in state["participants"]
                      if row["node_id"] == "member")
        self.assertEqual(member["chat"]["model"], "test-model")


    def test_a_dead_root_failure_is_cached_with_explicit_age(self):
        """A brief negative cache must remain visibly different from a probe."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import observatory

        node_calls = []
        original_json = observatory.http_json
        original_units = observatory.local_units
        original_ttl = observatory._ROOT_FAILURE_TTL

        def fake_json(url, *args, **kwargs):
            if url == "http://mailbox/health":
                return {}, None, 1.0, 200
            if "/tasks/window" in url:
                return {"chat": [], "work": [],
                        "counts": {"chat": 0, "work": 0, "work_open": 0}}, None, 1.0, 200
            if url == "http://dead:8790/nodes":
                node_calls.append(url)
                return None, "unreachable: timeout", 4000.0, 0
            raise AssertionError(f"unexpected probe: {url}")

        cfg = {
            "mailbox_url": "http://mailbox",
            "mailbox_token": "token",
            "roots": [{"name": "dead", "url": "http://dead:8790",
                       "token": "pairing"}],
        }
        observatory.http_json = fake_json
        observatory.local_units = lambda: {"host": "test", "services": []}
        with observatory._ROOT_FAILURE_LOCK:
            observatory._ROOT_FAILURE_CACHE.clear()
        try:
            first = observatory.collect_state(cfg)
            second = observatory.collect_state(cfg)
            self.assertEqual(len(node_calls), 1)
            self.assertTrue(second["roots"][0]["cached"])
            self.assertGreaterEqual(second["roots"][0]["cache_age_s"], 0)
            self.assertIn("cached ", second["roots"][0]["error"])
            cached_service = next(
                row for row in second["services"]
                if row["name"] == "root /nodes")
            self.assertTrue(cached_service["cached"])
            self.assertIn("cached ", cached_service["detail"])
            self.assertIsNone(cached_service["latency_ms"])
            self.assertNotIn("cached", first["roots"][0])

            observatory._ROOT_FAILURE_TTL = -1.0
            third = observatory.collect_state(cfg)
            self.assertEqual(len(node_calls), 2)
            self.assertNotIn("cached", third["roots"][0])
        finally:
            observatory.http_json = original_json
            observatory.local_units = original_units
            observatory._ROOT_FAILURE_TTL = original_ttl
            with observatory._ROOT_FAILURE_LOCK:
                observatory._ROOT_FAILURE_CACHE.clear()


    def test_an_execution_ledger_row_is_shown_unchanged(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import (attach_executions, execution_text,
                                 present_messages, read_execution_ledger)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execute-ledger.jsonl"
            path.write_text(
                '{"proposal_id": "p1", "action": "seed.export", "executed": true, "sealed": true}\n'
                '{"proposal_id": "p2", "action": "culture.refine", "executed": false, "skipped": "no-actuator"}\n'
                '{"proposal_id": "p3", "action": "seed.replicate", "executed": false, "skipped": "no-transport", "destination": "mac"}\n',
                encoding="utf-8")
            rows = read_execution_ledger(path)
            self.assertEqual(execution_text(rows["p1"]), "sealed")
            self.assertEqual(execution_text(rows["p2"]), "no-actuator")
            self.assertEqual(execution_text(rows["p3"]), "no-transport mac")
            self.assertEqual(read_execution_ledger(Path(tmp) / "missing"), {})
        messages = [{"fields": {"proposal_id": "p2", "action": "culture.refine",
                                "goal": "g", "expected_effect": "e"},
                     "kind": "proposal", "from": "mac-proposer"}]
        attach_executions(messages, {"p2": rows["p2"]})
        shown = present_messages(messages)[0]
        self.assertEqual(shown["execution"], "no-actuator")


class TestDelegatedOperatorGrant(unittest.TestCase):
    """A human approves the deployment action, not every future proposal.

    These tests keep the delegation narrow: one pinned signer, one catalogue
    resource, one host scope, seven days at most, and no recursive delegation.
    They generate disposable identities; production private keys are never
    read by the test suite.
    """

    @staticmethod
    def _delegation():
        from pando import grant_delegation
        return grant_delegation

    @staticmethod
    def _envelope(identity, now, **changes):
        import datetime as dt
        from pando.grants import format_timestamp

        payload = {
            "version": 1,
            "audience": "operator-host",
            "action": "deploy_operator",
            "resource": "profile-catalogue/v1",
            "scope": "operator-host",
            "approved_by": "test-user",
            "approved_at": format_timestamp(now - dt.timedelta(minutes=1)),
            "expires_at": format_timestamp(now + dt.timedelta(days=1)),
            "evidence": "test authorization",
        }
        payload.update(changes)
        return dict(payload, signature=identity.sign(payload))

    def _fixture(self, root: Path):
        import datetime as dt
        from unittest import mock
        from pando.grants import format_timestamp, save_grant
        from pando.identity import Identity

        identity = Identity.generate()
        identity_path = root / "identity.json"
        identity_path.write_text(json.dumps(identity.to_json()), encoding="utf-8")
        grants = root / "source-grants.toml"
        now = dt.datetime.now(dt.timezone.utc)
        save_grant("deploy_operator", "profile-catalogue/v1", "operator-host",
                   "test-user", format_timestamp(now + dt.timedelta(days=1)),
                   evidence="test authorization", path=grants)
        with mock.patch.object(self._delegation(), "MAC_OPERATOR_PUBKEY",
                               identity.public_key):
            envelope = self._delegation().issue_envelope(
                grants, identity=identity, now=now)
        return identity, identity_path, grants, now, envelope

    def test_signed_delegation_installs_once_and_is_idempotent(self):
        from pando.grants import audit_status, grant_status, load_grants

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "target-grants.toml"
            first = self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            second = self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)

            self.assertEqual(first["status"], "installed")
            self.assertTrue(first["changed"])
            self.assertEqual(second["status"], "unchanged")
            self.assertFalse(second["changed"])
            grant = load_grants(target)["deploy_operator"]
            self.assertEqual(grant["origin"], "mac-delegation/v1")
            self.assertEqual(grant["resource"], "profile-catalogue/v1")
            self.assertEqual(grant["scope"], "operator-host")
            self.assertEqual(grant_status("deploy_operator", target, now=now)
                             ["status"], "active")
            audit = audit_status(Path(str(target) + ".audit.jsonl"))
            self.assertEqual(audit["status"], "valid")
            self.assertEqual(audit["entries"], 1)

    def test_wrong_action_resource_scope_or_audience_is_refused(self):
        import datetime as dt
        from pando.identity import Identity

        now = dt.datetime.now(dt.timezone.utc)
        identity = Identity.generate()
        bad_fields = (
            ("action", "run_anything"),
            ("resource", "profile-catalogue/v2"),
            ("scope", "all-hosts"),
            ("audience", "some-other-host"),
        )
        for key, value in bad_fields:
            with self.subTest(field=key):
                envelope = self._envelope(identity, now, **{key: value})
                with self.assertRaises(self._delegation().DelegationError):
                    self._delegation().verify_envelope(
                        envelope, expected_pubkey=identity.public_key, now=now)

    def test_expiry_future_ttl_signature_and_tampering_all_fail_closed(self):
        import datetime as dt
        from pando.grants import format_timestamp
        from pando.identity import Identity

        now = dt.datetime.now(dt.timezone.utc)
        identity = Identity.generate()
        cases = {
            "expired": self._envelope(
                identity, now,
                approved_at=format_timestamp(now - dt.timedelta(days=2)),
                expires_at=format_timestamp(now - dt.timedelta(days=1))),
            "future": self._envelope(
                identity, now,
                approved_at=format_timestamp(now + dt.timedelta(minutes=10))),
            "ttl": self._envelope(
                identity, now,
                approved_at=format_timestamp(now - dt.timedelta(minutes=1)),
                expires_at=format_timestamp(now + dt.timedelta(days=8))),
            "wrong-signer": self._envelope(Identity.generate(), now),
        }
        tampered = self._envelope(identity, now)
        tampered["evidence"] = "changed after signing"
        cases["tampered"] = tampered
        for name, envelope in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(self._delegation().DelegationError):
                    self._delegation().verify_envelope(
                        envelope, expected_pubkey=identity.public_key, now=now)

    def test_only_the_pinned_identity_can_issue_a_delegation(self):
        import datetime as dt
        from unittest import mock
        from pando.grants import format_timestamp, save_grant
        from pando.identity import Identity

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pinned, attacker = Identity.generate(), Identity.generate()
            grants = root / "grants.toml"
            now = dt.datetime.now(dt.timezone.utc)
            save_grant("deploy_operator", "profile-catalogue/v1", "operator-host",
                       "test-user", format_timestamp(now + dt.timedelta(days=1)),
                       evidence="test authorization", path=grants)
            with mock.patch.object(self._delegation(), "MAC_OPERATOR_PUBKEY",
                                   pinned.public_key):
                with self.assertRaises(self._delegation().DelegationError):
                    self._delegation().issue_envelope(
                        grants, identity=attacker, now=now)

    def test_a_delegated_grant_cannot_mint_another_delegation(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "target-grants.toml"
            self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            with mock.patch.object(self._delegation(), "MAC_OPERATOR_PUBKEY",
                                   identity.public_key):
                with self.assertRaisesRegex(
                        self._delegation().DelegationError,
                        "cannot issue another delegation"):
                    self._delegation().issue_envelope(
                        target, identity=identity, now=now)

    def test_a_damaged_audit_chain_blocks_install_and_revocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "target-grants.toml"
            self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            audit = Path(str(target) + ".audit.jsonl")
            audit.write_text(audit.read_text(encoding="utf-8") + "broken\n",
                             encoding="utf-8")
            before = target.read_bytes()
            changed = self._envelope(identity, now, evidence="rotated evidence")

            with self.assertRaises(PermissionError):
                self._delegation().install_envelope(
                    changed, target, expected_pubkey=identity.public_key, now=now)
            with self.assertRaises(PermissionError):
                from pando.grants import remove_grant_if_origin
                remove_grant_if_origin("deploy_operator", "mac-delegation/v1",
                                       target)
            self.assertEqual(target.read_bytes(), before)

    def test_a_manual_grant_is_never_overwritten_or_revoked(self):
        import datetime as dt
        from pando.grants import format_timestamp, grant_status, save_grant

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "manual-grants.toml"
            save_grant("deploy_operator", "profile-catalogue/v1", "operator-host",
                       "human", format_timestamp(now + dt.timedelta(days=2)),
                       evidence="human grant", path=target)
            before = target.read_bytes()
            installed = self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            from pando.grants import remove_grant_if_origin
            removed = remove_grant_if_origin(
                "deploy_operator", "mac-delegation/v1", target)

            self.assertEqual(installed["status"], "manual-preserved")
            self.assertFalse(installed["changed"])
            self.assertFalse(removed)
            self.assertEqual(target.read_bytes(), before)
            self.assertNotIn("origin", grant_status(
                "deploy_operator", target, now=now)["grant"])

    def test_network_failure_preserves_and_only_all_404_revokes(self):
        import socket
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):  # noqa: N802
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "target-grants.toml"
            token = root / "mailbox.token"
            token.write_text("tok\n", encoding="utf-8")
            self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{probe.getsockname()[1]}/operator-grant"
            probe.close()
            try:
                mixed = self._delegation().refresh_and_install(
                    [dead, base + "/missing"], token_file=token,
                    grants_path=target, expected_pubkey=identity.public_key,
                    timeout=0.5, now=now)
                self.assertFalse(mixed["changed"])
                self.assertTrue(target.exists())

                absent = self._delegation().refresh_and_install(
                    [base + "/missing-a", base + "/missing-b"], token_file=token,
                    grants_path=target, expected_pubkey=identity.public_key,
                    timeout=0.5, now=now)
                self.assertEqual(absent["status"], "revoked")
                self.assertTrue(absent["changed"])
                self.assertFalse(target.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_bad_http_200_preserves_the_last_verified_grant(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, _, _, now, envelope = self._fixture(root)
            target = root / "target-grants.toml"
            token = root / "mailbox.token"
            token.write_text("tok\n", encoding="utf-8")
            self._delegation().install_envelope(
                envelope, target, expected_pubkey=identity.public_key, now=now)
            before = target.read_bytes()
            tampered = dict(envelope, approved_by="tampered after signing")
            bodies = {
                "/bad-json": b"not a json envelope",
                "/bad-signature": json.dumps(tampered).encode("utf-8"),
            }

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, fmt, *args):
                    pass

                def do_GET(self):  # noqa: N802
                    body = bodies.get(self.path, b"{}")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                for path in bodies:
                    with self.subTest(path=path):
                        result = self._delegation().refresh_and_install(
                            [base + path], token_file=token,
                            grants_path=target,
                            expected_pubkey=identity.public_key,
                            timeout=0.5, now=now)
                        self.assertEqual(result["status"], "error")
                        self.assertFalse(result["changed"])
                        self.assertTrue(target.exists())
                        self.assertEqual(target.read_bytes(), before)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_mailbox_operator_routes_are_authenticated_and_secret_free(self):
        import importlib.util
        from unittest import mock

        path = Path(__file__).resolve().parents[1] / "tools" / "agent_mailbox.py"
        spec = importlib.util.spec_from_file_location(
            "agent_mailbox_operator_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, identity_path, grants, now, envelope = self._fixture(root)
            box = module.Mailbox(root / "box")
            server = module.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                module.build_handler(box, "tok", grant_file=grants,
                                     identity_file=identity_path))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(base + "/operator-grant", timeout=5)
                self.assertEqual(denied.exception.code, 403)
                denied.exception.close()

                request = urllib.request.Request(
                    base + "/operator-status",
                    headers={"Authorization": "Bearer tok"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    status = json.loads(response.read())
                self.assertTrue(status["authorized"])
                self.assertEqual(status["signer_fingerprint"],
                                 identity.node_fingerprint)
                self.assertNotIn("secret_key", json.dumps(status))

                with mock.patch.object(self._delegation(), "MAC_OPERATOR_PUBKEY",
                                       identity.public_key):
                    request = urllib.request.Request(
                        base + "/operator-grant",
                        headers={"Authorization": "Bearer tok"})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        served = json.loads(response.read())
                self._delegation().verify_envelope(
                    served, expected_pubkey=identity.public_key, now=now)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_host_templates_use_the_same_signed_grant_paths(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "tools" / "win_node_supervisor.ps1").read_text(
            encoding="utf-8")
        self.assertIn("$stateDir = if ($env:PANDO_STATE_DIR)", windows)
        self.assertIn("Join-Path $env:USERPROFILE '.pando'", windows)
        self.assertIn("'--state-dir', \"$stateDir\"", windows)
        self.assertIn("'--grant-url', 'http://100.92.184.4:8776/operator-grant'",
                      windows)
        self.assertIn("'--grant-url', 'http://192.168.31.41:8776/operator-grant'",
                      windows)

        unit = (root / "tools" / "pi5-systemd" /
                "pandosi-execute.service").read_text(encoding="utf-8")
        self.assertIn("--state-dir /home/admin/.pando", unit)
        self.assertIn("--grant-url http://100.92.184.4:8776/operator-grant", unit)
        self.assertIn("--grant-url http://192.168.31.41:8776/operator-grant", unit)

        for name in ("com.jewelroam.pandosi-execute.plist",
                     "com.jewelroam.pandosi-mailbox.plist"):
            plist = plistlib.loads((root / "tools" / "mac-launchd" / name)
                                   .read_bytes())
            arguments = plist["ProgramArguments"]
            self.assertIn("/Users/zhengenrong/.pandosi/grants.toml", arguments)
            self.assertIn("/Users/zhengenrong/.pando/identity.json", arguments)


class TestVoteAblation(unittest.TestCase):
    """tools/vote_ablation.py - the questions the headline hides.

    The equal-budget control reported one number per arm. That number cannot
    say whether a member died on one distribution, whether a subset of the
    same votes would have won, or whether the weakest member is weak or merely
    answering with two labels swapped. This tool asks those three, so the test
    that matters is: does a genuinely relabelled member get *rescued*, and is
    a member that is simply bad never reported as rescued.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "vote_ablation.py"
        spec = importlib.util.spec_from_file_location("vote_ablation", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _rows(expected: list[str], choice: list[str]) -> list[dict]:
        return [{"task_id": f"big6-{i}", "ok": True,
                 "expected": e, "choice": c}
                for i, (e, c) in enumerate(zip(expected, choice))]

    def _raw(self, expected, choice) -> dict:
        """The smallest harvest shape vote_ablation will accept."""
        return {
            "suite": "synthetic",
            "n_rows": len(expected),
            "arm_a_url": "http://a",
            "arm_a": {"http://a": self._rows(expected, expected)},
            "arm_b": {"http://b": self._rows(expected, choice)},
        }

    def test_a_swapped_member_is_rescued_and_a_weak_member_is_not(self):
        """The two members this test must tell apart.

        ``swapped`` answers the truth with exactly two labels exchanged, so a
        single named swap restores it. ``weak`` answers at one third by
        guessing, so no swap rescues it and some should be reported as
        significantly *worse*. Collapsing these into one "survivors" list is
        the bug this asserts against.
        """
        va = self._tool()
        labels = ["a", "b", "c"]
        expected = [labels[i % 3] for i in range(90)]
        swapped = [{"a": "b", "b": "a"}.get(e, e) for e in expected]

        swap_arm = self._raw(expected, swapped)["arm_b"]["http://b"]
        verdict = va.swap_test(swap_arm)
        self.assertEqual(verdict["rescued"], [["a", "b"]])
        self.assertEqual(verdict["worsened"], [])

        # A member that is merely confused must not be reported as a fix.
        weak = [expected[i] if i % 3 else labels[(i + 1) % 3]
                for i in range(len(expected))]
        weak_verdict = va.swap_test(self._raw(expected, weak)["arm_b"]["http://b"])
        self.assertEqual(weak_verdict["rescued"], [],
                         "a chance-level member must never be called rescued")

    def test_rows_for_does_not_depend_on_the_arm_letter_case(self):
        """Regression: the raw harvest keys arms lower case, readers do not.

        ``per_member`` names members ``"B <url>"``; the harvest stores them
        under ``arm_b``. Joining those two spellings by hand is what crashed
        this tool on every run, so the lookup goes through the member's own
        fields and this pins that they stay paired.
        """
        va = self._tool()
        # Arm B is deliberately the weaker one, so ``weakest`` has one answer.
        raw = self._raw(["a", "b"], ["a", "c"])
        per = va.per_member(raw)
        self.assertIn("B http://b", per)
        rows = va.rows_for(raw, per["B http://b"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["expected"], "a")
        self.assertEqual(per["B http://b"]["accuracy"], 0.5)
        self.assertEqual(va.weakest(per), "B http://b")

    def test_the_bonferroni_bar_rises_with_the_number_of_hypotheses(self):
        """More variants, a higher bar -- so a subset cannot win by shopping."""
        va = self._tool()
        self.assertLess(va.bonferroni_z(1), va.bonferroni_z(4))
        self.assertLess(va.bonferroni_z(4), va.bonferroni_z(20))


class TestPairedDifferenceStandardError(unittest.TestCase):
    """The paired SE is sqrt(b + c), never half of it.

    Both equal-budget tools computed the McNemar standard error as
    ``sqrt(n) / 2``. That is the SE of a *difference of two independent*
    proportions; for a *paired* comparison the difference is b - c out of
    b + c discordant rows, so var(b - c) = b + c and SE = sqrt(b + c).
    Halving it doubled every z and shipped two "beyond noise" verdicts that
    were actually ties: 002 is b=21, c=15, which is z = -1.00 / exact
    p = 0.41, not the published z = -2.00. The test is deliberately a
    falsifier for the cheap fix ("just divide by two"), so it pins the
    arithmetic *and* one regenerated report, not only the code path.
    """

    def test_the_paired_se_is_the_root_of_the_discordant_count(self):
        """21 vs 15 is one SE of 6, so a six-row gap is one SE, not two."""
        paired = mcnemar_test(21, 15)
        self.assertEqual(paired["discordant"], 36)
        self.assertEqual(paired["delta"], -6)
        self.assertAlmostEqual(paired["se"], 6.0)
        self.assertAlmostEqual(paired["z"], -1.0)
        self.assertAlmostEqual(paired["p_exact"], 0.405, places=3)
        # The exact test must agree with the direction, not the old claim.
        self.assertGreater(paired["p_exact"], 0.05,
                           "002 was published as beyond noise; it is a tie")

    def test_a_significant_pair_stays_significant_but_weaker(self):
        """005 (23 vs 7) survives the correction -- it was not a fluke of it.

        The fix must not be a blanket "everything is noise" either: this pins
        a case where the honest SE still clears 5%.
        """
        paired = mcnemar_test(23, 7)
        self.assertAlmostEqual(paired["se"], 5.477, places=3)
        self.assertAlmostEqual(paired["z"], -2.921, places=3)
        self.assertLess(paired["p_exact"], 0.01)

    def test_no_discordant_rows_is_not_a_division_by_zero(self):
        """Unanimous arms have no paired evidence, so there is no z to report."""
        paired = mcnemar_test(0, 0)
        self.assertEqual(paired["discordant"], 0)
        self.assertEqual(paired["delta"], 0)
        self.assertIsNone(paired["z"])
        self.assertIsNone(paired["p_exact"])

    def test_the_stored_report_carries_the_corrected_standard_error(self):
        """The regenerated artifact, not just the function, must be fixed.

        Reading this from the committed report is the point: the bug lived in
        a *number on the page*, so a unit test that only calls the helper
        would have passed while the published table stayed wrong.
        """
        report = json.loads((Path(__file__).resolve().parents[1] /
                             "assets" / "votes" /
                             "equal-budget-002.json").read_text(encoding="utf-8"))
        paired = report["paired"]
        self.assertEqual(paired["a_right_b_wrong"], 21)
        self.assertEqual(paired["b_right_a_wrong"], 15)
        self.assertAlmostEqual(paired["se_delta"], 6.0)
        self.assertAlmostEqual(paired["delta_in_se"], -1.0)
        self.assertGreater(paired["mcnemar_exact_p"], 0.05)

    def test_every_stored_bracket_report_carries_the_corrected_standard_error(self):
        """002 was the report the 07:35 correction was written into.

        003/004/005 were regenerated in the same commit and nothing pinned
        them, so the next edit to any of the four could have put the halved SE
        back on a page with every test still green -- the failure the class
        exists for is a *published number*, so the assertion belongs on the
        page. Each report is re-derived from its own two discordant counts:
        the helper and the artifact must agree, and the direction must be the
        one the decision rule read.
        """
        votes_dir = Path(__file__).resolve().parents[1] / "assets" / "votes"
        expected = {
            # name: (a_right_b_wrong, b_right_a_wrong, delta_b_minus_a)
            "002": (21, 15, -6),
            "003": (22, 15, -7),
            "004": (14, 32, 18),
            "005": (23, 7, -16),
        }
        for name, (a_wrong, b_wrong, delta) in expected.items():
            with self.subTest(report=name):
                report = json.loads(
                    (votes_dir / f"equal-budget-{name}.json").read_text(
                        encoding="utf-8"))
                paired = report["paired"]
                self.assertEqual(paired["a_right_b_wrong"], a_wrong)
                self.assertEqual(paired["b_right_a_wrong"], b_wrong)
                self.assertEqual(paired["delta_b_minus_a"], delta)
                derived = mcnemar_test(a_wrong, b_wrong)
                self.assertEqual(derived["delta"], delta)
                self.assertAlmostEqual(paired["se_delta"], derived["se"],
                                       places=3)
                self.assertAlmostEqual(paired["delta_in_se"], derived["z"],
                                       places=3)
                self.assertAlmostEqual(paired["mcnemar_exact_p"],
                                       derived["p_exact"], places=6)


class TestBracketHoldsTheHeldArmStill(unittest.TestCase):
    """tools/equal_budget_check.py --reuse-arm-b - a bracket moves one arm.

    equal-budget-002 pre-registered a bracket arm (8B-Q4_K_M, 4.683 GiB) that
    sits on the *other* side of the committee's 5.112 GiB, so parity would be
    bracketed instead of asserted from a single 1.066x point estimate. Running
    that bracket means varying arm A and holding arm B to the votes of the run
    being bracketed. Re-harvesting the held arm would re-roll it -- the same
    configuration can answer differently on a second pass, and a timeout on one
    pass would then read as a difference between arms. So the held arm is
    adopted from a named file, and adoption is refused the moment that file
    answers a different question.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "equal_budget_check.py"
        spec = importlib.util.spec_from_file_location("equal_budget_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    SUITE = "assets/heldout4_test.jsonl,assets/heldout5_test.jsonl"
    URLS = ["http://b1", "http://b2", "http://b3", "http://b4"]

    def _donor(self, td, suite=None, n_rows=4, urls=None):
        path = Path(td) / "donor.raw.json"
        urls = self.URLS if urls is None else list(urls)
        path.write_text(json.dumps({
            "suite": self.SUITE if suite is None else suite,
            "n_rows": n_rows,
            "arm_a_url": "http://a-the-old-big-arm",
            "arm_b_urls": urls,
            "arm_a": {},
            "arm_b": {u: [{"ok": True, "task_id": f"big6-{i:04d}",
                           "expected": "a", "choice": "a"}
                          for i in range(n_rows)] for u in urls},
        }), encoding="utf-8")
        return path

    def test_a_matching_donor_is_adopted_verbatim(self):
        """The held arm must be the *same votes*, not an equivalent config."""
        eb = self._tool()
        with tempfile.TemporaryDirectory() as td:
            donor = self._donor(td)
            arm_b, reason = eb.adopt_arm_b(donor, self.SUITE, 4, self.URLS)
        self.assertEqual(reason, "")
        self.assertEqual(sorted(arm_b), sorted(self.URLS))
        self.assertEqual(len(arm_b["http://b1"]), 4)

    def test_a_donor_that_answered_a_different_question_is_refused(self):
        """A longer suite or another member set is a different experiment.

        Adopting one of these would silently compare arm A against votes from a
        question arm A never saw, which is the failure this refusal exists to
        prevent -- and it must be a refusal, not a quiet fallback to a fresh
        arm B, because a fresh arm B is not a bracket.
        """
        eb = self._tool()
        with tempfile.TemporaryDirectory() as td:
            short = self._donor(td, n_rows=3)
            arm_b, reason = eb.adopt_arm_b(short, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("n_rows 3 != 4", reason)

            other = self._donor(td, suite="assets/heldout5_test.jsonl")
            arm_b, reason = eb.adopt_arm_b(other, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("suite", reason)

            fewer = self._donor(td, urls=self.URLS[:3])
            arm_b, reason = eb.adopt_arm_b(fewer, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("arm_b_urls differ", reason)

    def test_a_missing_or_unreadable_donor_is_refused_not_improvised(self):
        """No donor means no bracket. Saying so beats a plausible number."""
        eb = self._tool()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "absent.json"
            arm_b, reason = eb.adopt_arm_b(missing, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("does not exist", reason)

            broken = Path(td) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            arm_b, reason = eb.adopt_arm_b(broken, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("not JSON", reason)

            empty = Path(td) / "empty.json"
            empty.write_text(json.dumps({"suite": self.SUITE, "n_rows": 4,
                                         "arm_b_urls": self.URLS}), encoding="utf-8")
            arm_b, reason = eb.adopt_arm_b(empty, self.SUITE, 4, self.URLS)
            self.assertIsNone(arm_b)
            self.assertIn("no arm_b", reason)


class TestVoteVerdictOnThePage(unittest.TestCase):
    """The page reads the judgement; it must never re-derive it.

    ``CHAT_VIEW_LIMIT`` is 120 messages -- about seventeen minutes of feed --
    while ``vote_watchdog`` fits the same question over the whole mailbox. Two
    fitters, one column name, and the disagreement lands on the reassuring
    side: fewer samples means ``unknown``, and ``unknown`` reads like "no
    alarm". So the judgement has exactly one owner, it publishes on every pass
    including the all-clear ones, and the page renders a *missing* judgement as
    missing rather than as a clean column.
    """

    def _obs(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "observatory.py"
        spec = importlib.util.spec_from_file_location(
            "observatory_verdict_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _wd(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "vote_watchdog.py"
        spec = importlib.util.spec_from_file_location(
            "vote_watchdog_verdict_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _write(self, path, *, written, members, decision=()):
        path.write_text(json.dumps({
            "written": written, "written_at": "2026-10-19 03:00:00",
            "window_s": 5400.0, "decision": list(decision), "members": members,
        }, ensure_ascii=False), encoding="utf-8")

    # --- what the watchdog publishes ---------------------------------

    def test_an_all_clear_pass_is_published_too(self):
        """A file that only appears when something breaks cannot be told apart
        from a watchdog that stopped running."""
        wd = self._wd()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "vote-watchdog.json"
            wd.write_snapshot(out, {"mac-text": {"status": wd.OK, "slope": -0.6}},
                              1_800_000_000.0, ["mac-text"], 5400.0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], ["mac-text"])
            self.assertEqual(payload["members"]["mac-text"]["status"], wd.OK)
            self.assertEqual(payload["written"], 1_800_000_000.0)

    def test_the_snapshot_and_the_state_file_remember_opposite_things(self):
        """The state file forgets a recovered member so a second wedge pages
        again; the snapshot has to go on saying ``ok``, because that is the
        reading. Same member, same recovery, two files, two rules."""
        wd = self._wd()
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            snap = Path(tmp) / "snap.json"
            wedged = {"m": {"status": wd.WEDGED, "slope": 0.5}}
            wd.track_first_seen(state, wedged, 1000.0)
            wd.write_snapshot(snap, wedged, 1000.0, [], 5400.0)
            healed = {"m": {"status": wd.OK, "slope": -0.1}}
            wd.track_first_seen(state, healed, 1060.0)
            wd.write_snapshot(snap, healed, 1060.0, [], 5400.0)
            self.assertEqual(json.loads(state.read_text())["members"], {})
            self.assertEqual(
                json.loads(snap.read_text())["members"]["m"]["status"], wd.OK)

    def test_no_snapshot_path_publishes_nothing_and_does_not_raise(self):
        wd = self._wd()
        wd.write_snapshot(None, {"m": {"status": wd.OK}}, 1000.0, [], 5400.0)

    # --- what the page does with it ----------------------------------

    def test_a_judgement_nobody_wrote_is_unavailable_not_clean(self):
        obs = self._obs()
        with tempfile.TemporaryDirectory() as tmp:
            verdict = obs.read_vote_verdict(Path(tmp) / "nothing.json")
            self.assertFalse(verdict["available"])
            self.assertEqual(verdict["error"], "never written")
            self.assertIn("不可用", obs.verdict_line(verdict))

    def test_a_judgement_that_stopped_arriving_is_stale(self):
        """The watchdog runs every 300s; three missed passes is a stopped
        judge, and a stopped judge must not read as an all-clear."""
        obs = self._obs()
        now = 1_800_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            self._write(path, written=now - 60, members={"m": {"status": "ok"}})
            fresh = obs.read_vote_verdict(path, now)
            self.assertTrue(fresh["available"])
            self.assertFalse(fresh["stale"])
            self._write(path, written=now - 5090, members={"m": {"status": "ok"}})
            old = obs.read_vote_verdict(path, now)
            self.assertTrue(old["stale"])
            self.assertGreater(old["age_s"], obs.VOTE_VERDICT_STALE_S)

    def test_a_corrupt_judgement_is_unavailable_not_a_guess(self):
        obs = self._obs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            path.write_text("{not json", encoding="utf-8")
            verdict = obs.read_vote_verdict(path)
            self.assertFalse(verdict["available"])
            self.assertIn("JSONDecodeError", verdict["error"])

    def test_a_decision_member_with_no_judgement_is_a_gap_not_a_dash(self):
        obs = self._obs()
        cell = obs._verdict_cell(None, "decision")
        self.assertEqual(cell["verdict"], "无判")
        self.assertTrue(cell["verdict_bad"])
        self.assertEqual(obs._verdict_cell(None, "proposal")["verdict"], "—")

    def test_a_proposer_is_not_badged_unknown_for_not_voting(self):
        """An LLM casts no vote by design. A permanent "未知" on every proposer
        would put a meaningless badge on the page and train the reader to
        ignore the column; a *decision* member that could not be measured is
        the opposite, and keeps its label."""
        obs = self._obs()
        cell = obs._verdict_cell({"status": "unknown"}, "proposal")
        self.assertEqual(cell["verdict"], "—")
        self.assertFalse(cell["verdict_bad"])
        judged = obs._verdict_cell({"status": "unknown"}, "decision")
        self.assertEqual(judged["verdict"], "未知")

    def test_a_wedge_and_a_silence_are_both_alarms(self):
        obs = self._obs()
        wedge = obs._verdict_cell({"status": "wedge"}, "decision")
        self.assertEqual(wedge["verdict"], "落后")
        self.assertTrue(wedge["verdict_bad"])
        self.assertTrue(obs._verdict_cell({"status": "silent"}, "decision")["verdict_bad"])
        self.assertFalse(obs._verdict_cell({"status": "ok"}, "decision")["verdict_bad"])

    def test_the_audit_catches_a_row_whose_verdict_is_blank(self):
        obs = self._obs()
        view = {"people": [{"name": "win-laya-int8", "from": "win",
                            "host": "192.168.31.100", "model": "laya-en",
                            "role": "决策", "record": "票 3/0",
                            "fresh": "35.8 小时", "verdict": ""}],
                "messages": []}
        self.assertTrue(any("verdict is blank" in p
                            for p in obs.audit_view(view)))
        view["people"][0]["verdict"] = "—"
        self.assertTrue(any("decision member has no verdict" in p
                            for p in obs.audit_view(view)))

    def test_a_short_feed_cannot_answer_under_the_same_column_name(self):
        """The drift this design exists to prevent, as a fixture.

        Same member, same moment, two window lengths. The short window -- what
        the page could see if it judged for itself -- has too few samples and
        says ``unknown``, which renders as no alarm. The published judgement,
        fitted over the whole feed, says ``wedge``. One column name must not
        hold both answers, so the page keeps the one the watchdog published.
        """
        obs, wd = self._obs(), self._wd()
        now = 1_800_000_000.0
        frozen = int(now - 1800)
        full = [(now - 5400 + i * 300, (now - 5400 + i * 300) - frozen)
                for i in range(18)]
        self.assertEqual(wd.member_slope(full, now)["status"], wd.WEDGED)
        self.assertEqual(wd.member_slope(full[-3:], now)["status"], wd.UNKNOWN)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            self._write(path, written=now - 30,
                        members={"win-laya-int8": wd.member_slope(full, now)})
            verdict = obs.read_vote_verdict(path, now)
            cell = obs._verdict_cell(verdict["members"]["win-laya-int8"], "decision")
            self.assertEqual(cell["verdict"], "落后")
            self.assertTrue(cell["verdict_bad"])


class TestHistoricalActionReplay(unittest.TestCase):
    """tools/historical_action_replay.py - frozen real-mailbox replay."""

    def _tool(self):
        import importlib.util
        path = (Path(__file__).resolve().parents[1]
                / "tools" / "historical_action_replay.py")
        spec = importlib.util.spec_from_file_location(
            "historical_action_replay", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _write_mailbox(self, root: Path):
        import json
        tool = self._tool()
        tasks = root / "tasks"
        tasks.mkdir()
        duplicate = "同一份跨主机共享目标，必须全局去重，不能带两个标签。"
        for action_index, action in enumerate(tool.ACTIONS):
            for index in range(10):
                author = "author-a" if index % 2 == 0 else "author-b"
                goal = duplicate if index == 0 and action_index > 0 else (
                    f"第{action_index}类动作的第{index}个真实历史目标，"
                    "描述跨主机资源与治理选择。")
                created = ("2026-10-28T12:00:00" if index < 9
                           else "2026-10-29T00:00:00")
                task_id = f"t-{action_index:02d}-{index:02d}"
                body = json.dumps({
                    "kind": "proposal",
                    "payload": {"action": action, "goal": goal},
                }, ensure_ascii=False)
                record = {
                    "id": task_id,
                    "from": author,
                    "title": f"[proposal] {task_id} {action} by {author}",
                    "body": body,
                    "created_at": created,
                }
                (tasks / f"{task_id}.json").write_text(
                    json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return tasks, tool

    def test_selection_is_balanced_deduplicated_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, tool = self._write_mailbox(Path(tmp))
            manifest = tool.select_historical(tasks)
            self.assertEqual(len(manifest["items"]), tool.TOTAL_SAMPLES)
            self.assertEqual(
                {item["action"] for item in manifest["items"]},
                set(tool.ACTIONS))
            self.assertEqual(
                {sum(item["action"] == action
                     for item in manifest["items"])
                 for action in tool.ACTIONS},
                {tool.PER_ACTION})
            self.assertTrue(all(
                item["created_at"] <= tool.SOURCE_CUTOFF
                for item in manifest["items"]))
            self.assertEqual(
                len({item["goal"] for item in manifest["items"]}),
                tool.TOTAL_SAMPLES)
            self.assertTrue(tool.audit_manifest(manifest, tasks)["ok"])

            manifest["items"][0]["goal"] += "篡改"
            audit = tool.audit_manifest(manifest, tasks)
            self.assertFalse(audit["ok"])
            self.assertFalse(audit["checks"]["manifest_digest"])
            self.assertFalse(audit["checks"]["source_match"])

    def test_summary_requires_both_thresholds_for_support(self):
        tool = self._tool()

        def rows(n, opaque_correct):
            out = []
            for index in range(n):
                out.append({
                    "pair": index, "arm": "named", "valid": True,
                    "correct": True, "author": "a",
                    "task_id": f"t{index}",
                })
                out.append({
                    "pair": index, "arm": "opaque", "valid": True,
                    "correct": index < opaque_correct, "author": "a",
                    "task_id": f"t{index}",
                })
            return out

        supported = tool.summarise(rows(64, 64), 64)
        self.assertEqual(supported["outcome"],
                         "historical_label_agreement_supported")
        self.assertFalse(supported["named_label_effect"])

        ambiguous = tool.summarise(rows(64, 47), 64)
        self.assertEqual(ambiguous["arms"]["opaque"]["accuracy"], 47 / 64)
        self.assertEqual(ambiguous["outcome"], "inconclusive")

        named_effect = tool.summarise(rows(64, 0), 64)
        self.assertTrue(named_effect["named_label_effect"])
        self.assertEqual(named_effect["outcome"], "inconclusive")


