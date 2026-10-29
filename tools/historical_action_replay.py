#!/usr/bin/env python3
"""Replay real historical proposal goals against the current action menu.

The synthetic action-semantics checks are useful mechanism probes, but their
goals were written by the same lineage that ran the experiment.  This tool
adds a source-held-out check: it mechanically samples signed proposal goals
from the colony mailbox, freezes those rows, and asks the current local model
to recover the action label each historical proposal carried.

This is *label agreement*, not expert correctness.  Historical labels came
from proposer models, so a high score can mean stable rule use; it cannot make
the historical label a human gold standard.  The named arm estimates how much
the action identifier itself helps; the opaque arm is the primary endpoint.

The selection rule, sample size, cutoff, and outcome rule are constants below.
Run ``prepare`` and commit the generated manifest before running ``run`` so the
source rows cannot move after model outputs are observed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from pando.thinkd import OpenAIProposer

ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = Path(__file__).with_name("action_semantics_check.py")
_spec = importlib.util.spec_from_file_location(
    "action_semantics_check_for_history", SEMANTICS_PATH)
SEM = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(SEM)

ACTIONS = tuple(SEM.ACTION_IDS)
SOURCE_CUTOFF = "2026-10-28T23:59:59"
PER_ACTION = 8
TOTAL_SAMPLES = PER_ACTION * len(ACTIONS)
MIN_GOAL_CHARS = 20
MANIFEST_SCHEMA = "historical-action-replay-manifest-v1"
ARTIFACT_SCHEMA = "historical-action-replay-artifact-v1"
SELECTION_RULE = (
    "parse mailbox proposal payloads through cutoff; keep current eight actions "
    "with non-empty normalized goals of at least 20 characters; deduplicate "
    "exactly equal normalized goals globally; rank each action's rows by SHA-256 of "
    "task_id + NUL + normalized goal; round-robin across authors (author order "
    "also SHA-256 ranked); take eight distinct task rows per action"
)
PREREGISTRATION = {
    "schema": "historical-action-replay-preregistration-v1",
    "source": "signed proposal payload.goal values in ~/.pandosi/mailbox/tasks",
    "source_cutoff": SOURCE_CUTOFF,
    "selection_rule": SELECTION_RULE,
    "per_action": PER_ACTION,
    "total_samples": TOTAL_SAMPLES,
    "primary_arm": "opaque",
    "control_arm": "named",
    "prompt_control": (
        "same goal, menu order, model seed, and token count in both arms; only "
        "the full action identifiers are replaced by per-item opaque labels"
    ),
    "outcome_rule": {
        "historical_label_agreement_supported_if": {
            "minimum_opaque_accuracy": 0.75,
            "wilson_95_lower_bound_above": 0.60,
        },
        "named_label_effect_if": {
            "minimum_named_minus_opaque_accuracy": 0.15,
            "directional_exact_mcnemar_alpha": 0.05,
        },
        "otherwise": "inconclusive",
    },
    "claim_boundary": (
        "agreement with historical proposer labels, not correctness, semantic "
        "understanding, generalization to other models, or collective intelligence"
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _normalise_goal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _stable_rank(*parts: str) -> str:
    return _sha256_text("\0".join(parts))


def _load_tasks(tasks_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            raw = path.read_bytes()
            task = json.loads(raw)
        except (OSError, ValueError):
            continue
        title = str(task.get("title") or "")
        if not title.startswith("[proposal]"):
            continue
        try:
            body = json.loads(str(task.get("body") or "{}"))
            payload = body.get("payload") or {}
        except (TypeError, ValueError):
            continue
        action = str(payload.get("action") or "")
        goal = _normalise_goal(str(payload.get("goal") or ""))
        created_at = str(task.get("created_at") or "")
        if action not in ACTIONS or len(goal) < MIN_GOAL_CHARS:
            continue
        if created_at > SOURCE_CUTOFF:
            continue
        rows.append({
            "task_id": str(task.get("id") or path.stem),
            "action": action,
            "goal": goal,
            "author": str(task.get("from") or ""),
            "created_at": created_at,
            "task_sha256": _sha256_bytes(raw),
            "goal_sha256": _sha256_text(goal),
            "source_file": path.name,
        })
    return rows


def select_historical(tasks_dir: Path) -> dict:
    """Apply the frozen, deterministic source selection rule."""
    candidates = _load_tasks(tasks_dir)
    selected = []
    used_goals = set()
    for action in ACTIONS:
        # One representative per exact-normalized goal.  A goal used by another
        # action is excluded globally because its gold label would be ambiguous.
        by_goal = {}
        for row in sorted(candidates, key=lambda item: (
                item["created_at"], item["task_id"])):
            if row["action"] != action or row["goal"] in used_goals:
                continue
            by_goal.setdefault(row["goal"], row)
        groups = defaultdict(list)
        for row in by_goal.values():
            groups[row["author"]].append(row)
        author_order = sorted(
            groups, key=lambda author: _stable_rank("author", action, author))
        for author in author_order:
            groups[author].sort(key=lambda row: _stable_rank(
                "goal", row["task_id"], row["goal"]))
        cursors = {author: 0 for author in author_order}
        picked = 0
        while picked < PER_ACTION:
            advanced = False
            for author in author_order:
                cursor = cursors[author]
                while cursor < len(groups[author]):
                    row = groups[author][cursor]
                    cursor += 1
                    if row["goal"] in used_goals:
                        continue
                    cursors[author] = cursor
                    selected.append(row)
                    used_goals.add(row["goal"])
                    picked += 1
                    advanced = True
                    break
                if picked >= PER_ACTION:
                    break
            if not advanced:
                break
    items = selected
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_cutoff": SOURCE_CUTOFF,
        "selection_rule": SELECTION_RULE,
        "per_action": PER_ACTION,
        "total_samples": len(items),
        "items": items,
    }
    manifest["manifest_sha256"] = _sha256_text(
        json.dumps(items, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")))
    return manifest


def audit_manifest(manifest: dict, tasks_dir: Path,
                   require_source_match: bool = True) -> dict:
    items = manifest.get("items") or []
    ids = [str(item.get("task_id") or "") for item in items]
    goals = [str(item.get("goal") or "") for item in items]
    counts = Counter(str(item.get("action") or "") for item in items)
    digest_ok = manifest.get("manifest_sha256") == _sha256_text(
        json.dumps(items, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")))
    source_match = None
    if require_source_match:
        try:
            expected = select_historical(tasks_dir)
            source_match = expected.get("items") == items
        except OSError:
            source_match = False
    checks = {
        "schema": manifest.get("schema") == MANIFEST_SCHEMA,
        "cutoff": manifest.get("source_cutoff") == SOURCE_CUTOFF,
        "sample_count": len(items) == TOTAL_SAMPLES,
        "per_action_counts": all(counts.get(action) == PER_ACTION
                                 for action in ACTIONS),
        "unique_task_ids": len(ids) == len(set(ids)) and all(ids),
        "unique_goals": len(goals) == len(set(goals)) and all(goals),
        "hashes_present": all(
            len(str(item.get("task_sha256") or "")) == 64
            and len(str(item.get("goal_sha256") or "")) == 64
            for item in items),
        "after_cutoff_absent": all(
            str(item.get("created_at") or "") <= SOURCE_CUTOFF
            for item in items),
        "manifest_digest": digest_ok,
    }
    if source_match is not None:
        checks["source_match"] = source_match
    return {"ok": all(checks.values()), "checks": checks,
            "action_counts": dict(sorted(counts.items()))}


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_item_prompts(item: dict, index: int,
                          tokenize: Callable[[str], int]) -> dict:
    seed = 5100 + index
    order, named_labels = SEM.menu_for_pair(index, seed, opaque=False)
    _, opaque_labels = SEM.menu_for_pair(index, seed, opaque=True)
    labels = {"named": named_labels, "opaque": opaque_labels}
    builds = {
        arm: (lambda pad, label_of=labels[arm]:
              SEM.build_prompt(item["goal"], order, label_of, pad))
        for arm in ("named", "opaque")
    }
    raw = {arm: builds[arm]("") for arm in ("named", "opaque")}
    raw_tokens = {arm: tokenize(raw[arm]) for arm in ("named", "opaque")}
    target_tokens = max(raw_tokens.values())
    built = {}
    for arm in ("named", "opaque"):
        text, count, pad = SEM.pad_to_tokens(
            builds[arm], target_tokens, tokenize)
        built[arm] = {
            "prompt": text,
            "prompt_sha256": _sha256_text(text),
            "tokens": count,
            "raw_tokens": raw_tokens[arm],
            "pad": pad,
            "order": order,
            "label_to_action": {
                label: action for action, label in labels[arm].items()},
        }
    return {
        "item_index": index,
        "item": item,
        "model_seed": seed,
        "target_tokens": target_tokens,
        "arms": built,
    }


def _tokenizer(base_url: str, api_key: str) -> Callable[[str], int]:
    url = SEM._default_tokenize_url(base_url)  # noqa: SLF001
    return lambda text: SEM.tokenize_text(text, url, api_key)


def run_replay(manifest: dict, *, base_url: str, model: str,
               api_key: str, temperature: float, max_tokens: int,
               timeout: float, seed_base: int = 5100) -> dict:
    items = manifest.get("items") or []
    proposer = OpenAIProposer(
        base_url, model, api_key=api_key, max_tokens=max_tokens,
        temperature=temperature, timeout=timeout)
    tokenize = _tokenizer(base_url, api_key)
    rows = []
    for index, item in enumerate(items):
        prepared = _prepare_item_prompts(item, index, tokenize)
        arms = ("named", "opaque") if index % 2 == 0 else ("opaque", "named")
        for position, arm in enumerate(arms):
            prompt = prepared["arms"][arm]
            text, usage, latency_ms = SEM.sample_once(
                proposer, prompt["prompt"], seed_base + index,
                temperature, max_tokens)
            selected, label, parsed = SEM.parse_selection(
                text, prompt["label_to_action"])
            row = {
                "pair": index,
                "arm": arm,
                "position": position,
                "task_id": item["task_id"],
                "expected_action": item["action"],
                "author": item["author"],
                "selected_action": selected,
                "selected_label": label,
                "valid": selected != "(invalid)",
                "correct": selected == item["action"],
                "raw_response": text,
                "parsed": parsed,
                "usage": usage,
                "latency_ms": latency_ms,
                "model_seed": seed_base + index,
                "target_tokens": prepared["target_tokens"],
                "prompt_tokens": prompt["tokens"],
                "raw_tokens": prompt["raw_tokens"],
                "pad": prompt["pad"],
                "prompt_sha256": prompt["prompt_sha256"],
                "prompt": prompt["prompt"],
                "order": prompt["order"],
                "label_to_action": prompt["label_to_action"],
            }
            rows.append(row)
            print(f"{index + 1:02d}/{len(items)} {arm:6s} "
                  f"{selected:20s} expected={item['action']}", flush=True)
    summary = summarise(rows, len(items))
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "preregistration": PREREGISTRATION,
        "manifest": manifest,
        "run": {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed_base": seed_base,
            "sample_calls": len(rows),
        },
        "rows": rows,
        "summary": summary,
    }
    artifact["artifact_checks"] = audit_artifact(artifact, require_summary=True)["checks"]
    return artifact


def _wilson(correct: int, n: int) -> list[float]:
    return list(SEM.wilson_interval(correct, n))


def _arm_summary(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row.get("arm") == arm]
    n = len(selected)
    valid = sum(bool(row.get("valid")) for row in selected)
    correct = sum(bool(row.get("correct")) for row in selected)
    accuracy = correct / n if n else 0.0
    se = math.sqrt(accuracy * (1.0 - accuracy) / n) if n else 0.0
    by_action = {}
    for action in ACTIONS:
        values = [row for row in selected if row.get("expected_action") == action]
        hits = sum(bool(row.get("correct")) for row in values)
        by_action[action] = {
            "n": len(values),
            "correct": hits,
            "accuracy": hits / len(values) if values else 0.0,
        }
    return {
        "n": n,
        "valid": valid,
        "invalid": n - valid,
        "correct": correct,
        "accuracy": accuracy,
        "standard_error": se,
        "wilson_95": _wilson(correct, n),
        "chance_accuracy": 1.0 / len(ACTIONS),
        "chance_two_sided_p": SEM.binomial_two_sided(
            correct, n, 1.0 / len(ACTIONS)),
        "selected_actions": dict(Counter(
            row.get("selected_action") for row in selected)),
        "by_action": by_action,
    }


def summarise(rows: list[dict], samples: int) -> dict:
    arms = {arm: _arm_summary(rows, arm) for arm in ("named", "opaque")}
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[int(row["pair"])][str(row["arm"])] = row
    counts = Counter()
    by_author = defaultdict(lambda: {"n": 0, "correct": 0})
    for index in range(samples):
        named = by_pair[index].get("named") or {}
        opaque = by_pair[index].get("opaque") or {}
        if not named.get("valid") or not opaque.get("valid"):
            counts["invalid"] += 1
            continue
        left = bool(named.get("correct"))
        right = bool(opaque.get("correct"))
        if left and right:
            counts["both_correct"] += 1
        elif left:
            counts["named_only"] += 1
        elif right:
            counts["opaque_only"] += 1
        else:
            counts["neither_correct"] += 1
        sample_author = str(named.get("author") or opaque.get("author") or "")
        if sample_author:
            bucket = by_author[sample_author]
            bucket["n"] += 1
            bucket["correct"] += int(right)
    comparable = sum(counts[key] for key in (
        "both_correct", "named_only", "opaque_only", "neither_correct"))
    opaque_acc = arms["opaque"]["accuracy"]
    named_acc = arms["named"]["accuracy"]
    lower = arms["opaque"]["wilson_95"][0]
    rule = PREREGISTRATION["outcome_rule"]
    support = rule["historical_label_agreement_supported_if"]
    if (opaque_acc >= support["minimum_opaque_accuracy"]
            and lower > support["wilson_95_lower_bound_above"]):
        outcome = "historical_label_agreement_supported"
    else:
        outcome = "inconclusive"
    named_effect_rule = rule["named_label_effect_if"]
    named_effect = (
        named_acc - opaque_acc >= named_effect_rule[
            "minimum_named_minus_opaque_accuracy"]
        and SEM.binomial_one_sided_at_least(
            counts["named_only"],
            counts["named_only"] + counts["opaque_only"], 0.5)
        < named_effect_rule["directional_exact_mcnemar_alpha"])
    return {
        "samples": samples,
        "arms": arms,
        "paired": {
            "comparable": comparable,
            "discordant": counts["named_only"] + counts["opaque_only"],
            **dict(counts),
            "named_minus_opaque_accuracy": named_acc - opaque_acc,
            "named_better_exact_one_sided_p": SEM.binomial_one_sided_at_least(
                counts["named_only"],
                counts["named_only"] + counts["opaque_only"], 0.5),
            "mcnemar_exact_two_sided_p": SEM.binomial_two_sided(
                counts["named_only"],
                counts["named_only"] + counts["opaque_only"], 0.5),
        },
        "named_label_effect": bool(named_effect),
        "outcome": outcome,
        "outcome_rule": rule,
        "by_author": {
            author: {**values,
                     "accuracy": values["correct"] / values["n"]
                     if values["n"] else 0.0}
            for author, values in sorted(by_author.items())
        },
    }


def audit_artifact(artifact: dict, require_summary: bool = False) -> dict:
    rows = artifact.get("rows") or []
    manifest = artifact.get("manifest") or {}
    items = manifest.get("items") or []
    row_by_key = {}
    duplicate = False
    for row in rows:
        key = (int(row.get("pair", -1)), str(row.get("arm") or ""))
        if key in row_by_key:
            duplicate = True
        row_by_key[key] = row
    expected_keys = {
        (index, arm) for index in range(len(items)) for arm in ("named", "opaque")
    }
    prompt_hashes = all(
        row.get("prompt_sha256") == _sha256_text(str(row.get("prompt") or ""))
        for row in rows)
    pair_integrity = all(
        str((row_by_key.get((index, "named")) or {}).get("task_id") or "")
        == str(item.get("task_id") or "")
        == str((row_by_key.get((index, "opaque")) or {}).get("task_id") or "")
        for index, item in enumerate(items))
    manifest_audit = audit_manifest(manifest, Path(""), require_source_match=False)
    checks = {
        "schema": artifact.get("schema") == ARTIFACT_SCHEMA,
        "preregistration": (
            artifact.get("preregistration") or {}).get("schema")
            == PREREGISTRATION["schema"],
        "manifest_valid": manifest_audit["ok"],
        "row_count": len(rows) == len(items) * 2,
        "no_duplicate_rows": not duplicate,
        "complete_arm_pairs": set(row_by_key) == expected_keys,
        "prompt_hashes": prompt_hashes,
        "same_task_within_pair": pair_integrity,
        "token_match": all(
            (row_by_key.get((index, "named")) or {}).get("prompt_tokens")
            == (row_by_key.get((index, "opaque")) or {}).get("prompt_tokens")
            for index in range(len(items))),
    }
    if require_summary:
        expected = summarise(rows, len(items))
        checks["summary_matches"] = (
            expected.get("outcome") == (artifact.get("summary") or {}).get("outcome")
            and expected["arms"]["opaque"]["correct"]
            == (artifact.get("summary") or {}).get("arms", {}).get(
                "opaque", {}).get("correct"))
    return {"ok": all(checks.values()), "checks": checks}


def print_report(summary: dict) -> None:
    print(f"\nhistorical replay: samples={summary['samples']} "
          f"outcome={summary['outcome']}")
    for arm in ("named", "opaque"):
        values = summary["arms"][arm]
        low, high = values["wilson_95"]
        print(f"{arm:6s}: {values['correct']}/{values['n']} = "
              f"{values['accuracy']:.4f} "
              f"(SE {values['standard_error']:.4f}, "
              f"Wilson95 [{low:.4f}, {high:.4f}])")
    paired = summary["paired"]
    print(f"paired: named-only={paired['named_only']} "
          f"opaque-only={paired['opaque_only']} "
          f"both={paired['both_correct']} neither={paired['neither_correct']}")
    print("opaque by action:")
    for action, values in summary["arms"]["opaque"]["by_action"].items():
        print(f"  {action:20s} {values['correct']}/{values['n']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir",
                        default=str(Path.home() / ".pandosi/mailbox/tasks"))
    parser.add_argument("--source-manifest",
                        default=str(ROOT / "assets" /
                                    "historical-action-heldout-20261029.json"))
    parser.add_argument("--output", default="")
    parser.add_argument("--audit-artifact", default="")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="qwen3-8b-q5km-instruct")
    parser.add_argument("--api-key-file", default="~/.pandosi/llm-api-8b-q6.token")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed-base", type=int, default=5100)
    args = parser.parse_args(argv)
    if not any((args.prepare, args.audit_source, args.run,
                args.audit_artifact)):
        parser.error("choose --prepare, --audit-source, --run, or --audit-artifact")
    tasks_dir = Path(args.tasks_dir).expanduser()
    manifest_path = Path(args.source_manifest).expanduser()
    if args.prepare:
        manifest = select_historical(tasks_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        audit = audit_manifest(manifest, tasks_dir)
        print(f"wrote {manifest_path}: {len(manifest['items'])} items; "
              f"audit={'PASS' if audit['ok'] else 'FAIL'}")
        if not audit["ok"]:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 1
    if args.audit_source:
        audit = audit_manifest(_load_manifest(manifest_path), tasks_dir)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 1
    if args.audit_artifact:
        artifact = _load_manifest(Path(args.audit_artifact).expanduser())
        audit = audit_artifact(artifact)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 1
    manifest = _load_manifest(manifest_path)
    source_audit = audit_manifest(manifest, tasks_dir)
    if not source_audit["ok"]:
        print(json.dumps(source_audit, ensure_ascii=False, indent=2))
        return 1
    token_path = Path(args.api_key_file).expanduser()
    api_key = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
    artifact = run_replay(
        manifest, base_url=args.base_url, model=args.model, api_key=api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        timeout=args.timeout, seed_base=args.seed_base)
    print_report(artifact["summary"])
    audit = audit_artifact(artifact, require_summary=True)
    print(f"artifact audit: {'PASS' if audit['ok'] else 'FAIL'}")
    if not audit["ok"]:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {output}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
