#!/usr/bin/env python3
"""Blind adjudication package for the frozen historical-action replay.

The replay found 22 pairs where both model arms disagreed with the historical
proposer label.  Those labels are not a human gold standard.  This tool builds
a reviewer packet from only the goal and eight shuffled, unnamed action
descriptions, keeps the hidden answer key outside the repository, and scores a
completed response against a preregistered rule.

The packet deliberately contains no task ids, authors, old labels, model
answers, raw responses, or action identifiers.  An independent reviewer must
see only the packet.  The key and replay artifact must stay hidden until the
response is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = Path(__file__).with_name("action_semantics_check.py")
HISTORICAL_PATH = Path(__file__).with_name("historical_action_replay.py")
EXPERIMENTS = Path.home() / ".pandosi" / "experiments"
DEFAULT_ARTIFACT = EXPERIMENTS / "historical-action-replay-20261029.json"
DEFAULT_PACKET = (ROOT / "assets" /
                  "historical-action-blind-review-20261029.packet.json")
DEFAULT_KEY = EXPERIMENTS / "historical-action-blind-review-20261029.key.json"
DEFAULT_PREREG = ROOT / "assets" / "historical-action-blind-review-20261029-prereg.json"
PACKET_SCHEMA = "historical-action-blind-review-packet-v1"
KEY_SCHEMA = "historical-action-blind-review-key-v1"
PREREG_SCHEMA = "historical-action-blind-review-preregistration-v1"
RESPONSE_SCHEMA = "historical-action-blind-review-response-v1"
RESULT_SCHEMA = "historical-action-blind-review-result-v1"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEM = _load_module("action_semantics_for_blind_review", SEMANTICS_PATH)
HIST = _load_module("historical_replay_for_blind_review", HISTORICAL_PATH)
ACTIONS = tuple(SEM.ACTION_IDS)


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_json(value) -> str:
    return _sha_text(_canonical_json(value))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def _stable_rank(*parts: str) -> str:
    return _sha_text("\0".join(parts))


def _blind_goal(goal: str) -> str:
    """Remove provenance references that historical proposers quoted in goals."""
    blinded = goal
    for action in ACTIONS:
        blinded = blinded.replace(action, "该动作")
    blinded = re.sub(r"\bt-\d{8}-\d{6}-[0-9a-f]{4}\b", "该任务", blinded)
    blinded = re.sub(r"\br\d{3,}\b", "该轮次", blinded)
    return blinded


def _both_wrong_rows(artifact: dict) -> list[dict]:
    """Return one pair record for every valid-but-wrong two-arm result."""
    items = (artifact.get("manifest") or {}).get("items") or []
    by_pair: dict[int, dict] = {}
    for row in artifact.get("rows") or []:
        pair = int(row.get("pair", -1))
        arm = str(row.get("arm") or "")
        if arm not in ("named", "opaque"):
            continue
        if arm in by_pair.setdefault(pair, {}):
            raise ValueError(f"duplicate {arm} row for pair {pair}")
        by_pair[pair][arm] = row
    selected = []
    for pair, item in enumerate(items):
        arms = by_pair.get(pair) or {}
        named = arms.get("named") or {}
        opaque = arms.get("opaque") or {}
        task_id = str(item.get("task_id") or "")
        expected = str(item.get("action") or "")
        if (not task_id or expected not in ACTIONS
                or not named or not opaque):
            raise ValueError(f"incomplete replay pair {pair}")
        if (str(named.get("task_id") or "") != task_id
                or str(opaque.get("task_id") or "") != task_id):
            raise ValueError(f"task mismatch within replay pair {pair}")
        if (str(named.get("expected_action") or "") != expected
                or str(opaque.get("expected_action") or "") != expected):
            raise ValueError(f"expected-action mismatch within replay pair {pair}")
        if not named.get("valid") or not opaque.get("valid"):
            continue
        if named.get("correct") or opaque.get("correct"):
            continue
        named_action = str(named.get("selected_action") or "")
        opaque_action = str(opaque.get("selected_action") or "")
        if named_action not in ACTIONS or opaque_action not in ACTIONS:
            raise ValueError(f"invalid selected action within replay pair {pair}")
        selected.append({
            "pair": pair,
            "task_id": task_id,
            "goal": str(item.get("goal") or ""),
            "historical_action": expected,
            "named_model_action": named_action,
            "opaque_model_action": opaque_action,
        })
    if not selected:
        raise ValueError("replay artifact has no valid both-wrong pairs")
    summary = artifact.get("summary") or {}
    paired = summary.get("paired") or {}
    if "neither_correct" in paired and int(paired["neither_correct"]) != len(selected):
        raise ValueError("both-wrong count disagrees with replay summary")
    return selected


def _preregistration(artifact: dict, artifact_sha256: str,
                     selected: list[dict]) -> dict:
    summary = artifact.get("summary") or {}
    paired = summary.get("paired") or {}
    return {
        "schema": PREREG_SCHEMA,
        "source_artifact": {
            "sha256": artifact_sha256,
            "schema": str(artifact.get("schema") or ""),
            "samples": int(summary.get("samples") or 0),
            "both_wrong_pairs": len(selected),
            "replay_named_only": int(paired.get("named_only") or 0),
            "replay_opaque_only": int(paired.get("opaque_only") or 0),
        },
        "selection_rule": (
            "include every replay pair for which both arms are valid and both "
            "are incorrect; preserve all such pairs without further filtering"
        ),
        "review_unit": (
            "one goal and the same eight action descriptions per pair; choice "
            "letters are independently shuffled and mapped only in the hidden key"
        ),
        "blinding": (
            "the reviewer packet omits task ids, authors, historical labels, "
            "named/opaque model choices, raw model responses, pair indices, and "
            "action identifiers; action identifiers, task ids, and round ids "
            "quoted inside goals are redacted, and only the hidden key can "
            "decode choices"
        ),
        "reviewer_task": (
            "select exactly one of eight unnamed action descriptions and rate "
            "confidence without consulting the replay artifact, key, notes, or "
            "another model"
        ),
        "primary_comparison": {
            "current_candidate": "the opaque arm's selected action",
            "historical_candidate": "the historical expected action",
            "informative_case": (
                "reviewer choice decodes to exactly one of the two candidates"
            ),
            "test": (
                "two-sided exact binomial test on current-vs-historical "
                "discordant choices at probability 0.5"
            ),
            "alpha": 0.05,
        },
        "primary_outcomes": {
            "historical_label_noise_supported": (
                "current candidate wins more informative cases and the two-sided "
                "exact binomial p is below alpha"
            ),
            "historical_label_noise_not_supported": (
                "historical candidate wins more informative cases and the "
                "two-sided exact binomial p is below alpha"
            ),
            "otherwise": "inconclusive",
        },
        "secondary_outcomes": [
            "reviewer match rate for historical label, opaque choice, named choice, and either current choice",
            "confidence-stratified counts without changing the primary outcome",
        ],
        "power_note": (
            "at most 22 items are reviewed, and non-informative reviewer choices "
            "further reduce discordant pairs; an inconclusive result does not "
            "mean the two candidates are equivalent"
        ),
        "claim_boundary": (
            "this adjudicates only the 22 both-wrong replay pairs; even a clear "
            "result cannot validate the replay overall, establish semantic "
            "understanding, or prove collective intelligence"
        ),
    }


def prepare_review(artifact: dict, artifact_sha256: str) -> tuple[dict, dict, dict]:
    """Build preregistration, blinded packet, and hidden key deterministically."""
    if len(artifact_sha256) != 64:
        raise ValueError("artifact SHA-256 must have 64 hex characters")
    selected = _both_wrong_rows(artifact)
    prereg = _preregistration(artifact, artifact_sha256, selected)
    prereg_sha256 = _sha_json(prereg)
    ordered = sorted(
        selected,
        key=lambda item: _stable_rank(
            artifact_sha256, item["task_id"], item["goal"]))
    packet_items = []
    key_items = {}
    for index, item in enumerate(ordered, start=1):
        review_id = f"R{index:02d}"
        action_order = sorted(
            ACTIONS,
            key=lambda action: _stable_rank(
                artifact_sha256, item["task_id"], review_id, action))
        choice_to_action = {
            _choice_letter(position): action
            for position, action in enumerate(action_order)
        }
        packet_items.append({
            "review_id": review_id,
            "goal": _blind_goal(item["goal"]),
            "choices": [
                {
                    "choice": choice,
                    "description": SEM.ACTION_DESCRIPTIONS[action],
                }
                for choice, action in choice_to_action.items()
            ],
        })
        key_items[review_id] = {
            "pair": item["pair"],
            "task_id": item["task_id"],
            "historical_action": item["historical_action"],
            "named_model_action": item["named_model_action"],
            "opaque_model_action": item["opaque_model_action"],
            "choice_to_action": choice_to_action,
        }
    packet = {
        "schema": PACKET_SCHEMA,
        "purpose": (
            "Blindly adjudicate the best action for each frozen historical goal."
        ),
        "review_instructions": [
            "Read only this packet. Do not open the replay artifact, hidden key, ledger, or repository notes.",
            "For each item, choose exactly one candidate description that best fits the goal.",
            "Do not infer the action identifier; descriptions are intentionally unnamed and reordered.",
            "Rate confidence as low, medium, or high. Notes are optional and may not name the old answer.",
            "Return the completed response_template as JSON; do not edit review ids or choice letters.",
        ],
        "preregistration_sha256": prereg_sha256,
        "source_artifact_sha256": artifact_sha256,
        "items": packet_items,
        "response_template": {
            "schema": RESPONSE_SCHEMA,
            "reviewer": "",
            "completed_at": "",
            "items": [
                {
                    "review_id": item["review_id"],
                    "choice": None,
                    "confidence": "",
                    "notes": "",
                }
                for item in packet_items
            ],
        },
    }
    packet_sha256 = _sha_json(packet)
    key = {
        "schema": KEY_SCHEMA,
        "artifact_sha256": artifact_sha256,
        "preregistration_sha256": prereg_sha256,
        "packet_sha256": packet_sha256,
        "items": key_items,
    }
    return prereg, packet, key


def _walk_keys(value) -> list[str]:
    if isinstance(value, dict):
        out = list(value.keys())
        for child in value.values():
            out.extend(_walk_keys(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_keys(child))
        return out
    return []


def audit_review(packet: dict, key: dict, prereg: dict,
                 artifact: dict | None = None) -> dict:
    """Verify packet/key correspondence and that the packet has no labels."""
    packet_items = packet.get("items") or []
    key_items = key.get("items") or {}
    review_ids = [str(item.get("review_id") or "") for item in packet_items]
    forbidden_keys = {
        "pair", "task_id", "author", "historical_action",
        "named_model_action", "opaque_model_action", "expected_action",
        "selected_action", "correct", "raw_response", "prompt",
    }
    packet_blob = _canonical_json(packet)
    leaked_action_ids = [action for action in ACTIONS if action in packet_blob]
    leaked_keys = sorted(set(_walk_keys(packet)) & forbidden_keys)
    item_checks = []
    for item in packet_items:
        review_id = str(item.get("review_id") or "")
        mapping = (key_items.get(review_id) or {}).get("choice_to_action") or {}
        choices = item.get("choices") or []
        labels = [str(choice.get("choice") or "") for choice in choices]
        descriptions = {
            str(choice.get("choice") or ""): str(choice.get("description") or "")
            for choice in choices
        }
        expected_descriptions = {
            choice: SEM.ACTION_DESCRIPTIONS.get(action, "")
            for choice, action in mapping.items()
        }
        item_checks.append(
            bool(review_id)
            and len(choices) == len(ACTIONS)
            and labels == [_choice_letter(i) for i in range(len(ACTIONS))]
            and set(mapping.values()) == set(ACTIONS)
            and len(set(mapping.values())) == len(ACTIONS)
            and descriptions == expected_descriptions
            and bool(str(item.get("goal") or "").strip())
        )
    prereg_sha256 = _sha_json(prereg)
    checks = {
        "packet_schema": packet.get("schema") == PACKET_SCHEMA,
        "key_schema": key.get("schema") == KEY_SCHEMA,
        "prereg_schema": prereg.get("schema") == PREREG_SCHEMA,
        "artifact_hash": len(str(key.get("artifact_sha256") or "")) == 64,
        "prereg_hash": (
            key.get("preregistration_sha256") == prereg_sha256
            == packet.get("preregistration_sha256")),
        "artifact_hash_match": (
            key.get("artifact_sha256") == packet.get("source_artifact_sha256")),
        "packet_hash": key.get("packet_sha256") == _sha_json(packet),
        "item_count": len(packet_items) == len(key_items),
        "unique_review_ids": (
            len(review_ids) == len(set(review_ids)) and all(review_ids)),
        "packet_items_valid": bool(item_checks) and all(item_checks),
        "no_action_ids_in_packet": not leaked_action_ids,
        "no_label_keys_in_packet": not leaked_keys,
    }
    if artifact is not None:
        selected = _both_wrong_rows(artifact)
        selected_by_task = {item["task_id"]: item for item in selected}
        checks["source_pairs_match"] = all(
            (selected_by_task.get(str(values.get("task_id") or "")) or {}).get(
                "historical_action") == values.get("historical_action")
            and (selected_by_task.get(str(values.get("task_id") or "")) or {}).get(
                "named_model_action") == values.get("named_model_action")
            and (selected_by_task.get(str(values.get("task_id") or "")) or {}).get(
                "opaque_model_action") == values.get("opaque_model_action")
            for values in key_items.values()
        ) and len(key_items) == len(selected_by_task)
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "leaked_action_ids": leaked_action_ids,
        "leaked_keys": leaked_keys,
    }


def decide_review(current_only: int, historical_only: int) -> dict:
    """Apply the preregistered exact test to the two exclusive choices."""
    n = current_only + historical_only
    p = SEM.binomial_two_sided(current_only, n, 0.5)
    if n and p < 0.05 and current_only > historical_only:
        outcome = "historical_label_noise_supported"
    elif n and p < 0.05 and historical_only > current_only:
        outcome = "historical_label_noise_not_supported"
    else:
        outcome = "inconclusive"
    return {
        "informative": n,
        "current_only": current_only,
        "historical_only": historical_only,
        "exact_two_sided_p": p,
        "alpha": 0.05,
        "outcome": outcome,
    }


def _match_summary(correct: int, n: int) -> dict:
    low, high = SEM.wilson_interval(correct, n)
    return {
        "correct": correct,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "wilson_95": [low, high],
    }


def score_review(packet: dict, key: dict, prereg: dict,
                 response: dict) -> dict:
    """Decode and score a completed blind response under the frozen rule."""
    audit = audit_review(packet, key, prereg)
    if not audit["ok"]:
        raise ValueError(f"invalid packet/key/prereg: {audit['checks']}")
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("response schema mismatch")
    if not str(response.get("reviewer") or "").strip():
        raise ValueError("response reviewer is required")
    if not str(response.get("completed_at") or "").strip():
        raise ValueError("response completed_at is required")
    response_items = response.get("items") or []
    by_review = {}
    for item in response_items:
        review_id = str(item.get("review_id") or "")
        if review_id in by_review:
            raise ValueError(f"duplicate response item {review_id}")
        by_review[review_id] = item
    expected_ids = set(key["items"])
    if set(by_review) != expected_ids:
        raise ValueError("response item ids do not match packet")
    counts = Counter()
    rows = []
    for review_id in sorted(expected_ids):
        answer = by_review[review_id]
        hidden = key["items"][review_id]
        choice = str(answer.get("choice") or "")
        mapping = hidden["choice_to_action"]
        if choice not in mapping:
            raise ValueError(f"invalid choice {choice!r} for {review_id}")
        selected = mapping[choice]
        historical = hidden["historical_action"]
        opaque = hidden["opaque_model_action"]
        named = hidden["named_model_action"]
        historical_match = selected == historical
        opaque_match = selected == opaque
        named_match = selected == named
        any_current_match = opaque_match or named_match
        if historical_match:
            category = "historical"
        elif opaque_match:
            category = "opaque_current"
        elif named_match:
            category = "named_current_only"
        else:
            category = "neither"
        counts[category] += 1
        counts["historical_match"] += int(historical_match)
        counts["opaque_match"] += int(opaque_match)
        counts["named_match"] += int(named_match)
        counts["any_current_match"] += int(any_current_match)
        confidence = str(answer.get("confidence") or "").lower()
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"invalid confidence for {review_id}")
        counts[f"confidence_{confidence}"] += 1
        rows.append({
            "review_id": review_id,
            "choice": choice,
            "selected_action": selected,
            "confidence": confidence,
            "category": category,
            "historical_match": historical_match,
            "opaque_match": opaque_match,
            "named_match": named_match,
            "any_current_match": any_current_match,
        })
    n = len(expected_ids)
    decision = decide_review(counts["opaque_current"], counts["historical"])
    return {
        "schema": RESULT_SCHEMA,
        "reviewer": str(response["reviewer"]),
        "completed_at": str(response["completed_at"]),
        "packet_sha256": _sha_json(packet),
        "key_sha256": _sha_json(key),
        "preregistration_sha256": _sha_json(prereg),
        "items": n,
        "categories": {
            "historical": counts["historical"],
            "opaque_current": counts["opaque_current"],
            "named_current_only": counts["named_current_only"],
            "neither": counts["neither"],
        },
        "matches": {
            "historical": _match_summary(counts["historical_match"], n),
            "opaque_current": _match_summary(counts["opaque_match"], n),
            "named_current": _match_summary(counts["named_match"], n),
            "any_current": _match_summary(counts["any_current_match"], n),
        },
        "confidence_counts": {
            level: counts[f"confidence_{level}"]
            for level in ("low", "medium", "high")
        },
        "primary_decision": decision,
        "rows": rows,
        "claim_boundary": prereg["claim_boundary"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--prereg", default=str(DEFAULT_PREREG))
    parser.add_argument("--response", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.prepare, args.audit, args.score)) != 1:
        parser.error("choose exactly one of --prepare, --audit, or --score")
    artifact_path = Path(args.artifact).expanduser()
    packet_path = Path(args.packet).expanduser()
    key_path = Path(args.key).expanduser()
    prereg_path = Path(args.prereg).expanduser()

    if args.prepare:
        artifact = _load_json(artifact_path)
        source_audit = HIST.audit_artifact(artifact, require_summary=True)
        if not source_audit["ok"]:
            print(json.dumps(source_audit, ensure_ascii=False, indent=2))
            return 1
        artifact_sha256 = _sha_file(artifact_path)
        prereg, packet, key = prepare_review(artifact, artifact_sha256)
        audit = audit_review(packet, key, prereg, artifact)
        if not audit["ok"]:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 1
        _write_json(prereg_path, prereg)
        _write_json(packet_path, packet)
        _write_json(key_path, key)
        key_path.chmod(0o600)
        print(f"wrote preregistration {prereg_path}")
        print(f"wrote blinded packet {packet_path}: {len(packet['items'])} items")
        print(f"wrote hidden key {key_path}")
        print("audit: PASS")
        return 0

    prereg = _load_json(prereg_path)
    packet = _load_json(packet_path)
    key = _load_json(key_path)
    artifact = _load_json(artifact_path) if args.artifact else None
    if args.audit:
        audit = audit_review(packet, key, prereg, artifact)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 1
    if not args.response:
        parser.error("--score requires --response")
    response = _load_json(Path(args.response).expanduser())
    result = score_review(packet, key, prereg, response)
    if args.output:
        _write_json(Path(args.output).expanduser(), result)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
