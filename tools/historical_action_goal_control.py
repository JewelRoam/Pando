#!/usr/bin/env python3
"""Goal-permutation control for the frozen same-host blind review.

The original 22-item review found that the existing Qwen3-4B reviewer matched
the opaque arm's current candidate on 15/22 items, versus 1/22 historical
labels.  That result could still reflect an action prior or choice-position
effect rather than using the goal.

This tool pre-registers and collects a negative control: keep every item's
eight candidate descriptions and choice order fixed, but replace its goal with
a goal from another item under every non-zero cyclic shift.  The original arm
is then compared with the 21 mismatched-goal arms by an exact cyclic-shift
randomization test.  Generation uses only the public packet; the hidden key is
opened only after collection is complete, during offline scoring.

This remains a same-host, same-model internal control.  It is not an external
replication and cannot establish semantic understanding or collective
intelligence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from pando.proposer import OpenAIProposer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tools" / "historical_action_model_review.py"
DEFAULT_PACKET = (ROOT / "assets" /
                  "historical-action-blind-review-20261029.packet.json")
DEFAULT_KEY = (Path.home() / ".pandosi" / "experiments" /
               "historical-action-blind-review-20261029.key.json")
DEFAULT_BASE_PROTOCOL = (ROOT / "assets" /
                         "historical-action-blind-review-20261029-qwen3-4b-protocol.json")
DEFAULT_BASE_RESULT = (Path.home() / ".pandosi" / "experiments" /
                       "historical-action-blind-review-20261029.qwen3-4b.result.json")
DEFAULT_PROTOCOL = (ROOT / "assets" /
                    "historical-action-goal-control-20261029-protocol.json")
DEFAULT_RAW = (Path.home() / ".pandosi" / "experiments" /
               "historical-action-goal-control-20261029.raw.json")
DEFAULT_RESULT = (Path.home() / ".pandosi" / "experiments" /
                  "historical-action-goal-control-20261029.result.json")
PROTOCOL_SCHEMA = "historical-action-goal-control-protocol-v1"
COLLECTION_SCHEMA = "historical-action-goal-control-collection-v1"
RESULT_SCHEMA = "historical-action-goal-control-result-v1"
ALPHA = 0.05


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODEL = _load_module("historical_action_model_review_for_goal_control", MODEL_PATH)
BLIND = MODEL.BLIND


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_json(value) -> str:
    return _sha_text(_canonical_json(value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _rotated_item(packet: dict, destination_index: int, offset: int) -> dict:
    """Copy one packet item but substitute a source goal from another item."""
    items = packet.get("items") or []
    if not items:
        raise ValueError("packet has no items")
    if not 0 < offset < len(items):
        raise ValueError("offset must be non-zero and smaller than item count")
    item = dict(items[destination_index])
    source = items[(destination_index + offset) % len(items)]
    item["goal"] = source["goal"]
    return item


def control_prompts(packet: dict) -> list[dict]:
    """Return the fixed control inventory in collection order."""
    items = packet.get("items") or []
    rows = []
    for offset in range(1, len(items)):
        for destination_index, destination in enumerate(items):
            source = items[(destination_index + offset) % len(items)]
            if str(destination.get("goal") or "") == str(source.get("goal") or ""):
                raise ValueError("control would reuse an identical goal")
            prompt = MODEL.item_prompt(
                _rotated_item(packet, destination_index, offset))
            rows.append({
                "offset": offset,
                "review_id": str(destination["review_id"]),
                "goal_source_review_id": str(source["review_id"]),
                "prompt": prompt,
            })
    return rows


def _observed_counts(base_result: dict) -> dict:
    rows = base_result.get("rows") or []
    if not rows:
        raise ValueError("base result has no rows")
    categories = base_result.get("categories") or {}
    observed_current = sum(bool(row.get("opaque_match")) for row in rows)
    observed_historical = sum(bool(row.get("historical_match")) for row in rows)
    if int(categories.get("opaque_current", -1)) != observed_current:
        raise ValueError("base result opaque count is inconsistent")
    if int(categories.get("historical", -1)) != observed_historical:
        raise ValueError("base result historical count is inconsistent")
    return {
        "items": len(rows),
        "opaque_current_matches": observed_current,
        "historical_matches": observed_historical,
    }


def build_protocol(packet: dict, base_protocol: dict,
                   base_result: dict) -> dict:
    """Build a deterministic protocol that must exist before collection."""
    items = packet.get("items") or []
    if len(items) < 2:
        raise ValueError("need at least two packet items")
    if base_protocol.get("packet_sha256") != _sha_json(packet):
        raise ValueError("base protocol does not match packet")
    if base_result.get("packet_sha256") != _sha_json(packet):
        raise ValueError("base result does not match packet")
    if base_result.get("reviewer") != base_protocol.get("reviewer", {}).get("name"):
        raise ValueError("base result reviewer does not match base protocol")
    inventory = control_prompts(packet)
    prompts = [row["prompt"] for row in inventory]
    leaked_actions = sorted({
        action for action in BLIND.ACTIONS
        if any(action in prompt for prompt in prompts)
    })
    if leaked_actions:
        raise ValueError(f"action ids leaked into control prompts: {leaked_actions}")
    observed = _observed_counts(base_result)
    return {
        "schema": PROTOCOL_SCHEMA,
        "source": {
            "packet_sha256": _sha_json(packet),
            "packet_preregistration_sha256": packet.get("preregistration_sha256"),
            "base_protocol_sha256": _sha_json(base_protocol),
            "base_result_sha256": _sha_json(base_result),
        },
        "reviewer": dict(base_protocol["reviewer"]),
        "generation": dict(base_protocol["generation"]),
        "system_prompt": base_protocol["system_prompt"],
        "system_prompt_sha256": base_protocol["system_prompt_sha256"],
        "item_count": len(items),
        "control": {
            "rule": (
                "for each non-zero cyclic offset k, keep every destination "
                "item's eight choices and choice order unchanged but replace "
                "its goal with the goal from source item (destination + k) "
                "mod item_count"
            ),
            "offsets": list(range(1, len(items))),
            "arms": len(items) - 1,
            "prompts": len(inventory),
            "goal_prompts_sha256": _sha_text("\0".join(prompts)),
        },
        "observed": observed,
        "primary_comparison": {
            "statistic": (
                "number of destination items where the reviewer-selected "
                "action equals the destination item's opaque-arm current "
                "candidate"
            ),
            "observed_arm": "offset 0, frozen base result (not re-sampled)",
            "null_arms": "all 21 non-zero cyclic goal substitutions",
            "test": (
                "one-sided exact cyclic-shift randomization: "
                "(1 + number of null arms with count >= observed) / "
                "(1 + number of null arms)"
            ),
            "alpha": ALPHA,
            "supported_outcome": (
                "goal_specificity_supported_in_same_model_control when "
                "p < alpha and the observed count is greater than every "
                "non-zero-shift count"
            ),
            "otherwise": "goal_specificity_not_supported",
        },
        "secondary_outcomes": [
            "match counts against the destination historical and named candidates",
            "reviewer action distribution across all control calls",
        ],
        "claim_boundary": (
            "internal same-host, same-family negative control only; it cannot "
            "establish external replication, semantic understanding, or "
            "collective intelligence"
        ),
    }


def audit_protocol(packet: dict, base_protocol: dict, base_result: dict,
                   protocol: dict) -> dict:
    try:
        rebuilt = build_protocol(packet, base_protocol, base_result)
    except ValueError as exc:
        return {"ok": False, "checks": {"protocol_rebuilt": False},
                "error": str(exc)}
    checks = {
        "schema": protocol.get("schema") == PROTOCOL_SCHEMA,
        "protocol_rebuilt": rebuilt == protocol,
        "source_packet_hash": (
            protocol.get("source", {}).get("packet_sha256")
            == _sha_json(packet)),
        "base_protocol_hash": (
            protocol.get("source", {}).get("base_protocol_sha256")
            == _sha_json(base_protocol)),
        "base_result_hash": (
            protocol.get("source", {}).get("base_result_sha256")
            == _sha_json(base_result)),
        "offset_count": (
            len(protocol.get("control", {}).get("offsets") or [])
            == max(len(packet.get("items") or []) - 1, 0)),
    }
    return {"ok": all(checks.values()), "checks": checks}


def _inventory_by_key(inventory: list[dict]) -> dict[tuple[int, str], str]:
    return {
        (int(row["offset"]), str(row["review_id"])): str(row["prompt"])
        for row in inventory
    }


def audit_collection(packet: dict, protocol: dict, collection: dict) -> dict:
    inventory = control_prompts(packet)
    expected = _inventory_by_key(inventory)
    rows = collection.get("rows") or []
    seen = set()
    prompt_checks = []
    raw_parse_checks = []
    usage_checks = []
    latency_total = 0.0
    usage_total: dict[str, int] = {}
    for row in rows:
        key = (int(row.get("offset", -1)), str(row.get("review_id") or ""))
        seen.add(key)
        prompt = expected.get(key)
        prompt_checks.append(
            prompt is not None
            and row.get("prompt") == prompt
            and row.get("prompt_sha256") == _sha_text(prompt)
        )
        try:
            parsed = MODEL.parse_reviewer_output(
                str(row.get("raw_response") or ""),
                {str(choice["choice"])
                 for item in packet["items"]
                 if str(item["review_id"]) == key[1]
                 for choice in item["choices"]},
            )
        except (ValueError, TypeError):
            parsed = None
        raw_parse_checks.append(parsed == row.get("parsed"))
        usage = row.get("usage") or {}
        usage_checks.append(all(isinstance(value, int)
                                for value in usage.values()))
        for name, value in usage.items():
            if isinstance(value, int):
                usage_total[name] = usage_total.get(name, 0) + value
        latency_total += float(row.get("latency_ms") or 0.0)
    totals = collection.get("totals") or {}
    checks = {
        "schema": collection.get("schema") == COLLECTION_SCHEMA,
        "protocol_hash": (
            collection.get("protocol_sha256") == _sha_json(protocol)),
        "packet_hash": collection.get("packet_sha256") == _sha_json(packet),
        "row_count": len(rows) == len(expected),
        "unique_rows": len(seen) == len(rows),
        "all_expected_rows": seen == set(expected),
        "prompts_match_protocol": bool(prompt_checks) and all(prompt_checks),
        "raw_parse_matches": bool(raw_parse_checks) and all(raw_parse_checks),
        "usage_values_integer": bool(usage_checks) and all(usage_checks),
        "totals_usage": totals.get("usage") == usage_total,
        "totals_latency": abs(float(totals.get("latency_ms") or 0.0)
                              - round(latency_total, 3)) < 0.001,
        "status_complete": collection.get("status") == "complete",
    }
    return {"ok": all(checks.values()), "checks": checks}


def collect(packet_path: Path, protocol_path: Path, raw_path: Path) -> dict:
    packet = _load_json(packet_path)
    protocol = _load_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("not a goal-control protocol")
    if protocol.get("source", {}).get("packet_sha256") != _sha_json(packet):
        raise ValueError("protocol does not match packet")
    rebuilt_prompts = control_prompts(packet)
    if _sha_text("\0".join(row["prompt"] for row in rebuilt_prompts)) != (
            protocol.get("control", {}).get("goal_prompts_sha256")):
        raise ValueError("protocol prompt hash does not match packet")

    protocol_sha = _sha_json(protocol)
    if raw_path.exists():
        collection = _load_json(raw_path)
        if collection.get("protocol_sha256") != protocol_sha:
            raise ValueError("existing collection belongs to another protocol")
        if collection.get("status") == "complete":
            audit = audit_collection(packet, protocol, collection)
            if not audit["ok"]:
                raise ValueError("completed collection failed audit")
            return {"collection": collection, "audit": audit, "resumed": True}
    else:
        collection = {
            "schema": COLLECTION_SCHEMA,
            "status": "in_progress",
            "packet_path": str(packet_path),
            "packet_sha256": _sha_json(packet),
            "protocol_sha256": protocol_sha,
            "reviewer_name": protocol["reviewer"]["name"],
            "generation": protocol["generation"],
            "started_at": dt.datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "rows": [],
            "totals": {},
        }

    reviewer = protocol["reviewer"]
    generation = protocol["generation"]
    api_key = Path(reviewer["api_key_file"]).expanduser().read_text(
        encoding="utf-8").strip()
    proposer = OpenAIProposer(
        base_url=reviewer["base_url"],
        model=reviewer["model"],
        api_key=api_key,
        max_tokens=int(generation["max_tokens"]),
        temperature=float(generation["temperature"]),
        timeout=float(generation["timeout_seconds"]),
    )
    completed = {
        (int(row.get("offset", -1)), str(row.get("review_id") or ""))
        for row in collection["rows"]
    }
    items_by_id = {
        str(item["review_id"]): item for item in packet["items"]
    }
    for expected in rebuilt_prompts:
        key = (int(expected["offset"]), str(expected["review_id"]))
        if key in completed:
            continue
        item = items_by_id[key[1]]
        valid_choices = {
            str(choice["choice"]) for choice in item.get("choices") or []
        }
        payload = {
            "model": reviewer["model"],
            "max_tokens": int(generation["max_tokens"]),
            "temperature": float(generation["temperature"]),
            "seed": int(generation["seed"]),
            "messages": [
                {"role": "system", "content": protocol["system_prompt"]},
                {"role": "user", "content": expected["prompt"]},
            ],
        }
        call_started = time.perf_counter()
        data = proposer._post(payload)
        latency_ms = (time.perf_counter() - call_started) * 1000.0
        choices = data.get("choices") or []
        raw = ""
        if choices:
            raw = str((choices[0].get("message") or {}).get("content") or "")
        parsed = MODEL.parse_reviewer_output(raw, valid_choices)
        collection["rows"].append({
            "offset": key[0],
            "review_id": key[1],
            "goal_source_review_id": expected["goal_source_review_id"],
            "prompt": expected["prompt"],
            "prompt_sha256": _sha_text(expected["prompt"]),
            "raw_response": raw,
            "parsed": parsed,
            "usage": data.get("usage") or {},
            "latency_ms": round(latency_ms, 3),
            "response_model": data.get("model") or reviewer["model"],
        })
        total_usage: dict[str, int] = {}
        total_latency = 0.0
        for row in collection["rows"]:
            total_latency += float(row.get("latency_ms") or 0.0)
            for name, value in (row.get("usage") or {}).items():
                if isinstance(value, int):
                    total_usage[name] = total_usage.get(name, 0) + value
        collection["totals"] = {
            "calls": len(collection["rows"]),
            "latency_ms": round(total_latency, 3),
            "usage": total_usage,
        }
        _write_json(raw_path, collection)

    collection["status"] = "complete"
    collection["completed_at"] = dt.datetime.now().astimezone().isoformat(
        timespec="seconds")
    _write_json(raw_path, collection)
    audit = audit_collection(packet, protocol, collection)
    if not audit["ok"]:
        raise ValueError(
            f"collection audit failed: {json.dumps(audit, ensure_ascii=False)}")
    return {"collection": collection, "audit": audit, "resumed": False}


def _decode_action(key: dict, review_id: str, choice: str) -> str:
    mapping = (key.get("items", {}).get(review_id, {}).get(
        "choice_to_action") or {})
    action = mapping.get(choice)
    if not action:
        raise ValueError(f"invalid choice {choice!r} for {review_id}")
    return str(action)


def cyclic_shift_p(observed: int, null_counts: list[int]) -> float:
    """Exact one-sided randomization p using observed plus all null shifts."""
    if observed < 0:
        raise ValueError("observed count must be non-negative")
    if not null_counts or any(int(value) < 0 for value in null_counts):
        raise ValueError("null counts must be a non-empty non-negative list")
    return (1 + sum(int(value) >= observed for value in null_counts)) / (
        1 + len(null_counts))


def score(packet: dict, key: dict, protocol: dict, base_result: dict,
          collection: dict) -> dict:
    audit = audit_collection(packet, protocol, collection)
    if not audit["ok"]:
        raise ValueError(f"collection audit failed: {json.dumps(audit, ensure_ascii=False)}")
    if key.get("packet_sha256") != _sha_json(packet):
        raise ValueError("hidden key does not match packet")
    key_items = key.get("items") or {}
    observed = _observed_counts(base_result)

    control_by_offset: dict[int, list[dict]] = {}
    for row in collection["rows"]:
        control_by_offset.setdefault(int(row["offset"]), []).append(row)

    arms = []
    action_counts: dict[str, int] = {}
    for offset in protocol["control"]["offsets"]:
        current = 0
        historical = 0
        named = 0
        for row in control_by_offset[int(offset)]:
            review_id = str(row["review_id"])
            decode = key_items[review_id]
            selected = _decode_action(key, review_id, str(row["parsed"]["choice"]))
            current += int(selected == decode["opaque_model_action"])
            historical += int(selected == decode["historical_action"])
            named += int(selected == decode["named_model_action"])
            action_counts[selected] = action_counts.get(selected, 0) + 1
        arms.append({
            "offset": int(offset),
            "items": len(control_by_offset[int(offset)]),
            "opaque_current_matches": current,
            "historical_matches": historical,
            "named_current_matches": named,
        })
    null_current = [arm["opaque_current_matches"] for arm in arms]
    null_historical = [arm["historical_matches"] for arm in arms]
    p_current = cyclic_shift_p(observed["opaque_current_matches"], null_current)
    p_historical = cyclic_shift_p(
        observed["historical_matches"], null_historical)
    supported = (
        observed["opaque_current_matches"] > max(null_current)
        and p_current < ALPHA
    )
    mean_current = sum(null_current) / len(null_current)
    return {
        "schema": RESULT_SCHEMA,
        "packet_sha256": _sha_json(packet),
        "protocol_sha256": _sha_json(protocol),
        "collection_sha256": _sha_json(collection),
        "reviewer": collection.get("reviewer_name"),
        "items": observed["items"],
        "observed": {
            "offset": 0,
            "opaque_current_matches": observed["opaque_current_matches"],
            "historical_matches": observed["historical_matches"],
        },
        "control": {
            "arms": len(arms),
            "items_per_arm": observed["items"],
            "opaque_current_match_counts": null_current,
            "historical_match_counts": null_historical,
            "opaque_current_mean": mean_current,
            "opaque_current_min": min(null_current),
            "opaque_current_max": max(null_current),
            "action_distribution": dict(sorted(action_counts.items())),
        },
        "primary_decision": {
            "test": "one-sided exact cyclic-shift randomization",
            "observed": observed["opaque_current_matches"],
            "null_max": max(null_current),
            "p": p_current,
            "alpha": ALPHA,
            "outcome": (
                "goal_specificity_supported_in_same_model_control"
                if supported else "goal_specificity_not_supported"
            ),
        },
        "secondary": {
            "historical_randomization_p": p_historical,
            "historical_null_counts": null_historical,
        },
        "claim_boundary": protocol["claim_boundary"],
        "arms": arms,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--base-protocol", default=str(DEFAULT_BASE_PROTOCOL))
    parser.add_argument("--base-result", default=str(DEFAULT_BASE_RESULT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--output", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (
            args.prepare, args.collect, args.audit, args.score)) != 1:
        parser.error("choose exactly one of --prepare, --collect, --audit, or --score")

    packet_path = Path(args.packet).expanduser()
    key_path = Path(args.key).expanduser()
    base_protocol_path = Path(args.base_protocol).expanduser()
    base_result_path = Path(args.base_result).expanduser()
    protocol_path = Path(args.protocol).expanduser()
    raw_path = Path(args.raw).expanduser()
    output_path = Path(args.output).expanduser()
    packet = _load_json(packet_path)
    base_protocol = _load_json(base_protocol_path)
    base_result = _load_json(base_result_path)

    if args.prepare:
        protocol = build_protocol(packet, base_protocol, base_result)
        audit = audit_protocol(packet, base_protocol, base_result, protocol)
        if not audit["ok"]:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 1
        _write_json(protocol_path, protocol)
        print(f"wrote protocol {protocol_path}")
        print(f"control prompts: {protocol['control']['prompts']}")
        print(f"goal prompts sha256: {protocol['control']['goal_prompts_sha256']}")
        return 0

    protocol = _load_json(protocol_path)
    if args.collect:
        outcome = collect(packet_path, protocol_path, raw_path)
        print(f"wrote raw collection {raw_path}")
        print(json.dumps(outcome["audit"], ensure_ascii=False, indent=2))
        return 0

    if args.audit:
        audit = audit_protocol(packet, base_protocol, base_result, protocol)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 1

    key = _load_json(key_path)
    collection = _load_json(raw_path)
    result = score(packet, key, protocol, base_result, collection)
    _write_json(output_path, result)
    print(f"wrote {output_path}")
    print(json.dumps(result["primary_decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
