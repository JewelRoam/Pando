#!/usr/bin/env python3
"""Collect a sealed, same-host second-model review of the blind packet.

This is a deliberately narrow follow-up to ``historical_action_blind_review.py``.
The frozen reviewer packet is sent to an already-running local model one item at
a time.  The collector never opens the hidden answer key: it emits a response
file that the existing scorer can decode only after collection is complete.

The default reviewer is Qwen3-4B-Instruct-2507, an existing local model used by
the proposer failover chain.  That makes this a cross-checkpoint review, not an
independent host or human review.  It cannot replace the packet's preregistered
external adjudication.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from pando.thinkd import OpenAIProposer, extract_json

ROOT = Path(__file__).resolve().parents[1]
BLIND_PATH = ROOT / "tools" / "historical_action_blind_review.py"
DEFAULT_PACKET = (ROOT / "assets" /
                  "historical-action-blind-review-20261029.packet.json")
DEFAULT_PROTOCOL = (ROOT / "assets" /
                    "historical-action-blind-review-20261029-qwen3-4b-protocol.json")
EXPERIMENTS = Path.home() / ".pandosi" / "experiments"
DEFAULT_RAW = (EXPERIMENTS /
               "historical-action-blind-review-20261029.qwen3-4b.raw.json")
DEFAULT_RESPONSE = (EXPERIMENTS /
                    "historical-action-blind-review-20261029.qwen3-4b.response.json")
PROTOCOL_SCHEMA = "historical-action-blind-review-model-protocol-v1"
COLLECTION_SCHEMA = "historical-action-blind-review-model-collection-v1"
REVIEWER_NAME = "qwen3-4b-instruct-2507-same-host"
SYSTEM_PROMPT = (
    "你是独立盲评员。你只能使用用户消息里的目标与八个候选描述，"
    "不得猜测隐藏的动作编号，不得补充外部知识。必须从 A 到 H 中"
    "选择恰好一个最符合目标的描述，并给出 low、medium 或 high 的置信度。"
    "只输出一个合法 JSON 对象，格式为 "
    '{"choice":"A","confidence":"high","notes":"简短理由"}，'
    "不要代码块，不要输出任何额外文字。"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BLIND = _load_module("historical_blind_for_model_review", BLIND_PATH)


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


def item_prompt(item: dict) -> str:
    """Render exactly the public fields for one packet item."""
    lines = [
        f"审核项：{item['review_id']}",
        "目标：",
        str(item["goal"]),
        "",
        "候选动作：",
    ]
    for choice in item.get("choices") or []:
        lines.append(f"{choice['choice']}. {choice['description']}")
    lines.extend([
        "",
        "请选择最符合目标的一项，并按指定 JSON 格式回答。",
    ])
    return "\n".join(lines)


def build_protocol(packet: dict) -> dict:
    """Build the fixed protocol that must be committed before collection."""
    items = packet.get("items") or []
    if packet.get("schema") != BLIND.PACKET_SCHEMA or not items:
        raise ValueError("not a non-empty blind-review packet")
    expected_labels = [chr(ord("A") + index) for index in range(len(BLIND.ACTIONS))]
    prompts = []
    for item in items:
        choices = item.get("choices") or []
        labels = [str(choice.get("choice") or "") for choice in choices]
        descriptions = [str(choice.get("description") or "") for choice in choices]
        if labels != expected_labels or len(set(descriptions)) != len(expected_labels):
            raise ValueError(f"malformed choices in {item.get('review_id')}")
        prompts.append(item_prompt(item))
    return {
        "schema": PROTOCOL_SCHEMA,
        "packet_sha256": _sha_json(packet),
        "packet_preregistration_sha256": packet.get("preregistration_sha256"),
        "reviewer": {
            "name": REVIEWER_NAME,
            "model": "qwen3-4b-instruct-2507",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key_file": "~/.pandosi/llm-api.token",
            "scope": (
                "existing same-host second checkpoint in the Qwen family; "
                "not an independent host, not a human, and not a replacement "
                "for external preregistered review"
            ),
        },
        "generation": {
            "temperature": 0.0,
            "max_tokens": 96,
            "seed": 20261029,
            "timeout_seconds": 180,
            "retry_policy": "none",
        },
        "input_policy": (
            "send only one packet review_id, goal, and its eight shuffled "
            "unnamed descriptions; never open or include the hidden key, "
            "replay artifact, ledger, repository notes, or previous answers"
        ),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": _sha_text(SYSTEM_PROMPT),
        "item_count": len(items),
        "item_prompts_sha256": _sha_text("\0".join(prompts)),
        "decision_rule": (
            "delegate unchanged to historical_action_blind_review.py --score "
            "after this response is frozen"
        ),
        "claim_boundary": (
            "this same-host, same-family model review is an exploratory "
            "cross-checkpoint diagnostic; it cannot establish external "
            "replication, semantic understanding, or collective intelligence"
        ),
    }


def parse_reviewer_output(text: str, valid_choices: set[str]) -> dict:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("reviewer did not return a JSON object")
    choice = str(parsed.get("choice") or "").strip().upper()
    confidence = str(parsed.get("confidence") or "").strip().lower()
    notes = str(parsed.get("notes") or "").strip()
    if choice not in valid_choices:
        raise ValueError(f"invalid choice: {choice!r}")
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"invalid confidence: {confidence!r}")
    return {"choice": choice, "confidence": confidence, "notes": notes}


def assemble_response(packet: dict, protocol: dict, rows: list[dict]) -> dict:
    if len(rows) != len(packet.get("items") or []):
        raise ValueError("response row count does not match packet")
    by_id = {str(row.get("review_id") or ""): row for row in rows}
    items = []
    for packet_item in packet["items"]:
        review_id = str(packet_item["review_id"])
        row = by_id.get(review_id)
        if not row:
            raise ValueError(f"missing response row {review_id}")
        answer = row.get("parsed") or {}
        items.append({
            "review_id": review_id,
            "choice": answer["choice"],
            "confidence": answer["confidence"],
            "notes": answer.get("notes") or "",
        })
    return {
        "schema": BLIND.RESPONSE_SCHEMA,
        "reviewer": protocol["reviewer"]["name"],
        "completed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "items": items,
    }


def audit_collection(packet: dict, protocol: dict, collection: dict,
                     response: dict) -> dict:
    rebuilt = build_protocol(packet)
    expected_items = packet.get("items") or []
    rows = collection.get("rows") or []
    by_id = {str(row.get("review_id") or ""): row for row in rows}
    checks = {
        "protocol_schema": protocol.get("schema") == PROTOCOL_SCHEMA,
        "protocol_rebuilt": rebuilt == protocol,
        "protocol_hash": collection.get("protocol_sha256") == _sha_json(protocol),
        "packet_hash": (
            protocol.get("packet_sha256") == _sha_json(packet)
            == collection.get("packet_sha256")),
        "collection_complete": collection.get("status") == "complete",
        "row_count": len(rows) == len(expected_items),
        "unique_review_ids": (
            len(by_id) == len(rows)
            and all(str(item.get("review_id") or "") in by_id
                    for item in expected_items)),
        "response_schema": response.get("schema") == BLIND.RESPONSE_SCHEMA,
        "response_reviewer": response.get("reviewer") == REVIEWER_NAME,
        "response_row_count": len(response.get("items") or []) == len(expected_items),
    }
    prompt_checks = []
    response_by_id = {
        str(item.get("review_id") or ""): item
        for item in response.get("items") or []
    }
    all_prompts = []
    for packet_item in expected_items:
        review_id = str(packet_item["review_id"])
        row = by_id.get(review_id) or {}
        prompt = item_prompt(packet_item)
        all_prompts.append(prompt)
        answer = response_by_id.get(review_id) or {}
        parsed = row.get("parsed") or {}
        prompt_checks.append(
            row.get("prompt") == prompt
            and row.get("prompt_sha256") == _sha_text(prompt)
            and answer.get("choice") == parsed.get("choice")
            and answer.get("confidence") == parsed.get("confidence")
        )
    checks["prompts_match_packet"] = bool(prompt_checks) and all(prompt_checks)
    checks["prompt_set_hash"] = (
        _sha_text("\0".join(all_prompts))
        == protocol.get("item_prompts_sha256"))
    checks["no_action_ids_in_prompts"] = not any(
        action in prompt for action in BLIND.ACTIONS for prompt in all_prompts)
    return {"ok": all(checks.values()), "checks": checks}


def collect(packet_path: Path, protocol_path: Path, raw_path: Path,
            response_path: Path) -> dict:
    packet = _load_json(packet_path)
    protocol = _load_json(protocol_path)
    if build_protocol(packet) != protocol:
        raise ValueError("committed protocol does not match the frozen packet")
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
    collection = {
        "schema": COLLECTION_SCHEMA,
        "status": "in_progress",
        "packet_path": str(packet_path),
        "packet_sha256": _sha_json(packet),
        "protocol_sha256": _sha_json(protocol),
        "reviewer": reviewer,
        "generation": generation,
        "rows": [],
        "totals": {},
    }
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    collection["started_at"] = started
    total_usage: dict[str, int] = {}
    total_latency_ms = 0.0
    for packet_item in packet["items"]:
        prompt = item_prompt(packet_item)
        payload = {
            "model": reviewer["model"],
            "max_tokens": int(generation["max_tokens"]),
            "temperature": float(generation["temperature"]),
            "seed": int(generation["seed"]),
            "messages": [
                {"role": "system", "content": protocol["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
        }
        call_started = time.perf_counter()
        data = proposer._post(payload)
        latency_ms = (time.perf_counter() - call_started) * 1000.0
        choices = data.get("choices") or []
        raw = ""
        if choices:
            raw = str((choices[0].get("message") or {}).get("content") or "")
        valid_choices = {str(choice["choice"]) for choice in packet_item["choices"]}
        parsed = parse_reviewer_output(raw, valid_choices)
        usage = data.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                total_usage[key] = total_usage.get(key, 0) + value
        total_latency_ms += latency_ms
        collection["rows"].append({
            "review_id": packet_item["review_id"],
            "goal_sha256": _sha_text(str(packet_item["goal"])),
            "prompt": prompt,
            "prompt_sha256": _sha_text(prompt),
            "raw_response": raw,
            "parsed": parsed,
            "usage": usage,
            "latency_ms": round(latency_ms, 3),
            "response_model": data.get("model") or reviewer["model"],
            "timings": data.get("timings") or {},
        })
    response = assemble_response(packet, protocol, collection["rows"])
    collection.update({
        "status": "complete",
        "completed_at": response["completed_at"],
        "response_sha256": _sha_json(response),
        "totals": {
            "calls": len(collection["rows"]),
            "latency_ms": round(total_latency_ms, 3),
            "usage": total_usage,
        },
    })
    audit = audit_collection(packet, protocol, collection, response)
    if not audit["ok"]:
        _write_json(raw_path, collection)
        raise ValueError(f"collection audit failed: {json.dumps(audit, ensure_ascii=False)}")
    _write_json(raw_path, collection)
    _write_json(response_path, response)
    return {"collection": collection, "response": response, "audit": audit}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--response", default=str(DEFAULT_RESPONSE))
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.prepare, args.collect, args.audit)) != 1:
        parser.error("choose exactly one of --prepare, --collect, or --audit")
    packet_path = Path(args.packet).expanduser()
    protocol_path = Path(args.protocol).expanduser()
    raw_path = Path(args.raw).expanduser()
    response_path = Path(args.response).expanduser()
    packet = _load_json(packet_path)
    if args.prepare:
        protocol = build_protocol(packet)
        _write_json(protocol_path, protocol)
        print(f"wrote protocol {protocol_path}")
        print(f"packet sha256: {protocol['packet_sha256']}")
        print(f"item prompts sha256: {protocol['item_prompts_sha256']}")
        return 0
    if args.collect:
        outcome = collect(packet_path, protocol_path, raw_path, response_path)
        print(f"wrote raw collection {raw_path}")
        print(f"wrote response {response_path}")
        print(json.dumps(outcome["audit"], ensure_ascii=False, indent=2))
        return 0
    protocol = _load_json(protocol_path)
    collection = _load_json(raw_path)
    response = _load_json(response_path)
    audit = audit_collection(packet, protocol, collection, response)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
