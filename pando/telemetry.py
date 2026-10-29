"""Signed, push-only telemetry for a node whose root cannot be dialled.

Reading such a node's root table over a reverse tunnel made the observation
depend on a route in the opposite direction from the application traffic:
the node could keep proposing and voting while its observer called it
unreadable. This module makes the node push a small, signed observation over
the same path its mailbox clients already use.

The transport is authenticated twice. The mailbox bearer token says the sender
may call the authenticated API, and the identity signature says the payload
came from the pinned node key. The receiver stamps its own clock; a sender's
clock is stored for diagnosis but never decides whether an observation is
fresh. Pins (host, member, pubkey) arrive from the caller - typically the
untracked per-host config in ``pando.hostconfig`` - never from this module.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .identity import Identity, verify

PROTOCOL_VERSION = 1


class TelemetryError(ValueError):
    """A malformed or semantically invalid telemetry envelope."""


class TelemetrySignatureError(TelemetryError):
    """A well-shaped envelope whose signer or signature is not trusted."""


def build_envelope(identity: Identity, *, host: str, member: str,
                   nodes: dict, code_head: str,
                   think_round: int, status: dict | None = None,
                   sent_at: float | None = None) -> dict:
    """Build the exact JSON object a node's code-sync job posts."""
    payload = {
        "member": member,
        "sent_at": time.time() if sent_at is None else float(sent_at),
        "code_head": str(code_head),
        "nodes": nodes,
        "think_round": int(think_round),
    }
    if status is not None:
        payload["status"] = status
    return {
        "version": PROTOCOL_VERSION,
        "host": host,
        "pubkey": identity.public_key,
        "payload": payload,
        "sig": identity.sign(payload),
    }


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def validate_envelope(envelope: Any, expected_pubkey: str,
                      *, expected_host: str,
                      expected_member: str) -> dict:
    """Return a validated payload, or raise and write nothing.

    Unknown fields are refused rather than ignored. The protocol is versioned,
    so adding a field is a deliberate change and an old receiver must not
    silently accept bytes whose meaning it does not know.
    """
    if not isinstance(envelope, dict):
        raise TelemetryError("envelope must be an object")
    required = {"version", "host", "pubkey", "payload", "sig"}
    if set(envelope) != required:
        raise TelemetryError("envelope fields must be exactly "
                             + ", ".join(sorted(required)))
    if (not isinstance(envelope["version"], int)
            or isinstance(envelope["version"], bool)
            or envelope["version"] != PROTOCOL_VERSION):
        raise TelemetryError("unsupported telemetry version")
    if envelope["host"] != expected_host:
        raise TelemetryError("unexpected telemetry host")
    if envelope["pubkey"] != expected_pubkey:
        raise TelemetrySignatureError("telemetry signer is not pinned")
    if not isinstance(envelope["sig"], str) or not envelope["sig"]:
        raise TelemetrySignatureError("telemetry signature is missing")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise TelemetryError("payload must be an object")
    allowed = {"member", "sent_at", "code_head", "nodes", "think_round",
               "status"}
    required_payload = allowed - {"status"}
    if not required_payload.issubset(payload):
        raise TelemetryError("payload is missing required fields")
    if set(payload) - allowed:
        raise TelemetryError("payload contains unknown fields")
    if payload.get("member") != expected_member:
        raise TelemetryError("unexpected telemetry member")
    if not _is_number(payload.get("sent_at")):
        raise TelemetryError("sent_at must be a finite number")
    code_head = payload.get("code_head")
    if not isinstance(code_head, str) or not code_head or len(code_head) > 128:
        raise TelemetryError("code_head must be a non-empty string")
    nodes = payload.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise TelemetryError("nodes must be a non-empty object")
    if len(nodes) > 1000:
        raise TelemetryError("nodes is implausibly large")
    if any(not isinstance(node_id, str) or not isinstance(info, dict)
           for node_id, info in nodes.items()):
        raise TelemetryError("every node entry must be an object")
    think_round = payload.get("think_round")
    if (not isinstance(think_round, int) or isinstance(think_round, bool)
            or think_round < 0):
        raise TelemetryError("think_round must be a non-negative integer")
    if "status" in payload and not isinstance(payload["status"], dict):
        raise TelemetryError("status must be an object")
    if not verify(expected_pubkey, payload, envelope["sig"]):
        raise TelemetrySignatureError("telemetry signature is invalid")
    return payload


def _atomic_json(path: Path, document: dict) -> None:
    """Replace ``path`` atomically; a failed write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class TelemetryReceiver:
    """Validate one pinned signer and atomically publish its root snapshot."""

    def __init__(self, destination: str | Path, *,
                 expected_pubkey: str,
                 expected_host: str,
                 expected_member: str):
        self.destination = Path(destination).expanduser()
        self.expected_pubkey = expected_pubkey
        self.expected_host = expected_host
        self.expected_member = expected_member

    @classmethod
    def from_config(cls, destination: str | Path, config: dict
                    ) -> "TelemetryReceiver":
        """Build from the ``telemetry.accept`` section of a host config."""
        from . import hostconfig
        pin = hostconfig.telemetry_accept(config)
        if pin is None:
            raise TelemetryError(
                "host config has no [telemetry.accept] pin; the receiver "
                "cannot trust anyone without one")
        return cls(destination, expected_pubkey=pin["pubkey"],
                   expected_host=pin["host"], expected_member=pin["member"])

    def receive(self, envelope: Any, now: float | None = None) -> dict:
        payload = validate_envelope(
            envelope, self.expected_pubkey,
            expected_host=self.expected_host,
            expected_member=self.expected_member)
        stamp = time.time() if now is None else float(now)
        document = {
            "nodes": payload["nodes"],
            "mirrored_at": stamp,
            "observed_at": stamp,
            "sent_at": payload["sent_at"],
            "code_head": payload["code_head"],
            "think_round": payload["think_round"],
            "telemetry": {
                "version": PROTOCOL_VERSION,
                "host": self.expected_host,
                "member": payload["member"],
                "pubkey": self.expected_pubkey,
            },
        }
        if "status" in payload:
            document["status"] = payload["status"]
        _atomic_json(self.destination, document)
        return {"ok": True, "mirrored_at": stamp,
                "members": len(payload["nodes"]),
                "think_round": payload["think_round"]}
