"""Signed evidence that a node's code watcher actually ran.

A local code-sync ledger proves what one checkout did to itself. It is not
proof that a remote node received or observed the same commit: the Mac can
read the ledger only when a separate route happens to work. This protocol
lets the watcher push a small receipt over the authenticated mailbox route it
already uses.

The mailbox bearer token authorises the transport. The node identity signs the
payload, and the receiver stamps its own clock. A receiver therefore never has
to trust the sender's wall clock for freshness. The first valid key for a node
is recorded (trust on first use); a later key change is refused rather than
silently replacing the evidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .identity import Identity, verify

PROTOCOL_VERSION = 1
ACTIONS = frozenset({"none", "recorded", "client-restart", "node-restart",
                     "refused"})
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CodeReceiptError(ValueError):
    """A malformed or semantically invalid code receipt."""


class CodeReceiptSignatureError(CodeReceiptError):
    """A receipt whose signer is missing, changed, or not valid."""


def valid_node_id(node: str) -> bool:
    return bool(_NODE_ID.fullmatch(str(node or "")))


def build_envelope(identity: Identity, *, node: str, member: str,
                   head: str, action: str, platform: str = "",
                   acted: bool = False, reason: str = "",
                   sent_at: float | None = None) -> dict:
    """Build one signed observation from a code-watch round."""
    payload = {
        "node": str(node),
        "member": str(member),
        "sent_at": time.time() if sent_at is None else float(sent_at),
        "head": str(head),
        "action": str(action),
        "platform": str(platform),
        "acted": bool(acted),
        "reason": str(reason),
    }
    return {
        "version": PROTOCOL_VERSION,
        "node": str(node),
        "pubkey": identity.public_key,
        "payload": payload,
        "sig": identity.sign(payload),
    }


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def validate_envelope(envelope: Any, *, expected_node: str,
                      expected_member: str = "",
                      expected_pubkey: str = "") -> dict:
    """Return a validated payload, or raise without writing anything."""
    if not valid_node_id(expected_node):
        raise CodeReceiptError("invalid expected node id")
    if not isinstance(envelope, dict):
        raise CodeReceiptError("receipt must be an object")
    required = {"version", "node", "pubkey", "payload", "sig"}
    if set(envelope) != required:
        raise CodeReceiptError("receipt fields must be exactly "
                               + ", ".join(sorted(required)))
    if (not isinstance(envelope["version"], int)
            or isinstance(envelope["version"], bool)
            or envelope["version"] != PROTOCOL_VERSION):
        raise CodeReceiptError("unsupported receipt version")
    if envelope["node"] != expected_node:
        raise CodeReceiptError("unexpected receipt node")
    pubkey = envelope["pubkey"]
    if not isinstance(pubkey, str) or not pubkey:
        raise CodeReceiptSignatureError("receipt public key is missing")
    if expected_pubkey and pubkey != expected_pubkey:
        raise CodeReceiptSignatureError("receipt signer changed for node")
    if not isinstance(envelope["sig"], str) or not envelope["sig"]:
        raise CodeReceiptSignatureError("receipt signature is missing")

    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise CodeReceiptError("payload must be an object")
    allowed = {"node", "member", "sent_at", "head", "action", "platform",
               "acted", "reason"}
    required_payload = {"node", "member", "sent_at", "head", "action",
                        "platform", "acted"}
    if not required_payload.issubset(payload):
        raise CodeReceiptError("payload is missing required fields")
    if set(payload) - allowed:
        raise CodeReceiptError("payload contains unknown fields")
    if payload.get("node") != expected_node:
        raise CodeReceiptError("payload node does not match envelope")
    member = payload.get("member")
    if not isinstance(member, str) or not member or len(member) > 128:
        raise CodeReceiptError("member must be a non-empty string")
    if expected_member and member != expected_member:
        raise CodeReceiptError("unexpected receipt member")
    if not _is_number(payload.get("sent_at")):
        raise CodeReceiptError("sent_at must be a finite number")
    head = payload.get("head")
    if not isinstance(head, str) or len(head) > 128:
        raise CodeReceiptError("head must be a string of at most 128 chars")
    if payload.get("action") not in ACTIONS:
        raise CodeReceiptError("unknown code-watch action")
    platform = payload.get("platform")
    if not isinstance(platform, str) or len(platform) > 64:
        raise CodeReceiptError("platform must be a short string")
    if not isinstance(payload.get("acted"), bool):
        raise CodeReceiptError("acted must be a boolean")
    reason = payload.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 128:
        raise CodeReceiptError("reason must be a short string")
    if not verify(pubkey, payload, envelope["sig"]):
        raise CodeReceiptSignatureError("receipt signature is invalid")
    return payload


def _atomic_json(path: Path, document: dict) -> None:
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


class CodeReceiptReceiver:
    """Validate receipts and publish one latest document per node."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self._lock = threading.Lock()

    def path(self, node: str) -> Path:
        if not valid_node_id(node):
            raise CodeReceiptError("invalid node id")
        return self.root / f"{node}.json"

    def load(self, node: str) -> dict | None:
        path = self.path(node)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise CodeReceiptError(f"existing receipt is unreadable: {exc}") from exc

    def receive(self, envelope: Any, *, expected_node: str,
                expected_member: str = "", now: float | None = None) -> dict:
        path = self.path(expected_node)
        with self._lock:
            previous = self.load(expected_node)
            expected_pubkey = ""
            if previous is not None:
                expected_pubkey = str(previous.get("pubkey") or "")
                if not expected_pubkey:
                    raise CodeReceiptError("existing receipt has no pinned key")
            payload = validate_envelope(
                envelope, expected_node=expected_node,
                expected_member=expected_member,
                expected_pubkey=expected_pubkey)
            received_at = time.time() if now is None else float(now)
            document = {
                "node": expected_node,
                "pubkey": envelope["pubkey"],
                "received_at": received_at,
                "sent_at": payload["sent_at"],
                "head": payload["head"],
                "action": payload["action"],
                "platform": payload["platform"],
                "acted": payload["acted"],
                "reason": payload.get("reason", ""),
                "member": payload["member"],
            }
            _atomic_json(path, document)
            return {"ok": True, "node": expected_node,
                    "received_at": received_at,
                    "head": payload["head"],
                    "action": payload["action"]}
