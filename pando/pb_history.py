"""Playbook version history: an append-only hash chain.

Each entry: {seq, hash, parent, playbook_hash, ts}
where hash = sha256(parent + playbook_hash + ts + seq).

Honest security claim: this chain detects *tampering* with past entries
(it is an integrity check).  It does NOT prove *authenticity* — anyone can
compute a sha256 and forge a valid-looking chain.  If you need provenance,
sign the chain head with an out-of-band key.

This is the file layer of the evolution loop (Phase 5): playbook v(n+1)
declares parent = v(n) hash, so a lineage is auditable end to end.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

GENESIS_PARENT = "0" * 64


def _entry_hash(parent: str, playbook_hash: str, ts: str, seq: int) -> str:
    material = f"{parent}|{playbook_hash}|{ts}|{seq}".encode()
    return hashlib.sha256(material).hexdigest()


def append_entry(history_path: Path, playbook_hash: str,
                 parent: str | None = None) -> dict:
    """Append a new entry whose parent is the chain head (or genesis).

    Explicit parent is allowed to BUILD A FORK (new lineage); the fork's
    first entry must still carry parent = the hash it descends from.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_entries(history_path)
    # Idempotent append: identical playbook content at the head is a no-op
    # (re-serving the same playbook must not advance the chain).
    if entries and parent is None \
            and entries[-1]["playbook_hash"] == playbook_hash:
        return entries[-1]
    seq = (entries[-1]["seq"] + 1) if entries else 0
    head = entries[-1]["hash"] if entries else GENESIS_PARENT
    if parent is None:
        parent = head
    ts = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    entry = {"seq": seq, "parent": parent, "playbook_hash": playbook_hash,
             "ts": ts, "hash": _entry_hash(parent, playbook_hash, ts, seq)}
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def load_entries(history_path: Path) -> list[dict]:
    if not Path(history_path).exists():
        return []
    entries = []
    for line in Path(history_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def check_integrity(history_path: Path, genesis_parent: str = GENESIS_PARENT) -> tuple[bool, list[str]]:
    """Recompute every hash; return (ok, problems)."""
    problems: list[str] = []
    parent = genesis_parent
    for i, entry in enumerate(load_entries(history_path)):
        if entry["parent"] != parent:
            problems.append(f"entry {i}: parent {entry['parent'][:12]} != "
                            f"chain head {parent[:12]}")
        expected = _entry_hash(entry["parent"], entry["playbook_hash"],
                               entry["ts"], entry["seq"])
        if entry["hash"] != expected:
            problems.append(f"entry {i}: hash mismatch (content tampered)")
        parent = entry["hash"]
    return (not problems), problems


def latest(history_path: Path) -> dict | None:
    entries = load_entries(history_path)
    return entries[-1] if entries else None
