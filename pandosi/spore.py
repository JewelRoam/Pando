"""Spore payload integrity: chunking, Merkle tree, Ed25519 manifests.

Design (blueprint v0.5):
- The signed artifact is a small JSON manifest, never the weights themselves.
- Weights/files are split into 64KiB chunks; each chunk is SHA-256 hashed;
  chunk hashes form a binary Merkle tree whose root goes into the manifest.
- Ed25519 (PyNaCl) signs the canonical manifest bytes.
- Fixed-size chunking for v0; content-defined chunking (CDC) lands with
  Phase 5 lineages so evolved swarms transfer only deltas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nacl.signing

CHUNK_SIZE = 64 * 1024


def chunk_file(path: Path, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    data = path.read_bytes()
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def _hash_pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def merkle_root(chunk_hashes: list[bytes]) -> bytes:
    """Binary Merkle root; odd nodes are promoted (Bitcoin-style)."""
    if not chunk_hashes:
        raise ValueError("no chunks")
    level = list(chunk_hashes)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_hash_pair(level[i], level[i + 1])
                 for i in range(0, len(level), 2)]
    return level[0]


def build_manifest(files: dict[str, Path], signer_key: nacl.signing.SigningKey,
                   metadata: dict | None = None) -> dict:
    """Chunk every file, build the Merkle root, sign the canonical manifest."""
    entries = {}
    all_chunk_hashes: list[bytes] = []
    for name, path in files.items():
        hashes = [hashlib.sha256(c).digest() for c in chunk_file(path)]
        all_chunk_hashes.extend(hashes)
        entries[name] = {
            "size": path.stat().st_size,
            "chunks": len(hashes),
            "merkle": hashes and merkle_root(hashes).hex() or "",
        }
    manifest = {
        "version": 1,
        "metadata": metadata or {},
        "files": entries,
        "chunk_merkle_root": merkle_root(all_chunk_hashes).hex(),
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signature"] = signer_key.sign(payload).signature.hex()
    return manifest


def verify_manifest(manifest: dict, verify_key: nacl.signing.VerifyKey) -> bool:
    """Check the Ed25519 signature over the canonical unsigned manifest."""
    manifest = dict(manifest)
    signature = bytes.fromhex(manifest.pop("signature"))
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    try:
        verify_key.verify(payload, signature)
        return True
    except Exception:
        return False


def verify_file(path: Path, manifest: dict, name: str) -> bool:
    """Recompute chunk hashes and the per-file Merkle root for one file."""
    entry = manifest["files"].get(name)
    if entry is None:
        return False
    hashes = [hashlib.sha256(c).digest() for c in chunk_file(path)]
    if len(hashes) != entry["chunks"] or path.stat().st_size != entry["size"]:
        return False
    return merkle_root(hashes).hex() == entry["merkle"]


def generate_signing_key() -> nacl.signing.SigningKey:
    return nacl.signing.SigningKey.generate()
