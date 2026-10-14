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
import hmac
import json
import os
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
                   metadata: dict | None = None,
                   parent_manifest: dict | None = None) -> dict:
    """Chunk every file, build the Merkle root, sign the canonical manifest."""
    metadata = dict(metadata or {})
    if parent_manifest is not None:
        parent_generation = int(
            parent_manifest.get("metadata", {}).get("generation", 0))
        metadata["generation"] = parent_generation + 1
        metadata["parent_manifest_hash"] = manifest_hash(parent_manifest)
    else:
        metadata.setdefault("generation", 0)
    entries = {}
    all_chunk_hashes: list[bytes] = []
    for name, path in files.items():
        hashes = [hashlib.sha256(c).digest() for c in chunk_file(path)]
        all_chunk_hashes.extend(hashes)
        entries[name] = {
            "size": path.stat().st_size,
            "chunks": len(hashes),
            "chunk_hashes": [h.hex() for h in hashes],
            "merkle": hashes and merkle_root(hashes).hex() or "",
        }
    manifest = {
        "version": 1,
        "metadata": metadata,
        "files": entries,
        "chunk_merkle_root": merkle_root(all_chunk_hashes).hex(),
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signature"] = signer_key.sign(payload).signature.hex()
    return manifest


def manifest_hash(manifest: dict) -> str:
    """Content hash of the signed manifest excluding its Ed25519 signature."""
    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_reproduction(parent: dict, child: dict) -> tuple[bool, str]:
    """Verify monotonic generation and direct parent linkage."""
    parent_generation = int(parent.get("metadata", {}).get("generation", 0))
    child_metadata = child.get("metadata", {})
    child_generation = int(child_metadata.get("generation", -1))
    if child_generation != parent_generation + 1:
        return False, (
            f"generation {child_generation} != parent {parent_generation} + 1")
    expected_parent = manifest_hash(parent)
    actual_parent = child_metadata.get("parent_manifest_hash")
    if actual_parent != expected_parent:
        return False, (
            f"parent manifest hash mismatch: {actual_parent} != "
            f"{expected_parent}")
    return True, f"generation {child_generation} links to {expected_parent[:16]}"


def lineage_entry(manifest: dict) -> dict:
    """Return the portable reproduction record for one manifest."""
    metadata = manifest.get("metadata", {})
    return {
        "manifest_hash": manifest_hash(manifest),
        "parent_manifest_hash": metadata.get("parent_manifest_hash"),
        "generation": int(metadata.get("generation", 0)),
        "lineage": metadata.get("lineage"),
    }


def append_lineage(path: Path, manifest: dict) -> dict:
    """Append one local, auditable reproduction event."""
    entry = lineage_entry(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def load_lineage(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def store_manifest(cache_dir: Path, manifest: dict) -> Path:
    """Persist one signed manifest under its content hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = manifest_hash(manifest)
    path = cache_dir / f"{digest}.json"
    if not path.exists():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8")
        os.replace(temporary, path)
    return path


def load_cached_manifest(cache_dir: Path, digest: str) -> dict | None:
    path = cache_dir / f"{digest}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cached_lineage(cache_dir: Path, child: dict,
                           verify_key: nacl.signing.VerifyKey | None = None
                           ) -> tuple[str, str]:
    """Return (valid|parent-missing|invalid, detail) using a local cache."""
    store_manifest(cache_dir, child)
    metadata = child.get("metadata", {})
    generation = int(metadata.get("generation", 0))
    parent_hash = metadata.get("parent_manifest_hash")
    if generation == 0:
        return "valid", "generation-0 root"
    if not parent_hash:
        return "invalid", "non-root manifest has no parent hash"
    parent = load_cached_manifest(cache_dir, parent_hash)
    if parent is None:
        return "parent-missing", f"parent {parent_hash[:16]} not cached"
    if verify_key is not None and not verify_manifest(parent, verify_key):
        return "invalid", "cached parent signature verification failed"
    valid, detail = validate_reproduction(parent, child)
    return ("valid" if valid else "invalid"), detail


def validate_lineage_graph(entries: list[dict],
                           require_single_root: bool = True,
                           allow_forks: bool = True,
                           max_generation: int | None = None
                           ) -> tuple[bool, list[str], dict]:
    """Validate a single-parent, possibly branching reproduction DAG."""
    problems: list[str] = []
    by_hash: dict[str, dict] = {}
    child_counts: dict[str, int] = {}
    roots: list[str] = []

    for index, entry in enumerate(entries):
        manifest_hash_value = entry.get("manifest_hash")
        if not manifest_hash_value:
            problems.append(f"entry {index}: missing manifest_hash")
            continue
        if manifest_hash_value in by_hash:
            problems.append(f"duplicate manifest hash {manifest_hash_value[:16]}")
            continue
        try:
            generation = int(entry.get("generation", -1))
        except (TypeError, ValueError):
            problems.append(f"entry {index}: invalid generation")
            continue
        parent = entry.get("parent_manifest_hash")
        if generation == 0:
            if parent:
                problems.append(
                    f"root {manifest_hash_value[:16]} has a parent")
            roots.append(manifest_hash_value)
        elif not parent:
            problems.append(
                f"generation {generation} node {manifest_hash_value[:16]} "
                "has no parent")
        if max_generation is not None and generation > max_generation:
            problems.append(
                f"generation {generation} exceeds max {max_generation}")
        normalized = dict(entry)
        normalized["generation"] = generation
        by_hash[manifest_hash_value] = normalized
        child_counts.setdefault(manifest_hash_value, 0)

    for manifest_hash_value, entry in by_hash.items():
        generation = entry["generation"]
        parent = entry.get("parent_manifest_hash")
        if generation == 0:
            continue
        parent_entry = by_hash.get(parent)
        if parent_entry is None:
            problems.append(
                f"node {manifest_hash_value[:16]} references missing parent "
                f"{str(parent)[:16]}")
            continue
        if parent_entry["generation"] != generation - 1:
            problems.append(
                f"node {manifest_hash_value[:16]} generation {generation} "
                f"does not follow parent generation "
                f"{parent_entry['generation']}")
        child_counts[parent] = child_counts.get(parent, 0) + 1

    if require_single_root and len(roots) != 1:
        problems.append(f"expected one root generation-0 manifest, got {len(roots)}")
    if not allow_forks:
        for parent, children in child_counts.items():
            if children > 1:
                problems.append(
                    f"parent {parent[:16]} has {children} children")

    summary = {
        "manifests": len(by_hash),
        "roots": len(roots),
        "max_generation": max(
            (entry["generation"] for entry in by_hash.values()), default=-1),
        "forks": sum(1 for children in child_counts.values() if children > 1),
    }
    return not problems, problems, summary


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


# ---------------------------------------------------------------------------
# Spore transfer (Phase 2): serve + pull with verify-on-arrival.
#
# Sandbox gate (THREAT-MODEL.md): pull_spore refuses to run unless the caller
# passes sandbox_confirmed=True - which pandosi-lab sets only after detecting
# a container runtime. The protocol itself has no exploit surface: it only
# serves files the operator explicitly registered.
# ---------------------------------------------------------------------------

import http.server  # noqa: E402
import threading  # noqa: E402
import urllib.request  # noqa: E402


def split_chunks(data: bytes, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


class SporeServer:
    """Serves a signed manifest and file chunks over HTTP (LAN, trusted).

    files: {name: Path} - exactly the files listed in the manifest.
    """

    def __init__(self, manifest: dict, files: dict[str, Path],
                 host: str = "0.0.0.0", port: int = 0,
                 auth_token: str | None = None,
                 history: dict[str, dict] | None = None):
        self.manifest = manifest
        self.auth_token = auth_token
        # Signed manifests this node can serve by content hash. A peer that
        # holds only a descendant can fetch a missing parent from here.
        self._manifests: dict[str, dict] = dict(history or {})
        self._manifests[manifest_hash(manifest)] = manifest
        self._chunks: dict[tuple[str, int], bytes] = {}
        for name, path in files.items():
            for i, chunk in enumerate(chunk_file(path)):
                self._chunks[(name, i)] = chunk
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                expected = server.auth_token
                supplied = self.headers.get("X-Pairing-Token", "")
                authorization = self.headers.get("Authorization", "")
                if authorization.startswith("Bearer "):
                    supplied = authorization[7:]
                if expected and not hmac.compare_digest(
                        supplied.encode("utf-8"),
                        expected.encode("utf-8")):
                    self._send(403, b'{"error":"invalid or missing token"}')
                    return
                if self.path == "/spore/manifest":
                    self._send(200, json.dumps(manifest).encode())
                elif self.path.startswith("/spore/manifests/"):
                    digest = self.path.rsplit("/", 1)[-1]
                    cached = server._manifests.get(digest)
                    if cached is None:
                        self._send(404, b'{"error":"unknown manifest"}')
                    else:
                        self._send(200, json.dumps(cached).encode())
                elif self.path.startswith("/spore/chunk/"):
                    parts = self.path.split("/")
                    name = "/".join(parts[3:-1])
                    index = int(parts[-1])
                    chunk = server._chunks.get((name, index))
                    if chunk is None:
                        self._send(404, b"{}")
                    else:
                        self._send(200, chunk, "application/octet-stream")
                else:
                    self._send(404, b"{}")

        self._httpd = http.server.ThreadingHTTPServer((host, port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def pull_spore(manifest_url: str, out_dir: Path, verify_key,
               sandbox_confirmed: bool = False,
               auth_token: str | None = None) -> dict:
    """Pull a spore payload with verify-on-arrival for every chunk.

    Refuses without sandbox_confirmed=True (see THREAT-MODEL.md). Verifies:
    signature over the manifest, per-file Merkle roots, then on-disk files.
    """
    if not sandbox_confirmed:
        raise PermissionError(
            "spore pull requires sandbox_confirmed=True (container/VM target); "
            "see THREAT-MODEL.md")

    import urllib.request  # local import keeps module import light

    headers = ({"Authorization": f"Bearer {auth_token}"}
               if auth_token else {})
    with urllib.request.urlopen(urllib.request.Request(
            f"{manifest_url}/spore/manifest", headers=headers),
            timeout=30) as resp:
        manifest = json.loads(resp.read())
    if not verify_manifest(manifest, verify_key):
        raise ValueError("manifest signature verification FAILED")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": {}, "chunks_pulled": 0, "chunks_rejected": 0}

    for name, entry in manifest["files"].items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            for i in range(entry["chunks"]):
                expected = bytes.fromhex(entry["chunk_hashes"][i])
                chunk = None
                # Selective re-fetch: a corrupt chunk is re-requested up to
                # 3 times (transient corruption vs persistent tampering).
                for attempt in range(3):
                    url = f"{manifest_url}/spore/chunk/{name}/{i}"
                    with urllib.request.urlopen(urllib.request.Request(
                            url, headers=headers), timeout=30) as resp:
                        chunk = resp.read()
                    if hashlib.sha256(chunk).digest() == expected:
                        break
                else:
                    raise ValueError(
                        f"chunk {name}#{i} failed hash check after 3 attempts "
                        "(tampered or persistently corrupt)")
                fh.write(chunk)
                stats["chunks_pulled"] += 1
        stats["files"][name] = verify_file(target, manifest, name)
    return stats


def fetch_manifest(base_url: str, digest: str,
                   auth_token: str | None = None,
                   timeout: float = 15.0) -> dict:
    """Fetch one signed manifest from a peer by content hash."""
    headers = ({"Authorization": f"Bearer {auth_token}"}
               if auth_token else {})
    url = f"{base_url}/spore/manifests/{digest}"
    with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers),
            timeout=timeout) as resp:
        return json.loads(resp.read())


def ensure_parent_lineage(base_url: str, child: dict, cache_dir: Path,
                          verify_key: nacl.signing.VerifyKey,
                          auth_token: str | None = None
                          ) -> tuple[str, str]:
    """Validate cached lineage, fetching a missing parent from the peer.

    A node that holds only a descendant can pull the missing parent manifest
    from the serving peer and upgrade "parent-missing" to "valid". Only a
    hash-matching, signature-valid manifest is accepted: a hash mismatch or a
    bad signature is reported as "untrusted" and never upgrades the lineage.
    Any network failure leaves the status at "parent-missing".
    """
    status, detail = validate_cached_lineage(cache_dir, child, verify_key)
    if status != "parent-missing":
        return status, detail
    parent_hash = (child.get("metadata", {}) or {}).get("parent_manifest_hash")
    if not parent_hash:
        return status, detail
    try:
        parent = fetch_manifest(base_url, parent_hash, auth_token)
    except Exception as exc:  # network/HTTP failure: stay unverified
        return "parent-missing", (
            f"parent {parent_hash[:16]} not cached and fetch failed: {exc}")
    if manifest_hash(parent) != parent_hash:
        return "untrusted", (
            f"peer returned manifest {manifest_hash(parent)[:16]} but "
            f"{parent_hash[:16]} was requested")
    if not verify_manifest(parent, verify_key):
        return "untrusted", "fetched parent signature verification failed"
    store_manifest(cache_dir, parent)
    return validate_cached_lineage(cache_dir, child, verify_key)


