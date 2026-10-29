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
import time
from pathlib import Path

import nacl.signing

CHUNK_SIZE = 64 * 1024


def chunk_file(path: Path, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    data = path.read_bytes()
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


CDC_AVG_SIZE = 256 * 1024
CDC_MIN_SIZE = 64 * 1024
CDC_MAX_SIZE = 1024 * 1024
CDC_BLOCK = 64


def _cut_bits(avg_size: int) -> int:
    blocks = max(avg_size, CDC_BLOCK) // CDC_BLOCK
    return max(1, blocks.bit_length() - 1)


def chunk_spans(data: bytes, avg_size: int = CDC_AVG_SIZE,
                min_size: int = CDC_MIN_SIZE,
                max_size: int = CDC_MAX_SIZE) -> list[tuple[int, int]]:
    """Content-defined chunk boundaries as (start, end) offsets.

    Boundaries are chosen by hashing each 64-byte block and cutting where the
    low bits are zero, so a *substitution* changes only the blocks it
    overlaps and everything after it re-synchronises. Boundaries land on
    64-byte multiples, which is what makes this vectorisable: the pure-Python
    per-byte rolling hash needed 339 s for one 1.7 GB pass, while this needs
    a couple of seconds (round 107).

    Note the 64-byte granularity: an *insertion* shifts alignment like
    fixed-size chunking until the next cut point, whereas a same-length edit
    is confined to the chunks that overlap it. Weight blobs are fixed-size
    arrays, so that is the case that matters here.
    """
    total = len(data)
    if total == 0:
        return []
    try:
        import numpy as np
    except ImportError:
        return _chunk_spans_python(data, avg_size, min_size, max_size)

    blocks = total // CDC_BLOCK
    if blocks == 0:
        return [(0, total)]
    view = np.frombuffer(data, dtype=np.uint8,
                         count=blocks * CDC_BLOCK)
    view = view.reshape(blocks, CDC_BLOCK).view(np.uint64)
    hashed = view[:, 0].copy()
    for column in range(1, 8):
        rotated = (np.left_shift(hashed, np.uint64(7))
                   | np.right_shift(hashed, np.uint64(57)))
        hashed = np.bitwise_xor(rotated, view[:, column])
    mask = np.uint64((1 << _cut_bits(avg_size)) - 1)
    hits = np.nonzero(np.bitwise_and(hashed, mask) == 0)[0]

    spans: list[tuple[int, int]] = []
    cursor = 0
    for block_index in hits:
        position = int(block_index) * CDC_BLOCK + CDC_BLOCK
        if position - cursor < min_size:
            continue
        while position - cursor > max_size:
            spans.append((cursor, cursor + max_size))
            cursor += max_size
        spans.append((cursor, position))
        cursor = position
    while total - cursor > max_size:
        spans.append((cursor, cursor + max_size))
        cursor += max_size
    if total > cursor:
        spans.append((cursor, total))
    return spans


def _chunk_spans_python(data: bytes, avg_size: int, min_size: int,
                        max_size: int) -> list[tuple[int, int]]:
    """Fallback for environments without numpy; correct but far slower."""
    mask = (1 << _cut_bits(avg_size)) - 1
    spans: list[tuple[int, int]] = []
    start = 0
    rolling = 0
    for index, byte in enumerate(data):
        rolling = ((rolling << 1) + byte) & 0xFFFFFFFFFFFFFFFF
        size = index - start + 1
        if size < min_size:
            continue
        if size >= max_size or (rolling & mask) == 0:
            spans.append((start, index + 1))
            start = index + 1
            rolling = 0
    if start < len(data):
        spans.append((start, len(data)))
    return spans


def chunk_bytes_cdc(data: bytes, avg_size: int = CDC_AVG_SIZE,
                    min_size: int = CDC_MIN_SIZE,
                    max_size: int = CDC_MAX_SIZE) -> list[bytes]:
    return [data[start:end] for start, end in
            chunk_spans(data, avg_size, min_size, max_size)]


def chunk_file_cdc(path: Path, avg_size: int = CDC_AVG_SIZE) -> list[bytes]:
    return chunk_bytes_cdc(Path(path).read_bytes(), avg_size)


def build_manifest_cdc(files: dict, signer_key: nacl.signing.SigningKey,
                       metadata: dict | None = None,
                       parent_manifest: dict | None = None,
                       avg_size: int = CDC_AVG_SIZE) -> dict:
    """Manifest v2: content-defined chunks, addressed by content hash.

    Same shape as v1 (signed canonical JSON over per-file Merkle roots) plus
    ``chunking`` and per-chunk sizes, so a client can tell which chunks it
    already holds and fetch only the rest.
    """
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
        path = Path(path)
        data = path.read_bytes()
        spans = chunk_spans(data, avg_size)
        hashes = [hashlib.sha256(data[start:end]).digest()
                  for start, end in spans]
        all_chunk_hashes.extend(hashes)
        entries[name] = {
            "size": path.stat().st_size,
            "chunks": len(spans),
            "chunk_sizes": [end - start for start, end in spans],
            "chunk_hashes": [h.hex() for h in hashes],
            "merkle": hashes and merkle_root(hashes).hex() or "",
        }
    manifest = {
        "version": 2,
        "chunking": {"algorithm": "gear", "avg_size": avg_size},
        "metadata": metadata,
        "files": entries,
        "chunk_merkle_root": merkle_root(all_chunk_hashes).hex(),
    }
    payload = json.dumps(manifest, sort_keys=True,
                         separators=(",", ":")).encode()
    manifest["signature"] = signer_key.sign(payload).signature.hex()
    return manifest


def is_cdc_manifest(manifest: dict) -> bool:
    return (manifest.get("version", 1) >= 2
            and (manifest.get("chunking") or {}).get("algorithm") == "gear")


def verify_file_cdc(path: Path, manifest: dict, name: str) -> bool:
    """Recompute content-defined chunks and check the stored plan."""
    entry = manifest["files"].get(name)
    if entry is None:
        return False
    avg_size = int((manifest.get("chunking") or {}).get(
        "avg_size", CDC_AVG_SIZE))
    data = Path(path).read_bytes()
    spans = chunk_spans(data, avg_size)
    if len(spans) != entry["chunks"] or \
            path.stat().st_size != entry["size"]:
        return False
    hashes = [hashlib.sha256(data[start:end]).digest()
              for start, end in spans]
    if [h.hex() for h in hashes] != entry["chunk_hashes"]:
        return False
    return merkle_root(hashes).hex() == entry["merkle"]


class ChunkStore:
    """Content-addressed local chunk cache.

    A replica that already holds a parent lineage reuses every chunk the
    child still uses, so only genuinely new chunks are transferred.

    Chunks are packed into one append-only file with a text index. The first
    implementation stored one file per chunk, which made the delta path
    slower than a full transfer: assembling a file re-read thousands of
    small files (round 107). Recently written chunks are also kept in memory,
    because a pull reads back what it just wrote.
    """

    MEMORY_CACHE_BYTES = 64 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = Path(root)
        self._blobs = self.root / "chunks.bin"
        self._index_path = self.root / "chunks.index"
        self._index: dict[str, tuple[int, int]] | None = None
        self._cache: dict[str, bytes] = {}
        self._cache_bytes = 0
        self._reader = None

    def _load(self) -> dict:
        if self._index is not None:
            return self._index
        self.root.mkdir(parents=True, exist_ok=True)
        index: dict[str, tuple[int, int]] = {}
        if self._index_path.exists():
            for line in self._index_path.read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                digest, offset, length = line.split("\t")
                index[digest] = (int(offset), int(length))
        self._index = index
        return index

    def has(self, digest: str) -> bool:
        return digest in self._load() or digest in self._cache

    def get(self, digest: str) -> bytes:
        cached = self._cache.get(digest)
        if cached is not None:
            return cached
        entry = self._load().get(digest)
        if entry is None:
            raise KeyError(digest)
        offset, length = entry
        if self._reader is None:
            self._reader = self._blobs.open("rb")
        self._reader.seek(offset)
        return self._reader.read(length)

    def put(self, digest: str, chunk: bytes) -> None:
        index = self._load()
        if digest in index or digest in self._cache:
            return
        offset = self._blobs.stat().st_size if self._blobs.exists() else 0
        with self._blobs.open("ab") as handle:
            handle.write(chunk)
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{digest}\t{offset}\t{len(chunk)}\n")
        index[digest] = (offset, len(chunk))
        if len(chunk) <= self.MEMORY_CACHE_BYTES // 4:
            self._cache[digest] = chunk
            self._cache_bytes += len(chunk)
            while self._cache_bytes > self.MEMORY_CACHE_BYTES:
                oldest = next(iter(self._cache))
                self._cache_bytes -= len(self._cache.pop(oldest))

    def store_manifest(self, manifest: dict, files: dict) -> int:
        """Seed the store from a local artifact, reusing its chunking."""
        avg_size = int((manifest.get("chunking") or {}).get(
            "avg_size", CDC_AVG_SIZE))
        stored = 0
        for name, path in files.items():
            for chunk in chunk_file_cdc(Path(path), avg_size):
                digest = hashlib.sha256(chunk).hexdigest()
                if not self.has(digest):
                    self.put(digest, chunk)
                    stored += 1
        return stored


def artifact_files(path: Path) -> list[Path]:
    """Every file needed to reproduce an ONNX artifact.

    An exported graph with external weights is a small ``.onnx`` protobuf
    plus a multi-gigabyte ``.onnx.data`` blob. Attesting only the protobuf
    would say nothing about the weights, so the sibling data file is part of
    the artifact.
    """
    path = Path(path)
    files = [path]
    for sibling in sorted(path.parent.glob(path.name + ".data")):
        files.append(sibling)
    return files


def artifact_digest(path: Path, block_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 over an artifact and its external-data siblings.

    Chunked so a multi-gigabyte graph can be attested without holding it in
    memory. File names are mixed in, so swapping a weight blob for another
    file of the same bytes under a different name changes the digest.
    """
    digest = hashlib.sha256()
    for file in artifact_files(path):
        digest.update(file.name.encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                digest.update(block)
    return digest.hexdigest()


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
# Sandbox gate: pull_spore refuses to run unless the caller
# passes sandbox_confirmed=True - which pando-lab sets only after detecting
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
        self._blobs: dict[str, bytes] = {}
        if is_cdc_manifest(manifest):
            # Content-addressed table for delta pulls: a client asks for
            # exactly the chunk hashes it does not already hold.
            for path in files.values():
                for chunk in chunk_file_cdc(
                        Path(path), int(manifest["chunking"]["avg_size"])):
                    self._blobs[hashlib.sha256(chunk).hexdigest()] = chunk
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
                if not self._authorized():
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
                elif self.path.startswith("/spore/chunks/"):
                    # Batched transfer: /spore/chunks/<name>/<start>/<count>
                    # returns the concatenated chunks, so a multi-gigabyte
                    # payload costs hundreds of requests instead of tens of
                    # thousands (round 106). Per-chunk hashes still verify.
                    parts = self.path.split("/")
                    name = "/".join(parts[3:-2])
                    start, count = int(parts[-2]), int(parts[-1])
                    body = b"".join(
                        server._chunks.get((name, index), b"")
                        for index in range(start, start + count))
                    ok = all((name, index) in server._chunks
                             for index in range(start, start + count))
                    if not ok or count <= 0:
                        self._send(404, b"{}")
                    else:
                        self._send(200, body, "application/octet-stream")
                else:
                    self._send(404, b"{}")

            def do_POST(self):  # noqa: N802
                if not self._authorized():
                    return
                if self.path != "/spore/blobs":
                    self._send(404, b"{}")
                    return
                length = int(self.headers.get("Content-Length", 0))
                wanted = json.loads(self.rfile.read(length)).get("hashes", [])
                missing = [h for h in wanted if h not in server._blobs]
                if missing:
                    self._send(404, json.dumps(
                        {"error": "unknown chunks", "missing": missing[:8]}
                    ).encode())
                    return
                body = b"".join(server._blobs[h] for h in wanted)
                self._send(200, body, "application/octet-stream")

            def _authorized(self) -> bool:
                expected = server.auth_token
                supplied = self.headers.get("X-Pairing-Token", "")
                authorization = self.headers.get("Authorization", "")
                if authorization.startswith("Bearer "):
                    supplied = authorization[7:]
                if expected and not hmac.compare_digest(
                        supplied.encode("utf-8"),
                        expected.encode("utf-8")):
                    self._send(403, b'{"error":"invalid or missing token"}')
                    return False
                return True

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


def _pull_cdc(client: "SporeClient", manifest: dict, out_dir: Path,
              store: "ChunkStore | None", batch_chunks: int,
              stats: dict) -> dict:
    """Delta pull for a content-defined manifest.

    Every chunk the local store already holds is not requested at all, so a
    lineage that differs from its parent by a few megabytes moves only the
    chunks that overlap the difference (round 107).
    """
    for name, entry in manifest["files"].items():
        hashes = entry["chunk_hashes"]
        sizes = entry["chunk_sizes"]
        missing = []
        for index, digest in enumerate(hashes):
            if store is not None and store.has(digest):
                stats["chunks_reused"] += 1
            else:
                missing.append(index)

        fetched: dict[int, bytes] = {}
        transfer_started = time.perf_counter()
        for start in range(0, len(missing), batch_chunks):
            group = missing[start:start + batch_chunks]
            body = client.post_json(
                "/spore/blobs", {"hashes": [hashes[i] for i in group]})
            offset = 0
            for index in group:
                size = sizes[index]
                chunk = body[offset:offset + size]
                offset += size
                if hashlib.sha256(chunk).hexdigest() != hashes[index]:
                    stats["chunks_rejected"] += 1
                    chunk = None
                    for _ in range(3):
                        single = client.post_json(
                            "/spore/blobs", {"hashes": [hashes[index]]})
                        if hashlib.sha256(single).hexdigest() == \
                                hashes[index]:
                            chunk = single
                            break
                    if chunk is None:
                        raise ValueError(
                            f"chunk {name}#{index} failed hash check after "
                            "3 attempts (tampered or persistently corrupt)")
                fetched[index] = chunk
                stats["chunks_pulled"] += 1
                stats["bytes_transferred"] += len(chunk)
        stats["seconds_transfer"] = round(
            stats.get("seconds_transfer", 0.0)
            + time.perf_counter() - transfer_started, 3)

        assemble_started = time.perf_counter()
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for index, digest in enumerate(hashes):
                chunk = fetched.get(index)
                if chunk is None:
                    if store is None:
                        raise ValueError(f"chunk {digest[:16]} not available")
                    chunk = store.get(digest)
                handle.write(chunk)
                if store is not None:
                    store.put(digest, chunk)
        stats["files"][name] = verify_file_cdc(target, manifest, name)
        stats["seconds_assemble"] = round(
            stats.get("seconds_assemble", 0.0)
            + time.perf_counter() - assemble_started, 3)
    return stats


def _chunk_length(entry: dict, index: int) -> int:
    """Every chunk is CHUNK_SIZE except the last one of a file."""
    if index < entry["chunks"] - 1:
        return CHUNK_SIZE
    return entry["size"] - (entry["chunks"] - 1) * CHUNK_SIZE


class SporeClient:
    """A keep-alive HTTP client for one spore server."""

    def __init__(self, base_url: str, headers: dict, timeout: float = 60.0):
        import http.client
        from urllib.parse import urlsplit

        self._http = http.client
        self._timeout = timeout
        self._headers = dict(headers)
        parts = urlsplit(base_url)
        self._host = parts.hostname
        self._port = parts.port or 80
        self._connection = http.client.HTTPConnection(
            self._host, self._port, timeout=timeout)

    def get(self, path: str) -> bytes:
        return self._request("GET", path)

    def post_json(self, path: str, payload: dict) -> bytes:
        body = json.dumps(payload).encode("utf-8")
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        return self._request("POST", path, body, headers)

    def _request(self, method: str, path: str, body: bytes | None = None,
                 headers: dict | None = None) -> bytes:
        for attempt in range(2):
            try:
                self._connection.request(method, path, body=body,
                                         headers=headers or self._headers)
                response = self._connection.getresponse()
                data = response.read()
                if response.status != 200:
                    raise ValueError(
                        f"{method} {path} -> {response.status}")
                return data
            except (OSError, self._http.HTTPException, ValueError):
                if attempt:
                    raise
                # A dropped keep-alive connection is worth exactly one retry
                # on a fresh socket before the caller sees a failure.
                self.close()
                self._reconnect()
        raise AssertionError("unreachable")

    def _reconnect(self) -> None:
        self._connection = self._http.HTTPConnection(
            self._host, self._port, timeout=self._timeout)

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception:
            pass


def pull_spore(manifest_url: str, out_dir: Path, verify_key,
               sandbox_confirmed: bool = False,
               auth_token: str | None = None,
               batch_chunks: int = 64,
               chunk_store: "ChunkStore | None" = None) -> dict:
    """Pull a spore payload with verify-on-arrival for every chunk.

    Refuses without sandbox_confirmed=True. Verifies:
    signature over the manifest, per-file Merkle roots, then on-disk files.

    Chunks arrive ``batch_chunks`` at a time over one keep-alive connection
    (round 106); every chunk is still hash-checked before it is written, and
    a batch that fails verification falls back to per-chunk re-fetching.
    ``batch_chunks=1`` reproduces the original request-per-chunk behaviour.
    """
    if not sandbox_confirmed:
        raise PermissionError(
            "spore pull requires sandbox_confirmed=True (container/VM target)")

    headers = ({"Authorization": f"Bearer {auth_token}"}
               if auth_token else {})
    client = SporeClient(manifest_url, headers)
    manifest = json.loads(client.get("/spore/manifest"))
    if not verify_manifest(manifest, verify_key):
        raise ValueError("manifest signature verification FAILED")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": {}, "chunks_pulled": 0, "chunks_rejected": 0,
             "chunks_reused": 0, "bytes_transferred": 0,
             "seconds_transfer": 0.0, "seconds_assemble": 0.0}

    def fetch_chunk(name: str, entry: dict, index: int) -> bytes:
        """One chunk, with the selective re-fetch rule."""
        expected = bytes.fromhex(entry["chunk_hashes"][index])
        for _ in range(3):
            chunk = client.get(f"/spore/chunk/{name}/{index}")
            if hashlib.sha256(chunk).digest() == expected:
                return chunk
            stats["chunks_rejected"] += 1
        raise ValueError(
            f"chunk {name}#{index} failed hash check after 3 attempts "
            "(tampered or persistently corrupt)")

    try:
        if is_cdc_manifest(manifest):
            return _pull_cdc(client, manifest, out_dir, chunk_store,
                             batch_chunks, stats)
        for name, entry in manifest["files"].items():
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                start = 0
                while start < entry["chunks"]:
                    count = min(batch_chunks, entry["chunks"] - start)
                    body = client.get(f"/spore/chunks/{name}/{start}/{count}")
                    offset = 0
                    for index in range(start, start + count):
                        length = _chunk_length(entry, index)
                        chunk = body[offset:offset + length]
                        offset += length
                        expected = bytes.fromhex(entry["chunk_hashes"][index])
                        if len(chunk) != length or \
                                hashlib.sha256(chunk).digest() != expected:
                            # One bad chunk in a batch: refetch it alone so a
                            # single transient fault does not fail the file.
                            chunk = fetch_chunk(name, entry, index)
                        fh.write(chunk)
                        stats["chunks_pulled"] += 1
                    start += count
            stats["files"][name] = verify_file(target, manifest, name)
    finally:
        client.close()
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


