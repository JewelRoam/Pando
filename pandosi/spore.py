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
            "chunk_hashes": [h.hex() for h in hashes],
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
                 host: str = "0.0.0.0", port: int = 0):
        self.manifest = manifest
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
                if self.path == "/spore/manifest":
                    self._send(200, json.dumps(manifest).encode())
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
               sandbox_confirmed: bool = False) -> dict:
    """Pull a spore payload with verify-on-arrival for every chunk.

    Refuses without sandbox_confirmed=True (see THREAT-MODEL.md). Verifies:
    signature over the manifest, per-file Merkle roots, then on-disk files.
    """
    if not sandbox_confirmed:
        raise PermissionError(
            "spore pull requires sandbox_confirmed=True (container/VM target); "
            "see THREAT-MODEL.md")

    import urllib.request  # local import keeps module import light

    with urllib.request.urlopen(f"{manifest_url}/spore/manifest",
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
                    with urllib.request.urlopen(url, timeout=30) as resp:
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


