"""Measure spore transfer: one request per chunk versus batched.

Round 105 replicated a 1.68 GB artifact in about four minutes because
``pull_spore`` issued one HTTP request per 64 KiB chunk - roughly 26,000
requests, each on a fresh connection. This times both paths on the same
payload and verifies they produce identical bytes.

Usage:
    python tools/spore_transfer_check.py --size-mb 64
    python tools/spore_transfer_check.py --payload artifacts/laya-en.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.spore import (CHUNK_SIZE, SporeServer, artifact_digest,  # noqa: E402
                           artifact_files, build_manifest,
                           generate_signing_key, pull_spore)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", default=None,
                        help="artifact to serve (default: a synthetic file)")
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="pando-transfer-"))
    try:
        if args.payload:
            payload = Path(args.payload)
            source = payload
            files = {file.name: file for file in artifact_files(source)}
        else:
            payload = work / "model.bin"
            payload.write_bytes(b"\x5a" * (args.size_mb * 1024 * 1024))
            source = payload
            files = {source.name: source}
        # An exported graph is a small protobuf plus its weight blob; both
        # must move or the transfer is not measuring the artifact (round 106).
        total_bytes = sum(file.stat().st_size for file in files.values())
        size_mb = total_bytes / (1024 * 1024)
        chunks = sum(-(-file.stat().st_size // CHUNK_SIZE)
                     for file in files.values())
        digest = artifact_digest(source)
        print(f"artifact {source.name}: {len(files)} file(s), "
              f"{size_mb:.1f} MB, {chunks} chunks")

        key = generate_signing_key()
        manifest = build_manifest(files, key,
                                  metadata={"lineage": "transfer-check"})
        server = SporeServer(manifest, files)
        server.start()
        results = {}
        try:
            for label, batch in (("per-chunk", 1),
                                 ("batched", args.batch)):
                out = work / label
                started = time.perf_counter()
                stats = pull_spore(server.url(), out, key.verify_key,
                                   sandbox_confirmed=True,
                                   batch_chunks=batch)
                elapsed = time.perf_counter() - started
                pulled = out / source.name
                ok = artifact_digest(pulled) == digest
                results[label] = {
                    "seconds": round(elapsed, 2),
                    "mb_per_s": round(size_mb / elapsed, 1),
                    "chunks_pulled": stats["chunks_pulled"],
                    "bytes_identical": ok,
                }
                print(f"  {label:10s} {elapsed:6.2f}s "
                      f"({results[label]['mb_per_s']:6.1f} MB/s) "
                      f"identical={ok}")
                if not ok:
                    raise SystemExit(f"{label} transfer produced wrong bytes")
        finally:
            server.stop()

        speedup = (results["per-chunk"]["seconds"]
                   / max(results["batched"]["seconds"], 1e-9))
        summary = {"payload_mb": round(size_mb, 1), "chunks": chunks,
                   "batch_chunks": args.batch, **results,
                   "speedup": round(speedup, 1)}
        print(json.dumps(summary, indent=2))
        print(f"TRANSFER CHECK OK ({speedup:.1f}x faster batched)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
