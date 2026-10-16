"""Measure delta replication between two real lineages.

A replica that already holds a parent lineage should only receive the chunks
that actually changed. This builds content-defined manifests for two
artifacts, seeds a local chunk store from the first, and pulls the second
three ways:

1. cold (empty store, batched) - the round-106 behaviour;
2. warm (store seeded from the parent) - the delta path;
3. per-chunk cold, as the reference the batching work was measured against.

Every reconstruction is verified against the source artifact, so a smaller
number is only believable if the bytes are identical.

Usage:
    python tools/spore_delta_check.py \
        --parent artifacts/laya-en.onnx --child artifacts/laya-rlcd-v7.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.spore import (CDC_AVG_SIZE, ChunkStore, SporeServer,  # noqa: E402
                           artifact_digest, artifact_files,
                           build_manifest_cdc, generate_signing_key,
                           pull_spore)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default=str(ROOT / "artifacts"
                                                / "laya-en.onnx"))
    parser.add_argument("--child", default=str(ROOT / "artifacts"
                                               / "laya-rlcd-v7.onnx"))
    parser.add_argument("--avg-size", type=int, default=CDC_AVG_SIZE)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--include-per-chunk", action="store_true",
                        help="also run the round-106 reference leg (slow: one "
                             "request per chunk over the whole artifact)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    parent = Path(args.parent)
    child = Path(args.child)
    for path in (parent, child):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")

    key = generate_signing_key()
    parent_files = {f.name: f for f in artifact_files(parent)}
    child_files = {f.name: f for f in artifact_files(child)}
    size_mb = sum(f.stat().st_size for f in child_files.values()) / (1024 ** 2)
    print(f"child artifact {child.name}: {size_mb:.1f} MB, "
          f"{len(child_files)} file(s)", flush=True)
    print("building content-defined manifests ...", flush=True)
    started = time.perf_counter()
    parent_manifest = build_manifest_cdc(parent_files, key,
                                         metadata={"lineage": "parent"},
                                         avg_size=args.avg_size)
    child_manifest = build_manifest_cdc(child_files, key,
                                        metadata={"lineage": "child"},
                                        parent_manifest=parent_manifest,
                                        avg_size=args.avg_size)
    print(f"  manifests built in {time.perf_counter() - started:.1f}s "
          f"({sum(e['chunks'] for e in child_manifest['files'].values())} "
          f"child chunks)", flush=True)

    expected = artifact_digest(child)
    server = SporeServer(child_manifest, child_files)
    server.start()
    work = Path(tempfile.mkdtemp(prefix="pando-delta-"))
    results = {}
    try:
        legs = [("batched_cold", None, args.batch),
                ("batched_warm", work / "store", args.batch)]
        if args.include_per_chunk:
            legs.insert(0, ("per_chunk_cold", None, 1))
        for label, store, batch in legs:
            if label == "batched_warm":
                seeding = time.perf_counter()
                chunk_store = ChunkStore(store)
                seeded = chunk_store.store_manifest(parent_manifest,
                                                    parent_files)
                print(f"  seeded store from the parent: {seeded} chunks in "
                      f"{time.perf_counter() - seeding:.1f}s", flush=True)
            else:
                chunk_store = None
            out = work / label
            started = time.perf_counter()
            stats = pull_spore(server.url(), out, key.verify_key,
                               sandbox_confirmed=True, batch_chunks=batch,
                               chunk_store=chunk_store)
            elapsed = time.perf_counter() - started
            ok = artifact_digest(out / child.name) == expected
            results[label] = {
                "seconds": round(elapsed, 2),
                "mb": round(stats["bytes_transferred"] / (1024 ** 2), 2),
                "chunks_pulled": stats["chunks_pulled"],
                "chunks_reused": stats["chunks_reused"],
                "seconds_transfer": stats["seconds_transfer"],
                "seconds_assemble": stats["seconds_assemble"],
                "verified": ok,
            }
            print(f"  {label:15s} {elapsed:6.2f}s  "
                  f"{results[label]['mb']:7.2f} MB  "
                  f"pulled={stats['chunks_pulled']} "
                  f"reused={stats['chunks_reused']} "
                  f"(transfer {stats['seconds_transfer']:.1f}s, "
                  f"assemble {stats['seconds_assemble']:.1f}s) verified={ok}",
                  flush=True)
            if not ok:
                raise SystemExit(f"{label}: reconstructed artifact differs")
    finally:
        server.stop()
        shutil.rmtree(work, ignore_errors=True)

    cold = results["batched_cold"]["mb"]
    warm = results["batched_warm"]["mb"]
    summary = {
        "artifact_mb": round(size_mb, 1),
        "avg_chunk": args.avg_size,
        **results,
        "delta_reduction": round(cold / warm, 1) if warm else None,
        "batching_speedup": round(
            results["per_chunk_cold"]["seconds"]
            / max(results["batched_cold"]["seconds"], 1e-9), 1)
        if "per_chunk_cold" in results else None,
    }
    print(json.dumps(summary, indent=2))
    print(f"DELTA CHECK OK (cold {cold:.1f} MB -> warm {warm:.1f} MB, "
          f"{summary['delta_reduction']}x less data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
