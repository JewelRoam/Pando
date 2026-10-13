"""Multi-hop spore reproduction chain: A -> B -> C with integrity verification.

Proves that a spore can reproduce more than once (the "infinite" part of
"无限繁殖") while maintaining byte-level integrity across all hops.

Each hop: seed serves -> target pulls -> target verifies -> target becomes
the next seed. The chain is auditable: each hop's manifest is signed and
the final file is byte-identical to the original.
"""

from __future__ import annotations

import hashlib
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pandosi.spore import (SporeServer, build_manifest, generate_signing_key,
                           pull_spore, verify_file)


def main() -> None:
    # 1. Create the original payload
    tmp = Path(tempfile.mkdtemp(prefix="pandosi-chain-"))
    payload = tmp / "origin" / "model.bin"
    payload.parent.mkdir(parents=True, exist_ok=True)
    original_data = secrets.token_bytes(2 * 1024 * 1024)  # 2MB
    payload.write_bytes(original_data)
    original_hash = hashlib.sha256(original_data).hexdigest()
    print(f"original: {len(original_data)} bytes, sha256={original_hash[:16]}...",
          flush=True)

    # 2. Generate the root keypair (persisted for the chain)
    key = generate_signing_key()
    pub_hex = key.verify_key.encode().hex()

    # 3. Hop 1: origin -> nodeB
    print("\n--- Hop 1: origin -> nodeB ---", flush=True)
    manifest_1 = build_manifest({"model.bin": payload}, key,
                                metadata={"hop": 1, "origin": original_hash[:16]})
    server_1 = SporeServer(manifest_1, {"model.bin": payload})
    server_1.start()

    node_b_dir = tmp / "nodeB"
    stats_b = pull_spore(server_1.url(), node_b_dir, key.verify_key,
                         sandbox_confirmed=True)
    server_1.stop()
    node_b_file = node_b_dir / "model.bin"
    node_b_hash = hashlib.sha256(node_b_file.read_bytes()).hexdigest()
    hop1_ok = node_b_hash == original_hash
    print(f"  nodeB: {stats_b['chunks_pulled']} chunks | "
          f"hash match: {hop1_ok}", flush=True)
    assert hop1_ok, "Hop 1 integrity failed!"

    # 4. Hop 2: nodeB -> nodeC (nodeB becomes the seed)
    print("\n--- Hop 2: nodeB -> nodeC ---", flush=True)
    manifest_2 = build_manifest({"model.bin": node_b_file}, key,
                                metadata={"hop": 2, "origin": original_hash[:16]})
    server_2 = SporeServer(manifest_2, {"model.bin": node_b_file})
    server_2.start()

    node_c_dir = tmp / "nodeC"
    stats_c = pull_spore(server_2.url(), node_c_dir, key.verify_key,
                         sandbox_confirmed=True)
    server_2.stop()
    node_c_file = node_c_dir / "model.bin"
    node_c_hash = hashlib.sha256(node_c_file.read_bytes()).hexdigest()
    hop2_ok = node_c_hash == original_hash
    print(f"  nodeC: {stats_c['chunks_pulled']} chunks | "
          f"hash match: {hop2_ok}", flush=True)
    assert hop2_ok, "Hop 2 integrity failed!"

    # 5. Hop 3: nodeC -> nodeD (nodeC becomes the seed)
    print("\n--- Hop 3: nodeC -> nodeD ---", flush=True)
    manifest_3 = build_manifest({"model.bin": node_c_file}, key,
                                metadata={"hop": 3, "origin": original_hash[:16]})
    server_3 = SporeServer(manifest_3, {"model.bin": node_c_file})
    server_3.start()

    node_d_dir = tmp / "nodeD"
    stats_d = pull_spore(server_3.url(), node_d_dir, key.verify_key,
                         sandbox_confirmed=True)
    server_3.stop()
    node_d_file = node_d_dir / "model.bin"
    node_d_hash = hashlib.sha256(node_d_file.read_bytes()).hexdigest()
    hop3_ok = node_d_hash == original_hash
    print(f"  nodeD: {stats_d['chunks_pulled']} chunks | "
          f"hash match: {hop3_ok}", flush=True)
    assert hop3_ok, "Hop 3 integrity failed!"

    # 6. Summary
    print(f"\n=== MULTI-HOP CHAIN COMPLETE ===", flush=True)
    print(f"  origin -> B -> C -> D: all byte-identical to original", flush=True)
    print(f"  3 hops, {1 + stats_b['chunks_pulled'] + stats_c['chunks_pulled'] + stats_d['chunks_pulled']} total chunks, "
          f"0 integrity failures", flush=True)
    print(f"  key pinned: {pub_hex[:16]}...", flush=True)
    print(f"  MULTI-HOP SPORE CHAIN OK", flush=True)

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
