"""Build and validate a branching Spore reproduction DAG."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pandosi.spore import (SporeServer, build_manifest, ensure_parent_lineage,
                           generate_signing_key, lineage_entry, manifest_hash,
                           store_manifest, validate_cached_lineage,
                           validate_lineage_graph)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pandosi-spore-dag-") as td:
        root = Path(td)
        payload = root / "model.bin"
        payload.write_bytes(b"pandosi-reproduction-dag" * 4096)
        key = generate_signing_key()

        origin = build_manifest({"model.bin": payload}, key,
                                metadata={"lineage": "origin", "node": "origin"})
        branch_b = build_manifest({"model.bin": payload}, key,
                                  metadata={"lineage": "origin", "node": "B"},
                                  parent_manifest=origin)
        branch_c = build_manifest({"model.bin": payload}, key,
                                  metadata={"lineage": "origin", "node": "C"},
                                  parent_manifest=origin)
        leaf_d = build_manifest({"model.bin": payload}, key,
                                metadata={"lineage": "origin", "node": "D"},
                                parent_manifest=branch_b)
        leaf_e = build_manifest({"model.bin": payload}, key,
                                metadata={"lineage": "origin", "node": "E"},
                                parent_manifest=branch_c)

        entries = [lineage_entry(manifest)
                   for manifest in (origin, branch_b, branch_c,
                                    leaf_d, leaf_e)]
        valid, problems, summary = validate_lineage_graph(entries)
        assert valid, problems
        assert summary == {"manifests": 5, "roots": 1,
                           "max_generation": 2, "forks": 1}, summary
        print(f"DAG valid: {summary}")

        cache = root / "manifest-cache"
        store_manifest(cache, origin)
        store_manifest(cache, branch_b)
        cached, cached_detail = validate_cached_lineage(cache, branch_b)
        assert cached == "valid", cached_detail
        print("cached parent lineage verified")

        missing_cache = root / "missing-cache"
        missing_status, missing_detail = validate_cached_lineage(
            missing_cache, branch_b)
        assert missing_status == "parent-missing", missing_detail
        print("uncached parent reported as parent-missing")

        bad_child = build_manifest({"model.bin": payload}, key,
                                   metadata={"lineage": "origin",
                                             "node": "bad"},
                                   parent_manifest=origin)
        bad_child["metadata"]["generation"] = 3
        bad_status, bad_detail = validate_cached_lineage(
            cache, bad_child)
        assert bad_status == "invalid", bad_detail
        print("cached generation mismatch rejected")

        # A node that holds only a descendant can pull the missing parent
        # manifest from a peer, verifying hash and signature before storing.
        peer_cache = root / "peer-cache"
        peer_status, _ = validate_cached_lineage(peer_cache, branch_b,
                                                 key.verify_key)
        assert peer_status == "parent-missing", peer_status

        server = SporeServer(branch_b, {"model.bin": payload},
                             history={manifest_hash(origin): origin})
        server.start()
        try:
            fetched_status, fetched_detail = ensure_parent_lineage(
                server.url(), branch_b, peer_cache, key.verify_key)
            assert fetched_status == "valid", fetched_detail
            print("missing parent fetched from peer and verified")

            # A peer that cannot serve the parent leaves lineage unverified.
            blind = SporeServer(branch_b, {"model.bin": payload})
            blind.start()
            try:
                missed, _ = ensure_parent_lineage(
                    blind.url(), branch_b, root / "blind-cache", key.verify_key)
                assert missed == "parent-missing", missed
                print("peer without the parent leaves lineage unverified")
            finally:
                blind.stop()

            # A peer returning a different manifest for the requested hash is
            # untrusted: content hash must match before anything is stored.
            server._manifests[manifest_hash(origin)] = dict(branch_c)
            wrong, _ = ensure_parent_lineage(
                server.url(), branch_b, root / "wrong-cache", key.verify_key)
            assert wrong == "untrusted", wrong
            print("peer returning a different manifest rejected")

            # manifest_hash ignores the signature, so a tampered signature
            # still hashes correctly and must be caught by verify_manifest.
            forged = dict(origin)
            forged["signature"] = "00" * 64
            server._manifests[manifest_hash(origin)] = forged
            unsigned, _ = ensure_parent_lineage(
                server.url(), branch_b, root / "forged-cache", key.verify_key)
            assert unsigned == "untrusted", unsigned
            print("peer returning a bad signature rejected")
        finally:
            server.stop()

        fork_denied, fork_problems, _ = validate_lineage_graph(
            entries, allow_forks=False)
        assert not fork_denied, "fork was accepted in single-chain mode"
        assert any("children" in problem for problem in fork_problems)
        print("fork rejected in single-chain mode")

        missing_parent = [dict(entry) for entry in entries
                          if entry["generation"] > 0]
        valid_missing, missing_problems, _ = validate_lineage_graph(
            missing_parent)
        assert not valid_missing
        assert any("missing parent" in problem for problem in missing_problems)
        print("missing parent rejected")

        rewound = [dict(entry) for entry in entries]
        for entry in rewound:
            if entry["manifest_hash"] == lineage_entry(leaf_d)["manifest_hash"]:
                entry["generation"] = 3
        valid_rewound, rewind_problems, _ = validate_lineage_graph(rewound)
        assert not valid_rewound
        assert any("does not follow parent" in problem
                   for problem in rewind_problems)
        print("generation mismatch rejected")

        double_root = entries + [
            dict(lineage_entry(origin), manifest_hash="f" * 64)]
        valid_roots, root_problems, _ = validate_lineage_graph(double_root)
        assert not valid_roots
        assert any("root" in problem for problem in root_problems)
        print("multiple roots rejected")

    print("SPORE REPRODUCTION DAG OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
