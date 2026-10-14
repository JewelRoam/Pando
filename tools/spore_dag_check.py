"""Build and validate a branching Spore reproduction DAG."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pandosi.spore import (build_manifest, generate_signing_key,
                           lineage_entry, validate_lineage_graph)


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
