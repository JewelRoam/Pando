"""Compute artifact digests for the local lineage graphs.

Prints a paste-ready ``[lineages.artifacts]`` block for a playbook, so a root
can require every stem to attest the artifact it actually loaded. The
artifact files are not in the repository (they are large and gitignored), so
the digests are host-specific: generate them here rather than copying
someone else's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.spore import artifact_digest  # noqa: E402

DEFAULT_GRAPHS = [
    ("english", "artifacts/laya-en.onnx"),
    ("multilingual", "artifacts/laya-ml.onnx"),
    ("typed-decisions", "artifacts/laya-td.onnx"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="append", default=None,
                        help="name=path; repeatable (defaults to the three "
                             "exported lineages)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    specs = []
    for entry in args.graph or [f"{n}={p}" for n, p in DEFAULT_GRAPHS]:
        name, _, path = entry.partition("=")
        specs.append((name, Path(path) if Path(path).is_absolute()
                      else ROOT / path))

    digests = {}
    for name, path in specs:
        if not path.exists():
            raise SystemExit(f"missing artifact for {name}: {path}")
        digest = artifact_digest(path)
        digests[name] = digest
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"# {name}: {path.name} ({size_mb:.0f} MB)", file=sys.stderr)

    if args.json:
        print(json.dumps(digests, indent=2))
        return 0
    print("[lineages.artifacts]")
    for name, digest in digests.items():
        print(f'{name} = "{digest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
