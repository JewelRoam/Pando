"""Verify source distributions contain source, assets, and no heavy artifacts."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REQUIRED_SUFFIXES = (
    "pyproject.toml",
    "README.md",
    "pandosi/__init__.py",
    "pandosi/cli.py",
    "pandosi/resources.py",
    "assets/playbook.example.toml",
    "assets/graft_suite.jsonl",
    "assets/calibration/english.json",
)

FORBIDDEN_PARTS = (
    "/artifacts/",
    "/.venv",
    "/build/",
    "/dist/",
)


def archives(arguments: list[str]) -> list[Path]:
    found: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            found.extend(sorted(path.glob("*.tar.gz")))
        elif path.exists():
            found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args()

    paths = archives(args.archives)
    if not paths:
        raise SystemExit("no source distribution found")
    failed = False
    for path in paths:
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
        missing = [
            suffix for suffix in REQUIRED_SUFFIXES
            if not any(name.endswith(suffix) for name in names)
        ]
        forbidden = [
            name for name in names
            if any(part in f"/{name}/" for part in FORBIDDEN_PARTS)
            or name.endswith((".onnx", ".safetensors"))
        ]
        print(f"{path.name}: {path.stat().st_size} bytes, "
              f"{len(names)} entries")
        if missing:
            failed = True
            for suffix in missing:
                print(f"  MISSING {suffix}")
        if forbidden:
            failed = True
            for name in forbidden[:20]:
                print(f"  FORBIDDEN {name}")
        if not missing and not forbidden:
            print("  source and assets present; heavyweight artifacts absent")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
