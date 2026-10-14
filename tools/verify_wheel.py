"""Verify that a built PandoSI wheel contains code and small runtime assets."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REQUIRED_SUFFIXES = (
    "pandosi/__init__.py",
    "pandosi/cli.py",
    "pandosi/resources.py",
    "pandosi/doctor.py",
    "share/pandosi/assets/playbook.example.toml",
    "share/pandosi/assets/calibration/english.json",
)


def wheels(arguments: list[str]) -> list[Path]:
    found: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            found.extend(sorted(path.glob("*.whl")))
        elif path.exists():
            found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()

    paths = wheels(args.wheels)
    if not paths:
        raise SystemExit("no wheel found")
    failed = False
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        missing = [
            suffix for suffix in REQUIRED_SUFFIXES
            if not any(name.endswith(suffix) for name in names)
        ]
        print(f"{path.name}: {path.stat().st_size} bytes, "
              f"{len(names)} entries")
        if missing:
            failed = True
            for suffix in missing:
                print(f"  MISSING {suffix}")
        else:
            print("  required code and asset entries present")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
