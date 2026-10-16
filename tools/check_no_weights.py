#!/usr/bin/env python3
"""Weights do not go to GitHub.

User directive, 2026-10-17: "权重之类的不要上github". A weight file is not
source. Once a blob is in a commit it is in every clone forever, and
`.gitignore` cannot take it back out - it only stops the *next* one.

So the culture story keeps a hard line: git carries code, docs, corpora and
*digests*; weight bytes move out-of-band (direct link / NAS / SD card), keyed
by the digest that git records. `docs/SEED.md` states the rule; this script is
where it can actually be enforced.

    python3 tools/check_no_weights.py              # files git tracks now
    python3 tools/check_no_weights.py --history    # every blob ever committed

Exit 0 = clean, 1 = a weight-like or oversized artifact is reachable from git.
Outside a git work tree it exits 0 and says so (an unpacked sdist has no
tracked-file set to check).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Suffixes that mean "these are weight bytes". Kept deliberately explicit so a
# new runtime is a one-line change someone has to look at, not a regex that
# silently stops matching.
WEIGHT_SUFFIXES = {
    ".onnx",
    ".rknn",
    ".safetensors",
    ".gguf",
    ".ggml",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".h5",
    ".npz",
    ".npy",
    ".tflite",
    ".engine",
    ".mlmodel",
    ".mlpackage",
    ".pb",
    ".pkl",
    ".joblib",
    ".msgpack",
}

# Directory names that mean the same thing for the whole subtree (these are
# bundles, so the files inside may carry an innocent suffix).
WEIGHT_DIRS = {
    ".hf-cache",
    ".mlpackage",
    "models",
    "spore-cache",
    "weights",
}

DEFAULT_MAX_BYTES = 10 * 1024 * 1024


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):
        return None


def in_work_tree(root: Path) -> bool:
    r = _git(root, "rev-parse", "--is-inside-work-tree")
    return bool(r and r.returncode == 0 and r.stdout.strip() == b"true")


def reason_for(path: str, size: int | None, max_bytes: int) -> str | None:
    """Why this path is refused, or None if it is fine."""
    p = Path(path)
    parts = p.parts
    if p.suffix.lower() in WEIGHT_SUFFIXES:
        return f"weight-like suffix {p.suffix.lower()}"
    for part in parts:
        if part.lower() in WEIGHT_DIRS:
            return f"inside weight-like directory {part!r}"
        if Path(part).suffix.lower() in {".mlpackage", ".xcodeproj"}:
            return f"inside bundle {part!r}"
    if size is not None and size > max_bytes:
        return f"{size / 1048576:.1f} MB exceeds the {max_bytes / 1048576:.0f} MB limit"
    return None


def scan_tracked(root: Path, max_bytes: int) -> tuple[list[tuple[str, str]], int]:
    r = _git(root, "ls-files", "-z")
    if r is None or r.returncode != 0:
        return [], 0
    names = [n for n in r.stdout.decode("utf-8", "replace").split("\0") if n]
    bad: list[tuple[str, str]] = []
    for name in names:
        size: int | None = None
        try:
            size = (root / name).stat().st_size
        except OSError:
            pass  # deleted-but-still-staged; suffix check still applies
        why = reason_for(name, size, max_bytes)
        if why:
            bad.append((name, why))
    return bad, len(names)


def scan_history(root: Path, max_bytes: int) -> tuple[list[tuple[str, str]], int]:
    """Every blob in the object database, not just HEAD.

    Uses rev-list --objects --all so each blob gets a name; a commit that was
    later deleted still shows up, which is the whole point.
    """
    r = _git(root, "rev-list", "--objects", "--all")
    if r is None or r.returncode != 0:
        return [], 0
    sha_to_name: dict[str, str] = {}
    sha_to_size: dict[str, int] = {}
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        sha, _, name = line.partition(" ")
        if sha:
            sha_to_name[sha] = name or "(tree)"
    if not sha_to_name:
        return [], 0
    # Default --batch-check output is "<object> <type> <size>"; ask for it
    # explicitly so the field order is a contract, not a memory.
    checker = subprocess.run(
        ["git", "-C", str(root), "cat-file",
         "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input="\n".join(sha_to_name).encode(),
        capture_output=True,
        check=False,
    )
    for line in checker.stdout.decode("utf-8", "replace").splitlines():
        sha, _, rest = line.partition(" ")
        kind, _, size_s = rest.partition(" ")
        if kind != "blob":
            continue
        try:
            sha_to_size[sha] = int(size_s)
        except ValueError:
            continue
    bad: list[tuple[str, str]] = []
    for sha, size in sha_to_size.items():
        name = sha_to_name.get(sha, sha)
        why = reason_for(name, size, max_bytes)
        if why:
            bad.append((f"{name} @ {sha[:12]}", why))
    return bad, len(sha_to_size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refuse weight artifacts in git.")
    ap.add_argument("--history", action="store_true",
                    help="scan every blob ever committed, not just the current tree")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                    help=f"size ceiling per file (default {DEFAULT_MAX_BYTES})")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not in_work_tree(root):
        print("not a git work tree; nothing to check")
        return 0

    bad, scanned = (scan_history if args.history else scan_tracked)(root, args.max_bytes)
    scope = "every blob in history" if args.history else "files tracked now"
    if not scanned:
        print(f"scanned 0 objects ({scope}) - refusing to report a pass "
              "on an empty scan", file=sys.stderr)
        return 1
    if not bad:
        print(f"no weight artifacts in git ({scope}); {scanned} object(s) scanned")
        return 0

    print(f"REFUSED - weight-like artifacts found in git "
          f"({scope}; {scanned} scanned):\n")
    for name, why in bad:
        print(f"  {name}\n      {why}")
    print(
        "\nRule (docs/SEED.md): weights do not go to GitHub. Move the bytes\n"
        "out-of-band and record only the digest in git. To take a file back\n"
        "out of history you need a rewrite (filter-repo) + force push, which\n"
        "affects every clone - so the fix belongs before the commit, here."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
