#!/usr/bin/env python3
"""Seed - carry the civilization off a single machine.

Genes and culture already live in git: weights, the playbook, the registry,
the rollout corpus. What does NOT live in git is the small, irreplaceable part:
the member's secret key, the authorization ledger with its audit hash chain,
the root's node_id -> pubkey bindings, and the group mailbox history. Today all
of that exists on exactly one box, and losing the key is irreversible: the root
still binds the old pubkey to the name, so a freshly generated key is refused as
an impostor and the member can never return under its own name.

A seed is the minimum state that lets a *different* box become the same member
of the same group.

    python3 tools/seed.py export    --out <dir> [--home ~/.pandosi] \
                                     [--identity ~/.pando/identity.json] \
                                     [--rollout-dir rollouts]
    python3 tools/seed.py verify    --from <dir>
    python3 tools/seed.py rehydrate --from <dir> --home <dir> \
                                     --identity <path> [--rollout-dir <dir>]

The bundle contains a private key. Never commit it. export refuses to write
inside a git working tree unless --force is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = 1
MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _identity_module():
    # Standalone runs (`python tools/seed.py ...`) put tools/ - not the repo
    # root - on sys.path, so `from pando import identity` failed there and the
    # export silently lost its signer and its source fingerprint (drill on
    # Windows found it, 2026-10-17). The drill inserted the path itself, which
    # is why only the standalone CLI was affected.
    import sys as _sys
    from pathlib import Path as _Path
    root = str(_Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    try:
        from pando import identity  # noqa: PLC0415
        return identity
    except Exception:
        return None


def _signer(identity_path: Path):
    mod = _identity_module()
    if mod is None:
        return None
    try:
        return mod.load_or_create(identity_path)
    except Exception:
        return None


def _inside_git_worktree(path: Path) -> bool:
    probe = path if path.exists() else path.parent
    try:
        out = subprocess.run(["git", "-C", str(probe), "rev-parse",
                              "--is-inside-work-tree"],
                             capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except Exception:
        return False


def collect(home: Path, identity_path: Path, rollout_dir: Path) -> list[dict]:
    """(kind, absolute source, bundle-relative target, mode)"""
    items: list[tuple[str, Path, str, str]] = []

    if identity_path.exists():
        items.append(("identity", identity_path, "identity/identity.json", "600"))

    grants = home / "grants.toml"
    if not grants.exists():
        legacy = home / "grants.toml"
        if legacy.exists():
            grants = legacy
    if grants.exists():
        items.append(("ledger", grants, "ledger/grants.toml", "644"))
        audit = grants.with_name(grants.name + ".audit.jsonl")
        if audit.exists():
            items.append(("ledger", audit, "ledger/grants.toml.audit.jsonl", "644"))

    for name, mode in (("pairing.token", "600"), ("node.env", "644")):
        p = home / name
        if p.exists():
            items.append(("config", p, f"config/{name}", mode))

    tasks = home / "mailbox" / "tasks"
    if tasks.is_dir():
        for f in sorted(tasks.glob("*.json")):
            items.append(("mailbox", f, f"mailbox/tasks/{f.name}", "644"))

    bindings = rollout_dir / "identity-bindings.json"
    if bindings.exists():
        items.append(("bindings", bindings, "root/identity-bindings.json", "644"))

    manifest = []
    for kind, src, rel, mode in items:
        manifest.append({"kind": kind, "path": rel, "sha256": sha256_file(src),
                         "size": src.stat().st_size, "mode": mode,
                         "_src": str(src)})
    return manifest


def cmd_export(args) -> int:
    home = Path(args.home).expanduser()
    identity_path = Path(args.identity).expanduser()
    rollout_dir = Path(args.rollout_dir).expanduser()
    out = Path(args.out).expanduser()

    files = collect(home, identity_path, rollout_dir)
    if not files:
        print("nothing to seed: no identity, ledger, config or mailbox found",
              file=sys.stderr)
        return 2

    if _inside_git_worktree(out) and not args.force:
        print(f"refusing to write a private key inside a git worktree: {out}\n"
              f"pick a path outside the repo, or pass --force if you accept the "
              f"risk of committing it", file=sys.stderr)
        return 3

    digests = [(f["kind"], Path(f["_src"]), f["path"], f["mode"]) for f in files]
    for entry in files:
        entry.pop("_src", None)

    signer = _signer(identity_path)
    body = {"version": VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {"node_fingerprint": getattr(signer, "node_fingerprint", None),
                       "pubkey": getattr(signer, "public_key", None)},
            "files": files,
            "totals": {"files": len(files),
                       "bytes": sum(f["size"] for f in files)}}
    signature = None
    if signer is not None:
        signature = signer.sign(body)
    manifest = dict(body)
    manifest["signature"] = signature

    if out.exists():
        if not args.force:
            print(f"{out} already exists; pass --force to overwrite",
                  file=sys.stderr)
            return 4
        shutil.rmtree(out)
    for _kind, src, rel, mode in digests:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, int(mode, 8))
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"seed written: {out}")
    print(f"  files      : {manifest['totals']['files']}")
    print(f"  bytes      : {manifest['totals']['bytes']}")
    print(f"  fingerprint: {manifest['source']['node_fingerprint']}")
    print(f"  signed     : {'yes' if signature else 'no (pando.identity unavailable)'}")
    kinds: dict[str, int] = {}
    for f in files:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    print("  contents   : " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print("  WARNING    : this bundle contains a PRIVATE KEY (identity/identity.json)")
    return 0


def load_bundle(src: Path) -> tuple[dict, Path]:
    src = Path(src).expanduser()
    path = src / MANIFEST
    if not path.exists():
        raise SystemExit(f"no {MANIFEST} under {src}")
    return json.loads(path.read_text()), src


def cmd_verify(args) -> int:
    manifest, src = load_bundle(Path(args.from_))
    bad = []
    for entry in manifest["files"]:
        p = src / entry["path"]
        if not p.exists():
            bad.append(f"missing: {entry['path']}")
        elif sha256_file(p) != entry["sha256"]:
            bad.append(f"digest mismatch: {entry['path']}")
    mod = _identity_module()
    sig_state = "not checked"
    if manifest.get("signature") and mod is not None:
        body = {k: v for k, v in manifest.items() if k != "signature"}
        ok = mod.verify(manifest["source"]["pubkey"], body,
                        manifest["signature"])
        sig_state = "valid" if ok else "INVALID"
        if not ok:
            bad.append("manifest signature invalid")
    elif manifest.get("signature"):
        sig_state = "unverifiable (pando.identity unavailable)"
    print(f"bundle   : {src}")
    print(f"files    : {len(manifest['files'])} checked, {len(bad)} problem(s)")
    print(f"signature: {sig_state}")
    for b in bad:
        print(f"  - {b}")
    return 0 if not bad else 1


def cmd_rehydrate(args) -> int:
    manifest, src = load_bundle(Path(args.from_))
    rc = cmd_verify(argparse.Namespace(from_=args.from_))
    if rc != 0:
        print("refusing to rehydrate a bundle that failed verification",
              file=sys.stderr)
        return rc

    home = Path(args.home).expanduser()
    identity_path = Path(args.identity).expanduser()
    rollout_dir = Path(args.rollout_dir).expanduser()
    restored = []
    for entry in manifest["files"]:
        kind = entry["kind"]
        rel = entry["path"]
        if kind == "identity":
            dst = identity_path
        elif kind == "ledger":
            dst = home / Path(rel).name
        elif kind == "config":
            dst = home / Path(rel).name
        elif kind == "mailbox":
            dst = home / "mailbox" / "tasks" / Path(rel).name
        elif kind == "bindings":
            dst = rollout_dir / "identity-bindings.json"
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dst)
        os.chmod(dst, int(entry["mode"], 8))
        restored.append((kind, dst))
    print(f"rehydrated into {home}")
    print(f"  identity   : {identity_path}")
    print(f"  rollout dir: {rollout_dir}")
    kinds: dict[str, int] = {}
    for kind, _dst in restored:
        kinds[kind] = kinds.get(kind, 0) + 1
    print("  restored   : " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"  fingerprint: {manifest['source']['node_fingerprint']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    home_default = str(Path.home() / ".pandosi")
    identity_default = str(Path.home() / ".pando" / "identity.json")

    p = sub.add_parser("export", help="write a seed bundle")
    p.add_argument("--out", default=str(Path.home() / ".pandosi" / "seed"))
    p.add_argument("--home", default=home_default)
    p.add_argument("--identity", default=identity_default)
    p.add_argument("--rollout-dir", default="rollouts")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="check digests and the signature")
    p.add_argument("--from", dest="from_", required=True)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("rehydrate", help="restore a bundle onto this machine")
    p.add_argument("--from", dest="from_", required=True)
    p.add_argument("--home", default=home_default)
    p.add_argument("--identity", default=identity_default)
    p.add_argument("--rollout-dir", default="rollouts")
    p.set_defaults(func=cmd_rehydrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
