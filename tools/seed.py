#!/usr/bin/env python3
"""Seed - carry the civilization off a single machine.

Genes and culture mostly live in git: the registry, the rollout corpus, the
playbook example, the corpora. Weight *bytes* do not - weights stay off GitHub
by rule (docs/operations/SEED.md), and git carries their digests instead. What does NOT
live in git at all is the small, irreplaceable part:
the member's secret key, the authorization ledger with its audit hash chain,
the root's node_id -> pubkey bindings, and the group mailbox history. Today all
of that exists on exactly one box, and losing the key is irreversible: the root
still binds the old pubkey to the name, so a freshly generated key is refused as
an impostor and the member can never return under its own name.

A seed is the minimum state that lets a *different* box become the same member
of the same group.

    python3 tools/seed.py export    --out <dir> [--home ~/.pandosi] \
                                     [--identity ~/.pando/identity.json] \
                                     [--rollout-dir rollouts] \
                                     [--passphrase-file ~/.seed-pass]
    python3 tools/seed.py verify    --from <dir> [--passphrase-file ~/.seed-pass]
    python3 tools/seed.py receive   --from <dir> [--inbox ~/.pandosi/inbox]
    python3 tools/seed.py rehydrate --from <dir> --home <dir> \
                                     --identity <path> [--rollout-dir <dir>] \
                                     [--passphrase-file ~/.seed-pass]

The bundle contains a private key. Never commit it. export refuses to write
inside a git working tree unless --force is given.

A seed that travels should not travel as a plaintext key. Pass
--passphrase-file on export and the identity payload is wrapped
scrypt -> XSalsa20-Poly1305 (PyNaCl SecretBox); the same file is required to
verify or rehydrate it. Everything else in the bundle stays readable so a
digest check still means something without the passphrase.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = 1
MANIFEST = "manifest.json"

# scrypt cost. 128 * n * r = 16 MiB, so this stays cheap enough to run on a
# Raspberry Pi 5 while still making an offline guess of a human passphrase
# cost something. Recorded in the bundle so a later change cannot silently
# make an old bundle undecryptable.
SCRYPT = {"n": 1 << 14, "r": 8, "p": 1, "dklen": 32}
SCRYPT_MAXMEM = 64 * 1024 * 1024
SEED_PASSPHRASE_ENV = "PANDO_SEED_PASSPHRASE"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def _derive_key(passphrase: str, salt: bytes, params: dict) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                          n=params["n"], r=params["r"], p=params["p"],
                          dklen=params.get("dklen", 32),
                          maxmem=SCRYPT_MAXMEM)


def wrap_identity(raw: bytes, passphrase: str) -> bytes:
    """Wrap the plaintext identity JSON in scrypt + XSalsa20-Poly1305."""
    from nacl.secret import SecretBox  # noqa: PLC0415
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt, SCRYPT)
    nonce = os.urandom(SecretBox.NONCE_SIZE)
    sealed = SecretBox(key).encrypt(raw, nonce)
    payload = {"kind": "encrypted-identity", "version": 1,
               "cipher": "xsalsa20-poly1305",
               "kdf": "scrypt", "salt": _b64(salt),
               "n": SCRYPT["n"], "r": SCRYPT["r"], "p": SCRYPT["p"],
               "dklen": SCRYPT["dklen"],
               "nonce": _b64(sealed.nonce),
               "ciphertext": _b64(sealed.ciphertext)}
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def unwrap_identity(blob: bytes, passphrase: str) -> bytes:
    """Reverse wrap_identity. Raises ValueError on a wrong passphrase."""
    from nacl.exceptions import CryptoError  # noqa: PLC0415
    from nacl.secret import SecretBox  # noqa: PLC0415
    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not an encrypted identity payload: {exc}") from exc
    if payload.get("kind") != "encrypted-identity":
        raise ValueError("not an encrypted identity payload")
    if payload.get("kdf") != "scrypt" or payload.get("cipher") != "xsalsa20-poly1305":
        raise ValueError(f"unsupported wrap: {payload.get('kdf')}/{payload.get('cipher')}")
    params = {"n": payload["n"], "r": payload["r"], "p": payload["p"],
              "dklen": payload.get("dklen", 32)}
    key = _derive_key(passphrase, _unb64(payload["salt"]), params)
    try:
        return SecretBox(key).decrypt(_unb64(payload["ciphertext"]),
                                      _unb64(payload["nonce"]))
    except CryptoError as exc:
        raise ValueError("wrong passphrase, or the wrapped key was tampered "
                         "with") from exc


def _read_passphrase(path: str | None) -> str | None:
    """Passphrase from a file, or from the environment. Never argv."""
    if path:
        text = Path(path).expanduser().read_text(encoding="utf-8")
        if text.endswith("\r\n"):
            text = text[:-2]
        elif text.endswith("\n") or text.endswith("\r"):
            text = text[:-1]
        return text
    env = os.environ.get(SEED_PASSPHRASE_ENV)
    return env if env else None


def _passphrase_from_args(args) -> str | None:
    return _read_passphrase(getattr(args, "passphrase_file", None))


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
    """True when ``path`` itself is inside a worktree.

    A pre-push hook exports ``GIT_DIR``. Left in the environment, git would
    report that a temporary inbox is inside this repo, and a sealed bundle
    would be refused for the wrong reason.
    """
    probe = path if path.exists() else path.parent
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    try:
        out = subprocess.run(["git", "-C", str(probe), "rev-parse",
                              "--is-inside-work-tree"],
                             capture_output=True, text=True, timeout=5,
                             env=env)
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

    passphrase = _passphrase_from_args(args)
    if passphrase == "":
        print("refusing an empty passphrase; unset it or use a real one",
              file=sys.stderr)
        return 5

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

    # The identity is the only plaintext secret in the bundle. When a
    # passphrase is given we replace its bytes with a sealed envelope, so the
    # manifest digest describes what actually lands on disk.
    sealed: dict[str, bytes] = {}
    if passphrase:
        for entry in files:
            if entry["kind"] != "identity":
                continue
            blob = wrap_identity(Path(entry["_src"]).read_bytes(), passphrase)
            sealed[entry["path"]] = blob
            entry["sha256"] = hashlib.sha256(blob).hexdigest()
            entry["size"] = len(blob)
            entry["encrypted"] = "scrypt+xsalsa20-poly1305"

    digests = [(f["kind"], Path(f["_src"]), f["path"], f["mode"]) for f in files]
    for entry in files:
        entry.pop("_src", None)

    signer = _signer(identity_path)
    body = {"version": VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {"node_fingerprint": getattr(signer, "node_fingerprint", None),
                       "pubkey": getattr(signer, "public_key", None)},
            "files": files,
            "protection": ({"identity": "scrypt+xsalsa20-poly1305",
                            "kdf": SCRYPT, "compare": ["node_fingerprint"]}
                           if sealed else {"identity": "plaintext"}),
            "totals": {"files": len(files),
                       "bytes": sum(f["size"] for f in files)}}
    # Optional. Absent on older bundles. Covered by the signature when present,
    # so a directory named after someone else cannot be quietly relabeled.
    proposal_id = getattr(args, "proposal_id", None)
    author = getattr(args, "author", None)
    if proposal_id:
        body["proposal_id"] = str(proposal_id)
    if author:
        body["author"] = str(author)
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
        if rel in sealed:
            dst.write_bytes(sealed[rel])
        else:
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
    if sealed:
        print("  identity   : ENCRYPTED (scrypt+xsalsa20-poly1305); "
              "keep the passphrase somewhere else")
    else:
        print("  identity   : PLAINTEXT - this bundle is a bare private key; "
              "use --passphrase-file before moving it anywhere")
        print("  WARNING    : this bundle contains an UNPROTECTED PRIVATE KEY "
              "(identity/identity.json)")
    return 0


def load_bundle(src: Path) -> tuple[dict, Path]:
    src = Path(src).expanduser()
    path = src / MANIFEST
    if not path.exists():
        raise SystemExit(f"no {MANIFEST} under {src}")
    return json.loads(path.read_text()), src


def fingerprint_of_identity_json(raw: bytes) -> str:
    """The same 16-hex fingerprint pando.identity derives, from raw bytes."""
    data = json.loads(raw.decode("utf-8"))
    return hashlib.sha256(_unb64(data["public_key"])).hexdigest()[:16]


def cmd_verify(args) -> int:
    manifest, src = load_bundle(Path(args.from_))
    passphrase = _passphrase_from_args(args)
    bad = []
    for entry in manifest["files"]:
        p = src / entry["path"]
        if not p.exists():
            bad.append(f"missing: {entry['path']}")
        elif sha256_file(p) != entry["sha256"]:
            bad.append(f"digest mismatch: {entry['path']}")

    encrypted = [e for e in manifest["files"] if e.get("encrypted")]
    identity_state = "plaintext"
    if encrypted:
        entry = encrypted[0]
        path = src / entry["path"]
        if passphrase is None:
            identity_state = "encrypted - NOT verified (no passphrase given)"
            bad.append("identity payload is encrypted; pass --passphrase-file "
                       "to check that it unwraps to the key the manifest claims")
        elif not path.exists():
            identity_state = "encrypted - missing"
        else:
            expected = (manifest.get("source") or {}).get("node_fingerprint")
            try:
                raw = unwrap_identity(path.read_bytes(), passphrase)
                got = fingerprint_of_identity_json(raw)
            except ValueError as exc:
                identity_state = "decrypt FAILED"
                bad.append(f"identity payload: {exc}")
            else:
                identity_state = f"decrypted, fingerprint {got}"
                if expected and got != expected:
                    identity_state += f" != manifest {expected}"
                    bad.append(f"unwrapped identity is {got} but the manifest "
                               f"says {expected}")
                elif not expected:
                    identity_state += " (manifest carries no fingerprint)"

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
    print(f"identity : {identity_state}")
    for b in bad:
        print(f"  - {b}")
    return 0 if not bad else 1


def cmd_rehydrate(args) -> int:
    manifest, src = load_bundle(Path(args.from_))
    passphrase = _passphrase_from_args(args)
    rc = cmd_verify(args)
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
        if entry.get("encrypted"):
            # cmd_verify already proved this unwraps to the claimed key.
            dst.write_bytes(unwrap_identity((src / rel).read_bytes(),
                                            passphrase or ""))
        else:
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


_INBOX_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def read_bundle(src: Path) -> tuple[dict, Path] | None:
    """Manifest plus its directory. None when the bundle is not readable.

    Unlike ``load_bundle`` this does not exit the process. A receiver that
    refuses a bad directory must return a reason, not take the interpreter
    down with it.
    """
    src = Path(src).expanduser()
    path = src / MANIFEST
    if path.is_symlink() or not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest, src


def _signature_reason(manifest: dict) -> str | None:
    """None when the manifest signature checks out. Does not read file bytes."""
    signature = manifest.get("signature")
    source = manifest.get("source")
    pubkey = source.get("pubkey") if isinstance(source, dict) else None
    if not signature or not pubkey:
        return "unsigned"
    mod = _identity_module()
    if mod is None:
        return "unverifiable"
    body = {key: value for key, value in manifest.items() if key != "signature"}
    if not mod.verify(str(pubkey), body, str(signature)):
        return "bad-signature"
    return None


def _rel_is_safe(rel: object) -> bool:
    if not isinstance(rel, str) or not rel:
        return False
    path = Path(rel)
    if path.is_absolute():
        return False
    return ".." not in path.parts


def _digest_reason(src: Path, manifest: dict) -> str | None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return "bad-digest"
    for entry in files:
        if not isinstance(entry, dict) or not _rel_is_safe(entry.get("path")):
            return "bad-path"
        path = src / entry["path"]
        if path.is_symlink() or not path.is_file():
            return "bad-digest"
        if sha256_file(path) != entry.get("sha256"):
            return "bad-digest"
    return None


def _json_has_secret(value: object) -> bool:
    if isinstance(value, dict):
        if "secret_key" in value:
            return True
        return any(_json_has_secret(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_has_secret(item) for item in value)
    return False


def _key_reason(src: Path, manifest: dict) -> str | None:
    """Why this bundle must not be kept. None when the key is sealed."""
    protection = manifest.get("protection")
    identity_flag = protection.get("identity") if isinstance(protection, dict) else None
    if identity_flag == "plaintext":
        return "plaintext-identity"
    files = [entry for entry in manifest.get("files") or [] if isinstance(entry, dict)]
    identities = [entry for entry in files if entry.get("kind") == "identity"]
    if not identities:
        return "no-identity"
    for entry in identities:
        if not entry.get("encrypted"):
            return "plaintext-identity"
        try:
            payload = json.loads((src / entry["path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return "plaintext-identity"
        if not isinstance(payload, dict) or payload.get("kind") != "encrypted-identity":
            return "plaintext-identity"
        if _json_has_secret(payload):
            return "plaintext-secret"
    for entry in files:
        if entry.get("kind") == "identity":
            continue
        try:
            payload = json.loads((src / entry["path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _json_has_secret(payload):
            return "plaintext-secret"
    return None


def judge_bundle(src: Path) -> dict:
    """Decide whether a bundle already on disk may be kept.

    The signature is checked before the private-key rule, and before any
    copy. Nothing here decrypts, connects, or starts a process.
    """
    result = {"accepted": False, "copied": False, "started_model": False,
              "unwrapped": False, "reason": "no-manifest"}
    loaded = read_bundle(src)
    if loaded is None:
        return result
    manifest, src = loaded
    reason = _signature_reason(manifest)
    if reason:
        result["reason"] = reason
        return result
    reason = _digest_reason(src, manifest)
    if reason:
        result["reason"] = reason
        return result
    reason = _key_reason(src, manifest)
    if reason:
        result["reason"] = reason
        return result
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    result.update({
        "accepted": True,
        "reason": "sealed",
        "proposal_id": str(manifest.get("proposal_id") or ""),
        "author": str(manifest.get("author") or ""),
        "fingerprint": str(source.get("node_fingerprint") or ""),
    })
    return result


def _inbox_name(manifest: dict) -> str | None:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    raw = str(manifest.get("proposal_id") or source.get("node_fingerprint") or "")
    if not _INBOX_NAME.fullmatch(raw):
        return None
    return raw


def _copy_sealed(src: Path, manifest: dict, dest: Path) -> None:
    """Copy the manifest and the files it names. Extra files stay behind.

    The previous copy stays in place until the new bytes are complete.
    """
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    backup = None
    try:
        for entry in manifest["files"]:
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((src / entry["path"]).read_bytes())
            mode = int(str(entry.get("mode") or "644"), 8) & 0o666
            os.chmod(target, mode or 0o600)
        manifest_path = staging / MANIFEST
        manifest_path.write_bytes((src / MANIFEST).read_bytes())
        os.chmod(manifest_path, 0o644)
        if dest.exists():
            backup = dest.parent / f".{dest.name}.old"
            if backup.exists():
                shutil.rmtree(backup)
            dest.rename(backup)
        staging.rename(dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not dest.exists():
            backup.rename(dest)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _append_receive_ledger(ledger, result: dict) -> None:
    if not ledger:
        return
    path = Path(ledger).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: result[key] for key in
           ("accepted", "author", "copied", "fingerprint", "path",
            "proposal_id", "reason") if key in result}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def receive_bundle(src, inbox, *, force: bool = False, ledger=None,
                   name: str | None = None) -> dict:
    """Keep a sealed bundle under ``inbox``.

    Refuses a missing signature, a bad signature, a digest mismatch, a
    plaintext ``secret_key``, and a destination inside a git worktree.
    The kept bytes stay encrypted. This does not open a socket and it does
    not unwrap the key. ``name`` chooses the directory; the default is the
    proposal id already signed into the manifest.
    """
    verdict = judge_bundle(src)
    if not verdict["accepted"]:
        return verdict
    loaded = read_bundle(src)
    if loaded is None:
        verdict["accepted"] = False
        verdict["reason"] = "no-manifest"
        return verdict
    manifest, src = loaded
    inbox = Path(inbox).expanduser()
    if _inside_git_worktree(inbox):
        verdict["accepted"] = False
        verdict["reason"] = "git-worktree"
        return verdict
    name = str(name) if name else _inbox_name(manifest)
    if not name or _INBOX_NAME.fullmatch(name) is None:
        verdict["accepted"] = False
        verdict["reason"] = "bad-name"
        return verdict
    dest = inbox / name
    if dest.exists() and not force:
        verdict["accepted"] = False
        verdict["copied"] = False
        verdict["reason"] = "already-received"
        verdict["path"] = str(dest)
        return verdict
    inbox.mkdir(parents=True, exist_ok=True)
    try:
        _copy_sealed(src, manifest, dest)
    except OSError:
        shutil.rmtree(dest, ignore_errors=True)
        verdict["accepted"] = False
        verdict["reason"] = "copy-failed"
        return verdict
    kept = judge_bundle(dest)
    if not kept["accepted"]:
        shutil.rmtree(dest, ignore_errors=True)
        verdict["accepted"] = False
        verdict["reason"] = "copy-failed"
        return verdict
    kept["copied"] = True
    kept["path"] = str(dest)
    _append_receive_ledger(ledger, kept)
    return kept


def cmd_receive(args) -> int:
    result = receive_bundle(args.from_, args.inbox, force=bool(args.force),
                            ledger=args.ledger)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result.get("copied") else 1


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
    p.add_argument("--passphrase-file", default=None,
                   help="wrap the identity with this passphrase "
                        f"(or set ${SEED_PASSPHRASE_ENV})")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="check digests and the signature")
    p.add_argument("--from", dest="from_", required=True)
    p.add_argument("--passphrase-file", default=None,
                   help="required when the bundle's identity is encrypted")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("receive", help="keep a sealed bundle already on disk")
    p.add_argument("--from", dest="from_", required=True)
    p.add_argument("--inbox", default=str(Path.home() / ".pandosi" / "inbox"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--ledger", default=str(Path.home() / ".pandosi" / "receive-ledger.jsonl"))
    p.set_defaults(func=cmd_receive)

    p = sub.add_parser("rehydrate", help="restore a bundle onto this machine")
    p.add_argument("--from", dest="from_", required=True)
    p.add_argument("--home", default=home_default)
    p.add_argument("--identity", default=identity_default)
    p.add_argument("--rollout-dir", default="rollouts")
    p.add_argument("--passphrase-file", default=None,
                   help="required when the bundle's identity is encrypted")
    p.set_defaults(func=cmd_rehydrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
