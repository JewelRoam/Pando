#!/usr/bin/env python3
"""Seed drill - prove a *different* box can become the same member.

This is the falsifiable half of the "expansion is survival" claim. The drill
builds a seed from this machine, rehydrates it into a fresh HOME that this
machine's live services have never touched, starts an ISOLATED root whose only
source of truth is the seed, and then asks the two questions that matter:

  1. does the rehydrated key re-register under its own name?   (expect: yes)
  2. does a freshly generated key get that name?               (expect: 403)

If (2) succeeds, identity is not really carried by the seed and the whole
"member survives the loss of its box" story is false. Nothing here talks to the
live island: the drill uses its own port, home and rollout dir.

The seed is going to travel (Mac -> Windows D: -> pi5 SD card), so the drill
runs the *whole* cycle twice: once with a plaintext bundle, once with the
identity wrapped under a passphrase. The wrapped run adds three negative
controls, because a lock nobody tries to pick is decoration:

  3. a sealed bundle with no passphrase is refused
  4. the right passphrase, wrong bytes (a flipped ciphertext) is refused
  5. the wrong passphrase is refused

The same drill can be pointed at a bundle that already exists - the one that
actually crossed the wire - with ``--seed-src``. Give it a passphrase and the
sealed run plus all three negative controls are performed against *that*
artifact instead of one exported here, which is the difference between "the
export path works" and "the thing that traveled still revives the member".

    python3 tools/seed_drill.py [--keep] [--json <path>]
    python3 tools/seed_drill.py --seed-src <dir> --seed-src-passphrase-file <p>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PASSPHRASE = "drill passphrase, not a secret"


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("pandosi_seed", HERE / "seed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def register(url: str, node_id: str, identity, token: str,
             checkpoint: str = "drill") -> tuple[int, str]:
    body = {"node_id": node_id, "url": "http://127.0.0.1:1",
            "checkpoint": checkpoint, "pairing_token": token,
            "artifact_sha256": "deadbeef"}
    body["identity"] = identity.proof(dict(body))
    req = urllib.request.Request(
        f"{url}/register", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def export_bundle(seed, bundle: Path, passphrase_file: Path | None) -> int:
    return seed.cmd_export(argparse.Namespace(
        out=str(bundle), home=str(Path.home() / ".pandosi"),
        identity=str(Path.home() / ".pando" / "identity.json"),
        rollout_dir=str(REPO / "rollouts"), force=True,
        passphrase_file=(str(passphrase_file) if passphrase_file else None)))


def rehydrate(seed, bundle: Path, work: Path, passphrase_file=None) -> int:
    return seed.cmd_rehydrate(argparse.Namespace(
        from_=str(bundle), home=str(work / "home"),
        identity=str(work / "pando" / "identity.json"),
        rollout_dir=str(work / "rollouts"),
        passphrase_file=(str(passphrase_file) if passphrase_file else None)))


def tamper_ciphertext(bundle: Path, out: Path) -> bool:
    """Flip one byte of the sealed identity. Returns False if there is none."""
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(bundle, out)
    target = out / "identity" / "identity.json"
    payload = json.loads(target.read_text())
    if payload.get("kind") != "encrypted-identity":
        return False
    blob = bytearray(base64.urlsafe_b64decode(payload["ciphertext"]))
    blob[0] ^= 0x01
    payload["ciphertext"] = base64.urlsafe_b64encode(bytes(blob)).decode()
    target.write_text(json.dumps(payload, indent=2))
    return True


def tree_digest(bundle: Path) -> str:
    """One digest over the whole tree, so a report names the exact artifact."""
    h = hashlib.sha256()
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(bundle)).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def note_provenance(seed, bundle: Path, report: dict) -> None:
    """Record which artifact was drilled. A drill that cannot say what it
    drilled is an anecdote, not evidence."""
    judged = seed.judge_bundle(bundle)
    report["bundle_path"] = str(bundle)
    report["bundle_digest"] = tree_digest(bundle)
    report["bundle_accepted"] = judged.get("accepted")
    report["bundle_reason"] = judged.get("reason")
    report["bundle_proposal_id"] = judged.get("proposal_id")
    report["bundle_author"] = judged.get("author")
    report["bundle_fingerprint"] = judged.get("fingerprint")


def run_member_cycle(seed, bundle: Path, work: Path, report: dict,
                     passphrase_file=None) -> None:
    """Rehydrate one bundle into `work` and ask the two membership questions."""
    from pando.identity import load_or_create
    from pando.rootd import Coordinator, Root

    manifest, _src = seed.load_bundle(bundle)
    want = (manifest.get("source") or {}).get("node_fingerprint")

    t0 = time.time()
    rc = rehydrate(seed, bundle, work, passphrase_file)
    report["rehydrate_rc"] = rc
    if rc != 0:
        return
    report["restore_seconds"] = round(time.time() - t0, 3)

    identity = load_or_create(work / "pando" / "identity.json")
    report["fingerprint_expected"] = want
    report["fingerprint_rehydrated"] = identity.node_fingerprint
    # When the manifest carries no fingerprint (no signer at export time) the
    # only honest check is that the key still round-trips.
    report["identity_carried"] = (identity.node_fingerprint == want
                                  if want else True)

    token = "drill-pairing-token"
    port = free_port()
    coord = Coordinator(rollout_dir=work / "rollouts",
                        identity_store=work / "rollouts" /
                        "identity-bindings.json")
    root = Root(coordinator=coord, host="127.0.0.1", port=port)
    root.set_pairing_token(token)
    root.set_require_identity(True)
    root.start()
    try:
        code, _body = register(root.url(), "mac-text", identity, token)
        report["same_key_status"] = code
        report["same_key_accepted"] = code == 200

        impostor = load_or_create(work / "pando" / "impostor.json")
        report["impostor_fingerprint"] = impostor.node_fingerprint
        code2, _body2 = register(root.url(), "mac-text", impostor, token)
        report["new_key_status"] = code2
        report["new_key_refused"] = code2 == 403
    finally:
        root.stop()

    report["bundle_bytes"] = manifest["totals"]["bytes"]
    report["bundle_files"] = manifest["totals"]["files"]
    report["ok"] = bool(report.get("identity_carried")
                        and report.get("same_key_accepted")
                        and report.get("new_key_refused"))


def print_cycle(label: str, report: dict) -> None:
    print(f"[{label}]")
    print(f"  bundle          : {report.get('bundle_files')} files, "
          f"{report.get('bundle_bytes')} bytes")
    print(f"  restore time    : {report.get('restore_seconds')}s")
    print(f"  identity carried: {report.get('identity_carried')} "
          f"({report.get('fingerprint_rehydrated')})")
    print(f"  same key        : HTTP {report.get('same_key_status')} "
          f"-> {'accepted' if report.get('same_key_accepted') else 'REFUSED'}")
    print(f"  new key         : HTTP {report.get('new_key_status')} "
          f"-> {'refused' if report.get('new_key_refused') else 'ACCEPTED (bad!)'}")
    print(f"  verdict         : {'PASS' if report.get('ok') else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the temp workspace")
    parser.add_argument("--json", default=None,
                        help="write the drill report as JSON to this path")
    parser.add_argument("--seed-src", default=None,
                        help="drill an existing bundle - the one that actually "
                             "traveled - instead of exporting one here")
    parser.add_argument("--seed-src-passphrase-file", default=None,
                        help="passphrase for --seed-src. With it the bundle "
                             "must be sealed and the sealed run plus all three "
                             "negative controls are done against that artifact")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    seed = _load_seed_module()

    if args.seed_src_passphrase_file and not args.seed_src:
        parser.error("--seed-src-passphrase-file only makes sense with "
                     "--seed-src")

    work = Path(tempfile.mkdtemp(prefix="pandosi-drill."))
    report: dict = {"ok": False, "workdir": str(work), "modes": {}}

    traveled = Path(args.seed_src).expanduser() if args.seed_src else None
    traveled_pass = (Path(args.seed_src_passphrase_file).expanduser()
                     if args.seed_src_passphrase_file else None)
    if traveled is not None:
        report["traveled"] = {"path": str(traveled)}

    # ---- mode 1: plaintext bundle, the original drill ----------------------
    plain = report["modes"].setdefault("plaintext", {})
    plain_attempted = True
    if traveled is not None and traveled_pass is not None:
        # A sealed bundle is not a plaintext one; asking the plaintext question
        # of it would only produce a misleading FAIL.
        plain["skipped"] = "traveled bundle is sealed; see the sealed slot"
        plain_attempted = False
        print("[plaintext]\n  skipped: --seed-src is sealed, so there is no "
              "plaintext run; the sealed slot below drills it")
    else:
        plain_dir = traveled if traveled is not None else work / "plain"
        if traveled is None:
            rc = export_bundle(seed, plain_dir, None)
            if rc != 0:
                print("plaintext export failed", file=sys.stderr)
                return rc
        else:
            note_provenance(seed, plain_dir, plain)
        run_member_cycle(seed, plain_dir, work / "member-plain", plain)
        if traveled is not None and not plain.get("ok"):
            # A sealed bundle dropped into the plaintext slot fails for a
            # reason that has nothing to do with the artifact being bad.
            on_disk = (plain_dir / "identity" / "identity.json")
            sealed_on_disk = False
            if on_disk.is_file():
                try:
                    sealed_on_disk = (json.loads(on_disk.read_text())
                                      .get("kind") == "encrypted-identity")
                except (OSError, ValueError):
                    sealed_on_disk = False
            if sealed_on_disk:
                plain["hint"] = ("this bundle is sealed; re-run with "
                                 "--seed-src-passphrase-file to drill it")
                print("  hint            : the bundle is sealed; re-run with "
                      "--seed-src-passphrase-file to drill it")
        print_cycle("plaintext", plain)

    # ---- mode 2: sealed bundle, plus the negative controls ------------------
    sealed = report["modes"].setdefault("sealed", {})
    sealed_attempted = True
    if traveled is not None and traveled_pass is None:
        sealed["skipped"] = "no passphrase for the traveled bundle"
        sealed_attempted = False
        print("[sealed]\n  skipped: --seed-src was given without "
              "--seed-src-passphrase-file, so the drill cannot open it")
    else:
        if traveled is not None:
            enc_dir = traveled
            pass_file = traveled_pass
            note_provenance(seed, enc_dir, sealed)
        else:
            enc_dir = work / "sealed"
            pass_file = work / "passphrase.txt"
            pass_file.write_text(PASSPHRASE + "\n")
            rc = export_bundle(seed, enc_dir, pass_file)
            if rc != 0:
                print("sealed export failed", file=sys.stderr)
                return rc

        # 3. no passphrase -> the whole run is refused before anything lands.
        sealed["no_passphrase_rc"] = rehydrate(seed, enc_dir,
                                               work / "member-silent")
        sealed["no_passphrase_refused"] = sealed["no_passphrase_rc"] != 0

        # 4. flipped ciphertext with the right passphrase -> refused.
        broken = work / "sealed-tampered"
        sealed["tampered"] = tamper_ciphertext(enc_dir, broken)
        if sealed["tampered"]:
            sealed["tampered_rc"] = rehydrate(seed, broken,
                                              work / "member-tampered",
                                              pass_file)
            sealed["tampered_refused"] = sealed["tampered_rc"] != 0

        # 5. wrong passphrase -> refused.
        wrong_file = work / "wrong-passphrase.txt"
        wrong_file.write_text("not the drill passphrase\n")
        sealed["wrong_passphrase_rc"] = rehydrate(seed, enc_dir,
                                                  work / "member-wrong",
                                                  wrong_file)
        sealed["wrong_passphrase_refused"] = sealed["wrong_passphrase_rc"] != 0

        # ... and the honest path still works end to end.
        run_member_cycle(seed, enc_dir, work / "member-sealed", sealed,
                         pass_file)
        print_cycle("sealed", sealed)
        print(f"  no passphrase   : rc={sealed['no_passphrase_rc']} "
              f"-> {'refused' if sealed['no_passphrase_refused'] else 'ACCEPTED (bad!)'}")
        if sealed.get("tampered"):
            print(f"  tampered bytes  : rc={sealed['tampered_rc']} "
                  f"-> {'refused' if sealed['tampered_refused'] else 'ACCEPTED (bad!)'}")
        print(f"  wrong passphrase: rc={sealed['wrong_passphrase_rc']} "
              f"-> {'refused' if sealed['wrong_passphrase_refused'] else 'ACCEPTED (bad!)'}")

    # A skipped slot is not a failure: the drill asked a question it could not
    # ask of this artifact, and said so out loud.
    plain_ok = bool(plain.get("ok")) if plain_attempted else True
    sealed_ok = True
    if sealed_attempted:
        sealed_ok = bool(sealed.get("ok")
                         and sealed.get("no_passphrase_refused")
                         and sealed.get("tampered_refused")
                         and sealed.get("wrong_passphrase_refused"))
    report["ok"] = plain_ok and sealed_ok
    print(f"verdict: {'PASS' if report['ok'] else 'FAIL'} "
          f"(plaintext={plain_ok}, sealed={sealed_ok})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
