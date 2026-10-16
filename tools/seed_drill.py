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

    python3 tools/seed_drill.py [--keep] [--json <path>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the temp workspace")
    parser.add_argument("--json", default=None,
                        help="write the drill report as JSON to this path")
    parser.add_argument("--seed-src", default=None,
                        help="use an existing seed bundle instead of exporting")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    seed = _load_seed_module()
    from pando.identity import load_or_create
    from pando.rootd import Coordinator, Root

    work = Path(tempfile.mkdtemp(prefix="pandosi-drill."))
    bundle = Path(args.seed_src).expanduser() if args.seed_src else work / "seed"
    home = work / "home"
    identity_path = work / "pando" / "identity.json"
    rollout = work / "rollouts"
    report: dict = {"ok": False, "workdir": str(work)}

    t0 = time.time()
    if not args.seed_src:
        rc = seed.cmd_export(argparse.Namespace(
            out=str(bundle), home=str(Path.home() / ".pandosi"),
            identity=str(Path.home() / ".pando" / "identity.json"),
            rollout_dir=str(REPO / "rollouts"), force=True))
        if rc != 0:
            print("export failed", file=sys.stderr)
            return rc
    manifest, _src = seed.load_bundle(bundle)
    rc = seed.cmd_rehydrate(argparse.Namespace(
        from_=str(bundle), home=str(home), identity=str(identity_path),
        rollout_dir=str(rollout)))
    if rc != 0:
        print("rehydrate failed", file=sys.stderr)
        return rc
    t_restore = time.time() - t0

    identity = load_or_create(identity_path)
    want = manifest["source"]["node_fingerprint"]
    report["fingerprint_expected"] = want
    report["fingerprint_rehydrated"] = identity.node_fingerprint
    report["identity_carried"] = identity.node_fingerprint == want

    token = "drill-pairing-token"
    port = free_port()
    coord = Coordinator(rollout_dir=rollout,
                        identity_store=rollout / "identity-bindings.json")
    root = Root(coordinator=coord, host="127.0.0.1", port=port)
    root.set_pairing_token(token)
    root.set_require_identity(True)
    root.start()
    url = root.url()
    try:
        code, body = register(url, "mac-text", identity, token)
        report["same_key_status"] = code
        report["same_key_accepted"] = code == 200

        impostor = load_or_create(work / "pando" / "impostor.json")
        report["impostor_fingerprint"] = impostor.node_fingerprint
        code2, body2 = register(url, "mac-text", impostor, token)
        report["new_key_status"] = code2
        report["new_key_refused"] = code2 == 403

        report["restore_seconds"] = round(t_restore, 3)
        report["bundle_bytes"] = manifest["totals"]["bytes"]
        report["bundle_files"] = manifest["totals"]["files"]
        report["ok"] = bool(report["identity_carried"]
                            and report["same_key_accepted"]
                            and report["new_key_refused"])
    finally:
        root.stop()

    print("seed drill")
    print(f"  bundle          : {manifest['totals']['files']} files, "
          f"{manifest['totals']['bytes']} bytes")
    print(f"  restore time    : {report['restore_seconds']}s")
    print(f"  identity carried: {report['identity_carried']} "
          f"({identity.node_fingerprint})")
    print(f"  same key        : HTTP {report['same_key_status']} "
          f"-> {'accepted' if report['same_key_accepted'] else 'REFUSED'}")
    print(f"  new key         : HTTP {report['new_key_status']} "
          f"-> {'refused' if report['new_key_refused'] else 'ACCEPTED (bad!)'}")
    print(f"  verdict         : {'PASS' if report['ok'] else 'FAIL'}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    if not args.keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
