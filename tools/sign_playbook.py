"""Sign (or verify) a playbook so its declarations are attributable.

A playbook declares which lineages a swarm admits and which artifact digest
each must attest. Those declarations only mean something if someone can check
who published them, so this signs the canonical playbook bytes with Ed25519
and writes a detached ``<playbook>.sig``.

Usage:
    # make a key and sign, pinning that key on the root
    python tools/sign_playbook.py --playbook assets/playbook.example.toml \
        --generate-key
    python -m pandosi.rootd --playbook assets/playbook.example.toml \
        --playbook-key <pubkey hex>

The private key is printed once and never written unless --key-out is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nacl.signing  # noqa: E402

from pandosi.playbook import (SIGNATURE_SUFFIX, load_playbook,  # noqa: E402
                              sign_playbook, verify_playbook_signature)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--key", default=None,
                        help="Ed25519 seed hex (default: with --generate-key "
                             "a fresh key is created)")
    parser.add_argument("--generate-key", action="store_true")
    parser.add_argument("--key-out", default=None,
                        help="write the seed hex to this file (0600 intent)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing signature instead of signing")
    parser.add_argument("--pinned-key", default=None,
                        help="require the signature to come from this key")
    args = parser.parse_args()

    path = Path(args.playbook)
    signature_path = Path(args.out) if args.out else Path(
        str(path) + SIGNATURE_SUFFIX)
    pb = load_playbook(path)

    if args.verify:
        if not signature_path.exists():
            raise SystemExit(f"no signature at {signature_path}")
        attestation = json.loads(signature_path.read_text(encoding="utf-8"))
        valid, detail = verify_playbook_signature(pb, attestation,
                                                 args.pinned_key)
        print(f"{'valid' if valid else 'INVALID'}: {detail}")
        return 0 if valid else 1

    if args.key:
        signing_key = nacl.signing.SigningKey(bytes.fromhex(args.key))
    elif args.generate_key:
        signing_key = nacl.signing.SigningKey.generate()
    else:
        raise SystemExit("pass --key <hex> or --generate-key")

    attestation = sign_playbook(pb, signing_key)
    signature_path.write_text(json.dumps(attestation, indent=2,
                                         sort_keys=True), encoding="utf-8")
    pubkey = attestation["pubkey"]
    print(f"signed {path.name} -> {signature_path.name}")
    print(f"playbook hash : {pb.hash()}")
    print(f"pubkey to pin : {pubkey}")
    if args.key_out:
        Path(args.key_out).write_text(signing_key.encode().hex(),
                                      encoding="utf-8")
        print(f"seed written  : {args.key_out}")
    elif args.generate_key:
        print(f"seed (keep it): {signing_key.encode().hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
