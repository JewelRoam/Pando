"""Drill: a proposal survives transport as a signed record, or it is dropped.

GenesisLoop (pando/genesis.py) makes the agenda endogenous in-process. For the
agenda to be the *group's*, a proposal must travel between members and be
checked on arrival -- the same rule as registration: the public key is the
claim, the node name is only a label (docs/design/IDENTITY.md).

This drill is the negative-control set that proves the record is actually
verified and not merely parsed: a clean record passes, every single-field
tamper is dropped, and a key that is not the pinned member's is refused even
though its signature is perfectly valid.

Run:  .venv/bin/python tools/genesis_log_check.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.genesis import (Proposal, proposals_from_log, sign_proposal,
                           verify_proposal)
from pando.identity import Identity


def main() -> int:
    print("== Genesis log drill: signed proposals survive transport ==")

    author = Identity.generate()   # temp identities, nothing on disk
    forger = Identity.generate()
    proposal = Proposal(proposal_id="p1", author="mac-codex",
                        goal="keep the group alive: copy ourselves",
                        action="reproduce.spawn",
                        expected_effect="a second member exists elsewhere",
                        cites=["r1:seed.export"],
                        pubkey=author.public_key)
    record = sign_proposal(proposal, author)
    print(f"author pubkey : {author.public_key[:16]}...")
    print(f"forger pubkey : {forger.public_key[:16]}...")
    print()

    back = verify_proposal(record)

    tampered = copy.deepcopy(record)
    tampered["payload"]["action"] = "recruit.invite"

    unsigned = copy.deepcopy(record)
    unsigned.pop("sig", None)

    wrong_kind = copy.deepcopy(record)
    wrong_kind["kind"] = "chat"

    # The forger signs the *same payload* with a key that is not the author's.
    # The signature is valid under the forger's key, so signature checking alone
    # accepts it; only pinning the author's key refuses it.
    forged = copy.deepcopy(record)
    forged.update(forger.proof(forged["payload"]))

    pinned = {"mac-codex": author.public_key}

    mixed_log = [record, tampered, forged, wrong_kind, unsigned]
    kept = proposals_from_log(mixed_log, pinned=pinned)

    checks = [
        ("clean record round-trips to the same proposal",
         back is not None and back == proposal),
        ("a tampered payload is rejected",
         verify_proposal(tampered) is None),
        ("a missing signature is rejected",
         verify_proposal(unsigned) is None),
        ("a non-proposal record is rejected",
         verify_proposal(wrong_kind) is None),
        ("a validly-signed foreign key passes signature but fails the pin",
         verify_proposal(forged) is not None
         and verify_proposal(forged, expected_pubkey=author.public_key) is None),
        ("only the clean proposal survives a mixed, pinned log",
         kept == [proposal]),
    ]

    print("checks:")
    ok = True
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print()
    print(f"kept {len(kept)}/{len(mixed_log)} entr(ies) from the mixed log")
    print(f"verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
