#!/usr/bin/env python3
"""Pull sealed offers and keep them. Do not unwrap, and do not start a model."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def _seed():
    import importlib.util
    tool = Path(__file__).resolve().parent / "seed.py"
    spec = importlib.util.spec_from_file_location("pandosi_seed_pull_tool", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("seed tool is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(url: str, token: str, method: str = "GET", body: bytes | None = None):
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()


def _extract(packed: bytes, dest: Path, seed) -> bool:
    """Unpack the offered tar. A tar that rotted on the way is not an exception.

    ``tarfile`` raises a truncated archive only while the members are being
    walked, so the guard has to cover the loop and not just the open. One
    truncated offer used to raise ``ReadError`` out of ``pull_once`` and stop
    the whole round: every offer behind it in sort order was never fetched,
    which is how one bad artifact starves the landing point.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(packed), mode="r:")
    except tarfile.TarError:
        return False
    try:
        for member in archive.getmembers():
            if not seed._rel_is_safe(member.name) or not member.isfile():
                return False
            target = dest / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                return False
            target.write_bytes(source.read())
    except (tarfile.TarError, OSError):
        return False
    return (dest / seed.MANIFEST).is_file()


def pull_once(base: str, token: str, inbox: Path, ledger: Path | None = None) -> dict:
    """Download every sealed offer and keep the ones that still verify."""
    seed = _seed()
    base = base.rstrip("/")
    listing = json.loads(_request(f"{base}/offers", token).decode("utf-8"))
    copied = []
    refused = []
    duplicates = []
    for row in listing.get("offers") or []:
        proposal_id = str(row.get("proposal_id") or "")
        if seed._INBOX_NAME.fullmatch(proposal_id) is None:
            refused.append(proposal_id or "?")
            continue
        try:
            packed = _request(f"{base}/offer/{proposal_id}.tar", token)
        except urllib.error.HTTPError:
            # The peer listed an offer it will not hand over. That is one bad
            # offer, not a dead mailbox: record it and keep pulling the rest.
            refused.append(proposal_id)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bundle"
            if not _extract(packed, src, seed):
                refused.append(proposal_id)
                continue
            kept = seed.receive_bundle(src, inbox, ledger=ledger)
        if kept.get("reason") == "already-received":
            # The bytes verify and the delivery is complete; the inbox simply
            # holds this bundle already. Counted, not acked -- nothing prunes
            # the sender's shelf, so a duplicate offer would sit there unacked
            # anyway (measured on the mac shelf 2026-10-19: 51 offers carrying
            # 16 distinct bundles, 16 acks). Skipping it silently made a round
            # that dropped 35 deliveries print "copied: []" and nothing else.
            duplicates.append(proposal_id)
            continue
        if not kept.get("copied"):
            refused.append(proposal_id)
            continue
        ack = json.dumps({"by": "pull"}).encode("utf-8")
        _request(f"{base}/offer/{proposal_id}/ack", token, method="POST", body=ack)
        copied.append(proposal_id)
    return {"copied": copied, "refused": refused, "duplicate": duplicates,
            "unwrapped": False, "started_model": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="base", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--inbox", default="~/.pandosi/inbox")
    parser.add_argument("--ledger", default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    ledger = Path(args.ledger).expanduser() if args.ledger else None

    def run(announce: bool) -> int:
        result = pull_once(args.base, token, Path(args.inbox), ledger)
        if (announce or result["copied"] or result["refused"]
                or result["duplicate"]):
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.once:
        return run(True)
    import time
    while True:
        try:
            run(False)
        except Exception as exc:  # noqa: BLE001 - a pull failure must not kill the loop
            print(json.dumps({"error": str(exc)[:200]}, ensure_ascii=False), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
