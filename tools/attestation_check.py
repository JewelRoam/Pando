"""End-to-end: a root that declares an artifact accepts only matching stems.

Starts a real coordinator from a generated playbook and one real ONNX lineage stem,
twice:

1. the playbook declares the digest of the graph actually on disk -> the
   stem attests it and registers;
2. the playbook declares a different digest -> the same honest stem is
   refused, because it cannot attest an artifact it is not running.

Case 2 is the point: without attestation the stem would have been admitted on
the strength of its own claim.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.spore import artifact_digest  # noqa: E402

GRAPH = ROOT / "artifacts" / "laya-en.onnx"
LINEAGE = "english"

PLAYBOOK = """[meta]
name = "{name}"
version = 1

[lineages]
admit = ["{lineage}"]

[lineages.artifacts]
{lineage} = "{digest}"
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def nodes(url: str, token: str) -> dict:
    request = urllib.request.Request(
        f"{url}/nodes", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:
            return json.loads(resp.read()).get("nodes", {})
    except (urllib.error.URLError, urllib.error.HTTPError):
        return {}


def attempt(work: Path, digest: str, expect_registered: bool,
            device: str) -> dict:
    playbook = work / f"playbook-{digest[:8]}.toml"
    playbook.write_text(
        PLAYBOOK.format(name=f"attest-{digest[:8]}", lineage=LINEAGE,
                        digest=digest), encoding="utf-8")
    token = secrets.token_urlsafe(18)
    token_file = work / f"token-{digest[:8]}"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    log = (work / f"coordinator-{digest[:8]}.log").open("wb")
    procs = [subprocess.Popen(
        [sys.executable, "-m", "pando.coordinator", "--port", str(port),
         "--playbook", str(playbook), "--pairing-token", token],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=creationflags)]
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/pairing",
                                            timeout=2) as resp:
                    if json.loads(resp.read()).get("required"):
                        break
            except OSError:
                time.sleep(0.2)

        procs.append(subprocess.Popen(
            [sys.executable, "-m", "pando.worker",
             "--node-id", "honest", "--root", url,
             "--pairing-token-file", str(token_file),
             "--backend", "onnx", "--onnx-path", str(GRAPH),
             "--checkpoint", LINEAGE, "--device", device],
            cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creationflags))

        # Give the stem time to compute the digest, load, and register.
        registered = False
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if "honest" in nodes(url, token):
                registered = True
                break
            time.sleep(1.0)
        if registered != expect_registered:
            raise SystemExit(
                f"digest {digest[:8]}: registered={registered}, "
                f"expected {expect_registered}")
        return {"digest": digest[:16], "registered": registered}
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this loads a real model worker; pass --yes")
    if not GRAPH.exists():
        raise SystemExit(f"missing graph: {GRAPH}")

    digest = artifact_digest(GRAPH)
    print(f"{GRAPH.name} digest {digest}")
    work = Path(tempfile.mkdtemp(prefix="pando-attestation-"))
    try:
        accepted = attempt(work, digest, True, args.device)
        print(f"declared correct digest -> registered: {accepted}")
        refused = attempt(work, "0" * 64, False, args.device)
        print(f"declared wrong digest   -> registered: {refused}")
        summary = {"graph": GRAPH.name, "digest": digest,
                   "correct_digest_accepted": accepted["registered"],
                   "wrong_digest_refused": not refused["registered"]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(summary, indent=2))
    print("ATTESTATION CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
