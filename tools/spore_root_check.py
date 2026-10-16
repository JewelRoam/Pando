"""Smoke-test rootd's spore serving surface over real HTTP.

Verifies the routes a joiner depends on against a live root process:
  /spore/pubkey            root verify key
  /spore/manifest          current signed manifest (signature checked)
  /spore/manifests/<hash>  a manifest by content hash (round 91 peer fetch)
  auth gate                a wrong pairing token is refused

No mocks: a real rootd subprocess serves a real payload file.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.spore import manifest_hash, verify_manifest  # noqa: E402

import nacl.signing  # noqa: E402

TOKEN = "spore-root-check-token"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, token: str | None = TOKEN) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_ready(url: str, deadline: float = 20.0) -> None:
    end = time.time() + deadline
    while time.time() < end:
        try:
            status, _ = get(f"{url}/spore/pubkey")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise SystemExit("rootd did not become ready")


def main() -> int:
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="pando-spore-root-") as td:
        tmp = Path(td)
        payload = tmp / "payload.onnx"
        payload.write_bytes(secrets.token_bytes(300 * 1024))
        proc = subprocess.Popen(
            [sys.executable, "-m", "pando.rootd",
             "--port", str(port),
             "--serve-spore", str(payload),
             "--pairing-token", TOKEN,
             "--rollout-dir", str(tmp / "rollouts")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            wait_ready(url)

            status, body = get(f"{url}/spore/pubkey")
            assert status == 200, status
            pub_hex = json.loads(body)["pubkey"]
            verify_key = nacl.signing.VerifyKey(bytes.fromhex(pub_hex))
            print("root pubkey served")

            status, body = get(f"{url}/spore/manifest")
            assert status == 200, status
            manifest = json.loads(body)
            assert verify_manifest(manifest, verify_key), "bad manifest signature"
            assert manifest["metadata"]["generation"] == 0
            print("root manifest verified (generation 0)")

            digest = manifest_hash(manifest)
            status, body = get(f"{url}/spore/manifests/{digest}")
            assert status == 200, status
            assert json.loads(body) == manifest, "served manifest differs"
            print("manifest served by content hash")

            status, _ = get(f"{url}/spore/manifests/{'0' * 64}")
            assert status == 404, status
            print("unknown manifest hash -> 404")

            status, _ = get(f"{url}/spore/manifests/{digest}", token="wrong")
            assert status == 403, status
            print("wrong pairing token -> 403")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("SPORE ROOT HTTP SURFACE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
