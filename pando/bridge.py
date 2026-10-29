"""Expose a remote root as one normal HTTP stem.

The bridge lets two groups keep independent local quorums while a parent
group treats the remote group's aggregate decision as one checkpoint. It
uses the existing Stem server and heartbeat loop, so the parent root needs
no special protocol support.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

from .common import (DecisionRequest, Vote, bypass_proxy_for_lan,
                     read_secret_file)
from .worker import Stem, heartbeat_loop


class RemoteRootBackend:
    def __init__(self, node_id: str, upstream_root: str,
                 token: str | None = None, checkpoint: str = "bridge",
                 timeout: float = 10.0):
        self.node_id = node_id
        self.upstream_root = upstream_root.rstrip("/")
        self._token = token
        self.checkpoint = checkpoint
        self.timeout = timeout

    def decide(self, req: DecisionRequest) -> Vote:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(
            f"{self.upstream_root}/decide", data=req.to_json(),
            headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"upstream root unavailable: {exc}") from exc
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        choice = payload.get("choice")
        if choice is None:
            raise RuntimeError(
                f"upstream root returned no choice (tier={payload.get('tier')})")
        probability = payload.get("mean_probability")
        if probability is None:
            probability = payload.get("agreement", 0.5)
        return Vote(self.node_id, self.checkpoint, str(choice),
                    float(probability))


def _token(literal: str | None, path: str | None, label: str) -> str | None:
    if literal and path:
        raise SystemExit(f"{label}: token and token-file are mutually exclusive")
    if path:
        try:
            return read_secret_file(path)
        except ValueError as exc:
            raise SystemExit(str(exc))
    return literal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default="group-bridge")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-pairing-token")
    parser.add_argument("--upstream-pairing-token-file")
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--local-pairing-token")
    parser.add_argument("--local-pairing-token-file")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream-timeout", type=float, default=10.0)
    args = parser.parse_args()

    upstream_token = _token(args.upstream_pairing_token,
                            args.upstream_pairing_token_file, "upstream")
    local_token = _token(args.local_pairing_token,
                         args.local_pairing_token_file, "local")
    checkpoint = args.checkpoint or f"bridge-{args.node_id}"
    bypass_proxy_for_lan()
    backend = RemoteRootBackend(
        args.node_id, args.upstream_root, upstream_token, checkpoint,
        args.upstream_timeout)
    stem = Stem(args.node_id, backend, host=args.host, port=args.port,
                advertise_host=args.advertise_host)
    stem.start()
    print(f"bridge {args.node_id} listening on {stem.url()} -> "
          f"{args.upstream_root} (checkpoint={checkpoint})", flush=True)
    threading.Thread(
        target=heartbeat_loop,
        args=(stem, args.local_root, local_token),
        daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stem.stop()


if __name__ == "__main__":
    main()
