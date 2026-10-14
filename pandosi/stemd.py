"""stemd: a PandoSI decision worker node.

Runs an HTTP service exposing /decide, /health, /graft. Registers with a
rootd coordinator and heartbeats periodically.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _PandoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

from .backend import MockBackend, OnnxLayaBackend
from .common import DecisionRequest
from .graft import assign_role, capability


def build_handler(backend, node_id: str):
    class StemHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet by default
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "node_id": node_id,
                                 "checkpoint": backend.checkpoint})
            elif self.path == "/graft":
                cap = capability()
                self._send(200, {"node_id": node_id, "capability": cap,
                                 "role": assign_role(cap)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/decide":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            req = DecisionRequest.from_json(self.rfile.read(length))
            try:
                vote = backend.decide(req)
                self._send(200, {"vote": vote.to_json()})
            except Exception as exc:  # surface backend errors to root
                self._send(500, {"error": str(exc)})

    return StemHandler


class Stem:
    """One worker node bound to a decision backend."""

    def __init__(self, node_id: str, backend, host: str = "127.0.0.1",
                 port: int = 0, advertise_host: str | None = None):
        self.node_id = node_id
        self.backend = backend
        self.host = host
        self.advertise_host = advertise_host or host
        self.port = port
        self._httpd = _PandoHTTPServer((host, port), build_handler(backend, node_id))
        self.port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()


def heartbeat_loop(stem: Stem, root_url: str,
                   pairing_token: str | None = None,
                   interval: float = 5.0) -> None:
    """Register with root and heartbeat until the process exits."""
    cap = capability()
    registration = {"node_id": stem.node_id, "url": stem.url(),
                    "checkpoint": stem.backend.checkpoint,
                    "capability": cap}
    headers = {"Content-Type": "application/json"}
    if pairing_token:
        registration["pairing_token"] = pairing_token
        headers["Authorization"] = f"Bearer {pairing_token}"
    payload = json.dumps(registration).encode("utf-8")
    while True:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{root_url}/register", data=payload,
                                       headers=headers),
                timeout=3) as resp:
                resp.read()
        except Exception:
            pass  # root may be down; retry next interval
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoSI stem worker")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--checkpoint", default="mock")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind; use 0.0.0.0 for LAN")
    parser.add_argument("--advertise-host", default=None,
                        help="address reported to root (defaults to --host)")
    parser.add_argument("--root", default=None, help="rootd URL to register with")
    parser.add_argument("--pairing-token", default=None,
                        help="secret required by a pairing-enabled root")
    parser.add_argument("--pairing-token-file", default=None,
                        help="read the pairing token from a user-owned file")
    parser.add_argument("--backend", default="mock", choices=["mock", "onnx"])
    parser.add_argument("--onnx-path", default="artifacts/laya-en.int8.onnx")
    parser.add_argument("--repo", default="convaiinnovations/laya")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--pad-to", type=int, default=None,
                        help="static-shape bucket for non-CPU EUs (default 256 on GPU/NPU)")
    args = parser.parse_args()
    if args.pairing_token and args.pairing_token_file:
        raise SystemExit("--pairing-token and --pairing-token-file are mutually exclusive")
    if args.pairing_token_file:
        from .common import read_secret_file
        try:
            args.pairing_token = read_secret_file(args.pairing_token_file)
        except ValueError as exc:
            raise SystemExit(str(exc))

    if args.backend == "onnx":
        print(f"loading ONNX Laya from {args.onnx_path} ...", flush=True)
        pad_to = args.pad_to if args.pad_to is not None \
            else (256 if args.device != "CPU" else None)
        backend = OnnxLayaBackend(args.node_id, args.onnx_path,
                                  repo=args.repo, device=args.device,
                                  pad_to=pad_to)
        print("model loaded", flush=True)
    else:
        backend = MockBackend(args.node_id, args.checkpoint)
    stem = Stem(args.node_id, backend, host=args.host, port=args.port,
                advertise_host=args.advertise_host)
    stem.start()
    print(f"stemd {args.node_id} listening on {stem.url()} "
          f"(checkpoint={args.checkpoint})")
    if args.root:
        threading.Thread(target=heartbeat_loop,
                         args=(stem, args.root, args.pairing_token),
                         daemon=True).start()
        print(f"registering with root at {args.root}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stem.stop()


if __name__ == "__main__":
    main()
