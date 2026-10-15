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
    # The default backlog is 5. A device whose host is busy - which is the
    # normal state of a personal computer, and the case this project is built
    # for - cannot always accept that fast, and once the queue is full the
    # kernel refuses connections: the root counts a failed call, drops the
    # vote, and escalates without saying why. Measured in round 134: 85% of
    # decisions escalated on a loaded host until this was raised.
    request_queue_size = 128

from contextlib import nullcontext

from .backend import MockBackend, OnnxLayaBackend, TorchLayaBackend
from .common import DecisionRequest
from .graft import assign_role, capability


def host_ms_per_decision(usage: dict | None) -> float | None:
    """Host CPU this process has spent per decision served."""
    if not usage or not usage.get("decisions"):
        return None
    used = time.process_time() - usage["started"]
    return round(1000.0 * used / usage["decisions"], 2)


def build_handler(backend, node_id: str, usage: dict | None = None,
                  gate: threading.Lock | None = None):
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
                                 "checkpoint": backend.checkpoint,
                                 "host_ms_per_decision":
                                     host_ms_per_decision(usage)})
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
                # One decision at a time is what a device is (round 109), and
                # round 135 measured what happens without it: the OpenVINO
                # stems failed about half their calls under four concurrent
                # clients while the ONNX CPU stem never did. A backend that
                # proves itself thread-safe can be served in parallel with
                # --parallel; the default is the honest one.
                with (gate if gate is not None else nullcontext()):
                    vote = backend.decide(req)
                if usage is not None:
                    usage["decisions"] += 1
                self._send(200, {"vote": vote.to_json()})
            except Exception as exc:  # surface backend errors to root
                self._send(500, {"error": str(exc)})

    return StemHandler


class Stem:
    """One worker node bound to a decision backend."""

    def __init__(self, node_id: str, backend, host: str = "127.0.0.1",
                 port: int = 0, advertise_host: str | None = None,
                 serialize: bool = True):
        self.node_id = node_id
        self.backend = backend
        self.host = host
        self.advertise_host = advertise_host or host
        self.port = port
        # A device serves one decision at a time unless its backend can prove
        # otherwise (round 135: the OpenVINO path failed half its calls under
        # four concurrent clients; the ONNX CPU path never failed).
        self.serialize = bool(serialize)
        self._gate = threading.Lock() if self.serialize else None
        # What this device costs the machine it lives on, measured rather than
        # claimed: round 134 found the CPU unit spends 791-1647 ms of host CPU
        # per decision against the NPU's 1.4-1.8 ms, so a fleet that picks by
        # latency alone keeps waking the most expensive member.
        self.usage = {"decisions": 0, "started": time.process_time()}
        self._httpd = _PandoHTTPServer((host, port),
                                       build_handler(backend, node_id,
                                                     self.usage, self._gate))
        self.port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()

    def host_ms_per_decision(self) -> float | None:
        """Host CPU this device has spent per decision (whole process)."""
        return host_ms_per_decision(self.usage)


def heartbeat_loop(stem: Stem, root_url: str,
                   pairing_token: str | None = None,
                   interval: float = 5.0) -> None:
    """Register with root and heartbeat until the process exits."""
    cap = capability()
    registration = {"node_id": stem.node_id, "url": stem.url(),
                    "checkpoint": stem.backend.checkpoint,
                    "capability": cap}
    # Attestation: report the artifact this worker actually loaded, so a
    # root that declares an expected artifact can refuse a stem that claims a
    # lineage it is not running (round 102).
    digest = getattr(stem.backend, "artifact_digest", None)
    if digest:
        registration["artifact_sha256"] = digest
    headers = {"Content-Type": "application/json"}
    if pairing_token:
        registration["pairing_token"] = pairing_token
        headers["Authorization"] = f"Bearer {pairing_token}"
    payload = json.dumps(registration).encode("utf-8")
    while True:
        # Refresh what the device costs the machine it lives on, so a root can
        # prefer a cheap member over an expensive one (round 134). It has to be
        # re-measured every beat: a stem that has served nothing yet has no
        # cost to report.
        measured = stem.host_ms_per_decision()
        if measured is not None:
            registration["capability"] = dict(cap,
                                              host_ms_per_decision=measured)
            payload = json.dumps(registration).encode("utf-8")
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
    parser.add_argument("--checkpoint", default=None,
                        help="lineage/checkpoint identity; defaults by backend")
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
    parser.add_argument("--backend", default="mock",
                        choices=["mock", "onnx", "laya", "text"],
                        help="mock | onnx (ONNX Runtime/OpenVINO) | "
                             "laya (native PyTorch) | text (naive Bayes)")
    parser.add_argument("--onnx-path", default="artifacts/laya-en.int8.onnx")
    parser.add_argument("--repo", default="convaiinnovations/laya")
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="HF id or local path for the torch (laya) backend")
    parser.add_argument("--text-model", default="assets/textmodel.json",
                        help="fitted TextModel for --backend text")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--delay-ms", type=float, default=0.0,
                        help="mock backend only: simulated per-decision "
                             "device speed, for scaling measurements")
    parser.add_argument("--serialize", action="store_true",
                        help="deprecated: stems serialise by default now")
    parser.add_argument("--parallel", action="store_true",
                        help="serve decisions concurrently. Only for a backend "
                             "proven thread-safe - ONNX Runtime's CPU session "
                             "is, the OpenVINO path is not (round 135)")
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
        from .spore import artifact_digest
        from pathlib import Path as _Path
        digest = None
        onnx_file = _Path(args.onnx_path)
        if onnx_file.exists():
            digest = artifact_digest(onnx_file)
            print(f"artifact sha256 {digest}", flush=True)
        pad_to = args.pad_to if args.pad_to is not None \
            else (256 if args.device != "CPU" else None)
        backend = OnnxLayaBackend(args.node_id, args.onnx_path,
                                  repo=args.repo, device=args.device,
                                  pad_to=pad_to,
                                  checkpoint=args.checkpoint)
        backend.artifact_digest = digest
        print("model loaded", flush=True)
    elif args.backend == "laya":
        print(f"loading torch Laya from {args.model} on {args.device} ...",
              flush=True)
        backend = TorchLayaBackend(args.node_id, model=args.model,
                                   device=args.device,
                                   checkpoint=args.checkpoint)
        print("model loaded", flush=True)
    elif args.backend == "text":
        # The cheap, independent member (rounds 122-125): stdlib only, so a
        # device that cannot hold a neural graph can still contribute a vote
        # whose errors do not travel with the family's.
        from .textmodel import TextModel
        print(f"loading text model from {args.text_model} ...", flush=True)
        backend = TextModel.load(args.text_model)
        if args.checkpoint:
            backend.checkpoint = args.checkpoint
        print(f"text model loaded: {backend.summary()}", flush=True)
    else:
        backend = MockBackend(args.node_id, args.checkpoint or "mock",
                              delay_ms=args.delay_ms,
                              serialize=args.serialize)
    stem = Stem(args.node_id, backend, host=args.host, port=args.port,
                advertise_host=args.advertise_host,
                serialize=not args.parallel)
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
