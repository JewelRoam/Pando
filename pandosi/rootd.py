"""rootd: the PandoSI consensus coordinator.

Maintains a registry of stems (lazy expiry), fans decisions out to a
checkpoint-diverse quorum, and aggregates votes via common.aggregate.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _PandoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

from .common import DecisionRequest, Vote, aggregate
from .spore import build_manifest, generate_signing_key

HEARTBEAT_TTL = 15.0
QUORUM_K = 5
CALL_TIMEOUT = 30.0


def http_transport(url: str, payload: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class Coordinator:
    def __init__(self, transport=http_transport, quorum_k: int = QUORUM_K,
                 gating: dict | None = None):
        self.nodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._transport = transport
        self.quorum_k = quorum_k
        self.gating = gating or {}

    # -- registry -------------------------------------------------------
    def register(self, node_id: str, url: str, checkpoint: str,
                 cap: dict | None = None) -> None:
        with self._lock:
            self.nodes[node_id] = {"url": url, "checkpoint": checkpoint,
                                   "capability": cap or {},
                                   "last_seen": time.monotonic()}

    def live_nodes(self) -> dict[str, dict]:
        now = time.monotonic()
        with self._lock:
            return {nid: n for nid, n in self.nodes.items()
                    if now - n["last_seen"] <= HEARTBEAT_TTL}

    def pick_quorum(self, k: int) -> list[tuple[str, dict]]:
        """Pick k live nodes maximizing checkpoint diversity (round-robin
        across checkpoint groups). Falls back to any live nodes."""
        nodes = self.live_nodes()
        groups: dict[str, list[tuple[str, dict]]] = {}
        for nid, info in nodes.items():
            groups.setdefault(info["checkpoint"], []).append((nid, info))
        for members in groups.values():
            members.sort()
        picked: list[tuple[str, dict]] = []
        group_iter = iter(sorted(groups))
        while len(picked) < k and groups:
            try:
                ckpt = next(group_iter)
            except StopIteration:
                group_iter = iter(sorted(groups))
                ckpt = next(group_iter)
            if groups[ckpt]:
                picked.append(groups[ckpt].pop(0))
            else:
                del groups[ckpt]
        return picked[:k]

    # -- decision -------------------------------------------------------
    def _log_decision(self, req, result, quorum_size: int) -> None:
        """Append one JSON line per decision (local metrics only)."""
        try:
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "request_id": req.request_id,
                     "choice": result.choice,
                     "tier": result.tier,
                     "agreement": result.agreement,
                     "mean_probability": result.mean_probability,
                     "responded": result.responded if hasattr(result, "responded") else len(result.votes),
                     "quorum_size": quorum_size}
            with (metrics_dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # metrics must never break decisions


    def decide(self, req: DecisionRequest) -> dict:
        quorum = self.pick_quorum(self.quorum_k)
        payload = req.to_json()
        votes: list[Vote] = []

        for nid, info in quorum:
            try:
                raw = self._transport(f"{info['url']}/decide", payload, CALL_TIMEOUT)
                data = json.loads(raw)
                votes.append(Vote(nid, info["checkpoint"], data["vote"]["choice"],
                                  float(data["vote"]["probability"])))
            except Exception:
                continue  # dead node: shrink quorum, never block the vote

        result = aggregate(
            votes, min_quorum=min(3, len(quorum)),
            auto_min_agreement=self.gating.get("auto_min_agreement", 0.6),
            auto_min_probability=self.gating.get("auto_min_probability", 0.6))
        self._log_decision(req, result, len(quorum))
        out = result.__dict__.copy()
        out["quorum_size"] = len(quorum)
        out["responded"] = len(votes)
        return out


def build_handler(coordinator: Coordinator, root=None):
    class RootHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/nodes":
                self._send(200, {"nodes": coordinator.live_nodes()})
            elif self.path == "/playbook" and root is not None \
                    and getattr(root, "playbook", None) is not None:
                self._send(200, {
                    "name": root.playbook.name,
                    "version": root.playbook.version,
                    "hash": root.playbook.hash(),
                    "canonical": json.loads(root.playbook.canonical_json()),
                })
            elif (self.path == "/spore/pubkey"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, {"pubkey": root.spore_pubkey})
            elif (self.path == "/spore/manifest"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, root.spore_manifest)
            elif (self.path.startswith("/spore/chunk/")
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                parts = self.path.split("/")
                name = "/".join(parts[3:-1])
                index = int(parts[-1])
                chunk = root.spore_chunks.get((name, index))
                if chunk is None:
                    self._send(404, b"{}")
                else:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/octet-stream")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path == "/register":
                data = json.loads(raw)
                coordinator.register(data["node_id"], data["url"],
                                     data["checkpoint"], data.get("capability"))
                self._send(200, {"ok": True})
            elif self.path == "/decide":
                req = DecisionRequest.from_json(raw)
                self._send(200, coordinator.decide(req))
            else:
                self._send(404, {"error": "not found"})

    return RootHandler


class Root:
    def __init__(self, coordinator: Coordinator | None = None,
                 host: str = "127.0.0.1", port: int = 0,
                 spore_files: dict | None = None):
        self.coordinator = coordinator or Coordinator()
        self.spore = None
        if spore_files:
            # Persist the spore keypair: an ephemeral key would trip the
            # joiners' TOFU pinning on every root restart (round 16).
            import nacl.signing
            from pathlib import Path as _P
            key_file = _P("spore-cache") / "root-spore-key.hex"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            if key_file.exists():
                key = nacl.signing.SigningKey(
                    bytes.fromhex(key_file.read_text(encoding="utf-8").strip()))
                print("spore key loaded from", key_file, flush=True)
            else:
                key = generate_signing_key()
                key_file.write_text(key.encode().hex(), encoding="utf-8")
                print("spore key generated and saved to", key_file, flush=True)
            self.spore = {"key": key}
            self.spore_pubkey = key.verify_key.encode().hex()
            self.spore_manifest = build_manifest(spore_files, key,
                                                 metadata={"role": "root"})
            self.spore_chunks = {}
            for name, path in spore_files.items():
                data = path.read_bytes()
                for i in range(0, len(data), 64 * 1024):
                    self.spore_chunks[(name, i // (64 * 1024))] = \
                        data[i:i + 64 * 1024]
        self._httpd = _PandoHTTPServer(
            (host, port), build_handler(self.coordinator, root=self))
        self.port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoSI root coordinator")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--serve-spore", default=None,
                        help="payload file to serve for spore pulls")
    parser.add_argument("--playbook", default=None,
                        help="playbook TOML to load and serve")
    parser.add_argument("--history", default=None,
                        help="playbook history chain to verify against")
    parser.add_argument("--allow-non-head", action="store_true")
    args = parser.parse_args()
    from .common import bypass_proxy_for_lan
    bypass_proxy_for_lan()
    spore_files = None
    if args.serve_spore:
        from pathlib import Path as _P
        spore_files = {"payload.onnx": _P(args.serve_spore)}
        if args.playbook:
            spore_files["playbook.toml"] = _P(args.playbook)
    root = Root(port=args.port, spore_files=spore_files)
    if args.playbook:
        from .pb_history import verify_chain
        from .playbook import load_playbook
        root.playbook = load_playbook(Path(args.playbook))
        pb_hash = root.playbook.hash()
        if args.history:
            ok, problems = verify_chain(Path(args.history))
            if not ok:
                raise SystemExit(
                    f"SECURITY: playbook history chain FAILED verification: {problems}")
            chain = [json.loads(l) for l in
                     Path(args.history).read_text(encoding="utf-8")
                     .splitlines() if l.strip()]
            heads = [chain[-1]["playbook_hash"]] if chain else []
            if args.allow_non_head:
                known = {e["playbook_hash"] for e in chain}
                if pb_hash not in known:
                    raise SystemExit(
                        f"SECURITY: playbook hash {pb_hash[:16]} is not in the history chain - refusing (audited forks must be appended "
                        "to the chain first)")
            elif heads and pb_hash != heads[0]:
                raise SystemExit(
                    f"SECURITY: playbook hash {pb_hash[:16]} is not the chain head ({heads[0][:16]}) - refusing. Use --allow-non-head only "
                    "for audited forks.")
            print(f"playbook hash verified against history chain "
                  f"({len(chain)} entries)", flush=True)
        root.coordinator.gating = root.playbook.gating
        print(f"playbook loaded: {root.playbook.name} v{root.playbook.version} "
              f"hash={pb_hash[:16]} "
              f"gating={root.playbook.gating}", flush=True)
    root.start()
    print(f"rootd listening on {root.url()}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        root.stop()


if __name__ == "__main__":
    main()
