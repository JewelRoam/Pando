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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _PandoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

from .common import DecisionRequest, Vote, aggregate

HEARTBEAT_TTL = 15.0
QUORUM_K = 5
CALL_TIMEOUT = 5.0


def http_transport(url: str, payload: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class Coordinator:
    def __init__(self, transport=http_transport, quorum_k: int = QUORUM_K):
        self.nodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._transport = transport
        self.quorum_k = quorum_k

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

        result = aggregate(votes)
        out = result.__dict__.copy()
        out["quorum_size"] = len(quorum)
        out["responded"] = len(votes)
        return out


def build_handler(coordinator: Coordinator):
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
                 host: str = "127.0.0.1", port: int = 0):
        self.coordinator = coordinator or Coordinator()
        self._httpd = _PandoHTTPServer((host, port), build_handler(self.coordinator))
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
    args = parser.parse_args()
    root = Root(port=args.port)
    root.start()
    print(f"rootd listening on {root.url()}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        root.stop()


if __name__ == "__main__":
    main()
