"""rootd: the PandoSI consensus coordinator.

Maintains a registry of stems (lazy expiry), fans decisions out concurrently
to a checkpoint-diverse quorum, and aggregates votes via common.aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _PandoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

from .common import DecisionRequest, Vote, aggregate
from .spore import build_manifest, generate_signing_key
from .rollout import RolloutRecorder
from .calibrate import Calibrator

HEARTBEAT_TTL = 15.0
QUORUM_K = 5
CALL_TIMEOUT = 30.0
INITIAL_STEM_TIMEOUT = 5.0
MIN_STEM_TIMEOUT = 0.5
STEM_TIMEOUT_MULTIPLIER = 4.0
STEM_TIMEOUT_MARGIN = 0.25
STEM_EWMA_ALPHA = 0.3
FAILURE_BACKOFF_BASE = 1.0
FAILURE_BACKOFF_MAX = 30.0


def _empty_stem_stats() -> dict:
    return {
        "successes": 0,
        "failures": 0,
        "last_latency_ms": None,
        "latency_ewma_ms": None,
        "last_error": None,
        "consecutive_failures": 0,
        "unhealthy_until": 0.0,
    }


def http_transport(url: str, payload: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class Coordinator:
    def __init__(self, transport=http_transport, quorum_k: int = QUORUM_K,
                 gating: dict | None = None,
                 call_timeout: float = CALL_TIMEOUT,
                 initial_stem_timeout: float = INITIAL_STEM_TIMEOUT,
                 min_stem_timeout: float = MIN_STEM_TIMEOUT,
                 timeout_multiplier: float = STEM_TIMEOUT_MULTIPLIER,
                 timeout_margin: float = STEM_TIMEOUT_MARGIN,
                 failure_backoff_base: float = FAILURE_BACKOFF_BASE,
                 failure_backoff_max: float = FAILURE_BACKOFF_MAX,
                 rollout_dir: str | Path = "rollouts"):
        self.nodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._pick_rotation = 0
        self._transport = transport
        self.quorum_k = quorum_k
        self.gating = gating or {}
        self.call_timeout = call_timeout
        self.initial_stem_timeout = initial_stem_timeout
        self.min_stem_timeout = min_stem_timeout
        self.timeout_multiplier = timeout_multiplier
        self.timeout_margin = timeout_margin
        self.failure_backoff_base = failure_backoff_base
        self.failure_backoff_max = failure_backoff_max
        self.playbook_hash: str | None = None
        self._calibrator = Calibrator()
        self._uncalibrated_warned: set[str] = set()
        self.known_questions: set[str] | None = None  # None = accept any
        self._recorder = RolloutRecorder(rollout_dir)

    # -- registry -------------------------------------------------------
    def register(self, node_id: str, url: str, checkpoint: str,
                 cap: dict | None = None) -> None:
        # Calibration is looked up by checkpoint string, so a misnamed stem
        # silently changes the gate. Say so once, at join time, rather than
        # letting it alter every later decision in silence (round 100).
        _, is_calibrated = self._calibrator.calibrate(checkpoint, 0.5)
        if not is_calibrated and checkpoint not in self._uncalibrated_warned:
            self._uncalibrated_warned.add(checkpoint)
            print(f"[calibration] checkpoint {checkpoint!r} has no "
                  f"temperature table; its votes cannot reach tier 'auto'",
                  flush=True)
        with self._lock:
            stats = self.nodes.get(node_id, {}).get("stats")
            self.nodes[node_id] = {
                "url": url,
                "checkpoint": checkpoint,
                "capability": cap or {},
                "last_seen": time.monotonic(),
                "stats": stats or _empty_stem_stats(),
            }

    def live_nodes(self) -> dict[str, dict]:
        now = time.monotonic()
        with self._lock:
            live = {}
            for node_id, info in self.nodes.items():
                if now - info["last_seen"] > HEARTBEAT_TTL:
                    continue
                copy = dict(info)
                stats = dict(info.get("stats", _empty_stem_stats()))
                stats["cooldown_remaining_s"] = round(
                    max(0.0, stats.get("unhealthy_until", 0.0) - now), 3)
                copy["stats"] = stats
                live[node_id] = copy
            return live

    def _selectable_nodes(self) -> dict[str, dict]:
        live = self.live_nodes()
        now = time.monotonic()
        healthy = {
            node_id: info for node_id, info in live.items()
            if info["stats"].get("unhealthy_until", 0.0) <= now
        }
        return healthy or live

    def pick_quorum(self, k: int) -> list[tuple[str, dict]]:
        """Pick k live nodes maximizing checkpoint diversity (round-robin
        across checkpoint groups). Falls back to any live nodes.

        Within a checkpoint group the starting member rotates per decision, so
        concurrent decisions can use disjoint subsets. Without that rotation
        every decision used the same voters and adding replicas bought no
        capacity at all (round 109).
        """
        nodes = self._selectable_nodes()
        groups: dict[str, list[tuple[str, dict]]] = {}
        for nid, info in nodes.items():
            groups.setdefault(info["checkpoint"], []).append((nid, info))
        for members in groups.values():
            members.sort(key=lambda item: (
                item[1]["stats"].get("latency_ewma_ms")
                if item[1]["stats"].get("latency_ewma_ms") is not None
                else float("inf"),
                item[0]))
        with self._lock:
            rotation = self._pick_rotation
            self._pick_rotation = (rotation + 1) % 1_000_003
        for checkpoint, members in groups.items():
            if len(members) > 1:
                shift = rotation % len(members)
                groups[checkpoint] = members[shift:] + members[:shift]
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
    def _log_decision(self, req, result, quorum_size: int,
                      fanout_ms: float) -> None:
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
                     "quorum_size": quorum_size,
                     "fanout_ms": fanout_ms}
            with (metrics_dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # metrics must never break decisions

    def _record_stem_call(self, node_id: str, ok: bool,
                          latency_ms: float, error: str | None = None) -> None:
        with self._lock:
            info = self.nodes.get(node_id)
            if info is None:
                return
            stats = info.setdefault("stats", _empty_stem_stats())
            stats["last_latency_ms"] = round(latency_ms, 3)
            if ok:
                stats["successes"] += 1
                stats["consecutive_failures"] = 0
                stats["unhealthy_until"] = 0.0
                previous = stats.get("latency_ewma_ms")
                stats["latency_ewma_ms"] = round(
                    latency_ms if previous is None else
                    STEM_EWMA_ALPHA * latency_ms
                    + (1 - STEM_EWMA_ALPHA) * previous,
                    3)
                stats["last_error"] = None
            else:
                stats["failures"] += 1
                failures = int(stats.get("consecutive_failures", 0)) + 1
                stats["consecutive_failures"] = failures
                backoff = min(
                    self.failure_backoff_max,
                    self.failure_backoff_base
                    * (2 ** min(failures - 1, 50)))
                stats["unhealthy_until"] = time.monotonic() + backoff
                stats["last_error"] = error

    def _timeout_for(self, node_id: str) -> float:
        with self._lock:
            info = self.nodes.get(node_id)
            stats = info.get("stats", {}) if info else {}
            ewma_ms = stats.get("latency_ewma_ms")
        if ewma_ms is None:
            return self.initial_stem_timeout
        learned_s = (float(ewma_ms) / 1000.0) * self.timeout_multiplier \
            + self.timeout_margin
        return min(self.call_timeout,
                   max(self.min_stem_timeout, learned_s))

    def _call_stem(self, node_id: str, info: dict,
                   payload: bytes) -> Vote | None:
        timeout = self._timeout_for(node_id)
        started = time.perf_counter()
        try:
            raw = self._transport(f"{info['url']}/decide", payload,
                                  timeout)
            data = json.loads(raw)
            vote = Vote(node_id, info["checkpoint"],
                        data["vote"]["choice"],
                        float(data["vote"]["probability"]))
            self._record_stem_call(
                node_id, True, (time.perf_counter() - started) * 1000)
            return vote
        except Exception as exc:
            self._record_stem_call(
                node_id, False, (time.perf_counter() - started) * 1000,
                f"{type(exc).__name__}: {exc}")
            return None  # dead node: shrink quorum, never block the vote


    def decide(self, req: DecisionRequest) -> dict:
        if self.known_questions and req.question not in self.known_questions:
            return {"error": f"question {req.question!r} not in playbook; "
                    f"known: {sorted(self.known_questions)}",
                    "responded": 0, "quorum_size": 0}
        quorum = self.pick_quorum(self.quorum_k)
        payload = req.to_json()
        fanout_started = time.perf_counter()
        if quorum:
            with ThreadPoolExecutor(
                    max_workers=len(quorum),
                    thread_name_prefix="pandosi-quorum") as pool:
                results = list(pool.map(
                    lambda item: self._call_stem(item[0], item[1], payload),
                    quorum))
            votes = [vote for vote in results if vote is not None]
        else:
            votes = []
        fanout_ms = round((time.perf_counter() - fanout_started) * 1000, 3)

        # Calibrate probabilities before aggregation (blueprint 3.2.1)
        calibrated_votes = []
        uncalibrated: set[str] = set()
        for v in votes:
            cp, is_calibrated = self._calibrator.calibrate(
                v.checkpoint, v.probability)
            if not is_calibrated:
                uncalibrated.add(v.checkpoint)
            calibrated_votes.append(Vote(v.node_id, v.checkpoint, v.choice, cp))

        # Security questions get stricter gating (lower thresholds)
        is_security = any(
            kw in req.question.lower()
            for kw in ("security", "suspicious", "phishing", "threat"))
        eff_gating = dict(self.gating)
        if is_security and self.gating.get("security"):
            sec = self.gating["security"]
            eff_gating["auto_min_agreement"] = sec.get(
                "auto_min_agreement", 0.5)
            eff_gating["auto_min_probability"] = sec.get(
                "auto_min_probability", 0.4)
        result = aggregate(
            calibrated_votes, min_quorum=min(3, len(quorum)),
            auto_min_agreement=eff_gating.get("auto_min_agreement", 0.6),
            auto_min_probability=eff_gating.get("auto_min_probability", 0.6))
        self._log_decision(req, result, len(quorum), fanout_ms)
        try:
            self._recorder.record(req.request_id, req.state, req.question,
                                  req.candidates,
                                  [v.to_json() for v in votes],
                                  result.choice, result.tier,
                                  playbook_hash=self.playbook_hash)
        except OSError:
            pass
        out = result.__dict__.copy()
        out["quorum_size"] = len(quorum)
        out["responded"] = len(votes)
        out["fanout_ms"] = fanout_ms
        # Report uncalibrated voters rather than acting on them silently. The
        # tier is left alone so a synthetic test harness is unaffected; the
        # policy belongs to whatever acts on the answer (see AutonomyLoop).
        if uncalibrated:
            out["uncalibrated_checkpoints"] = sorted(uncalibrated)
        return out


def build_handler(coordinator: Coordinator, root=None):
    class RootHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload) -> None:
            body = (payload if isinstance(payload, bytes)
                    else json.dumps(payload).encode("utf-8"))
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _pairing_authorized(self) -> bool:
            expected = getattr(root, "pairing_token", None) if root else None
            if not expected:
                return True
            supplied = self.headers.get("X-Pairing-Token", "")
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                supplied = authorization[7:]
            return hmac.compare_digest(
                supplied.encode("utf-8"), expected.encode("utf-8"))

        def do_GET(self):  # noqa: N802
            if self.path == "/pairing":
                required = bool(root and root.pairing_token)
                self._send(200, {"required": required})
                return
            if not self._pairing_authorized():
                self._send(403, {"error": "invalid or missing pairing token"})
                return
            if self.path == "/nodes":
                self._send(200, {"nodes": coordinator.live_nodes()})
            elif self.path == "/playbook" and root is not None \
                    and getattr(root, "playbook", None) is not None:
                self._send(200, {
                    "name": root.playbook.name,
                    "version": root.playbook.version,
                    "hash": root.playbook.hash(),
                    "canonical": json.loads(root.playbook.canonical_json()),
                    "gating": root.playbook.gating,
                    "questions": [
                        {"id": q["id"], "type": q["type"],
                         "instructions": q.get("instructions", ""),
                         "criteria": q.get("criteria", []),
                         "informational": q.get("informational", False)}
                        for q in root.playbook.questions
                    ],
                })
            elif (self.path == "/spore/pubkey"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, {"pubkey": root.spore_pubkey})
            elif (self.path == "/spore/manifest"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, root.spore_manifest)
            elif (self.path.startswith("/spore/manifests/")
                    and root is not None):
                digest = self.path.rsplit("/", 1)[-1]
                cached = getattr(root, "spore_history", {}).get(digest)
                if cached is None:
                    self._send(404, {"error": "unknown manifest"})
                else:
                    self._send(200, cached)
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
            elif (self.path.startswith("/spore/chunks/")
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                # Batched transfer, same as SporeServer (round 106).
                parts = self.path.split("/")
                name = "/".join(parts[3:-2])
                start, count = int(parts[-2]), int(parts[-1])
                present = all((name, index) in root.spore_chunks
                              for index in range(start, start + count))
                if not present or count <= 0:
                    self._send(404, b"{}")
                else:
                    body = b"".join(root.spore_chunks[(name, index)]
                                    for index in range(start, start + count))
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path != "/register" and not self._pairing_authorized():
                self._send(403, {"error": "invalid or missing pairing token"})
                return
            if self.path == "/register":
                data = json.loads(raw)
                expected = getattr(root, "pairing_token", None) if root else None
                supplied = data.get("pairing_token", "")
                if expected and not hmac.compare_digest(
                        str(supplied).encode("utf-8"),
                        str(expected).encode("utf-8")):
                    self._send(403, {"error": "invalid or missing pairing token"})
                    return
                admitted = getattr(root, "admitted_lineages", None)
                if admitted is not None and data.get("checkpoint") not in admitted:
                    self._send(403, {
                        "error": f"lineage {data.get('checkpoint')!r} is not "
                                 f"declared by this playbook",
                        "admitted": sorted(admitted)})
                    return
                expected_artifact = getattr(
                    root, "lineage_artifacts", {}).get(data.get("checkpoint"))
                if expected_artifact and \
                        data.get("artifact_sha256") != expected_artifact:
                    self._send(403, {
                        "error": f"lineage {data.get('checkpoint')!r} requires "
                                 "artifact attestation",
                        "expected_sha256": expected_artifact,
                        "reported_sha256": data.get("artifact_sha256")})
                    return
                coordinator.register(data["node_id"], data["url"],
                                     data["checkpoint"], data.get("capability"))
                self._send(200, {"ok": True})
            elif self.path == "/spore/blobs":
                # Delta pulls: return only the chunks a replica asked for and
                # does not already hold (round 107).
                wanted = json.loads(raw).get("hashes", [])
                blobs = getattr(root, "spore_blobs", {}) if root else {}
                missing = [digest for digest in wanted if digest not in blobs]
                if missing:
                    self._send(404, {"error": "unknown chunks",
                                     "missing": missing[:8]})
                    return
                body = b"".join(blobs[digest] for digest in wanted)
                self.send_response(200)
                self.send_header("Content-Type",
                                 "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/decide":
                req = DecisionRequest.from_json(raw)
                try:
                    self._send(200, coordinator.decide(req))
                except Exception as exc:
                    self._send(503, {"error": f"decide failed: {exc}"})
            elif self.path == "/shutdown":
                data = json.loads(raw)
                if root and root._auth_token and \
                        data.get("token") == root._auth_token:
                    print("SHUTDOWN requested via /shutdown (token valid)",
                          flush=True)
                    self._send(200, {"ok": True, "message": "shutting down"})
                    threading.Thread(target=root.stop, daemon=True).start()
                else:
                    self._send(403, {"error": "invalid or missing token"})
            else:
                self._send(404, {"error": "not found"})

    return RootHandler


class Root:
    def __init__(self, coordinator: Coordinator | None = None,
                 host: str = "127.0.0.1", port: int = 0,
                 spore_files: dict | None = None,
                 chunking: str = "fixed"):
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
            pb_meta = {"role": "root"}
            if getattr(self, "playbook", None) is not None:
                pb_meta["playbook_hash"] = self.playbook.hash()
                pb_meta["playbook_name"] = self.playbook.name
                pb_meta["playbook_version"] = self.playbook.version
            from .spore import build_manifest_cdc
            if chunking == "cdc":
                # Content-defined chunking lets a joiner that already holds a
                # related lineage fetch only the chunks it lacks (round 107).
                self.spore_manifest = build_manifest_cdc(
                    spore_files, key, metadata=pb_meta)
            else:
                self.spore_manifest = build_manifest(spore_files, key,
                                                     metadata=pb_meta)
            # Signed manifests this root can serve by content hash, so a
            # joiner that already holds a descendant can fetch its parent.
            from .spore import manifest_hash
            self.spore_history: dict[str, dict] = {
                manifest_hash(self.spore_manifest): self.spore_manifest}
            self.spore_chunks = {}
            self.spore_blobs: dict[str, bytes] = {}
            if chunking == "cdc":
                from .spore import chunk_spans
                avg = int(self.spore_manifest["chunking"]["avg_size"])
                for name, path in spore_files.items():
                    data = path.read_bytes()
                    for start, end in chunk_spans(data, avg):
                        chunk = data[start:end]
                        self.spore_blobs[
                            hashlib.sha256(chunk).hexdigest()] = chunk
            else:
                for name, path in spore_files.items():
                    data = path.read_bytes()
                    for i in range(0, len(data), 64 * 1024):
                        self.spore_chunks[(name, i // (64 * 1024))] = \
                            data[i:i + 64 * 1024]
        self.shutdown_event = threading.Event()
        self._auth_token = None
        self._pairing_token = None
        self._httpd = _PandoHTTPServer(
            (host, port), build_handler(self.coordinator, root=self))
        self.port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def set_auth_token(self, token: str) -> None:
        self._auth_token = token

    @property
    def pairing_token(self) -> str | None:
        return self._pairing_token

    def set_pairing_token(self, token: str) -> None:
        """Require this secret on every HTTP /register request."""
        if not token:
            raise ValueError("pairing token must not be empty")
        self._pairing_token = token

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
    parser.add_argument("--chunking", default="fixed",
                        choices=["fixed", "cdc"],
                        help="fixed = v1 manifests; cdc = content-defined "
                             "chunking so an existing replica pulls a delta")
    parser.add_argument("--playbook", default=None,
                        help="playbook TOML to load and serve")
    parser.add_argument("--playbook-key", default=None,
                        help="pinned Ed25519 pubkey (hex) for the playbook; "
                             "with it, an unsigned or invalid playbook is "
                             "refused")
    parser.add_argument("--playbook-signature", default=None,
                        help="detached signature path "
                             "(default: <playbook>.sig)")
    parser.add_argument("--history", default=None,
                        help="playbook history chain to verify against")
    parser.add_argument("--allow-non-head", action="store_true")
    parser.add_argument("--auth-token", default=None,
                        help="enable /shutdown with this token")
    parser.add_argument("--pairing-token", default=None,
                        help="fixed stem admission token; generated when omitted")
    parser.add_argument("--pairing-token-file", default=None,
                        help="read the fixed admission token from a user-owned file")
    parser.add_argument("--rollout-dir", default="rollouts",
                        help="directory for local rollout evidence")
    parser.add_argument("--stem-call-timeout", type=float, default=CALL_TIMEOUT,
                        help="maximum per-stem call timeout")
    parser.add_argument("--quorum-k", type=int, default=QUORUM_K,
                        help="voters per decision; keep it fixed while adding "
                             "replicas or every decision uses every replica")
    parser.add_argument("--initial-stem-timeout", type=float,
                        default=INITIAL_STEM_TIMEOUT,
                        help="discovery timeout before latency is learned")
    parser.add_argument("--lan", action="store_true",
                        help="bind 0.0.0.0; requires an active lan_listen grant")
    parser.add_argument("--reveal-pairing-token", action="store_true",
                        help="print the full token even when stdout is not a TTY")
    args = parser.parse_args()
    if args.pairing_token and args.pairing_token_file:
        raise SystemExit("--pairing-token and --pairing-token-file are mutually exclusive")
    if args.pairing_token_file:
        from .common import read_secret_file
        try:
            args.pairing_token = read_secret_file(args.pairing_token_file)
        except ValueError as exc:
            raise SystemExit(str(exc))
    from .common import bypass_proxy_for_lan
    bypass_proxy_for_lan()
    spore_files = None
    if args.serve_spore:
        from pathlib import Path as _P
        from .spore import artifact_files
        payload = _P(args.serve_spore)
        # An exported graph may be a small protobuf plus a multi-gigabyte
        # external weight blob that the protobuf references *by name*, so
        # every file is served under its original name or the replica cannot
        # resolve its weights (round 105).
        spore_files = {file.name: file for file in artifact_files(payload)}
        if args.playbook:
            spore_files["playbook.toml"] = _P(args.playbook)
    listen_host = "127.0.0.1"
    if args.lan:
        from .grants import require_grant
        try:
            require_grant("lan_listen")
        except PermissionError as exc:
            raise SystemExit(f"[locked] {exc}")
        listen_host = "0.0.0.0"
    coordinator = Coordinator(
        call_timeout=args.stem_call_timeout,
        initial_stem_timeout=args.initial_stem_timeout,
        quorum_k=args.quorum_k,
        rollout_dir=args.rollout_dir)
    root = Root(coordinator=coordinator, host=listen_host, port=args.port,
                spore_files=spore_files, chunking=args.chunking)
    import secrets
    root.set_pairing_token(args.pairing_token or secrets.token_urlsafe(18))
    token_digest = hashlib.sha256(
        root.pairing_token.encode("utf-8")).hexdigest()[:16]
    if args.reveal_pairing_token or sys.stdout.isatty():
        print(f"stem pairing token: {root.pairing_token}", flush=True)
    else:
        print(f"stem pairing token fingerprint: sha256:{token_digest}",
              flush=True)
    if args.auth_token:
        root.set_auth_token(args.auth_token)
        print(f"kill-switch enabled (token set)", flush=True)
    if args.playbook:
        from .pb_history import verify_chain
        from .playbook import (load_verified_playbook,
                               resolve_admitted_lineages)
        try:
            root.playbook, signature_status = load_verified_playbook(
                Path(args.playbook), args.playbook_key,
                args.playbook_signature)
        except ValueError as exc:
            raise SystemExit(f"[playbook] {exc}")
        if not args.playbook_key:
            print(f"[playbook] not pinned: {signature_status} - anyone who "
                  f"can write this file chooses what this swarm admits",
                  flush=True)
        else:
            print(f"[playbook] {signature_status}", flush=True)
        admitted, unusable = resolve_admitted_lineages(root.playbook)
        if root.playbook.lineages:
            if not admitted:
                raise SystemExit(
                    "playbook declares no usable lineage: "
                    f"{sorted(unusable)} have no calibration table, so no "
                    "stem could register")
            root.admitted_lineages = admitted
            artifacts = root.playbook.lineages.get("artifacts") or {}
            root.lineage_artifacts = artifacts
            for name in unusable:
                print(f"[lineages] declared {name!r} has no calibration "
                      f"table and is NOT admitted", flush=True)
            print(f"[lineages] admitting {sorted(admitted)}", flush=True)
            if artifacts:
                attested = sorted(artifacts)
                print(f"[lineages] requiring artifact attestation for "
                      f"{attested}", flush=True)
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
        root.coordinator.playbook_hash = root.playbook.hash()
        root.coordinator.known_questions = {
            q["id"] for q in root.playbook.questions}
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
