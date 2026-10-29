"""pando join: Ring-0 one-command onboarding (zero privilege).

Flow:
1. Capability probe (graft) - what can this machine do?
2. Browse mDNS for an existing PandoSI root (_pando._tcp.local.).
3. If a root is found: join as a stem. If not: become root AND stem.
4. Best available backend: ONNX Laya if weights exist, else mock.

No elevation, no drivers, no network beyond the LAN. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .common import bypass_proxy_for_lan
from .graft import assign_role, capability, enumerate_execution_units
from .grants import require_grant
from .identity import load_or_create
from .resources import asset_path
from .rootd import Coordinator

SERVICE_TYPE = "_pando._tcp.local."
ARTIFACT_CANDIDATES = [
    Path("artifacts/laya-en.int8.onnx"),
    Path("artifacts/laya-en.onnx"),
]


def _resolve_text_model() -> Path:
    """The packaged asset when installed; the source-checkout path otherwise."""
    packaged = asset_path("textmodel.json")
    if packaged.exists():
        return packaged
    return Path("assets/textmodel.json")


TEXT_MODEL = _resolve_text_model()


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def discover_root(timeout_s: float = 3.0) -> dict | None:
    """Browse mDNS for a PandoSI root; return {'ip', 'port'} or None."""
    from zeroconf import ServiceBrowser, Zeroconf

    found: dict | None = None

    class _Listener:
        def add_service(self, zc, type_, name):
            nonlocal found
            info = zc.get_service_info(type_, name, timeout=2000)
            if info and info.addresses:
                found = {"ip": socket.inet_ntoa(info.addresses[0]),
                         "port": info.port,
                         "name": name}

        def update_service(self, zc, type_, name): ...
        def remove_service(self, zc, type_, name): ...

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, _Listener())
        time.sleep(timeout_s)
    finally:
        zc.close()
    return found


def advertise_root(port: int) -> threading.Event:
    """Advertise this root over mDNS until the returned event is set."""
    from zeroconf import ServiceInfo, Zeroconf

    zc = Zeroconf()
    info = ServiceInfo(SERVICE_TYPE, f"pando-root-{port}.{SERVICE_TYPE}",
                       addresses=[socket.inet_aton(lan_ip())], port=port,
                       properties={"version": "1", "pairing": "required"})
    zc.register_service(info)
    stop = threading.Event()

    def _wait():
        stop.wait()
        zc.unregister_service(info)
        zc.close()

    threading.Thread(target=_wait, daemon=True).start()
    return stop


def pairing_required(root_url: str) -> bool:
    """Ask a root whether it requires a token without exposing the secret."""
    try:
        with urllib.request.urlopen(f"{root_url}/pairing", timeout=2) as resp:
            return bool(json.loads(resp.read()).get("required"))
    except Exception:
        return False


def auth_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def pick_backend(node_id: str) -> tuple[str, dict]:
    for cand in ARTIFACT_CANDIDATES:
        if cand.exists():
            return "onnx", {"onnx_path": str(cand)}
    # A fresh clone has no weights and would otherwise fall back to a mock,
    # which votes from a hash and means nothing. The repository carries a
    # 19 KB text member (round 126) trained from the swarm's own labels, so
    # the honest zero-download default is a real voter.
    if TEXT_MODEL.exists():
        return "text", {"text_model": str(TEXT_MODEL)}
    return "mock", {"checkpoint": "mock"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Join or start a PandoSI swarm")
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", default=None,
                        choices=["mock", "onnx", "text"],
                        help="mock (a hash, for protocol tests) | onnx (the "
                             "Laya weights) | text (the 19 KB member shipped "
                             "in assets/, no download)")
    parser.add_argument("--multi-eu", action="store_true",
                        help="start one stem per execution unit (iGPU + CPU on an AI PC); the main --onnx-path must be a DYNAMIC onnx (CPU stem); the GPU stem auto-uses artifacts/laya-en-static-fixed.onnx")
    parser.add_argument("--spore-url", default=None,
                        help="pull the payload from a SporeServer first "
                             "(GRAFT/SPROUT; implies --backend onnx)")
    parser.add_argument("--spore", action="store_true",
                        help="pull from the root's own spore endpoints")
    parser.add_argument("--sandbox-confirmed", action="store_true",
                        help="confirm this target is a sandbox (required for spore)")
    parser.add_argument("--root", default=None,
                        help="explicit root URL (skips mDNS discovery)")
    parser.add_argument("--pairing-token", default=os.getenv("PANDOSI_PAIRING_TOKEN"),
                        help="secret required by a pairing-enabled root "
                             "(default: PANDOSI_PAIRING_TOKEN)")
    parser.add_argument("--onnx-path", default=None,
                        help="use this ONNX file as the decision backend")
    parser.add_argument("--checkpoint", default=None,
                        help="explicit lineage/checkpoint identity for ONNX stems")
    parser.add_argument("--chunk-store", default=None,
                        help="content-addressed chunk cache for delta pulls "
                             "(default: spore-cache/<node>/chunks); point two "
                             "replicas at the same store to share a lineage")
    parser.add_argument("--lan", action="store_true",
                        help="bind/advertise beyond loopback; requires an "
                             "active lan_listen grant")
    args = parser.parse_args()
    bypass_proxy_for_lan()

    if args.lan:
        try:
            require_grant("lan_listen")
        except PermissionError as exc:
            raise SystemExit(f"[locked] {exc}")
    if (args.spore_url or args.spore) and not args.sandbox_confirmed:
        raise SystemExit("spore pull requires --sandbox-confirmed "
                         "(see THREAT-MODEL.md)")

    cap = capability()
    eus = enumerate_execution_units()
    role = assign_role(cap)
    node_id = args.node_id or f"{socket.gethostname().lower()}-{args.port or 'x'}"
    if args.onnx_path:
        backend_name, backend_cfg = "onnx", {"onnx_path": args.onnx_path}
    else:
        backend_name, backend_cfg = (args.backend, {}) if args.backend \
            else pick_backend(node_id)

    print("capability probe:", flush=True)
    print(f"  arch={cap['arch']} ram={cap['ram_gb']}GB cpus={cap['cpu_count']} "
          f"-> role={role}", flush=True)
    print(f"  execution units: {[u['id'] for u in eus]}", flush=True)
    print(f"  backend: {backend_name} {backend_cfg or ''}", flush=True)

    root_info = {"ip": args.root.split("//")[1].split(":")[0],
                 "port": int(args.root.rsplit(":", 1)[1])} if args.root \
        else (discover_root() if args.lan else None)
    from .rootd import Coordinator, Root
    from .stemd import Stem, heartbeat_loop

    if root_info:
        root_url = f"http://{root_info['ip']}:{root_info['port']}"
        print(f"found existing root at {root_url} - joining as stem", flush=True)
        root = None
        if pairing_required(root_url) and not args.pairing_token:
            raise SystemExit(
                f"root {root_url} requires --pairing-token "
                "(or PANDOSI_PAIRING_TOKEN); refusing automatic join")
    else:
        listen_host = "0.0.0.0" if args.lan else "127.0.0.1"
        scope = "LAN" if args.lan else "loopback"
        print(f"no existing root - becoming root + stem ({scope})", flush=True)
        args.pairing_token = args.pairing_token or secrets.token_urlsafe(18)
        # Load the advisory combiner when the repository carries one, so the
        # one-command path runs with the swarm's best available rule
        # (rounds 139-142) instead of the plain gate alone.
        combiner = None
        combiner_path = Path("assets/combiner.json")
        if combiner_path.exists():
            try:
                from .combiner import Combiner
                combiner = Combiner.load(combiner_path)
                print(f"advisory combiner loaded from {combiner_path} "
                      f"(spec {combiner.meta.get('rows')} training rows)",
                      flush=True)
            except (OSError, ValueError, KeyError) as exc:
                print(f"[combiner] {combiner_path}: {exc}", flush=True)
        root = Root(coordinator=Coordinator(combiner=combiner),
                    host=listen_host, port=args.port or 8080)
        root.set_pairing_token(args.pairing_token)
        root.start()
        root_url = root.url()
        if args.lan:
            advertise_root(root.port)
            print(f"rootd listening on {root_url} "
                  "(advertised via mDNS)", flush=True)
        else:
            print(f"rootd listening on {root_url} (loopback only)", flush=True)
        print(f"stem pairing token: {args.pairing_token}", flush=True)
        print("other stems must pass the same --pairing-token", flush=True)

    from .spore import (artifact_digest, generate_signing_key, pull_spore,
                        verify_manifest)
    if args.spore and not args.spore_url and root_info:
        args.spore_url = f"http://{root_info['ip']}:{root_info['port']}"
        print(f"spore: pulling from root at {args.spore_url}", flush=True)
    if args.spore_url:
        import nacl.signing

        # 1) Obtain the seed's verify key.
        #    --spore (root mode): fetch from the root, pin on first use.
        #    --spore-url (explicit seed): must come from PANDOSI_SEED_PUBKEY.
        if args.spore:
            with urllib.request.urlopen(urllib.request.Request(
                    f"{args.spore_url}/spore/pubkey",
                    headers=auth_headers(args.pairing_token)),
                    timeout=15) as resp:
                seed_pub_hex = json.loads(resp.read())["pubkey"]
            pin_file = Path("spore-cache") / "pinned-pubkey.txt"
            pin_file.parent.mkdir(parents=True, exist_ok=True)
            if pin_file.exists():
                pinned = pin_file.read_text(encoding="utf-8").strip()
                if pinned and pinned != seed_pub_hex:
                    raise SystemExit(
                        "SECURITY: root spore pubkey CHANGED vs pinned key "
                        f"({pin_file}) - possible seed compromise; refusing. "
                        "Delete the pin file only if you verified the new key.")
                print("[TOFU] pubkey matches pin", flush=True)
            else:
                pin_file.write_text(seed_pub_hex, encoding="utf-8")
                print("[TOFU] root pubkey pinned:", seed_pub_hex, flush=True)
        else:
            seed_pub_hex = os.environ.get("PANDOSI_SEED_PUBKEY")
            if not seed_pub_hex:
                raise SystemExit(
                    "set PANDOSI_SEED_PUBKEY (seed verify key hex)")

        verify_key = nacl.signing.VerifyKey(bytes.fromhex(seed_pub_hex))

        # 2) Fetch and verify the manifest, then pull with per-chunk checks.
        with urllib.request.urlopen(urllib.request.Request(
                f"{args.spore_url}/spore/manifest",
                headers=auth_headers(args.pairing_token)),
                timeout=60) as resp:
            manifest = json.loads(resp.read())
        if not verify_manifest(manifest, verify_key):
            raise SystemExit("SECURITY: manifest signature verification FAILED")
        print(f"SPORE manifest: lineage="
              f"{manifest['metadata'].get('lineage')} "
              f"generation={manifest['metadata'].get('generation', 0)} "
              f"files={list(manifest['files'])}", flush=True)

        cache_dir = Path("spore-cache") / node_id
        from .spore import ensure_parent_lineage
        # A joiner that holds only a descendant fetches the missing parent
        # manifest from the serving peer; a hash/signature mismatch is
        # 'untrusted' and refuses instead of silently upgrading lineage.
        lineage_status, lineage_detail = ensure_parent_lineage(
            args.spore_url, manifest, cache_dir / "manifests", verify_key,
            args.pairing_token)
        if lineage_status in ("invalid", "untrusted"):
            raise SystemExit(
                f"SECURITY: Spore lineage {lineage_status}: {lineage_detail}")
        print(f"SPORE lineage status: {lineage_status} ({lineage_detail})",
              flush=True)
        from .spore import ChunkStore
        store_path = Path(args.chunk_store) if args.chunk_store \
            else cache_dir / "chunks"
        stats = pull_spore(args.spore_url, cache_dir, verify_key,
                           sandbox_confirmed=True,
                           auth_token=args.pairing_token,
                           chunk_store=ChunkStore(store_path))
        # Leave the numbers on disk: a caller supervising a reproduction
        # needs to know what the delta actually cost (round 107).
        (cache_dir / "pull-stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
        # A spore payload can carry a graph plus its external weight blobs;
        # the ONNX session must be pointed at the graph, not at a .data file.
        graphs = [name for name in manifest["files"]
                  if name.endswith(".onnx")]
        pulled_file = cache_dir / (graphs[0] if graphs
                                   else next(iter(manifest["files"])))
        print(f"SPROUT: pulled {stats['chunks_pulled']} chunks, reused "
              f"{stats['chunks_reused']}, "
              f"{stats['bytes_transferred'] / 1024 / 1024:.1f} MB over the "
              f"wire -> {pulled_file}", flush=True)
        from .spore import append_lineage
        lineage_entry = append_lineage(cache_dir / "lineage.jsonl", manifest)
        print(f"SPROUT: lineage generation={lineage_entry['generation']} "
              f"manifest={lineage_entry['manifest_hash'][:16]}", flush=True)
        # Strategy layer travels with the payload: a pulled playbook is
        # validated, its hash reported, and appended to this node's local
        # lineage chain (reproduction carries strategy, auditable).
        pulled_pb = cache_dir / "playbook.toml"
        if pulled_pb.exists():
            from .playbook import load_playbook
            from .pb_history import append_entry
            pulled_playbook = load_playbook(pulled_pb)
            pb_hash = pulled_playbook.hash()
            carried_parent = (manifest.get("metadata") or {}).get(
                "playbook_parent")
            entry = append_entry(Path("spore-cache") / node_id /
                                 "pb-history.jsonl",
                                 pb_hash, parent=carried_parent)
            print(f"SPROUT: playbook {pulled_playbook.name} "
                  f"v{pulled_playbook.version} hash={pb_hash[:16]} "
                  f"-> lineage entry {entry['hash'][:12]} "
                  f"(parent {entry['parent'][:12]})", flush=True)
        args.backend = "onnx"
        backend_cfg = {"onnx_path": str(pulled_file)}
        backend_name = "onnx"

    from .backend import create_backend
    if backend_name == "onnx":
        backend = create_backend("onnx", node_id,
                                 checkpoint=args.checkpoint, **backend_cfg)
        # Attest what this node is actually running, or a root that declares
        # artifact digests will refuse it (round 103).
        backend.artifact_digest = artifact_digest(
            Path(backend_cfg["onnx_path"]))
        print(f"artifact sha256 {backend.artifact_digest}", flush=True)
    else:
        backend = create_backend(backend_name, node_id,
                                 checkpoint=args.checkpoint, **backend_cfg)
        describe = getattr(backend, "summary", None)
        if callable(describe):
            print(f"text member loaded: {describe()}", flush=True)

    stem_host = "0.0.0.0" if args.lan else "127.0.0.1"
    stems = [Stem(node_id, backend, host=stem_host, port=0)]
    if args.multi_eu and backend_name == "onnx":
        # Auto-discover static surgery-graph buckets (any seq size).
        bucket_files = sorted(
            Path("artifacts").glob("laya-en-static*-fixed.onnx"))
        if bucket_files and any(u["kind"] == "GPU" for u in eus):
            from .backend import create_backend as _mk
            gpu_backend = _mk(
                "onnx", node_id + "-gpu", onnx_path=str(bucket_files[-1]),
                repo="convaiinnovations/laya", device="GPU",
                onnx_buckets=[str(b) for b in bucket_files],
                checkpoint=f"{args.checkpoint}-gpu" if args.checkpoint else None)
            gpu_backend.artifact_digest = artifact_digest(bucket_files[-1])
            stems.append(Stem(node_id + "-gpu", gpu_backend,
                              host=stem_host, port=0))
            print(f"multi-EU: iGPU stem added "
                  f"({len(bucket_files)} bucket(s))", flush=True)
        # NPU stem: needs the IsNaN-free static graph
        npu_file = Path("artifacts/laya-en-static-nonisnan.onnx")
        if npu_file.exists() and any(u["kind"] == "NPU" for u in eus):
            from .backend import create_backend as _mk
            npu_backend = _mk(
                "onnx", node_id + "-npu", onnx_path=str(npu_file),
                repo="convaiinnovations/laya", device="NPU",
                pad_to=256,
                checkpoint=f"{args.checkpoint}-npu" if args.checkpoint else None)
            npu_backend.artifact_digest = artifact_digest(npu_file)
            stems.append(Stem(node_id + "-npu", npu_backend,
                              host=stem_host, port=0))
            print("multi-EU: NPU stem added (IsNaN-free static graph)",
                  flush=True)
    for s in stems:
        s.start()
        print(f"stem {s.node_id} on {s.url()}", flush=True)
    if args.spore_url:
        from .common import DecisionRequest
        for i in range(3):
            for s in stems:
                r = s.backend.decide(DecisionRequest(
                    state=f"verify probe {i}", question="ok?",
                    candidates=["yes", "no"]))
                print(f"VERIFY {s.node_id} probe {i+1}/3: "
                      f"{r.choice} @ {r.probability}", flush=True)
        print("VERIFY passed - node is LIVE", flush=True)

    advertised_ip = lan_ip() if args.lan else "127.0.0.1"
    registrations = [{"node_id": s.node_id,
                      "url": f"http://{advertised_ip}:{s.port}",
                      "checkpoint": s.backend.checkpoint} for s in stems]
    for reg, stem in zip(registrations, stems):
        digest = getattr(stem.backend, "artifact_digest", None)
        if digest:
            reg["artifact_sha256"] = digest
    if args.pairing_token:
        for reg in registrations:
            reg["pairing_token"] = args.pairing_token
    try:
        node_identity = load_or_create()
    except Exception as exc:  # identity is an upgrade, not a startup blocker
        print(f"[identity] disabled: {exc}", flush=True)
        node_identity = None

    def signed_reg(reg: dict) -> bytes:
        body = {**reg, "capability": cap}
        if node_identity is not None:
            body["identity"] = node_identity.proof(dict(body))
        return json.dumps(body).encode()

    while True:
        try:
            for reg in registrations:
                payload = signed_reg(reg)
                urllib.request.urlopen(
                    urllib.request.Request(f"{root_url}/register",
                                           data=payload,
                                           headers=auth_headers(
                                               args.pairing_token)),
                    timeout=3)
            print(f"registered {len(registrations)} stem(s) with root "
                  f"{root_url}", flush=True)
            break
        except Exception as exc:
            print(f"register attempt failed: {exc}", flush=True)
            time.sleep(2)

    # Registration IS the heartbeat: rootd expires stems after 15s without
    # a /register refresh, so keep re-registering for the process lifetime.
    def heartbeat():
        while True:
            time.sleep(5)
            try:
                for reg in registrations:
                    payload = signed_reg(reg)
                    urllib.request.urlopen(
                        urllib.request.Request(
                            f"{root_url}/register", data=payload,
                            headers=auth_headers(args.pairing_token)),
                        timeout=3)
            except Exception:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()

    print("PandoSI is up. Try:", flush=True)
    auth_arg = (f' -H "Authorization: Bearer {args.pairing_token}"'
                if args.pairing_token else "")
    print(f'  curl -X POST {root_url}/decide{auth_arg} '
          f'-d "{{\\"state\\":\\"test\\",\\"question\\":\\"ok?\\",'
          f'\\"candidates\\":[\\"yes\\",\\"no\\"]}}"', flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        if root:
            root.stop()
        stem.stop()


if __name__ == "__main__":
    main()
