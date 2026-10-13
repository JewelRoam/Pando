"""pandosi join: Ring-0 one-command onboarding (zero privilege).

Flow:
1. Capability probe (graft) - what can this machine do?
2. Browse mDNS for an existing PandoSI root (_pandosi._tcp.local.).
3. If a root is found: join as a stem. If not: become root AND stem.
4. Best available backend: ONNX Laya if weights exist, else mock.

No elevation, no drivers, no network beyond the LAN. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .graft import assign_role, capability, enumerate_execution_units

SERVICE_TYPE = "_pandosi._tcp.local."
ARTIFACT_CANDIDATES = [
    Path("artifacts/laya-en.int8.onnx"),
    Path("artifacts/laya-en.onnx"),
]


def _bypass_proxy_for_lan() -> None:
    """All PandoSI traffic is LAN-local; a system proxy must never route it.

    urllib on Windows honours registry/system proxies, which black-hole
    loopback and LAN addresses (observed: register POSTs silently retrying
    forever when HTTP_PROXY was set in the environment).
    """
    bypass = {"localhost", "127.0.0.1", lan_ip(), "*"}
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        merged = ",".join(sorted(set(filter(None, current.split(","))) | bypass))
        os.environ[var] = merged


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
    info = ServiceInfo(SERVICE_TYPE, f"pandosi-root-{port}.{SERVICE_TYPE}",
                       addresses=[socket.inet_aton(lan_ip())], port=port,
                       properties={"version": "0"})
    zc.register_service(info)
    stop = threading.Event()

    def _wait():
        stop.wait()
        zc.unregister_service(info)
        zc.close()

    threading.Thread(target=_wait, daemon=True).start()
    return stop


def pick_backend(node_id: str) -> tuple[str, dict]:
    for cand in ARTIFACT_CANDIDATES:
        if cand.exists():
            return "onnx", {"onnx_path": str(cand)}
    return "mock", {"checkpoint": "mock"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Join or start a PandoSI swarm")
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", default=None, choices=["mock", "onnx"])
    parser.add_argument("--spore-url", default=None,
                        help="pull the payload from a SporeServer first "
                             "(GRAFT/SPROUT; implies --backend onnx)")
    parser.add_argument("--spore", action="store_true",
                        help="pull from the root's own spore endpoints")
    parser.add_argument("--sandbox-confirmed", action="store_true",
                        help="confirm this target is a sandbox (required for spore)")
    parser.add_argument("--root", default=None,
                        help="explicit root URL (skips mDNS discovery)")
    parser.add_argument("--onnx-path", default=None,
                        help="use this ONNX file as the decision backend")
    args = parser.parse_args()
    _bypass_proxy_for_lan()

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

    print("capability probe:")
    print(f"  arch={cap['arch']} ram={cap['ram_gb']}GB cpus={cap['cpu_count']} "
          f"-> role={role}")
    print(f"  execution units: {[u['id'] for u in eus]}")
    print(f"  backend: {backend_name} {backend_cfg or ''}")

    root_info = {"ip": args.root.split("//")[1].split(":")[0],
                 "port": int(args.root.rsplit(":", 1)[1])} if args.root \
        else discover_root()
    from .rootd import Coordinator, Root
    from .stemd import Stem, heartbeat_loop

    if root_info:
        root_url = f"http://{root_info['ip']}:{root_info['port']}"
        print(f"found existing root at {root_url} - joining as stem")
        root = None
    else:
        print("no root on this LAN - becoming root + stem")
        root = Root(host="0.0.0.0", port=args.port or 8080)
        root.start()
        root_url = root.url()
        advertise_root(root.port)
        print(f"rootd listening on {root_url} (advertised via mDNS)")

    if args.spore and not args.spore_url and root_info:
        args.spore_url = f"http://{root_info['ip']}:{root_info['port']}"
        print(f"spore: pulling from root at {args.spore_url}", flush=True)
    if args.spore_url:
        import nacl.signing

        # 1) Obtain the seed's verify key.
        #    --spore (root mode): fetch from the root, pin on first use.
        #    --spore-url (explicit seed): must come from PANDOSI_SEED_PUBKEY.
        if args.spore:
            with urllib.request.urlopen(f"{args.spore_url}/spore/pubkey",
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
            import os
            seed_pub_hex = os.environ.get("PANDOSI_SEED_PUBKEY")
            if not seed_pub_hex:
                raise SystemExit(
                    "set PANDOSI_SEED_PUBKEY (seed verify key hex)")

        verify_key = nacl.signing.VerifyKey(bytes.fromhex(seed_pub_hex))

        # 2) Fetch and verify the manifest, then pull with per-chunk checks.
        with urllib.request.urlopen(f"{args.spore_url}/spore/manifest",
                                    timeout=60) as resp:
            manifest = json.loads(resp.read())
        if not verify_manifest(manifest, verify_key):
            raise SystemExit("SECURITY: manifest signature verification FAILED")
        print(f"SPORE manifest: lineage="
              f"{manifest['metadata'].get('lineage')} "
              f"files={list(manifest['files'])}", flush=True)

        cache_dir = Path("spore-cache") / node_id
        stats = pull_spore(args.spore_url, cache_dir, verify_key,
                           sandbox_confirmed=True)
        pulled_file = cache_dir / next(iter(manifest["files"]))
        print(f"SPROUT: pulled {stats['chunks_pulled']} chunks -> {pulled_file}",
              flush=True)
        args.backend = "onnx"
        backend_cfg = {"onnx_path": str(pulled_file)}
        backend_name = "onnx"

    import importlib
    backend_mod = importlib.import_module(".backend", __package__)
    if backend_name == "onnx":
        backend = backend_mod.OnnxLayaBackend(node_id, **backend_cfg)
    else:
        backend = backend_mod.MockBackend(node_id, **backend_cfg)

    stem = Stem(node_id, backend, host="0.0.0.0", port=0)
    stem.start()
    print(f"stem {node_id} on {stem.url()}")
    if args.spore_url:
        from .common import DecisionRequest
        for i in range(3):
            r = backend.decide(DecisionRequest(
                state=f"verify probe {i}", question="ok?",
                candidates=["yes", "no"]))
            print(f"VERIFY probe {i+1}/3: {r.choice} @ {r.probability}",
                  flush=True)
        print("VERIFY passed - node is LIVE", flush=True)

    stem_url = f"http://{lan_ip()}:{stem.port}"
    payload = json.dumps({"node_id": node_id, "url": stem_url,
                          "checkpoint": backend.checkpoint,
                          "capability": cap}).encode()
    while True:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{root_url}/register", data=payload,
                                       headers={"Content-Type": "application/json"}),
                timeout=3)
            print(f"registered with root {root_url}")
            break
        except Exception as exc:
            print(f"register attempt failed: {exc}", flush=True)
            time.sleep(2)

    print("PandoSI is up. Try:")
    print(f'  curl -X POST {root_url}/decide -d "{{\\"state\\":\\"test\\",\\"question\\":\\"ok?\\",\\"candidates\\":[\\"yes\\",\\"no\\"]}}"')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        if root:
            root.stop()
        stem.stop()


if __name__ == "__main__":
    main()
