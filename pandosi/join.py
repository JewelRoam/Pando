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
    args = parser.parse_args()

    cap = capability()
    eus = enumerate_execution_units()
    role = assign_role(cap)
    node_id = args.node_id or f"{socket.gethostname().lower()}-{args.port or 'x'}"
    backend_name, backend_cfg = (args.backend, {}) if args.backend \
        else pick_backend(node_id)

    print("capability probe:")
    print(f"  arch={cap['arch']} ram={cap['ram_gb']}GB cpus={cap['cpu_count']} "
          f"-> role={role}")
    print(f"  execution units: {[u['id'] for u in eus]}")
    print(f"  backend: {backend_name} {backend_cfg or ''}")

    root_info = discover_root()
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

    import importlib
    backend_mod = importlib.import_module(".backend", __package__)
    if backend_name == "onnx":
        backend = backend_mod.OnnxLayaBackend(node_id, **backend_cfg)
    else:
        backend = backend_mod.MockBackend(node_id, **backend_cfg)

    stem = Stem(node_id, backend, host="0.0.0.0", port=0)
    stem.start()
    print(f"stem {node_id} on {stem.url()}")

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
        except Exception:
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
