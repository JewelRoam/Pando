"""pandosi lab: sandbox-gated replication experiments (Ring 2).

Per THREAT-MODEL.md, Spore experiments may only run where the target is
disposable by construction: inside a container or VM. This module is the
gate - it detects the sandbox and refuses everything otherwise. It contains
no exploit surface: the lab only runs the already-public seed/pull protocol
against the operator's own LAN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def detect_sandbox() -> dict:
    """Detect whether we are inside a disposable sandbox."""
    evidence: list[str] = []
    if Path("/.dockerenv").exists():
        evidence.append("/.dockerenv present")
    if Path("/run/.containerenv").exists():
        evidence.append("/run/.containerenv present")
    import os
    if os.environ.get("PANDOSI_CONTAINER") == "1":
        evidence.append("PANDOSI_CONTAINER=1 (operator attestation)")
    if os.name != "nt" and Path("/proc/version").exists():
        try:
            version = Path("/proc/version").read_text(errors="ignore").lower()
            if "microsoft" in version:
                evidence.append("WSL kernel detected")
        except OSError:
            pass

    kind = "container/wsl" if evidence else "none"
    return {"kind": kind, "sandboxed": bool(evidence), "evidence": evidence}


def require_sandbox() -> dict:
    info = detect_sandbox()
    if not info["sandboxed"]:
        raise PermissionError(
            "pandosi lab refused: no disposable sandbox detected "
            f"(kind={info['kind']}). Spore experiments require a container "
            "or VM - see THREAT-MODEL.md. Run inside Docker with "
            "PANDOSI_CONTAINER=1 or a WSL distro.")
    return info


def run_lab() -> dict:
    """Sandbox- and consent-gated replication verification round trip."""
    info = require_sandbox()
    from .grants import require_grant
    require_grant("lab_replication")
    import secrets
    import tempfile

    from .spore import (SporeServer, build_manifest, generate_signing_key,
                        pull_spore)

    key = generate_signing_key()
    tmp = Path(tempfile.mkdtemp(prefix="pandosi-lab-"))
    payload = tmp / "payload.bin"
    payload.write_bytes(secrets.token_bytes(2 * 1024 * 1024))

    manifest = build_manifest({"payload.bin": payload}, key,
                              metadata={"lab": info["kind"]})
    server = SporeServer(manifest, {"payload.bin": payload},
                         auth_token="local-lab")
    server.start()
    try:
        stats = pull_spore(server.url(), tmp / "pulled", key.verify_key,
                           sandbox_confirmed=True,
                           auth_token="local-lab")
        ok = all(stats["files"].values())
    finally:
        server.stop()

    return {"sandbox": info, "chunks_pulled": stats["chunks_pulled"],
            "verified": ok, "workdir": str(tmp)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoSI sandbox lab (Ring 2)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = run_lab()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("sandbox:", result["sandbox"]["kind"],
                  "| chunks pulled:", result["chunks_pulled"],
                  "| verified:", result["verified"])
    except PermissionError as exc:
        print(f"[locked] {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
