"""End to end: a second device replicates a child lineage as a delta.

Two roots each serve one artifact, and two real joiners replicate them:

1. root A serves the parent lineage; node A joins it (cold, whole artifact);
2. root B serves the child lineage; node B joins it **sharing node A's chunk
   store**, so it already holds every chunk the two lineages have in common.

The child's pull stats are read back from disk, so the numbers are what the
replica actually received over the wire - not what the library says it
should have.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.grants import save_grant  # noqa: E402
from pandosi.playbook import load_playbook, resolve_admitted_lineages, \
    sign_playbook  # noqa: E402
from pandosi.reproduce import Target, reproduce  # noqa: E402
from pandosi.spore import artifact_digest, generate_signing_key  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_root(url: str, deadline: float = 30.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/pairing", timeout=2) as resp:
                if json.loads(resp.read()).get("required"):
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise SystemExit("rootd did not become ready")


def start_root(work: Path, payload: Path, lineage: str, digest: str,
               token: str, key, chunking: str) -> str:
    playbook = work / f"playbook-{lineage}.toml"
    playbook.write_text(
        '[meta]\nname = "delta-replication"\nversion = 1\n\n'
        f'[lineages]\nadmit = ["{lineage}"]\n\n'
        f'[lineages.artifacts]\n{lineage} = "{digest}"\n',
        encoding="utf-8")
    loaded = load_playbook(playbook)
    (work / (playbook.name + ".sig")).write_text(
        json.dumps(sign_playbook(loaded, key)), encoding="utf-8")
    token_file = work / f"token-{lineage}"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    log = (work / f"rootd-{lineage}.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "pandosi.rootd", "--port", str(port),
         "--playbook", str(playbook),
         "--playbook-key", key.verify_key.encode().hex(),
         "--pairing-token", token,
         "--serve-spore", str(payload),
         "--chunking", chunking,
         "--rollout-dir", str(work / "rollouts")],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=creationflags)
    wait_root(url)
    time.sleep(1.5)
    return url, process, log, loaded, token_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--parent", default=str(ROOT / "artifacts"
                                                / "laya-en.onnx"))
    parser.add_argument("--child", default=str(ROOT / "artifacts"
                                               / "laya-rlcd-v7.onnx"))
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real rootd/join processes; pass --yes")
    parent, child = Path(args.parent), Path(args.child)
    for path in (parent, child):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")

    work = Path(tempfile.mkdtemp(prefix="pandosi-delta-repl-"))
    shared_store = work / "shared-chunks"
    key = generate_signing_key()
    grants = work / "grants.toml"
    expires = (dt.datetime.now(dt.timezone.utc)
               + dt.timedelta(hours=1)).isoformat()
    save_grant("reproduce", "localhost", "operator-host", "check", expires,
               path=grants)

    roots: list[tuple] = []
    children: list[subprocess.Popen] = []
    try:
        for label, payload, lineage in (("parent", parent, "english"),
                                        ("child", child, "rlcd-v7")):
            digest = artifact_digest(payload)
            token = secrets.token_urlsafe(18)
            url, process, log, playbook, token_file = start_root(
                work, payload, lineage, digest, token, key, "cdc")
            roots.append((label, url, process, log))
            admitted, _ = resolve_admitted_lineages(playbook)
            node_id = f"node-{label}"
            target = Target(
                node_id=node_id, host="localhost", cwd=str(ROOT),
                command=[sys.executable, "-m", "pandosi.join",
                         "--node-id", node_id,
                         "--spore", "--root", url,
                         "--sandbox-confirmed",
                         "--pairing-token", token,
                         "--checkpoint", lineage,
                         "--chunk-store", str(shared_store)])
            outcome, child_proc = reproduce(
                payload, lineage, target, url, token,
                declared_lineages=sorted(admitted),
                declared_artifacts={lineage: digest},
                grants_path=grants, sandbox_confirmed=True,
                log_path=work / f"{node_id}.log", timeout=600.0)
            if child_proc is not None:
                children.append(child_proc)
            stats_path = (ROOT / "spore-cache" / node_id
                          / "pull-stats.json")
            stats = json.loads(stats_path.read_text(encoding="utf-8")) \
                if stats_path.exists() else {}
            print(f"{label:6s} lineage={lineage:9s} registered="
                  f"{outcome.registered} wire="
                  f"{stats.get('bytes_transferred', 0) / 1024 / 1024:.1f} MB "
                  f"pulled={stats.get('chunks_pulled')} "
                  f"reused={stats.get('chunks_reused')}", flush=True)
            if not outcome.registered:
                tail = (work / f"{node_id}.log").read_text(
                    encoding="utf-8", errors="replace").splitlines()[-10:]
                print("child log tail:", *tail, sep="\n  ")
                raise SystemExit(f"{label} never registered")
            globals().setdefault("_results", {})[label] = stats
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for _, _, process, log in roots:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log.close()
        for node in ("node-parent", "node-child"):
            shutil.rmtree(ROOT / "spore-cache" / node, ignore_errors=True)

    results = globals().get("_results", {})
    cold = results.get("parent", {}).get("bytes_transferred", 0)
    warm = results.get("child", {}).get("bytes_transferred", 0)
    summary = {"parent_wire_mb": round(cold / 1024 / 1024, 2),
               "child_wire_mb": round(warm / 1024 / 1024, 2),
               "reduction": round(cold / warm, 1) if warm else None,
               "child_chunks_reused": results.get("child", {}).get(
                   "chunks_reused")}
    print(json.dumps(summary, indent=2))
    if not warm or warm >= cold * 0.5:
        raise SystemExit("the child replication did not reuse the parent")
    shutil.rmtree(work, ignore_errors=True)
    print("DELTA REPLICATION CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
