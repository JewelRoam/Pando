"""End to end: a swarm reproduces onto a new device, or refuses to.

Starts a real coordinator from a generated playbook that declares one lineage and
its artifact digest, then runs the reproduction controller three times:

1. without the ``reproduce`` grant -> refused, nothing transferred;
2. with the grant but a declared digest that does not match -> refused
   before anything moves;
3. with both -> the child pulls the payload over the Spore endpoints,
   attests the artifact, registers, and the quorum grows.

The child is launched as ``python -m pando.join --spore``, which is the
real replication path (pull, verify, become a worker), so case 3 exercises
transfer, attestation, lineage admission and elastic membership together.
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

from pando.grants import save_grant  # noqa: E402
from pando.playbook import load_playbook, resolve_admitted_lineages, \
    sign_playbook  # noqa: E402
from pando.reproduce import Target, plan_reproduction, reproduce  # noqa: E402
from pando.spore import artifact_digest, generate_signing_key  # noqa: E402

LINEAGE = "english"


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
    raise SystemExit("coordinator did not become ready")


def child_command(root_url: str, token_file: Path, node_id: str,
                  lineage: str) -> list[str]:
    return [sys.executable, "-m", "pando.join",
            "--node-id", node_id,
            "--spore", "--root", root_url,
            "--sandbox-confirmed",
            "--pairing-token", token_file.read_text(encoding="utf-8"),
            "--checkpoint", lineage]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--graph", default=str(ROOT / "artifacts"
                                               / "laya-en.onnx"))
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real coordinator/join processes; pass --yes")
    graph = Path(args.graph)
    if not graph.exists():
        raise SystemExit(f"missing graph: {graph}")

    digest = artifact_digest(graph)
    print(f"payload {graph.name} digest {digest}")
    work = Path(tempfile.mkdtemp(prefix="pando-reproduce-"))
    grants_path = work / "grants.toml"
    playbook_path = work / "playbook.toml"
    playbook_path.write_text(
        '[meta]\nname = "reproduce-check"\nversion = 1\n\n'
        f'[lineages]\nadmit = ["{LINEAGE}", "mock"]\n\n'
        f'[lineages.artifacts]\n{LINEAGE} = "{digest}"\n',
        encoding="utf-8")
    playbook = load_playbook(playbook_path)
    key = generate_signing_key()
    (work / (playbook_path.name + ".sig")).write_text(
        json.dumps(sign_playbook(playbook, key)), encoding="utf-8")
    admitted, _ = resolve_admitted_lineages(playbook)

    token = secrets.token_urlsafe(18)
    token_file = work / "pairing.token"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    root_url = f"http://127.0.0.1:{port}"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    root_log = (work / "coordinator.log").open("wb")
    # One parent voter, so the swarm exists before it reproduces.
    parent = subprocess.Popen(
        [sys.executable, "-m", "pando.worker", "--node-id", "parent",
         "--root", root_url, "--pairing-token-file", str(token_file),
         "--backend", "mock", "--checkpoint", "mock"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags)
    children: list[subprocess.Popen] = []
    try:
        subprocess.Popen(
            [sys.executable, "-m", "pando.coordinator", "--port", str(port),
             "--playbook", str(playbook_path),
             "--playbook-key", key.verify_key.encode().hex(),
             "--pairing-token", token,
             "--serve-spore", str(graph),
             "--rollout-dir", str(work / "rollouts")],
            cwd=ROOT, stdout=root_log, stderr=subprocess.STDOUT,
            creationflags=creationflags)
        wait_root(root_url)
        time.sleep(2.0)

        target = Target(node_id="child", host="localhost",
                        command=child_command(root_url, token_file, "child",
                                              LINEAGE),
                        cwd=str(ROOT))

        refused, _ = reproduce(graph, LINEAGE, target, root_url, token,
                               declared_lineages=sorted(admitted),
                               declared_artifacts={LINEAGE: digest},
                               grants_path=grants_path,
                               sandbox_confirmed=True)
        print(f"1. no grant        -> approved={refused.approved} "
              f"refusals={len(refused.refusals)}")
        if refused.approved:
            raise SystemExit("reproduction proceeded without a grant")

        bad = plan_reproduction(
            graph, LINEAGE, target, declared_lineages=sorted(admitted),
            declared_artifacts={LINEAGE: "0" * 64},
            grants_path=None, sandbox_confirmed=True)
        print(f"2. wrong digest    -> approved={bad.approved} "
              f"reason={bad.refusals[0][:60] if bad.refusals else None}")
        if bad.approved or not any("does not match" in r
                                   for r in bad.refusals):
            raise SystemExit("digest mismatch was not refused")

        # Approve the grant the way an operator would, then reproduce.
        expires = (dt.datetime.now(dt.timezone.utc)
                   + dt.timedelta(hours=1)).isoformat()
        save_grant("reproduce", "localhost", "operator-host", "check",
                   expires, path=grants_path)
        outcome, child = reproduce(
            graph, LINEAGE, target, root_url, token,
            declared_lineages=sorted(admitted),
            declared_artifacts={LINEAGE: digest},
            grants_path=grants_path, sandbox_confirmed=True,
            lineage_log=work / "lineage.jsonl", signing_key=key,
            log_path=work / "child.log", timeout=600.0)
        if child is not None:
            children.append(child)
        print(f"3. granted         -> started={outcome.started} "
              f"registered={outcome.registered} "
              f"voters {outcome.voters_before} -> {outcome.voters_after} "
              f"lineage={outcome.lineage_entry}")
        if not outcome.registered:
            tail = (work / "child.log").read_text(
                encoding="utf-8", errors="replace").splitlines()[-12:]
            print("child log tail:", *tail, sep="\n  ")
            raise SystemExit("child never registered")
        if outcome.voters_after <= outcome.voters_before:
            raise SystemExit("quorum did not grow")
        if not outcome.lineage_entry:
            raise SystemExit("no lineage record for the reproduction")
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        for proc in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        parent.terminate()
        root_log.close()

    summary = {"digest": digest,
               "refused_without_grant": not refused.approved,
               "refused_on_digest_mismatch": not bad.approved,
               "registered": outcome.registered,
               "voters_before": outcome.voters_before,
               "voters_after": outcome.voters_after,
               "lineage_generation": outcome.lineage_entry["generation"]
               if outcome.lineage_entry else None}
    print(json.dumps(summary, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print("REPRODUCTION CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
