"""Self-replication: make a new device part of the swarm.

Everything needed for this already existed separately - Spore transfers a
signed payload, a joiner verifies it and becomes a worker, the root admits
declared lineages and requires artifact attestation, and the grant ledger
records that a human approved something sensitive. This module is the loop
that puts them in order, and it refuses before it transfers.

The policy is deliberately front-loaded: every check below happens on the
*parent* side, before a payload moves or a process starts, because the
cheapest failure is the one that never transfers anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .grants import require_grant
from .net import open_url
from .spore import artifact_digest


@dataclass
class Target:
    """Where and how the child will run.

    ``command`` is the full argv to launch on that device. For a real
    second machine this is an ssh/Ansible invocation; for the local check it
    is a plain subprocess, which is the same boundary the root sees.
    """

    node_id: str
    command: list[str]
    host: str = "localhost"
    cwd: str | None = None


@dataclass
class Reproduction:
    """The outcome, including every refusal."""

    node_id: str
    lineage: str
    artifact_sha256: str | None = None
    approved: bool = False
    started: bool = False
    registered: bool = False
    voters_before: int = 0
    voters_after: int = 0
    lineage_entry: dict | None = None
    refusals: list[str] = field(default_factory=list)


def plan_reproduction(payload: Path, lineage: str, target: Target,
                      declared_lineages: list[str] | None = None,
                      declared_artifacts: dict | None = None,
                      grants_path: Path | str | None = None,
                      sandbox_confirmed: bool = False) -> Reproduction:
    """Decide whether this reproduction is allowed, without doing it."""
    outcome = Reproduction(node_id=target.node_id, lineage=lineage)
    payload = Path(payload)
    if not payload.exists():
        outcome.refusals.append(f"payload not found: {payload}")
        return outcome

    outcome.artifact_sha256 = artifact_digest(payload)

    try:
        require_grant("reproduce", path=grants_path)
    except PermissionError as exc:
        outcome.refusals.append(str(exc))

    if not sandbox_confirmed:
        outcome.refusals.append(
            "the receiving device must confirm a sandbox "
            "(container/VM) before a spore is transferred")

    if declared_lineages is not None and lineage not in declared_lineages:
        outcome.refusals.append(
            f"lineage {lineage!r} is not declared by this swarm")

    expected = (declared_artifacts or {}).get(lineage)
    if expected and expected != outcome.artifact_sha256:
        outcome.refusals.append(
            f"artifact digest {outcome.artifact_sha256[:16]}... does not "
            f"match the digest declared for {lineage!r} ({expected[:16]}...)")

    outcome.approved = not outcome.refusals
    return outcome


def _wait_for(url: str, token: str | None, node_id: str,
              timeout: float = 180.0,
              process: subprocess.Popen | None = None) -> bool:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False           # the child died: do not wait out the clock
        try:
            with open_url(
                    urllib.request.Request(f"{url}/nodes", headers=headers),
                    timeout=5) as resp:
                if node_id in json.loads(resp.read()).get("nodes", {}):
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    return False


def reproduce(payload: Path, lineage: str, target: Target, root_url: str,
              pairing_token: str | None = None,
              declared_lineages: list[str] | None = None,
              declared_artifacts: dict | None = None,
              grants_path: Path | str | None = None,
              sandbox_confirmed: bool = False,
              lineage_log: Path | None = None,
              signing_key=None,
              parent_manifest: dict | None = None,
              log_path: Path | None = None,
              timeout: float = 180.0,
              launch: bool = True) -> tuple[Reproduction, subprocess.Popen | None]:
    """Plan, then execute: start the child and confirm it joined.

    Returns the outcome and the child process (None when refused or when
    ``launch`` is False), so a caller can supervise or terminate it.
    """

    outcome = plan_reproduction(
        payload, lineage, target, declared_lineages=declared_lineages,
        declared_artifacts=declared_artifacts, grants_path=grants_path,
        sandbox_confirmed=sandbox_confirmed)
    outcome.voters_before = _count_voters(root_url, pairing_token)
    if not outcome.approved or not launch:
        return outcome, None

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0
    # The child prints while it pulls a multi-gigabyte payload; an unread
    # pipe would fill and block it forever, so its output goes to a file.
    log = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = Path(log_path).open("wb")
    child = subprocess.Popen(
        target.command, cwd=target.cwd or None,
        stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
        creationflags=creationflags)
    outcome.started = True
    outcome.registered = _wait_for(root_url, pairing_token, target.node_id,
                                  timeout, process=child)
    outcome.voters_after = _count_voters(root_url, pairing_token)
    if not outcome.registered:
        return outcome, child

    if lineage_log is not None and signing_key is not None:
        # A reproduction event is a signed manifest for the artifact that was
        # replicated, so it chains into the same lineage the rest of the
        # system verifies (rounds 82-91).
        from .spore import append_lineage, build_manifest
        manifest = build_manifest(
            {Path(payload).name: Path(payload)}, signing_key,
            metadata={"lineage": lineage, "node": target.node_id,
                      "event": "reproduction"},
            parent_manifest=parent_manifest)
        outcome.lineage_entry = append_lineage(Path(lineage_log), manifest)
    return outcome, child


def _count_voters(root_url: str, token: str | None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with open_url(
                urllib.request.Request(f"{root_url}/nodes", headers=headers),
                timeout=5) as resp:
            return len(json.loads(resp.read()).get("nodes", {}))
    except (urllib.error.URLError, urllib.error.HTTPError):
        return 0
