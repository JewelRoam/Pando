"""PandoSI smoke tests.

Policy (project decision): minimal mocks, minimal test count. Static analysis
and research are the default; this file runs only at milestones or on demand.
Two cases cover the contract: quorum gating semantics, and one real-HTTP
end-to-end vote including node death and checkpoint-diverse quorum picking.
"""

import json
import sys
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pandosi.backend import MockBackend
from pandosi.common import (DecisionRequest, Vote, aggregate,
                            canonical_question)
from pandosi.rootd import Coordinator, INITIAL_STEM_TIMEOUT, Root
from pandosi.stemd import Stem
from pandosi.onboard import build_plan
from pandosi.grants import (audit_status, grant_status, require_grant,
                            save_grant)
from pandosi.service import build_service_plan, detect_platform


class TestSmoke(unittest.TestCase):
    def test_quorum_gating_semantics(self):
        """auto on strong majority, verify on tie, escalate below quorum."""
        strong = [Vote(f"n{i}", "en", "yes", 0.9) for i in range(3)]
        strong.append(Vote("n3", "ml", "no", 0.4))
        strong.append(Vote("n4", "td", "yes", 0.85))
        self.assertEqual(aggregate(strong).tier, "auto")

        tie = [Vote("a", "en", "yes", 0.9), Vote("b", "ml", "no", 0.9),
               Vote("c", "en", "yes", 0.9), Vote("d", "ml", "no", 0.9)]
        result = aggregate(tie)
        self.assertIsNone(result.choice)
        self.assertEqual(result.tier, "verify")

        self.assertEqual(aggregate(strong[:2]).tier, "escalate")

    def test_question_schema_identity_includes_option_set(self):
        first, first_options = canonical_question(
            "intent", ["refund", "complaint", "greeting"])
        reordered, reordered_options = canonical_question(
            "intent", ["greeting", "refund", "complaint"])
        different, _ = canonical_question(
            "intent", ["refund", "greeting"])

        self.assertEqual(first, reordered)
        self.assertEqual(first_options, reordered_options)
        self.assertNotEqual(first, different)

    def test_quorum_fanout_is_parallel(self):
        delay_s = 0.25
        active = 0
        peak = 0
        lock = threading.Lock()

        def transport(url, payload, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(delay_s)
            with lock:
                active -= 1
            return json.dumps({
                "vote": {"choice": "approve", "probability": 0.9}
            }).encode()

        coordinator = Coordinator(transport=transport, quorum_k=3)
        for index, checkpoint in enumerate(("en", "ml", "td")):
            coordinator.register(
                f"node-{index}", f"http://node-{index}", checkpoint)

        started = time.perf_counter()
        result = coordinator.decide(
            DecisionRequest(state="parallel", question="approve?",
                            candidates=["approve", "reject"]))
        elapsed = time.perf_counter() - started

        self.assertEqual(result["responded"], 3)
        self.assertEqual(peak, 3)
        self.assertLess(elapsed, delay_s * 2)

    def test_stem_timeout_adapts_to_observed_latency(self):
        timeouts: list[float] = []

        def transport(url, payload, timeout):
            timeouts.append(timeout)
            time.sleep(0.05)
            return json.dumps({
                "vote": {"choice": "approve", "probability": 0.9}
            }).encode()

        coordinator = Coordinator(transport=transport, quorum_k=1)
        coordinator.register("node-0", "http://node-0", "en")
        request = DecisionRequest(state="timeout", question="approve?",
                                  candidates=["approve", "reject"])
        coordinator.decide(request)
        coordinator.decide(request)

        self.assertEqual(timeouts[0], INITIAL_STEM_TIMEOUT)
        self.assertLess(timeouts[1], INITIAL_STEM_TIMEOUT)
        self.assertGreaterEqual(timeouts[1], 0.5)

    def test_failed_stem_enters_cooldown_and_is_temporarily_skipped(self):
        slow_calls = 0

        def transport(url, payload, timeout):
            nonlocal slow_calls
            if "slow" in url:
                slow_calls += 1
                time.sleep(0.2)
                raise TimeoutError("controlled slow stem")
            return json.dumps({
                "vote": {"choice": "approve", "probability": 0.9}
            }).encode()

        coordinator = Coordinator(
            transport=transport, quorum_k=5,
            call_timeout=0.05, initial_stem_timeout=0.05,
            failure_backoff_base=1.0, failure_backoff_max=1.0)
        for index, checkpoint in enumerate(("en", "ml", "td")):
            coordinator.register(
                f"fast-{index}", f"http://fast-{index}", checkpoint)
        coordinator.register("slow-0", "http://slow-0", "slow")
        request = DecisionRequest(state="cooldown", question="approve?",
                                  candidates=["approve", "reject"])

        first = coordinator.decide(request)
        self.assertEqual(first["responded"], 3)
        self.assertEqual(first["quorum_size"], 4)
        slow_info = coordinator.live_nodes()["slow-0"]
        self.assertEqual(slow_info["stats"]["consecutive_failures"], 1)
        self.assertGreater(slow_info["stats"]["cooldown_remaining_s"], 0)

        second = coordinator.decide(request)
        self.assertEqual(second["responded"], 3)
        self.assertEqual(second["quorum_size"], 3)
        self.assertEqual(slow_calls, 1)

    def test_onboarding_plan_prioritizes_ring0_over_sandbox(self):
        rows = [
            {"status": "ok", "item": "python", "fix": ""},
            {"status": "ok", "item": "model backend", "fix": ""},
            {"status": "locked", "item": "lab sandbox: none",
             "fix": "run inside Docker/Podman/WSL"},
        ]
        plan = build_plan(rows)
        self.assertTrue(plan["ring0_ready"])
        self.assertFalse(plan["sandbox_ready"])
        self.assertEqual(plan["next_command"], "python tools/demo.py")
        self.assertEqual(plan["counts"]["sandbox_required"], 1)

    def test_root_pairing_token_gates_registration(self):
        root = Root()
        root.set_pairing_token("pairing-secret")
        root.start()
        try:
            with urllib.request.urlopen(f"{root.url()}/pairing",
                                        timeout=2) as resp:
                self.assertTrue(json.loads(resp.read())["required"])

            payload = json.dumps({
                "node_id": "unauthorized", "url": "http://127.0.0.1:1",
                "checkpoint": "mock", "pairing_token": "wrong"
            }).encode()
            request = urllib.request.Request(
                f"{root.url()}/register", data=payload,
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(caught.exception.code, 403)

            payload = json.dumps({
                "node_id": "paired", "url": "http://127.0.0.1:1",
                "checkpoint": "mock", "pairing_token": "pairing-secret"
            }).encode()
            request = urllib.request.Request(
                f"{root.url()}/register", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as resp:
                self.assertTrue(json.loads(resp.read())["ok"])

            # Pairing protects the full root surface, not just admission.
            for path in ("/nodes",):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(f"{root.url()}{path}", timeout=2)
                self.assertEqual(caught.exception.code, 403)
            headers = {"Authorization": "Bearer pairing-secret"}
            with urllib.request.urlopen(
                    urllib.request.Request(f"{root.url()}/nodes",
                                           headers=headers),
                    timeout=2) as resp:
                self.assertIn("paired", json.loads(resp.read())["nodes"])
            decision = json.dumps({"state": "x", "question": "ok?",
                                   "candidates": ["yes", "no"]}).encode()
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(
                    urllib.request.Request(f"{root.url()}/decide",
                                           data=decision), timeout=2)
            self.assertEqual(caught.exception.code, 403)
        finally:
            root.stop()

    def test_grant_ledger_expires_and_fails_closed(self):
        import datetime as dt
        import tempfile
        now = dt.datetime.now(dt.timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "grants.toml"
            with self.assertRaises(PermissionError):
                require_grant("lan_listen", path, now)

            save_grant(
                "lan_listen", "0.0.0.0:8080", "LAN", "test",
                (now - dt.timedelta(minutes=1)).isoformat(), path=path)
            self.assertEqual(grant_status("lan_listen", path, now)["status"],
                             "expired")
            with self.assertRaises(PermissionError):
                require_grant("lan_listen", path, now)

            save_grant(
                "lan_listen", "0.0.0.0:8080", "LAN", "test",
                (now + dt.timedelta(hours=1)).isoformat(), path=path)
            grant = require_grant("lan_listen", path, now)
            self.assertEqual(grant["resource"], "0.0.0.0:8080")

            audit = audit_status(path.with_name(path.name + ".audit.jsonl"))
            self.assertEqual(audit["status"], "valid")
            self.assertEqual(audit["entries"], 5)
            audit_file = path.with_name(path.name + ".audit.jsonl")
            audit_file.write_text(
                audit_file.read_text(encoding="utf-8") + '{"tampered":true}\n',
                encoding="utf-8")
            with self.assertRaises(PermissionError):
                require_grant("lan_listen", path, now)

    def test_service_plan_is_explicit_and_keeps_token_out_of_argv(self):
        import datetime as dt
        import tempfile
        platform = detect_platform("auto")
        if platform not in ("windows", "linux", "termux"):
            self.skipTest(f"no managed service adapter on {platform}")
        now = dt.datetime.now(dt.timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grants = root / "grants.toml"
            token = root / "pairing.token"
            token.write_text("pairing-secret", encoding="utf-8")
            blocked = build_service_plan(
                "root", platform=platform, lan=True, port=8090,
                token_path=str(token), grants_path=grants, audit=False)
            self.assertFalse(blocked["install_allowed"])
            self.assertFalse(blocked["commands"])
            self.assertTrue(any(r["name"] == "service_autostart"
                                and r["status"] != "active"
                                for r in blocked["requirements"]))

            save_grant("service_autostart", "user-agent/pandosi",
                       "operator-host", "test",
                       (now + dt.timedelta(hours=1)).isoformat(), path=grants)
            save_grant("lan_listen", "0.0.0.0:8090", "LAN", "test",
                       (now + dt.timedelta(hours=1)).isoformat(), path=grants)
            plan = build_service_plan(
                "root", platform=platform, lan=True, port=8090,
                token_path=str(token), grants_path=grants, audit=True)
            self.assertTrue(plan["install_allowed"])
            self.assertEqual(plan["platform"], platform)
            self.assertNotIn("pairing-secret", plan["process_argv"])
            self.assertIn("--pairing-token-file", plan["process_argv"])
            self.assertEqual(set(plan["commands"]), {
                "install", "start", "status", "stop", "uninstall"})
            self.assertTrue(plan["files"])
            if platform == "windows":
                self.assertNotIn("schtasks", plan["commands"]["install"])
                self.assertTrue(any("Startup" in item["path"]
                                    for item in plan["files"]))
            audit = audit_status(grants.with_name(
                grants.name + ".audit.jsonl"))
            self.assertEqual(audit["status"], "valid")

    def test_quorum_floor_is_per_category(self):
        """A security decision keeps its voters under latency pressure.

        The adaptive quorum trades checking power for throughput, but a
        playbook can make that trade impossible below a per-category floor
        (round 111). The floor is binding even when the latency target is
        not reachable.
        """
        from pandosi.rootd import Coordinator

        coordinator = Coordinator(
            adaptive_quorum=True, quorum_floor=1, quorum_k=5,
            latency_target_ms=10.0)
        coordinator.gating = {
            "quorum_floor": 2,
            "security": {"quorum_floor": 4},
        }
        self.assertEqual(coordinator._category_floor("approve?"), 2)
        self.assertEqual(coordinator._category_floor("is this phishing?"), 4)
        self.assertEqual(coordinator._category_floor("security threat?"), 4)

        # Under sustained pressure the ordinary floor is reached and held,
        # while the security floor stops the shrink sooner.
        coordinator._decision_ewma_ms = 500.0
        for _ in range(8):
            coordinator._adapt_quorum("approve?")
        self.assertEqual(coordinator.quorum_k, 2)
        coordinator.quorum_k = 5
        for _ in range(8):
            coordinator._adapt_quorum("security threat?")
        self.assertEqual(coordinator.quorum_k, 4)

        # A coordinator with no playbook falls back to the CLI floor.
        bare = Coordinator(adaptive_quorum=True, quorum_floor=3,
                           latency_target_ms=10.0)
        bare._decision_ewma_ms = 500.0
        for _ in range(8):
            bare._adapt_quorum("approve?")
        self.assertEqual(bare.quorum_k, 3)

    def test_content_defined_delta_transfer(self):
        """A replica that holds the parent should receive only the diff.

        Round 107: content-defined boundaries make a same-length edit local,
        so a lineage that shares most of its bytes with its parent ships a
        fraction of the artifact. Verified so a smaller number cannot be a
        different file.
        """
        import hashlib

        from pandosi.spore import (ChunkStore, SporeServer, artifact_digest,
                                   build_manifest_cdc, chunk_spans,
                                   generate_signing_key, pull_spore)

        tmp = Path(tempfile.mkdtemp(prefix="pandosi-delta-"))
        base = tmp / "parent"
        child_dir = tmp / "child"
        base.mkdir()
        child_dir.mkdir()
        body = bytearray(b"".join(
            hashlib.sha256(str(i).encode()).digest()
            for i in range(2 * 1024 * 1024 // 32)))
        (base / "model.bin").write_bytes(body)
        edited = bytearray(body)
        edited[600_000:604_096] = b"\x00" * 4_096
        (child_dir / "model.bin").write_bytes(edited)

        # Boundaries depend on content, so a same-length edit is local.
        parent_spans = {hashlib.sha256(body[a:b]).hexdigest()
                        for a, b in chunk_spans(bytes(body), 64 * 1024)}
        child_spans = {hashlib.sha256(bytes(edited)[a:b]).hexdigest()
                       for a, b in chunk_spans(bytes(edited), 64 * 1024)}
        shared = len(parent_spans & child_spans)
        self.assertGreater(shared, len(child_spans) * 0.5,
                           f"only {shared}/{len(child_spans)} chunks survived "
                           "a 4 KiB edit")

        key = generate_signing_key()
        parent_manifest = build_manifest_cdc(
            {"model.bin": base / "model.bin"}, key, avg_size=64 * 1024)
        child_manifest = build_manifest_cdc(
            {"model.bin": child_dir / "model.bin"}, key,
            parent_manifest=parent_manifest, avg_size=64 * 1024)
        expected = artifact_digest(child_dir / "model.bin")
        server = SporeServer(child_manifest, {"model.bin": child_dir / "model.bin"})
        server.start()
        try:
            cold = pull_spore(server.url(), tmp / "cold", key.verify_key,
                              sandbox_confirmed=True, batch_chunks=8)
            store = ChunkStore(tmp / "store")
            store.store_manifest(parent_manifest, {"model.bin": base / "model.bin"})
            warm = pull_spore(server.url(), tmp / "warm", key.verify_key,
                              sandbox_confirmed=True, batch_chunks=8,
                              chunk_store=store)
        finally:
            server.stop()

        self.assertEqual(artifact_digest(tmp / "cold" / "model.bin"), expected)
        self.assertEqual(artifact_digest(tmp / "warm" / "model.bin"), expected)
        self.assertEqual(warm["chunks_reused"], shared)
        self.assertLess(warm["bytes_transferred"],
                        cold["bytes_transferred"] * 0.5,
                        "the delta sent more than half the artifact")
        self.assertTrue(all(warm["files"].values()))

    def test_batched_spore_transfer(self):
        """Batches arrive verified per chunk, with a per-chunk fallback.

        Round 105 moved a 1.68 GB artifact in roughly four minutes because
        every 64 KiB chunk was its own request; batching cut the wall time
        8.8x (round 106). Correctness must not depend on the transport being
        polite, so one transient corruption inside a batch has to fall back
        to fetching that chunk alone.
        """
        import secrets

        from pandosi.spore import (SporeClient, SporeServer, build_manifest,
                                   generate_signing_key, pull_spore)

        tmp = Path(tempfile.mkdtemp(prefix="pandosi-batch-"))
        payload = tmp / "model.bin"
        # 151 chunks: several batches, with a short final chunk.
        body = secrets.token_bytes(150 * 64 * 1024 + 1234)
        payload.write_bytes(body)
        key = generate_signing_key()
        manifest = build_manifest({"model.bin": payload}, key)
        server = SporeServer(manifest, {"model.bin": payload})
        server.start()
        try:
            out = tmp / "plain"
            stats = pull_spore(server.url(), out, key.verify_key,
                               sandbox_confirmed=True, batch_chunks=16)
            self.assertEqual(stats["chunks_pulled"], 151)
            self.assertEqual((out / "model.bin").read_bytes(), body)

            original = SporeClient.get
            state = {"faulted": False}

            def flaky(client, path):
                data = original(client, path)
                if path.startswith("/spore/chunks/") and not state["faulted"]:
                    state["faulted"] = True
                    return data[:-1] + bytes([data[-1] ^ 0xFF])
                return data

            SporeClient.get = flaky
            try:
                recovered = tmp / "recovered"
                pull_spore(server.url(), recovered, key.verify_key,
                           sandbox_confirmed=True, batch_chunks=16)
            finally:
                SporeClient.get = original
            self.assertTrue(state["faulted"], "the fault was never injected")
            self.assertEqual((recovered / "model.bin").read_bytes(), body)
        finally:
            server.stop()

    def test_reproduction_policy_fails_closed(self):
        """Every reason to refuse is checked before anything transfers.

        Round 105: the parent decides, and the cheapest failure is the one
        that never moves a payload.
        """
        import datetime as dt

        from pandosi.grants import save_grant
        from pandosi.reproduce import Target, plan_reproduction
        from pandosi.spore import artifact_digest

        tmp = Path(tempfile.mkdtemp(prefix="pandosi-reproduce-"))
        payload = tmp / "model.onnx"
        payload.write_bytes(b"pandosi-reproduction-policy" * 512)
        digest = artifact_digest(payload)
        grants = tmp / "grants.toml"
        target = Target(node_id="child", command=["true"])

        plan = plan_reproduction(payload, "english", target,
                                 declared_lineages=["english"],
                                 declared_artifacts={"english": digest},
                                 grants_path=grants,
                                 sandbox_confirmed=True)
        self.assertFalse(plan.approved)
        self.assertTrue(any("reproduce" in r for r in plan.refusals),
                        plan.refusals)

        expires = (dt.datetime.now(dt.timezone.utc)
                   + dt.timedelta(hours=1)).isoformat()
        save_grant("reproduce", "localhost", "operator-host", "test",
                   expires, path=grants)

        # Granted, but the receiving device has not confirmed a sandbox.
        plan = plan_reproduction(payload, "english", target,
                                 declared_lineages=["english"],
                                 grants_path=grants, sandbox_confirmed=False)
        self.assertFalse(plan.approved)
        self.assertTrue(any("sandbox" in r for r in plan.refusals))

        # Granted and sandboxed, but the lineage is not declared.
        plan = plan_reproduction(payload, "rogue", target,
                                 declared_lineages=["english"],
                                 grants_path=grants, sandbox_confirmed=True)
        self.assertFalse(plan.approved)
        self.assertTrue(any("not declared" in r for r in plan.refusals))

        # Declared, but this is not the artifact the swarm pinned.
        plan = plan_reproduction(payload, "english", target,
                                 declared_lineages=["english"],
                                 declared_artifacts={"english": "0" * 64},
                                 grants_path=grants, sandbox_confirmed=True)
        self.assertFalse(plan.approved)
        self.assertTrue(any("does not match" in r for r in plan.refusals))

        plan = plan_reproduction(payload, "english", target,
                                 declared_lineages=["english"],
                                 declared_artifacts={"english": digest},
                                 grants_path=grants, sandbox_confirmed=True)
        self.assertTrue(plan.approved, plan.refusals)
        self.assertEqual(plan.artifact_sha256, digest)

    def test_join_cli_starts(self):
        """`join` is the replication path; it must at least parse.

        A local `import os` shadowed the module import and made every
        `pandosi.join` invocation die with UnboundLocalError, which no check
        covered until the reproduction loop ran it (round 105).
        """
        result = subprocess.run(
            [sys.executable, "-m", "pandosi.join", "--help"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--spore", result.stdout)

    def test_playbook_signature_covers_declarations(self):
        """A pinned playbook key makes the declaration chain checkable.

        The admitted lineages and their artifact digests live inside the
        signed canonical bytes, so tampering with what the swarm will accept
        invalidates the signature (rounds 102-104).
        """
        import nacl.signing

        from pandosi.playbook import (SIGNATURE_SUFFIX, load_playbook,
                                      load_verified_playbook,
                                      sign_playbook,
                                      verify_playbook_signature)

        tmp = Path(tempfile.mkdtemp(prefix="pandosi-signed-"))
        path = tmp / "signed.toml"
        path.write_text(
            '[meta]\nname = "signed"\nversion = 1\n\n'
            '[lineages]\nadmit = ["english"]\n\n'
            '[lineages.artifacts]\nenglish = "' + "a" * 64 + '"\n',
            encoding="utf-8")
        key = nacl.signing.SigningKey.generate()
        pubkey = key.verify_key.encode().hex()
        playbook = load_playbook(path)

        attestation = sign_playbook(playbook, key)
        valid, detail = verify_playbook_signature(playbook, attestation,
                                                  pubkey)
        self.assertTrue(valid, detail)
        (tmp / (path.name + SIGNATURE_SUFFIX)).write_text(
            json.dumps(attestation), encoding="utf-8")

        loaded, status = load_verified_playbook(path, pubkey)
        self.assertEqual(loaded.hash(), playbook.hash())
        self.assertIn("verified", status)

        # An unpinned signature is self-consistency, not trust.
        _, unpinned = load_verified_playbook(path)
        self.assertIn("unpinned", unpinned)

        # Widening what the swarm admits does not survive the signature.
        path.write_text(
            '[meta]\nname = "signed"\nversion = 1\n\n'
            '[lineages]\nadmit = ["english", "rogue"]\n',
            encoding="utf-8")
        tampered = load_playbook(path)
        valid, detail = verify_playbook_signature(tampered, attestation,
                                                  pubkey)
        self.assertFalse(valid)
        with self.assertRaises(ValueError):
            load_verified_playbook(path, pubkey)

        # A pinned key with no signature at all is also refused.
        (tmp / (path.name + SIGNATURE_SUFFIX)).unlink()
        with self.assertRaises(ValueError):
            load_verified_playbook(path, pubkey)
        _, status = load_verified_playbook(path)
        self.assertIn("unsigned", status)

    def test_declared_lineages_gate_registration(self):
        """A playbook that declares lineages admits only those lineages.

        Identity must be declared rather than inferred from a free-form
        checkpoint string: otherwise a misnamed device silently runs a
        different gate (round 100). A root with no declared lineages stays
        permissive so the synthetic harnesses keep working (round 101).
        """
        from pandosi.playbook import load_playbook, resolve_admitted_lineages

        tmp = Path(tempfile.mkdtemp(prefix="pandosi-lineage-"))
        playbook_path = tmp / "declared.toml"
        playbook_path.write_text(
            '[meta]\nname = "declared"\nversion = 1\n\n'
            '[lineages]\nadmit = ["english", "not-a-real-lineage"]\n',
            encoding="utf-8")
        playbook = load_playbook(playbook_path)
        admitted, unusable = resolve_admitted_lineages(playbook)
        self.assertEqual(admitted, {"english"})
        self.assertEqual(unusable, ["not-a-real-lineage"])

        root = Root()
        root.playbook = playbook
        root.admitted_lineages = admitted
        root.start()
        stems = []
        try:
            declared = Stem("ok", MockBackend("ok", "english"))
            declared.start()
            stems.append(declared)

            def register(node_id: str, checkpoint: str,
                         artifact: str | None = None) -> tuple[int, dict]:
                body = {"node_id": node_id, "url": "http://127.0.0.1:1",
                        "checkpoint": checkpoint}
                if artifact:
                    body["artifact_sha256"] = artifact
                payload = json.dumps(body)
                request = urllib.request.Request(
                    f"{root.url()}/register", data=payload.encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=10) as resp:
                        return resp.status, json.loads(resp.read())
                except urllib.error.HTTPError as exc:
                    return exc.code, json.loads(exc.read())

            status, payload = register("ok", "english")
            self.assertEqual(status, 200)

            status, payload = register("sneaky", "english-v2")
            self.assertEqual(status, 403)
            self.assertIn("not declared", payload["error"])
            self.assertEqual(payload["admitted"], ["english"])

            # Declaring a lineage is a claim; requiring a digest makes it
            # checkable. A stem that cannot attest the declared artifact is
            # refused, and so is one that attests a different artifact.
            expected = "a" * 64
            root.lineage_artifacts = {"english": expected}
            status, payload = register("unattested", "english")
            self.assertEqual(status, 403)
            self.assertIn("attestation", payload["error"])
            self.assertEqual(payload["expected_sha256"], expected)

            status, _ = register("liar", "english", "b" * 64)
            self.assertEqual(status, 403)

            status, _ = register("honest", "english", expected)
            self.assertEqual(status, 200)

            # The refused node never joined.
            with urllib.request.urlopen(f"{root.url()}/nodes",
                                        timeout=10) as resp:
                nodes = json.loads(resp.read())["nodes"]
            self.assertIn("ok", nodes)
            self.assertNotIn("sneaky", nodes)
        finally:
            for stem in stems:
                stem.stop()
            root.stop()

    def test_uncalibrated_checkpoint_cannot_reach_auto(self):
        """The acting loop refuses votes whose confidence scale is unknown.

        Calibration is looked up by the checkpoint string a stem reports, so
        a misnamed or brand-new device runs a different gate with no error.
        The coordinator reports that in ``uncalibrated_checkpoints``; the
        autonomy loop must refuse to act on it (round 100).
        """
        from pandosi.autonomy import AutonomyLoop, Observation, RootDecider

        root = Root()
        root.start()
        coord = root.coordinator
        stems = []
        acted = []
        try:
            for i in range(3):
                stem = Stem(f"u{i}", MockBackend(f"u{i}", "uncatalogued-v9",
                                                 correct_choice="approve",
                                                 agree_rate=1.0))
                stem.start()
                stems.append(stem)
                coord.register(stem.node_id, stem.url(), "uncatalogued-v9")

            loop = AutonomyLoop(
                decide=RootDecider(root.url()),
                actuator=lambda obs, choice, res: acted.append(choice),
                allowed_tiers=("auto",), min_responders=3)
            outcome = loop.step(Observation("obs", "smoke", "approve?",
                                            ["approve", "reject"]))

            # The consensus itself is strong and unanimous; only the missing
            # calibration table stops the loop from acting.
            self.assertNotEqual(outcome.tier, "escalate")
            self.assertFalse(outcome.acted)
            self.assertIn("uncalibrated", outcome.reason)
            self.assertEqual(acted, [])
        finally:
            for stem in stems:
                stem.stop()
            root.stop()

    def test_end_to_end_http_vote(self):
        """Real root + real HTTP stems; dead node shrinks quorum gracefully."""
        root = Root()
        root.start()
        coord = root.coordinator
        stems = []
        checkpoints = ["english", "multilingual", "english",
                       "typed-decisions", "english"]
        try:
            for i, ckpt in enumerate(checkpoints):
                stem = Stem(f"n{i}", MockBackend(f"n{i}", ckpt,
                                                 correct_choice="approve",
                                                 agree_rate=0.9))
                stem.start()
                stems.append(stem)
                coord.register(stem.node_id, stem.url(), ckpt)
            # one registered-but-dead node to prove graceful shrink
            coord.register("dead", "http://127.0.0.1:1", "english")

            picked = coord.pick_quorum(5)
            ckpts = {info["checkpoint"] for _, info in picked}
            self.assertGreaterEqual(len(ckpts), 2,  "quorum must be checkpoint-diverse")

            req = DecisionRequest(state="smoke", question="approve?",
                                  candidates=["approve", "reject"])
            with urllib.request.urlopen(
                urllib.request.Request(f"{root.url()}/decide",
                                       data=req.to_json(),
                                       headers={"Content-Type": "application/json"}),
                timeout=10) as resp:
                result = json.loads(resp.read())

            self.assertEqual(result["choice"], "approve")
            self.assertGreaterEqual(result["responded"], 4)  # 5 picked incl. 1 dead -> graceful shrink
            self.assertIn(result["tier"], ("auto", "verify"))
        finally:
            for stem in stems:
                stem.stop()
            root.stop()


if __name__ == "__main__":
    unittest.main()
