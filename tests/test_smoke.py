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

from pandosi.backend import (MockBackend, OpenAIChoiceBackend,
                            parse_openai_choice)
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

    def test_assent_gating_demotes_auto(self):
        """A named assent checkpoint can veto an otherwise-auto decision."""
        def decide(ft_choice: str, assent_checkpoint: str | None = None,
                   assent_fallback: str = "advisor"):
            coordinator = Coordinator(quorum_k=3,
                                      assent_checkpoint=assent_checkpoint,
                                      assent_fallback=assent_fallback)
            stems = []
            try:
                members = [
                    ("english-0", "english", "approve"),
                    ("english-1", "english", "approve"),
                    ("ft-v3", "ft-v3", ft_choice),
                ]
                for node_id, checkpoint, choice in members:
                    stem = Stem(node_id, MockBackend(
                        node_id, checkpoint, correct_choice=choice,
                        agree_rate=1.0))
                    stem.start()
                    stems.append(stem)
                    coordinator.register(stem.node_id, stem.url(), checkpoint)
                return coordinator.decide(DecisionRequest(
                    state="assent", question="approve?",
                    candidates=["approve", "reject"]))
            finally:
                for stem in stems:
                    stem.stop()

        agreed = decide("approve", "ft-v3")
        self.assertEqual(agreed["choice"], "approve")
        self.assertEqual(agreed["tier"], "auto")

        split = decide("reject", "ft-v3")
        self.assertEqual(split["choice"], "approve")
        self.assertEqual(split["tier"], "verify")
        self.assertNotIn("veto", split)

        veto = decide("reject", "ft-v3", assent_fallback="veto")
        self.assertEqual(veto["choice"], "reject")
        self.assertEqual(veto["tier"], "verify")
        self.assertEqual(veto["veto"], "ft-v3")
        self.assertEqual(veto["agreement"], split["agreement"])
        self.assertEqual(veto["mean_probability"], split["mean_probability"])

        control = decide("reject")
        self.assertEqual(control["choice"], "approve")
        self.assertEqual(control["tier"], "auto")

        root = Root(coordinator=Coordinator(
            quorum_k=3, assent_checkpoint="ft-v3",
            assent_fallback="veto"))
        root.start()
        try:
            with urllib.request.urlopen(f"{root.url()}/status",
                                        timeout=2) as resp:
                status = json.loads(resp.read())
        finally:
            root.stop()
        self.assertEqual(status["assent_fallback"], "veto")

    def test_openai_choice_backend_parsing_and_payload(self):
        """Variant B parsing is strict, and the request is the agreed shape."""
        candidates = ["approve", "reject", "escalate"]
        self.assertEqual(parse_openai_choice("approve", candidates), "approve")
        self.assertEqual(parse_openai_choice("  APPROVE  ", candidates),
                         "approve")
        self.assertEqual(parse_openai_choice("答案：reject。", candidates),
                         "reject")
        with self.assertRaises(ValueError):
            parse_openai_choice("approve or reject", candidates)
        with self.assertRaises(ValueError):
            parse_openai_choice("no idea", candidates)

        seen = {}

        def fake_transport(url, payload, api_key, timeout):
            seen.update(url=url, payload=payload, api_key=api_key,
                        timeout=timeout)
            return {"choices": [{"message": {"content": " escalate "}}]}

        backend = OpenAIChoiceBackend(
            "mac-test", "http://127.0.0.1:9999/v1/", api_key="secret",
            model="qwen-test", checkpoint="openai-v1",
            probability=0.6, transport=fake_transport)
        vote = backend.decide(DecisionRequest(
            state="state", question="what?", candidates=candidates))
        self.assertEqual(vote.choice, "escalate")
        self.assertEqual(vote.probability, 0.6)
        self.assertEqual(seen["url"],
                         "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(seen["api_key"], "secret")
        self.assertEqual(seen["payload"]["max_tokens"], 16)
        self.assertEqual(seen["payload"]["temperature"], 0.0)
        self.assertFalse(seen["payload"]["stream"])
        self.assertEqual(seen["payload"]["model"], "qwen-test")
        self.assertIn("只输出候选词本身", seen["payload"]["messages"][0]["content"])
        self.assertIn("只从以上候选中选择一个。",
                      seen["payload"]["messages"][1]["content"])

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

    def test_elastic_supervisor_policy(self):
        """Grow only when the other dials are exhausted.

        A slow decision with quorum left to spend is a policy question; slow
        at the floor with members to spare is a capacity one (round 113).
        """
        from pandosi.elastic import ElasticSupervisor
        from pandosi.elastic import Supervision

        supervisor = ElasticSupervisor("http://127.0.0.1:1", max_members=5,
                                      keep=2)

        def status(**overrides) -> dict:
            base = {"members": 3, "quorum_k": 1, "quorum_floor": 1,
                    "latency_target_ms": 50.0, "decision_ewma_ms": 200.0,
                    "adaptive": True}
            base.update(overrides)
            return base

        self.assertEqual(supervisor.decide(status()).action, "grow")
        # Quorum head-room: shrink the quorum before adding hardware.
        held = supervisor.decide(status(quorum_k=3))
        self.assertEqual(held.action, "hold")
        self.assertIn("can still shrink", held.reason)
        # At the member ceiling, more members are not the answer.
        capped = supervisor.decide(status(members=5))
        self.assertEqual(capped.action, "hold")
        self.assertIn("per-decision-cost", capped.reason)
        # Inside the target, leave it alone.
        calm = supervisor.decide(status(decision_ewma_ms=52.0))
        self.assertEqual(calm.action, "hold")
        self.assertIn("within", calm.reason)
        # A static quorum cannot shrink, so the floor test does not apply.
        static = supervisor.decide(status(adaptive=False, quorum_k=3))
        self.assertEqual(static.action, "grow")
        # No signal yet: never act on nothing.
        self.assertEqual(
            supervisor.decide(status(decision_ewma_ms=None)).action, "hold")

        # A decision nobody executed must not start a cooldown or be recorded
        # as an action (found in round 114: a supervisor with no retire
        # executor re-reported the same retirement every cycle).
        idle = ElasticSupervisor("http://127.0.0.1:1", max_members=2,
                                cooldown_s=60.0)
        idle.last_action = Supervision("retire", "test", target="member-9")
        idle.last_action_at = idle._clock()
        self.assertGreater(idle.cooling_down(), 0)
        self.assertEqual(idle.actions, [])

        # Host awareness: never grow a lineage onto a machine that already
        # carries one, because that adds compute contention and no
        # information (rounds 95 and 114, measured).
        from pandosi.elastic import Member, co_located
        members = [
            Member("a", "english", 40.0, 5, "127.0.0.1"),
            Member("b", "english", 45.0, 5, "10.0.0.9"),
            Member("c", "rlcd-v7", 50.0, 5, "127.0.0.1"),
        ]
        self.assertEqual(len(co_located(members, "127.0.0.1", "english")), 1)
        self.assertEqual(len(co_located(members, "127.0.0.1", "rlcd-v7")), 1)
        self.assertEqual(co_located(members, "10.0.0.9", "rlcd-v7"), [])
        self.assertEqual(co_located(members, None, "english"), [])
        blocked = supervisor.decide(status(), co_located_members=1,
                                   candidate="127.0.0.1/english")
        self.assertEqual(blocked.action, "hold")
        self.assertIn("add no information", blocked.reason)

        # Capacity accounting: a rollback must not remove a member the swarm
        # depends on (round 116).
        from pandosi.elastic import rollback_allowed
        serving = [Member("a", "english", 40.0, 9, "127.0.0.1"),
                   Member("b", "english", 45.0, 0, "127.0.0.1")]
        allowed, why = rollback_allowed("a", serving, keep=1)
        self.assertFalse(allowed)
        self.assertIn("no other member has served any", why)
        allowed, _ = rollback_allowed("b", serving, keep=1)
        self.assertTrue(allowed)
        allowed, why = rollback_allowed("a", serving, keep=2)
        self.assertFalse(allowed)
        self.assertIn("below keep", why)

        # A rolled-back change is remembered, so the loop does not repeat it
        # (found by running the loop long enough to oscillate).
        remembering = ElasticSupervisor("http://127.0.0.1:1", max_members=5)
        remembering.rolled_back[("127.0.0.1", "english")] = {
            "at": remembering._clock(), "observed": 420.0, "before": 106.0}
        self.assertIsNotNone(remembering._failed_before("127.0.0.1", "english"))
        repeated = remembering.decide(
            status(), failed_before=remembering._failed_before(
                "127.0.0.1", "english"))
        self.assertEqual(repeated.action, "hold")
        self.assertIn("measured and rejected", repeated.reason)
        # A different host is not tarred with the same brush.
        self.assertIsNone(remembering._failed_before("10.0.0.9", "english"))
        # ...and the memory expires.
        remembering.bad_candidate_ttl_s = 0.0
        self.assertIsNone(remembering._failed_before("127.0.0.1", "english"))

    def test_quorum_floor_is_per_category(self):
        """A security decision keeps its voters under latency pressure.

        The adaptive quorum trades checking power for throughput, but a
        playbook can make that trade impossible below a per-category floor
        (round 111). The floor is binding even when the latency target is
        not reachable.
        """
        from pandosi.rootd import Coordinator

        from pandosi.playbook import load_playbook

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

        # A declared category decides, so a question whose *text* looks
        # innocent still gets the demanding floor (round 112).
        tmp = Path(tempfile.mkdtemp(prefix="pandosi-category-"))
        path = tmp / "categorised.toml"
        path.write_text(
            '[meta]\nname = "categorised"\nversion = 1\n\n'
            '[gating]\nquorum_floor = 2\n\n'
            '[gating.security]\nquorum_floor = 4\n\n'
            '[[questions]]\nid = "risk_check"\ntype = "choice"\n'
            'category = "security"\ncriteria = ["freeze", "allow"]\n\n'
            '[[questions]]\nid = "intent"\ntype = "choice"\n'
            'category = "ordinary"\ncriteria = ["a", "b"]\n',
            encoding="utf-8")
        in_playbook = load_playbook(path)
        self.assertEqual(in_playbook.question_categories,
                         {"risk_check": "security", "intent": "ordinary"})
        declared = Coordinator(adaptive_quorum=True, quorum_floor=1,
                              latency_target_ms=10.0)
        declared.gating = in_playbook.gating
        declared.question_categories = in_playbook.question_categories
        self.assertEqual(declared._category_floor("risk_check"), 4)
        self.assertEqual(declared._category_floor("intent"), 2)
        # Unclassified in a categorised playbook: fail conservative.
        self.assertEqual(declared._category_floor("something_new"), 4)
        # A keyword match still raises the floor even if declared ordinary.
        self.assertEqual(declared._category_floor("intent"), 2)
        self.assertEqual(declared._category_floor("phishing?"), 4)

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

    def test_latency_target_buys_the_fast_twin(self):
        """With a latency target set, the seat goes to the fast member.

        Round 156 measured the cost-first order live: with a 100 ms target
        the root kept the 110 ms NPU seated (cheaper per host-CPU-second)
        while the 40 ms iGPU idled, p50 115 ms at the quorum floor. With a
        target configured the operator has said latency outranks host cost;
        without one, cost leads (round 134).
        """
        from pandosi.rootd import Coordinator

        def coordinator(target):
            coord = Coordinator(quorum_k=1, latency_target_ms=target)
            coord.register("cheap-slow", "http://127.0.0.1:1", "english",
                           cap={"host_ms_per_decision": 5.0})
            coord.register("fast-pricey", "http://127.0.0.1:2", "english",
                           cap={"host_ms_per_decision": 56.0})
            coord.nodes["cheap-slow"]["stats"]["latency_ewma_ms"] = 110.0
            coord.nodes["fast-pricey"]["stats"]["latency_ewma_ms"] = 40.0
            return coord

        picked = [node_id for node_id, _ in
                  coordinator(100.0).pick_quorum(1)]
        self.assertEqual(picked, ["fast-pricey"])
        picked = [node_id for node_id, _ in
                  coordinator(None).pick_quorum(1)]
        self.assertEqual(picked, ["cheap-slow"])

    def test_opinion_ceiling_under_collapse(self):
        """A lineage-counted quorum never expands k past the opinion count.

        Round 156 measured the unfixed dial live: a two-opinion fleet
        oscillated k 2<->3 every other decision, because landing under
        target*0.6 asked for one more voter - and the only voter left was a
        replica the lineage had already cast. The ceiling is the number of
        opinions, so the dial stops at 2.
        """
        from pandosi.rootd import Coordinator

        coord = Coordinator(adaptive_quorum=True, quorum_floor=2,
                            quorum_k=3, latency_target_ms=100.0,
                            count_lineages=True)
        coord.register("en-a", "http://127.0.0.1:1", "english")
        coord.register("en-b", "http://127.0.0.1:2", "english")
        coord.register("ft", "http://127.0.0.1:3", "ft-v3")
        coord.quorum_k = 2
        coord._decision_ewma_ms = 30.0  # comfortably under target*0.6
        for _ in range(4):
            coord._adapt_quorum("approve?")
        self.assertEqual(coord.quorum_k, 2)


if __name__ == "__main__":
    unittest.main()
