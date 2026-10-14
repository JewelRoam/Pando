"""PandoSI smoke tests.

Policy (project decision): minimal mocks, minimal test count. Static analysis
and research are the default; this file runs only at milestones or on demand.
Two cases cover the contract: quorum gating semantics, and one real-HTTP
end-to-end vote including node death and checkpoint-diverse quorum picking.
"""

import json
import sys
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
