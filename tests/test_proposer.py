"""PandoRSI smoke tests.

Policy (project decision): minimal mocks, minimal test count. Static analysis
and research are the default; this file runs only at milestones or on demand.
Two cases cover the contract: quorum gating semantics, and one real-HTTP
end-to-end vote including node death and checkpoint-diverse quorum picking.
"""

import argparse
import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import plistlib
import re
import shutil
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import _git_env, _git_in  # noqa: E402, F401

from pando.backend import (MockBackend, OpenAIChoiceBackend,
                            parse_openai_choice)
from pando.common import (DecisionRequest, Vote, aggregate,
                            canonical_question, mcnemar_test)
from pando.coordinator import (Coordinator, INITIAL_STEM_TIMEOUT,
                            Root, load_accuracy_registry)
from pando.worker import Stem
from pando.onboard import build_plan
from pando import graft
from pando.grants import (audit_status, grant_status, require_grant,
                            save_grant)
from pando.service import build_service_plan, detect_platform


class TestResidentThinker(unittest.TestCase):
    """pando.proposer - the proposer that has to keep running when nobody asks.

    The measured failure this guards (2026-10-17): one agent process died and
    the colony produced nothing for 7h47m, because every other device was a
    responder. The contract under test is deliberately narrow - a thinker that
    stops on a bad round is the same single point of failure in new clothes.
    """

    def _thinker(self, answers, ledger_path, published=None, **kw):
        from pando.identity import Identity
        from pando.proposer import Ledger, Thinker

        class ScriptedProposer:
            """Hands back canned answers; raises when the script says to."""

            def __init__(self, script):
                self.script = list(script)
                self.calls = 0
                self.users = []

            def complete(self, system, user):
                self.calls += 1
                self.users.append(user)
                item = self.script.pop(0) if self.script else {"action": ""}
                if isinstance(item, Exception):
                    raise item
                return json.dumps(item, ensure_ascii=False), {"completion_tokens": 7}

        sent = []

        def publish(proposal, identity, url, token):
            sent.append(proposal)
            return {"task": {"id": "t-x"}, "claimed": True}

        proposer = ScriptedProposer(answers)
        defaults = {"fetch_history": lambda: [], "fetch_activity": lambda: []}
        defaults.update(kw)
        thinker = Thinker(
            node_id="test-node", proposer=proposer, identity=Identity.generate(),
            ledger=Ledger(ledger_path), mailbox_url="http://invalid.example",
            interval=0, dry_run=False,
            pause=kw.pop("pause", lambda _s: None),
            publish=(publish if published is None else published), **defaults)
        return thinker, proposer, sent

    def test_proposer_pulls_json_out_of_a_chatty_answer(self):
        """A 4B model wraps its JSON in prose; that must not be a lost round."""
        from pando.proposer import extract_json

        text = ('Sure! Here is my proposal:\n'
                '```json\n'
                '{"action": "seed.export", "goal": "carry a seed", '
                '"cites": ["r1"], "nested": {"a": 1, "b": [2, 3]}}\n'
                '```\nLet me know if you want another one.')
        obj = extract_json(text)
        self.assertEqual(obj["action"], "seed.export")
        self.assertEqual(obj["nested"]["b"], [2, 3])
        self.assertIsNone(extract_json("no json at all here"))

    def test_proposer_accepts_a_comma_joined_cites_string(self):
        """Measured: qwen3-4b-instruct-2507 returns cites as one string."""
        from pando.proposer import _as_cites

        self.assertEqual(_as_cites("r1:seed.export, r2:recruit.invite"),
                         ["r1:seed.export", "r2:recruit.invite"])
        self.assertEqual(_as_cites(["a", " b "]), ["a", "b"])
        self.assertEqual(_as_cites(None), [])

    def test_proposer_refuses_an_action_outside_the_space(self):
        """The model proposes; it does not get to widen its own action space."""
        from pando.proposer import proposal_from_obj

        proposal, why = proposal_from_obj(
            {"action": "overthrow.the.operator", "goal": "take over the machine"},
            "n1", "n1", 1)
        self.assertIsNone(proposal)
        self.assertIn("not in the action space", why)
        proposal, why = proposal_from_obj(
            {"action": "seed.export", "goal": ""}, "n1", "n1", 1)
        self.assertIsNone(proposal)
        self.assertIn("no goal", why)
        proposal, why = proposal_from_obj(
            {"action": "seed.replicate", "goal": "carry it", "destination": "win"},
            "n1", "n1", 1)
        self.assertEqual(why, "")
        self.assertEqual(proposal.destination, "win")
        self.assertNotIn("destination", proposal_from_obj(
            {"action": "seed.export", "goal": "seal", "destination": "win"},
            "n1", "n1", 1)[0].payload())
        proposal, why = proposal_from_obj(
            {"action": "seed.replicate", "goal": "carry it", "destination": "http://win"},
            "n1", "n1", 1)
        self.assertIsNone(proposal)
        self.assertIn("not a host name", why)

    def test_proposer_requires_valid_signed_params_for_policy_set(self):
        """The proposer and executor must agree on the one typed action intent.

        A ``policy.set`` with prose but no valid params is not a rule change;
        publishing it would only create an unexecutable loop entry. The same
        validator is used here and by the executor, so the intent shape cannot
        drift between the two.
        """
        from pando.proposer import proposal_from_obj

        proposal, why = proposal_from_obj(
            {"action": "policy.set", "goal": "raise the adoption floor",
             "params": {"set": {"min_adopt": 3}}},
            "n1", "n1", 7)
        self.assertEqual(why, "")
        self.assertEqual(proposal.params, {"set": {"min_adopt": 3}})
        self.assertEqual(proposal.payload()["params"],
                         {"set": {"min_adopt": 3}})

        for raw in (None, {}, {"set": {}}, {"set": {"min_adopt": 9}},
                    {"set": {"nope": 1}}, {"min_adopt": 3},
                    {"set": {"min_adopt": True}}):
            bad, why = proposal_from_obj(
                {"action": "policy.set", "goal": "change a rule",
                 "params": raw},
                "n1", "n1", 7)
            self.assertIsNone(bad)
            self.assertIn("policy.set params invalid", why)

        bad, why = proposal_from_obj(
            {"action": "seed.export", "goal": "seal", "params": {"set": {}}},
            "n1", "n1", 7)
        self.assertIsNone(bad)
        self.assertIn("does not accept params", why)

    def test_the_policy_skeleton_is_a_params_block_the_executor_accepts(self):
        """The repair example must be the exact shape the executor validates.

        A skeleton that drifted from ``pando.policy.validate`` would be worse
        than none: the model would copy it faithfully and still be refused.
        """
        from pando.policy import validate as validate_policy
        from pando.proposer import policy_skeleton

        skeleton = policy_skeleton()
        checked, why = validate_policy(skeleton)
        self.assertEqual(why, "")
        self.assertEqual(checked, {"min_adopt": 3})

    def test_a_policy_set_without_params_is_repaired_once(self):
        """Measured on the Mac mailbox (2026-10-19): all 304 visible policy.set
        proposals carried no params, so the rule-change actuator could not run.
        The proposer is shown the typed object once; the validator is unchanged.
        """
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, _sent = self._thinker(
                [{"action": "policy.set", "goal": "lower the gate",
                  "params": None},
                 {"action": "policy.set", "goal": "lower the gate",
                  "expected_effect": "faster adoption",
                  "params": {"set": {"min_adopt": 3}}}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertTrue(round_.policy_params_repaired)
            self.assertTrue(round_.published)
            self.assertEqual(round_.proposal.action, "policy.set")
            self.assertEqual(round_.proposal.params, {"set": {"min_adopt": 3}})
            self.assertEqual(proposer.calls, 2)
            # The retry must carry the shape, not just repeat the prose.
            self.assertIn('"set"', proposer.users[1])
            self.assertIn("min_adopt", proposer.users[1])

    def test_a_policy_set_that_never_gets_params_is_held_not_signed(self):
        """The repair is one call, not a loop. If the model still cannot produce
        the typed object, the round is held and nothing signed reaches the wire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, sent = self._thinker(
                [{"action": "policy.set", "goal": "change a rule"},
                 {"action": "policy.set", "goal": "still prose"}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertFalse(round_.policy_params_repaired)
            self.assertFalse(round_.published)
            self.assertIsNone(round_.proposal)
            self.assertIn("policy.set params invalid", round_.error)
            self.assertEqual(proposer.calls, 2)
            self.assertEqual(sent, [])

    def test_the_operator_skeleton_is_a_params_block_the_executor_accepts(self):
        """The repair example must be the exact shape ``pando.operators`` takes.

        The skeleton is handed to a 1.7B model, and the model copies what it is
        shown. A skeleton ``pando.operators.validate`` refuses would be worse
        than none: the model would echo it faithfully and still be refused.
        """
        from pando.operators import validate as validate_operator
        from pando.proposer import operator_skeleton

        skeleton = operator_skeleton()
        checked, why = validate_operator(skeleton)
        self.assertEqual(why, "")
        self.assertTrue(checked["rationale"])
        # The example must be recognisable *as an example*. Measured
        # 2026-10-19: with a plausible label (``vote.weighted``) the first
        # post-repair rounds echoed it verbatim, and on pi5 the echo
        # contradicting the rationale ("screening operator" named
        # ``vote.weighted``) was invisible in ``requested.operator``.
        name = checked["operator"]
        self.assertIn("example", name.lower())
        self.assertNotIn(".vote", name)

    def test_a_meta_propose_without_an_operator_is_repaired_once(self):
        """Measured on the pi5 ledger (2026-10-19): 33 of the last 40 rounds
        chose ``meta.propose`` and every one was nameless - until this change
        the proposer refused *any* params on that action, so the only
        publishable form was the empty spec the actuator stages and refuses.
        """
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, _sent = self._thinker(
                [{"action": "meta.propose",
                  "goal": "let the group weigh how sure a vote is"},
                 {"action": "meta.propose",
                  "goal": "let the group weigh how sure a vote is",
                  "params": {"operator": "vote.weighted",
                             "rationale": "one vote each ignores confidence"}}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertTrue(round_.meta_params_repaired)
            self.assertTrue(round_.published)
            self.assertEqual(round_.proposal.action, "meta.propose")
            self.assertEqual(round_.proposal.params["operator"], "vote.weighted")
            self.assertEqual(proposer.calls, 2)
            # The retry must carry the shape, not just repeat the prose.
            from pando.proposer import operator_skeleton as skeleton_of
            self.assertIn('"operator"', proposer.users[1])
            self.assertIn(skeleton_of()["operator"], proposer.users[1])

    def test_a_meta_propose_that_never_names_an_operator_is_held_not_signed(self):
        """One repair, not a loop.

        The cost of signing the empty form instead is measured: every proposal
        collects one reaction from each decision model, so a nameless
        ``meta.propose`` buys four to seven inferences and produces a spec
        whose entire content is the refusal list.
        """
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, sent = self._thinker(
                [{"action": "meta.propose", "goal": "a new operator"},
                 {"action": "meta.propose", "goal": "a new operator"}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertFalse(round_.meta_params_repaired)
            self.assertFalse(round_.published)
            self.assertIsNone(round_.proposal)
            self.assertIn("meta.propose params invalid", round_.error)
            self.assertEqual(proposer.calls, 2)
            self.assertEqual(sent, [])

    def test_meta_propose_params_carry_a_name_and_prose_and_nothing_else(self):
        """The one action whose payload ends up being code keeps the rule the
        actuator enforces: labels and prose, never an executable surface.
        """
        from pando.proposer import proposal_from_obj

        for raw in ({"code": "import os"}, {"path": "/tmp/x"}, {"url": "http://x"},
                    {"operator": ""}, {"operator": "   "}, {"operator": 7},
                    {"operator": "x" * 200}):
            bad, why = proposal_from_obj(
                {"action": "meta.propose", "goal": "ask", "params": raw},
                "n1", "n1", 7)
            self.assertIsNone(bad, raw)
            self.assertIn("meta.propose params invalid", why)

        good, why = proposal_from_obj(
            {"action": "meta.propose", "goal": "ask",
             "params": {"operator": "vote.weighted", "rationale": "why"}},
            "n1", "n1", 7)
        self.assertEqual(why, "")
        self.assertEqual(good.params["operator"], "vote.weighted")

    def test_a_named_meta_propose_reaches_the_actuator(self):
        """End to end across the two sides.

        The intent the proposer now signs is the intent the executor stages,
        and the staged spec names the operator instead of carrying "no
        operator named" as its whole content. The implementation refusal stays:
        a name is not code, and nothing here installs anything.
        """
        from pando.identity import Identity
        from pando.operators import stage as stage_operator
        from pando.proposer import proposal_from_obj

        proposal, why = proposal_from_obj(
            {"action": "meta.propose", "goal": "ask for a new operator",
             "params": {"operator": "vote.weighted",
                        "rationale": "certainty should count"}},
            "n1", "n1", 3)
        self.assertEqual(why, "")

        with tempfile.TemporaryDirectory() as tmp:
            result = stage_operator(proposal, identity=Identity.generate(),
                                    outbox=tmp)
            spec = json.loads((Path(tmp) / proposal.proposal_id
                               / "spec.json").read_text(encoding="utf-8"))

        self.assertTrue(result["staged"])
        self.assertEqual(spec["requested"]["operator"], "vote.weighted")
        self.assertEqual(spec["rationale"], "certainty should count")
        self.assertFalse([r for r in spec["refusals"] if "no operator named" in r])
        self.assertTrue([r for r in spec["refusals"] if "no implementation" in r])
        self.assertFalse(spec["installed"])

    def test_proposer_requires_valid_signed_params_for_culture_refine(self):
        """The proposer and executor share one culture.refine intent shape.

        A ``culture.refine`` whose params are not ``{"playbook": <full
        playbook>}`` is not a culture change; publishing it would only create an
        unexecutable loop entry. Both sides call ``pando.culture.validate``, so
        the shape the model is asked for cannot drift from what runs.
        """
        from pando.proposer import proposal_from_obj

        good = {"meta": {"name": "loop", "version": 2},
                "gating": {"quorum_floor": 3}}
        proposal, why = proposal_from_obj(
            {"action": "culture.refine", "goal": "raise the culture floor",
             "params": {"playbook": good}},
            "n1", "n1", 9)
        self.assertEqual(why, "")
        self.assertEqual(proposal.params, {"playbook": good})
        self.assertEqual(proposal.payload()["params"], {"playbook": good})

        for raw in (None, {}, {"playbook": {}}, {"playbook": "patch"},
                    {"playbook": {"meta": {"name": "x", "version": 1},
                                  "unknown_table": 1}},
                    {"set": {"min_adopt": 3}}):
            bad, why = proposal_from_obj(
                {"action": "culture.refine", "goal": "refine", "params": raw},
                "n1", "n1", 9)
            self.assertIsNone(bad, raw)
            self.assertIn("culture.refine params invalid", why)

    def test_the_culture_skeleton_is_a_playbook_the_executor_accepts(self):
        """The example handed to the proposer must be the shape that runs.

        ``pando.culture.validate`` and ``pando.playbook.parse_playbook`` are
        the two doors the candidate has to walk through. A skeleton that
        drifted from them would be worse than no skeleton: the model would
        copy it faithfully and still be refused, and the round would look like
        another failed attempt to think rather than a broken prompt.
        """
        from pando.culture import validate as validate_culture
        from pando.proposer import culture_skeleton

        skeleton = culture_skeleton()
        checked, why = validate_culture({"playbook": skeleton})
        self.assertEqual(why, "")
        self.assertEqual(checked, skeleton)
        # Round-trippable as JSON too: the wire format is what gets signed.
        self.assertEqual(json.loads(json.dumps(skeleton)), skeleton)

    def test_a_culture_refine_without_the_shape_is_repaired_once(self):
        """Measured (pi5 r541, 2026-10-18): the model picked the action and had
        no way to guess the object, so every attempt was held. The repair is
        the same one seed.replicate already gets - show it, ask again - not a
        looser validator, because the validator is what stops prose from being
        signed as a culture artifact.
        """
        from pando.proposer import culture_skeleton

        with tempfile.TemporaryDirectory() as tmp:
            skeleton = culture_skeleton()
            thinker, proposer, _sent = self._thinker(
                [{"action": "culture.refine", "goal": "raise the floor",
                  "params": {"set": {"min_adopt": 3}}},
                 {"action": "culture.refine", "goal": "raise the floor",
                  "expected_effect": "a lower adoption bar",
                  "params": {"playbook": skeleton}}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertTrue(round_.params_repaired)
            self.assertTrue(round_.published)
            self.assertEqual(round_.proposal.action, "culture.refine")
            self.assertEqual(round_.proposal.params, {"playbook": skeleton})
            self.assertEqual(proposer.calls, 2)
            # The retry must actually carry the shape, or it is a plea again.
            self.assertIn('"meta"', proposer.users[1])
            self.assertIn("gating", proposer.users[1])

    def test_a_round_that_never_picks_up_the_shape_is_held_not_signed(self):
        """The repair is one call, not a loop. If the model still cannot do it,
        the round is held and recorded - and nothing signed reaches the wire."""
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, sent = self._thinker(
                [{"action": "culture.refine", "goal": "raise the floor",
                  "params": {"playbook": {"meta": {"name": "x"}}}},
                 {"action": "culture.refine", "goal": "still guessing",
                  "params": {"playbook": {"meta": {"name": "x"}}}}],
                Path(tmp) / "think.jsonl")
            round_ = thinker.think(1)

            self.assertFalse(round_.params_repaired)
            self.assertFalse(round_.published)
            self.assertIsNone(round_.proposal)
            self.assertIn("culture.refine params invalid", round_.error)
            self.assertEqual(proposer.calls, 2)
            self.assertEqual(sent, [])

    def test_proposer_keeps_thinking_through_model_and_mailbox_failures(self):
        """A dead model, dead mailbox, or bad JSON is a recorded round, not an end."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            thinker, _, _ = self._thinker(
                [OSError("connection refused"),
                 OSError("connection refused"),
                 "not json at all",
                 {"action": "seed.replicate", "goal": "carry the seed"},
                 {"action": "recruit.invite", "goal": "onboard a candidate"}],
                ledger)

            thinker.run_forever(max_rounds=4, sleep=lambda _s: None)

            rows = thinker.ledger.tail(10)
            self.assertEqual(len(rows), 4)
            self.assertIn("model call failed", rows[0]["error"])
            self.assertIn("did not return a JSON object", rows[1]["error"])
            self.assertEqual(rows[2]["action"], "seed.replicate")
            self.assertTrue(rows[2]["published"])
            self.assertEqual(rows[3]["action"], "recruit.invite")
            # The loop reached round 4 despite two dead rounds - that is the
            # whole point: nothing here is allowed to end the process.
            self.assertEqual([r["round"] for r in rows], [1, 2, 3, 4])

    def test_proposer_retries_one_dropped_model_call(self):
        """A respawned server should not cost the whole round."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            thinker, proposer, sent = self._thinker(
                [OSError("closed"),
                 {"action": "seed.export", "goal": "carry a seed"}],
                ledger)
            thinker.run_forever(max_rounds=1, sleep=lambda _s: None)
            row = thinker.ledger.tail(1)[0]
            self.assertEqual(row["action"], "seed.export")
            self.assertTrue(row["call_retried"])
            self.assertFalse(row["retried"])
            self.assertFalse(row["error"])
            self.assertEqual(proposer.calls, 2)
            self.assertEqual(len(sent), 1)

    def test_a_dropped_call_does_not_count_as_a_repaired_repeat(self):
        """Reconnect and novelty repair are different facts."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            thinker, proposer, sent = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"},
                 OSError("closed"),
                 {"action": "seed.export", "goal": "again"},
                 {"action": "seed.export", "goal": "still the same"}],
                ledger)
            thinker.run_forever(max_rounds=2, sleep=lambda _s: None)
            second = thinker.rounds[1]
            self.assertTrue(second.call_retried)
            self.assertTrue(second.repeated)
            self.assertFalse(second.retried)
            self.assertTrue(second.novelty_forced)
            self.assertEqual(second.proposal.action, "seed.export")
            self.assertEqual(proposer.calls, 4)
            self.assertEqual(len(sent), 2)

    def test_proposer_repairs_a_repeat_and_records_that_it_had_to(self):
        """Self-repetition is repaired once, and the repair is visible."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            same = {"action": "culture.refine", "goal": "tune the voter",
                    "params": {"playbook": {"meta": {"name": "tune", "version": 2}}}}
            thinker, proposer, sent = self._thinker(
                [same, dict(same),
                 {"action": "resource.rebalance", "goal": "move the member",
                  "expected_effect": "frees the saturated host"},
                 {"action": "seed.export", "goal": "carry a seed"}],
                ledger)

            # run_forever, not think(): the retry reads the round *already
            # written down*, so a ledger that is never appended to would make
            # the correction invisible - which is exactly the bug this covers.
            thinker.run_forever(max_rounds=2, sleep=lambda _s: None)
            first, second = thinker.rounds

            self.assertTrue(first.published)
            self.assertFalse(first.repeated)

            self.assertTrue(second.repeated)     # the repeat happened
            self.assertTrue(second.retried)      # and was repaired
            self.assertFalse(second.novelty_forced)
            self.assertEqual(second.proposal.action, "resource.rebalance")
            self.assertEqual(proposer.calls, 3)  # one answer + one forced retry
            self.assertEqual([p.action for p in sent],
                             ["culture.refine", "resource.rebalance"])

    def test_proposer_marks_an_unrepaired_repeat_instead_of_dressing_it_up(self):
        """Measured 2026-10-17: the model rewords the goal and keeps the verb.

        A reworded goal is not a new step. When the repair fails, the round must
        still carry the fact - otherwise the ledger says "24h of thought" while
        the organism walked into the same wall all day.
        """
        with tempfile.TemporaryDirectory() as tmp:
            same = {"action": "seed.export", "goal": "carry a seed"}
            reworded = {"action": "seed.export",
                        "goal": "carry a seed, but this time modular and portable",
                        "expected_effect": "a much longer explanation than before "
                                           "so that a length-based heuristic "
                                           "would wrongly accept it"}
            thinker, _, sent = self._thinker(
                [same, dict(same), reworded], Path(tmp) / "think.jsonl")

            thinker.run_forever(max_rounds=2, sleep=lambda _s: None)
            second = thinker.rounds[1]

            self.assertTrue(second.repeated)
            self.assertFalse(second.retried)      # the repair did NOT take
            self.assertTrue(second.novelty_forced)  # and the record says so
            self.assertEqual(second.proposal.action, "seed.export")

    def test_proposer_falls_over_to_the_surviving_endpoint(self):
        """The 8B is reaped first under memory pressure; the 4B must take over."""
        from pando.proposer import FailoverProposer, OpenAIProposer

        class Endpoint(OpenAIProposer):
            def __init__(self, model, answer):
                super().__init__(base_url="http://x", model=model)
                self.answer = answer
                self.calls = 0

            def complete(self, system, user):
                self.calls += 1
                if isinstance(self.answer, Exception):
                    raise self.answer
                return json.dumps(self.answer), {"completion_tokens": 3}

        big = Endpoint("qwen3-8b", OSError("model swapped out"))
        small = Endpoint("qwen3-4b", {"action": "seed.export", "goal": "carry a seed"})
        chain = FailoverProposer([big, small])

        text, _usage = chain.complete("s", "u")
        self.assertIn("seed.export", text)
        self.assertEqual(chain.served_by, "qwen3-4b")
        self.assertEqual((big.calls, small.calls), (1, 1))

        # Both down is still an exception the round recorder turns into a held
        # round - the loop does not end, and it does not pretend to have thought.
        dead = FailoverProposer([Endpoint("a", OSError("x")),
                                 Endpoint("b", OSError("y"))])
        with self.assertRaises(OSError):
            dead.complete("s", "u")

    def test_proposer_refreshes_the_outcome_feed_on_a_stride(self):
        """The mailbox listing is O(tasks): asking every round would be suicide."""
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def activity():
                calls.append(1)
                return [{"status": "done", "title": "carried a seed"}]

            actions = [{"action": a, "goal": f"goal for {a}"} for a in
                       ["seed.export", "seed.replicate", "recruit.invite",
                        "reproduce.spawn", "policy.set", "culture.refine",
                        "resource.rebalance", "meta.propose"]] * 2
            thinker, _, _ = self._thinker(
                actions, Path(tmp) / "think.jsonl",
                fetch_activity=activity, activity_every=4)

            thinker.run_forever(max_rounds=8, sleep=lambda _s: None)

            self.assertEqual(len(thinker.rounds), 8)
            self.assertEqual(len(calls), 2)  # 8 rounds, stride 4

    def test_proposer_reads_bounded_final_outcomes_not_the_full_message_feed(self):
        """Reaction traffic must not turn a trend read into a 9 MB poll."""
        from unittest import mock
        from pando import proposer

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _, _ = self._thinker(
                [], Path(tmp) / "think.jsonl", history_window=2)
            calls = []
            by_status = {
                "done": [
                    {"id": "t1", "status": "done", "created_at": "2026-10-27T10:00:00",
                     "title": "carried a seed"},
                    {"id": "t2", "status": "done", "created_at": "2026-10-27T12:00:00",
                     "title": "[proposal] p2 seed.export by mac-proposer"},
                ],
                "failed": [
                    {"id": "t3", "status": "failed", "created_at": "2026-10-27T11:00:00",
                     "title": "attempt that failed"},
                ],
                "blocked": [
                    {"id": "t4", "status": "blocked", "created_at": "2026-10-27T13:00:00",
                     "title": "[reaction] p3 赞同 by mac-text"},
                    {"id": "t5", "status": "blocked", "created_at": "2026-10-27T14:00:00",
                     "title": "[alarm] root offline"},
                ],
            }

            def fake_get(path):
                calls.append(path)
                status = path.split("status=", 1)[1].split("&", 1)[0]
                return {"tasks": by_status[status]}

            with mock.patch.object(thinker, "_mailbox_get", fake_get):
                activity = thinker._fetch_group_activity()

        self.assertEqual(calls, [
            f"/tasks?status={status}&limit={proposer.ACTIVITY_FEED_LIMIT}"
            for status in proposer.ACTIVITY_STATUSES])
        self.assertNotIn("/tasks?status=all", calls)
        self.assertEqual([row["title"] for row in activity],
                         ["attempt that failed", "[alarm] root offline"])

    def test_proposer_never_publishes_an_unreachable_mailbox(self):
        """Transport failure is held, recorded, and survived - not raised."""
        def publish(proposal, identity, url, token):
            raise OSError("mailbox down")

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", published=publish)

            round_ = thinker.think(1)
            self.assertFalse(round_.published)
            self.assertIn("publish failed", round_.error)
            self.assertIsNotNone(round_.proposal)  # thought happened, delivery did not


    def test_a_destination_is_kept_on_the_round_and_in_the_spool(self):
        from pando.genesis import Proposal
        from pando.proposer import Round, Spool, destination_hint, proposal_from_spool
        bare = Round(1, 0.0)
        self.assertNotIn("destination", bare.as_dict())
        bare.proposal = Proposal("p", "mac", "carry", "seed.replicate", destination="win")
        self.assertEqual(bare.as_dict()["destination"], "win")
        self.assertEqual(
            destination_hint(["mac", "win"]),
            "seed.replicate 的 destination 只能是这些名字之一，不知道就不要这个字段：mac, win")
        with tempfile.TemporaryDirectory() as tmp:
            spool = Spool(Path(tmp) / "held.jsonl")
            spool.add(bare.proposal, 4)
            restored = proposal_from_spool(spool.pending()[0])
            self.assertEqual(restored.destination, "win")
            self.assertEqual(restored.action, "seed.replicate")

    def test_a_replicate_without_a_destination_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            thinker, proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "carry"},
                 {"action": "seed.replicate", "goal": "carry", "destination": "win"}],
                Path(tmp) / "think.jsonl", host_names=["mac", "win"])
            round_ = thinker.think(1)
            thinker.ledger.append(round_)
            self.assertTrue(round_.published)
            self.assertEqual(sent[-1].destination, "win")
            self.assertEqual(proposer.calls, 2)
            self.assertIn("JSON 里的一个字段", proposer.users[-1])
            self.assertIn("只把名字写进 goal 不算", proposer.users[-1])

            thinker, proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "carry"},
                 {"action": "seed.replicate", "goal": "still nowhere"}],
                Path(tmp) / "think2.jsonl", host_names=["mac", "win"])
            round_ = thinker.think(1)
            self.assertFalse(round_.published)
            self.assertIn("needs a destination", round_.error)
            self.assertEqual(round_.proposal.action, "seed.replicate")
            self.assertEqual(round_.as_dict()["action"], "seed.replicate")
            self.assertEqual(sent, [])

            thinker, _proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "carry"}],
                Path(tmp) / "think3.jsonl")
            round_ = thinker.think(1)
            self.assertTrue(round_.published)
            self.assertEqual(sent[-1].destination, "")

    def test_one_host_named_only_in_the_goal_is_labeled_as_such(self):
        from pando.proposer import named_host
        self.assertEqual(named_host("在 win 主机上", ["mac", "win"]), "win")
        self.assertEqual(named_host("在 mac 和 win 上", ["mac", "win"]), "")
        self.assertEqual(named_host("window", ["win"]), "")
        from pando.proposer import Round, replicate_streak
        from pando.genesis import Proposal
        self.assertFalse(replicate_streak([{"action": "seed.replicate"}]))
        self.assertFalse(replicate_streak([
            {"action": "seed.replicate"}, {"action": "seed.replicate"}]))
        mixed = [{"action": "culture.refine"}] + [{"action": "seed.replicate"}] * 4
        self.assertTrue(replicate_streak(mixed))
        with tempfile.TemporaryDirectory() as tmp:
            thinker, _proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "在 win 主机上部署"},
                 {"action": "seed.replicate", "goal": "仍是在 win 主机上"}],
                Path(tmp) / "think.jsonl", host_names=["mac", "win"])
            round_ = thinker.think(1)
            self.assertTrue(round_.published)
            self.assertEqual(sent[-1].destination, "win")
            self.assertEqual(round_.destination_source, "sentence")
            self.assertEqual(round_.as_dict()["destination_source"], "sentence")

            # A repeat the model would not change does not inherit a host
            # from the sentence.
            from pando.proposer import Round
            from pando.genesis import Proposal
            prior = Round(1, 0.0)
            prior.proposal = Proposal("p", "mac", "old", "seed.replicate", destination="mac")
            prior.published = True
            prior.destination_source = "sentence"
            thinker.ledger.append(prior)
            thinker, _proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "在 win 主机上"},
                 {"action": "seed.replicate", "goal": "还是在 win 主机上"},
                 {"action": "seed.replicate", "goal": "仍然在 win 主机上"}],
                Path(tmp) / "repeat.jsonl", host_names=["mac", "win"])
            thinker.ledger.append(prior)
            held = thinker.think(2)
            self.assertTrue(held.novelty_forced)
            self.assertFalse(held.published)
            self.assertEqual(sent, [])

            thinker, _proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "在 mac 和 win 上"},
                 {"action": "seed.replicate", "goal": "还是 mac 和 win"}],
                Path(tmp) / "think2.jsonl", host_names=["mac", "win"])
            round_ = thinker.think(1)
            self.assertFalse(round_.published)
            self.assertEqual(sent, [])

            thinker.ledger.append(round_)
            thinker, _proposer, sent = self._thinker(
                [{"action": "seed.replicate", "goal": "还是 mac 和 win"},
                 {"action": "seed.replicate", "goal": "继续 mac 和 win"},
                 {"action": "seed.replicate", "goal": "仍然 mac 和 win"}],
                Path(tmp) / "think2.jsonl", host_names=["mac", "win"])
            # The ledger above belongs to the previous thinker. Repeat the
            # held row on this one so the second round can see it.
            thinker.ledger.append(round_)
            again = thinker.think(2)
            self.assertTrue(again.published)
            self.assertEqual(sent[-1].destination, "")
            self.assertEqual(again.destination_source, "missing")



    def test_a_replicate_streak_is_named_before_the_model_answers(self):
        from pando.genesis import Proposal
        from pando.proposer import Round
        with tempfile.TemporaryDirectory() as tmp:
            thinker, _, _ = self._thinker(
                [{"action": "culture.refine", "goal": "x"}],
                Path(tmp) / "think.jsonl")
            for i in range(1, 5):
                prior = Round(i, 0.0)
                prior.proposal = Proposal(f"p{i}", "mac", "g", "seed.replicate")
                prior.published = True
                thinker.ledger.append(prior)
            prompt = thinker._user_prompt(5, [])
            self.assertIn("这一轮不要再选它", prompt)
            self.assertIn("已经太多", prompt)

    def test_a_destination_retry_cannot_walk_back_onto_the_action_just_left(self):
        from pando.genesis import Proposal
        from pando.proposer import Round
        with tempfile.TemporaryDirectory() as tmp:
            thinker, _proposer, sent = self._thinker(
                [{"action": "culture.refine", "goal": "again",
                  "params": {"playbook": {"meta": {"name": "a", "version": 1}}}},
                 {"action": "seed.replicate", "goal": "carry"},
                 {"action": "culture.refine", "goal": "back again",
                  "params": {"playbook": {"meta": {"name": "b", "version": 1}}}}],
                Path(tmp) / "think.jsonl", host_names=["mac", "win"])
            prior = Round(1, 0.0)
            prior.proposal = Proposal("p0", "mac", "old", "culture.refine")
            prior.published = True
            thinker.ledger.append(prior)
            round_ = thinker.think(2)
            self.assertTrue(round_.retried)
            self.assertFalse(round_.published)
            self.assertEqual(round_.proposal.action, "seed.replicate")
            self.assertEqual(sent, [])


class TestThinkWatchdog(unittest.TestCase):
    """tools/think_watchdog.py - the difference between "running" and "thinking".

    KeepAlive answers "did it exit?". It does not answer "did it stop
    producing thoughts?" A proposer wedged on an unread socket keeps running
    with a frozen ledger, which is the same silent failure that cost this
    colony 7h47m. The contract: silence past the threshold triggers a kick,
    and a kick that has not helped yet must not become a restart loop.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "think_watchdog.py"
        spec = importlib.util.spec_from_file_location("think_watchdog", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_progress_not_liveness_is_the_signal(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            mtime = ledger.stat().st_mtime

            # A live process is implied: the ledger moved recently.
            verdict, _ = wd.decide(mtime + 10, ledger, stall_seconds=900)
            self.assertEqual(verdict, "ok")
            # Same process, same liveness - but it has stopped thinking.
            verdict, why = wd.decide(mtime + 901, ledger, stall_seconds=900)
            self.assertEqual(verdict, "stall")
            self.assertIn("ledger age", why)

    def test_a_kick_that_did_not_help_does_not_become_a_restart_loop(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            mtime = ledger.stat().st_mtime
            cooldown = Path(tmp) / "cooldown"

            verdict, _ = wd.decide(mtime + 901, ledger, 900, cooldown, 300)
            self.assertEqual(verdict, "stall")
            cooldown.write_text(f"{mtime + 901}\n", encoding="utf-8")
            # 100s later: we already kicked, it has not recovered. Kicking again
            # would just paper over a real stall with restart noise.
            verdict, why = wd.decide(mtime + 1001, ledger, 900, cooldown, 300)
            self.assertEqual(verdict, "cooldown")
            self.assertIn("kicked", why)  # names the earlier kick, not a new one
            # 400s later the cooldown has expired and a real kick is due again.
            verdict, _ = wd.decide(mtime + 1301, ledger, 900, cooldown, 300)
            self.assertEqual(verdict, "stall")

    def test_a_missing_ledger_is_a_stall_not_a_crash(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            verdict, why = wd.decide(
                time.time(), Path(tmp) / "never-written.jsonl", 900)
            self.assertEqual(verdict, "stall")
            self.assertIn("no ledger", why)


class TestThinkWatchdogVoteDelivery(unittest.TestCase):
    """think_watchdog's vote leg - the caller that turns a wedge into work.

    tools/vote_watchdog.py measured the win-laya-int8 wedge correctly for a
    full day and nobody acted on it, because a signal with no *caller* is a
    docstring. This is the caller. The contract it must keep:

      * exit 2 (a member started falling behind) becomes exactly one
        claimable ``[alarm]`` task in the mailbox - not a log line;
      * a feed that cannot be read creates **nothing**. Inventing an alarm
        from our own parse error is how a pager starts crying wolf;
      * no failure path may make the job exit non-zero. Its exit code is a
        health signal of its own, so a noisy red job would trade one silent
        fault for another;
      * with no ``--vote-cmd`` it behaves exactly as before.
    """

    WEDGE = {
        "now": 1_800_000_000.0,
        "members": {
            "win-laya-int8": {
                "status": "wedge", "slope": 0.54, "slope_max": 0.2,
                "samples": 43, "span_s": 5280.0, "window_s": 5400.0,
                "first_lag_s": 126000.0, "last_lag_s": 128880.0},
            "mac-text": {
                "status": "ok", "slope": -15.22, "samples": 129,
                "span_s": 5340.0, "window_s": 5400.0,
                "first_lag_s": 81720.0, "last_lag_s": 720.0},
        },
        "unnamed_silent": [],
    }

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "think_watchdog.py"
        spec = importlib.util.spec_from_file_location("think_watchdog", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _vote_cmd(self, tmp, payload, rc):
        """A stand-in for vote_watchdog: print one JSON line, exit with rc."""
        script = Path(tmp) / f"vote_{rc}.py"
        script.write_text(
            "import json, sys\n"
            f"sys.stdout.write(json.dumps({payload!r}))\n"
            f"sys.exit({rc})\n",
            encoding="utf-8")
        return f"{sys.executable} {script}"

    def _alarm_stub(self, tmp, token="tok"):
        """A real mailbox on a real port - the page has to actually land."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_mailbox_think_watchdog_test",
            Path(__file__).resolve().parents[1] / "tools" / "agent_mailbox.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        box = module.Mailbox(Path(tmp) / "box")
        server = module.ThreadingHTTPServer(
            ("127.0.0.1", 0), module.build_handler(box, token))
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        token_file = Path(tmp) / "alarm.token"
        token_file.write_text(token, encoding="utf-8")
        return f"http://127.0.0.1:{port}", token_file, box, server, thread

    def _run(self, wd, tmp, vote_cmd, alarm_url="", token_file="", extra=()):
        ledger = Path(tmp) / "think.jsonl"
        ledger.write_text("{}\n", encoding="utf-8")
        return wd.main([
            "--ledger", str(ledger),
            # Far past any real ledger age: the proposer leg must stay quiet so
            # anything we see in the box was the vote leg's doing.
            "--stall-seconds", "999999999",
            "--restart-cmd", "true",
            "--vote-cmd", vote_cmd,
            "--alarm-url", alarm_url,
            "--alarm-token-file", str(token_file) if token_file else "",
            *extra,
        ])

    def test_a_wedge_becomes_exactly_one_claimable_task(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                rc = self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2),
                               url, token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertTrue(task["title"].startswith("[alarm]"),
                            f"a wedged member is work, not a message: {task['title']!r}")
            # [chat] and [proposal] mean "message, never work" to the claim
            # automation, so an alarm must not borrow either prefix.
            self.assertNotIn("[chat]", task["title"])
            self.assertNotIn("[proposal]", task["title"])
            self.assertIn("win-laya-int8", task["title"])
            self.assertEqual(task["status"], "open")

    def test_an_unreadable_feed_pages_nobody(self):
        """rc=1 is *our* feed failing, not a member. Two different faults."""
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                # rc=1 and no JSON at all: the feed could not be read.
                script = Path(tmp) / "vote_broken.py"
                script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
                rc = self._run(wd, tmp, f"{sys.executable} {script}",
                               url, token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(tasks, [], "an unreadable feed must not page anyone")

    def test_a_clean_feed_creates_no_task(self):
        wd = self._tool()
        healthy = {"now": 1_800_000_000.0,
                   "members": {"mac-text": {"status": "ok", "slope": -15.2,
                                            "samples": 129}},
                   "unnamed_silent": []}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                rc = self._run(wd, tmp, self._vote_cmd(tmp, healthy, 0),
                               url, token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(tasks, [])

    def test_a_recovered_member_retires_the_open_alarm_task(self):
        """Clearing the state file must also close the mailbox task.

        On 2026-10-20 win-laya-int8 wedged, the watchdog paged a claimable
        ``[alarm]`` task, win-laya recovered, and the watchdog cleared its own
        state - but the task stayed open forever. An open list holding one
        stale alarm stops meaning "something is wrong", so recovery has to
        mark the task done, not just forget it.
        """
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state_file = Path(tmp) / "alarm-state.json"
            try:
                # First poll: win-laya wedges -> exactly one open alarm task.
                rc1 = self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2),
                                url, token_file,
                                extra=("--alarm-state-file", str(state_file)))
                tasks = box.list(None)
                # Second poll: bounded again -> the same task is retired.
                healthy = {"now": 1_800_000_000.0,
                           "members": {"win-laya-int8": {"status": "ok",
                                                         "slope": -0.01,
                                                         "samples": 43}},
                           "unnamed_silent": []}
                rc2 = self._run(wd, tmp, self._vote_cmd(tmp, healthy, 0),
                                url, token_file,
                                extra=("--alarm-state-file", str(state_file)))
                tasks_after = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["status"], "open")
            self.assertEqual(len(tasks_after), 1,
                             "recovery retires the task, it does not mint a new one")
            self.assertEqual(tasks_after[0]["status"], "done")
            self.assertIn("retired", tasks_after[0].get("output", ""))

    def test_a_suppressed_alarm_is_named_not_called_ok(self):
        """rc=0 with --page-new-only means "nothing new", not "all healthy".

        The live win-laya-int8 wedge is exactly this shape: already paged, so
        the poll exits 0. A bare "ok, 9 members measured" over it would be a
        green light on a dead node - the failure this whole file exists to
        prevent - so the log has to name what it is suppressing.
        """
        wd = self._tool()
        payload = {"members": {
            "win-laya-int8": {"status": "wedge", "slope": 0.54, "seen": "ongoing",
                              "slope_max": 0.2, "samples": 43, "span_s": 5280.0,
                              "window_s": 5400.0,
                              "first_lag_s": 126000.0, "last_lag_s": 128880.0},
            "mac-text": {"status": "ok", "slope": -15.2, "samples": 129}}}
        alarms = wd.vote_alarms(payload)
        ongoing = [row["member"] for row in alarms if row.get("seen") == "ongoing"]
        self.assertEqual(ongoing, ["win-laya-int8"])

    def test_a_dead_pager_still_exits_zero(self):
        """The job's exit code is a health signal; a delivery bug must not make
        it red, or the red *becomes* the story and the wedge is forgotten."""
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            # A port nothing is listening on: post_alarm must fail *inside*.
            dead = "http://127.0.0.1:1"
            rc = self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2), dead, "")
            self.assertEqual(rc, 0)

    def test_no_vote_cmd_keeps_the_original_behaviour(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                rc = self._run(wd, tmp, "", url, token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
            self.assertEqual(rc, 0)
            self.assertEqual(tasks, [])

    def test_the_body_carries_the_numbers_and_the_falsifier(self):
        """A reader has to be able to disagree with the alarm from the task."""
        wd = self._tool()
        alarms = wd.vote_alarms(self.WEDGE)
        self.assertEqual([row["member"] for row in alarms], ["win-laya-int8"])
        body = wd.alarm_body(alarms, json.dumps(self.WEDGE), 1_800_000_000.0)
        self.assertIn("Falsifier", body)
        self.assertIn("win-laya-int8", body)
        self.assertIn("+0.54", body)        # the slope that tripped it
        self.assertIn("43", body)           # the sample size behind it
        self.assertIn("35.0h", body)        # first lag, in hours
        self.assertIn("35.8h", body)        # last lag, in hours

    def test_silent_outranks_nothing_but_wedge_sorts_first(self):
        wd = self._tool()
        payload = {"members": {
            "quiet": {"status": "silent", "why": "named but never voted"},
            "slow": {"status": "wedge", "slope": 0.3},
            "slower": {"status": "wedge", "slope": 0.9}}}
        got = [row["member"] for row in wd.vote_alarms(payload)]
        self.assertEqual(got, ["slower", "slow", "quiet"])

    def test_a_garbage_payload_yields_no_alarms(self):
        wd = self._tool()
        self.assertEqual(wd.vote_alarms(None), [])
        self.assertEqual(wd.vote_alarms({"members": "nope"}), [])
        self.assertEqual(wd.vote_alarms({"members": {"m": {"status": "ok"}}}), [])

    # --- the alarm task has to stay true while the fault lasts ----------
    #
    # win-laya-int8 sat wedged from 2026-10-18 ~23:00 while its alarm task kept
    # reading as if it had just been filed, and the body kept instructing a
    # restart that nobody with access to the host ever performed. Both halves
    # of that are pager defects: a body with no age hides how long the fault
    # has run, and an instruction with no stated precondition asks for the
    # impossible. These four tests pin the repair.

    def _ongoing(self):
        """The same wedge, as a later poll sees it: rc=0, already paged."""
        payload = json.loads(json.dumps(self.WEDGE))
        payload["members"]["win-laya-int8"]["seen"] = "ongoing"
        return payload

    def test_an_open_alarm_is_rechecked_in_place_not_duplicated(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            try:
                first = self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2),
                                  url, token_file,
                                  extra=("--alarm-state-file", str(state)))
                after_first = box.list(None)
                # A later poll: same member, still wedged, already paged.
                second = self._run(wd, tmp, self._vote_cmd(tmp, self._ongoing(), 0),
                                   url, token_file,
                                   extra=("--alarm-state-file", str(state),
                                          "--alarm-refresh-seconds", "0"))
                after_second = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual((first, second), (0, 0))
            self.assertEqual(len(after_first), 1)
            self.assertEqual(len(after_second), 1,
                             "a still-open alarm must not open a second task")
            out = after_second[0].get("output") or ""
            self.assertIn("RE-CHECK", out,
                          "the open task must carry a dated re-check, not just "
                          "the numbers from the day it was filed")
            self.assertIn("still WEDGE", out)

    def test_a_recheck_inside_the_refresh_window_is_skipped(self):
        """Dedupe is the point: a job every 300s must not flood one task."""
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            try:
                self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                before = (box.list(None)[0].get("output") or "")
                self._run(wd, tmp, self._vote_cmd(tmp, self._ongoing(), 0), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                after = (box.list(None)[0].get("output") or "")
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
            self.assertEqual(before, after,
                             "a second poll minutes later must not append again")

    def test_an_overdue_alarm_says_nothing_automatic_will_clear_it(self):
        """Past the deadline the body must stop implying help is coming.

        There is no escalation target configured here, and the body has to say
        so rather than let a reader assume something ran.
        """
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            try:
                self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                self._run(wd, tmp, self._vote_cmd(tmp, self._ongoing(), 0), url,
                          token_file,
                          extra=("--alarm-state-file", str(state),
                                 "--alarm-refresh-seconds", "0",
                                 "--alarm-overdue-seconds", "0"))
                out = box.list(None)[0].get("output") or ""
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
            self.assertIn("Escalation", out)
            self.assertIn("human hand is needed", out)
            self.assertIn("no escalation target configured", out)

    def test_a_recovered_member_is_forgotten_and_can_page_again(self):
        """Clearing the state on recovery is what keeps the second wedge loud."""
        wd = self._tool()
        clean = {"members": {"win-laya-int8": {"status": "ok", "slope": -0.01,
                                               "samples": 40}}}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            try:
                self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                self.assertTrue(state.exists())
                self._run(wd, tmp, self._vote_cmd(tmp, clean, 0), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                self.assertEqual(json.loads(state.read_text()), {},
                                 "a recovered member must be forgotten")
                self._run(wd, tmp, self._vote_cmd(tmp, self.WEDGE, 2), url,
                          token_file, extra=("--alarm-state-file", str(state)))
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
            self.assertEqual(len(tasks), 2,
                             "a second wedge after a recovery is a new alarm")

    def test_the_body_states_who_can_actually_do_the_restart(self):
        """The instruction has a precondition; saying it is the whole fix."""
        wd = self._tool()
        body = wd.alarm_body(wd.vote_alarms(self.WEDGE), "{}", 1_800_000_000.0)
        self.assertIn("ON THE HOST THAT", body)
        self.assertIn("may have neither", body)
        self.assertIn("Waiting does not clear it", body)

    def test_the_recheck_names_the_age_and_the_issuer_note(self):
        wd = self._tool()
        alarms = wd.vote_alarms(self.WEDGE)
        block = wd.recheck_block(alarms, 1_800_000_000.0, 7200.0, False,
                                 note="mac cannot reach win")
        self.assertIn("2.0h", block)          # the age, not a bare "still open"
        self.assertIn("mac cannot reach win", block)
        self.assertNotIn("Escalation", block)  # not overdue yet: say nothing

    def test_an_overdue_body_says_no_escalation_target_is_configured(self):
        wd = self._tool()
        self.assertIn("no escalation target configured", wd.escalate("", True))
        self.assertIn("not due yet", wd.escalate("true", False))
        self.assertIn("rc=0", wd.escalate("exit 0", True))


class TestThinkWatchdogCodeDelivery(unittest.TestCase):
    """think_watchdog's code leg - the caller that pages a frozen checkout.

    tools/code_drift_check.py answers "is the code that produced the thought
    the code we think is running?". On 2026-10-19 win was frozen ~51 commits
    behind and its decision member answered ~39-hour-old proposals; every panel
    stayed green because nothing read ``capability.code_head``. A measurement
    with no *caller* is a docstring, so this is the caller, hung on the timer
    that already runs. The contract:

      * a stale member becomes exactly one claimable ``[alarm] code-drift``
        task - not a log line;
      * a transient unreadable root creates nothing, while a root that stays
        unreadable past the grace window becomes one ``[alarm] code-coverage``
        task - never a fabricated member-drift accusation;
      * the vote and code legs share one ``--alarm-state-file`` (the age and
        re-check logic lives there), so neither leg's clean poll may erase the
        other leg's open alarm.
    """

    DRIFT = {
        "now": 1_800_000_000.0,
        "min_persist_s": 900.0,
        "members": {
            "win-gemma-12b": {
                "member": "win-gemma-12b", "root": "win",
                "reported": "d" * 40, "head": "dddddddddddd",
                "behind": 51, "raw": "behind", "status": "stale",
                "why": "51 commit(s) behind the tip",
                "behind_for_s": 152000.0},
            "mac-text": {"member": "mac-text", "reported": "a" * 40,
                         "behind": 0, "raw": "current", "status": "ok",
                         "why": "on the tip"},
        },
        "unreadable": [],
        "stale": ["win-gemma-12b"],
    }
    WEDGE = TestThinkWatchdogVoteDelivery.WEDGE

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "think_watchdog.py"
        spec = importlib.util.spec_from_file_location("think_watchdog", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _probe_cmd(self, tmp, payload, rc, tag):
        """A stand-in for code_drift_check: one JSON line, exit with rc."""
        script = Path(tmp) / f"probe_{tag}_{rc}.py"
        script.write_text(
            "import json, sys\n"
            f"sys.stdout.write(json.dumps({payload!r}))\n"
            f"sys.exit({rc})\n",
            encoding="utf-8")
        return f"{sys.executable} {script}"

    def _alarm_stub(self, tmp, token="tok"):
        """A real mailbox on a real port - the page has to actually land."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_mailbox_think_watchdog_code_test",
            Path(__file__).resolve().parents[1] / "tools" / "agent_mailbox.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        box = module.Mailbox(Path(tmp) / "box")
        server = module.ThreadingHTTPServer(
            ("127.0.0.1", 0), module.build_handler(box, token))
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        token_file = Path(tmp) / "alarm.token"
        token_file.write_text(token, encoding="utf-8")
        return f"http://127.0.0.1:{port}", token_file, box, server, thread

    def _run(self, wd, tmp, code_cmd, vote_cmd="", alarm_url="", token_file="",
             extra=()):
        ledger = Path(tmp) / "think.jsonl"
        ledger.write_text("{}\n", encoding="utf-8")
        return wd.main([
            "--ledger", str(ledger),
            # Far past any real ledger age: the proposer leg must stay quiet so
            # anything we see in the box was a vote/code leg's doing.
            "--stall-seconds", "999999999",
            "--restart-cmd", "true",
            "--vote-cmd", vote_cmd,
            "--code-cmd", code_cmd,
            "--alarm-url", alarm_url,
            "--alarm-token-file", str(token_file) if token_file else "",
            *extra,
        ])

    def test_only_the_stale_rows_are_named_and_worst_first(self):
        wd = self._tool()
        payload = {"members": {
            "b": {"status": "stale", "behind": 3, "behind_for_s": 9.0},
            "a": {"status": "stale", "behind": 51, "behind_for_s": 1.0},
            "c": {"status": "lag", "behind": 2, "behind_for_s": 1.0},
            "d": {"status": "ok", "behind": 0},
        }}
        alarms = wd.code_alarms(payload)
        self.assertEqual([row["member"] for row in alarms], ["a", "b"])
        self.assertTrue(all(row["kind"] == "code-drift" for row in alarms))
        self.assertEqual(wd.alarm_title(alarms), "[alarm] code-drift: a, b")

    def test_a_stale_member_becomes_exactly_one_claimable_task(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                rc = self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "d"),
                               alarm_url=url, token_file=token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertTrue(task["title"].startswith("[alarm] code-drift:"),
                            f"a frozen checkout is work: {task['title']!r}")
            self.assertNotIn("[chat]", task["title"])
            self.assertNotIn("[proposal]", task["title"])
            self.assertIn("win-gemma-12b", task["title"])
            self.assertEqual(task["status"], "open")

    def test_an_unreadable_root_pages_nobody(self):
        """rc=1 with no payload is a whole feed we could not read."""
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                script = Path(tmp) / "drift_broken.py"
                script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = self._run(wd, tmp, f"{sys.executable} {script}",
                                   alarm_url=url, token_file=token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(tasks, [], "an unreadable root must not page anyone")
            self.assertIn("UNREADABLE", out.getvalue())
            self.assertNotIn("PARTIAL COVERAGE", out.getvalue())

    def test_a_clean_partial_report_names_the_blind_root_without_paging(self):
        """One unreadable root must not turn measured members into a fault.

        This is the 2026-10-19 Pi5-mirror case: six reachable members were
        clean, but the report said ``UNREADABLE`` because the whole feed was
        misread. A partial report is a coverage fact, not a member alarm.
        """
        wd = self._tool()
        partial = {
            "members": {"mac-text": {"status": "ok", "behind": 0}},
            "unreadable": [{"root": "pi5", "error": "mirror stale"}],
            "stale": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = self._run(
                        wd, tmp,
                        self._probe_cmd(tmp, partial, 1, "partial_clean"),
                        alarm_url=url, token_file=token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(tasks, [])
            text = out.getvalue()
            self.assertIn("PARTIAL COVERAGE (not paged)", text)
            self.assertIn("1 members measured, 1 root(s) unreadable: pi5", text)
            self.assertNotIn("UNREADABLE", text)

    def test_a_persistent_blind_root_becomes_a_coverage_task(self):
        """Blindness is actionable after three polls, without blaming members."""
        wd = self._tool()
        partial = {
            "members": {"mac-text": {"status": "ok", "behind": 0}},
            "unreadable": [
                {"root": "win", "error": "urlopen timed out"},
                {"root": "pi5", "error": "mirror stale"},
            ],
            "stale": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            cmd = self._probe_cmd(tmp, partial, 1, "partial_persistent")
            extra = ("--alarm-state-file", str(state),
                     "--code-coverage-grace-seconds", "900")
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self._run(wd, tmp, cmd, alarm_url=url,
                              token_file=token_file,
                              extra=extra + ("--now", "1000"))
                self.assertEqual(box.list(None), [],
                                 "one blind poll is not yet coverage proof")
                self.assertIn("tracking, not paged", out.getvalue())

                self._run(wd, tmp, cmd, alarm_url=url,
                          token_file=token_file,
                          extra=extra + ("--now", "1900"))
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertTrue(task["title"].startswith(
                "[alarm] code-coverage:"), task["title"])
            self.assertIn("win", task["title"])
            self.assertIn("pi5", task["title"])
            self.assertNotIn("code-drift:", task["title"])
            self.assertIn("does NOT mean", task["body"])
            self.assertIn("not accused of running old", task["body"])

    def test_measured_coverage_closes_the_coverage_task(self):
        """A fresh measurement is the only accepted recovery evidence."""
        wd = self._tool()
        partial = {
            "members": {"mac-text": {"status": "ok", "behind": 0}},
            "unreadable": [{"root": "win", "error": "offline"}],
            "stale": [],
        }
        clean = {"members": {
            "mac-text": {"status": "ok", "behind": 0},
            "win-gemma-12b": {"status": "ok", "behind": 0},
        }, "unreadable": [], "stale": []}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state),
                     "--code-coverage-grace-seconds", "0")
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, partial, 1, "blind"),
                          alarm_url=url, token_file=token_file,
                          extra=extra + ("--now", "1000"))
                self._run(wd, tmp, self._probe_cmd(tmp, clean, 0, "covered"),
                          alarm_url=url, token_file=token_file,
                          extra=extra + ("--now", "1001"))
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["status"], "done")
            self.assertNotIn(wd.COVERAGE_TRACKING_KEY,
                             json.loads(state.read_text()))

    def test_each_blind_root_waits_for_its_own_grace_window(self):
        """A long outage cannot drag a newly blind root into its alarm."""
        wd = self._tool()
        win_blind = {
            "members": {"mac-text": {"status": "ok", "behind": 0}},
            "unreadable": [{"root": "win", "error": "offline"}],
            "stale": [],
        }
        both_blind = {
            "members": {"mac-text": {"status": "ok", "behind": 0}},
            "unreadable": [
                {"root": "win", "error": "offline"},
                {"root": "pi5", "error": "mirror stale"},
            ],
            "stale": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state),
                     "--code-coverage-grace-seconds", "900")
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, win_blind, 1, "win"),
                          alarm_url=url, token_file=token_file,
                          extra=extra + ("--now", "1000"))
                self._run(wd, tmp, self._probe_cmd(tmp, both_blind, 1, "both"),
                          alarm_url=url, token_file=token_file,
                          extra=extra + ("--now", "1900"))
                tasks = box.list(None)
                tracking = json.loads(state.read_text())[
                    wd.COVERAGE_TRACKING_KEY]
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1)
            self.assertTrue(tasks[0]["title"].startswith(
                "[alarm] code-coverage: win"), tasks[0]["title"])
            self.assertNotIn("pi5", tasks[0]["title"])
            self.assertEqual(tracking["win"]["first_seen"], 1000.0)
            self.assertEqual(tracking["pi5"]["first_seen"], 1900.0)

    def test_a_partial_report_still_pages_a_stale_member(self):
        """An unreadable root does not hide a stale row we did measure."""
        wd = self._tool()
        partial = dict(self.DRIFT)
        partial["unreadable"] = [{"root": "pi5", "error": "mirror stale"}]
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            try:
                rc = self._run(
                    wd, tmp,
                    self._probe_cmd(tmp, partial, 1, "partial_stale"),
                    alarm_url=url, token_file=token_file)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(rc, 0)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertTrue(task["title"].startswith("[alarm] code-drift:"))
            self.assertIn("win-gemma-12b", task["title"])

    def test_a_clean_code_poll_does_not_forget_an_open_vote_alarm(self):
        """One shared state file, two families: a clean poll clears only its own.

        The code leg runs *after* the vote leg and, on a clean feed, drops its
        family's entries. If that drop were family-agnostic it would erase the
        vote alarm a moment after the vote leg filed it, and the next poll
        would file the same wedge a second time.
        """
        wd = self._tool()
        clean_code = {"members": {"mac-text": {"status": "ok", "behind": 0}}}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state))
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, clean_code, 0, "ok"),
                          vote_cmd=self._probe_cmd(tmp, self.WEDGE, 2, "w"),
                          alarm_url=url, token_file=token_file, extra=extra)
                self._run(wd, tmp, self._probe_cmd(tmp, clean_code, 0, "ok"),
                          vote_cmd=self._probe_cmd(tmp, self.WEDGE, 2, "w"),
                          alarm_url=url, token_file=token_file, extra=extra)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1,
                             "the second poll must re-check, not re-file")

    def test_a_clean_vote_poll_does_not_forget_an_open_drift_alarm(self):
        """The mirror case: the vote leg must not erase the code leg's alarm."""
        wd = self._tool()
        clean_vote = {"members": {"mac-text": {"status": "ok", "slope": -0.01,
                                               "samples": 40}}}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state))
            try:
                for _ in range(2):
                    self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "d"),
                              vote_cmd=self._probe_cmd(tmp, clean_vote, 0, "v"),
                              alarm_url=url, token_file=token_file, extra=extra)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1,
                             "the second poll must re-check, not re-file")

    def test_both_families_can_be_open_at_once(self):
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "d"),
                          vote_cmd=self._probe_cmd(tmp, self.WEDGE, 2, "w"),
                          alarm_url=url, token_file=token_file,
                          extra=("--alarm-state-file", str(state)))
                tasks = box.list(None)
                titles = sorted(task["title"] for task in tasks)
                saved = json.loads(state.read_text())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 2, titles)
            self.assertTrue(titles[0].startswith("[alarm] code-drift:"), titles)
            self.assertTrue(titles[1].startswith("[alarm] vote-wedge:"), titles)
            self.assertEqual(len(saved), 2)

    def test_a_recovered_drift_is_forgotten_and_can_page_again(self):
        wd = self._tool()
        clean = {"members": {"win-gemma-12b": {"status": "ok", "behind": 0}}}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state))
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "d"),
                          alarm_url=url, token_file=token_file, extra=extra)
                self._run(wd, tmp, self._probe_cmd(tmp, clean, 0, "c"),
                          alarm_url=url, token_file=token_file, extra=extra)
                self.assertEqual(json.loads(state.read_text()), {},
                                 "a caught-up member must be forgotten")
                self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "d"),
                          alarm_url=url, token_file=token_file, extra=extra)
                tasks = box.list(None)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 2,
                             "a drift after a catch-up is a new alarm")

    def test_a_churning_member_set_refreshes_one_task_not_two(self):
        """Identity is the family, never the title.

        On 2026-10-19 a two-minute LAN blip made the first poll name two
        members and the very next poll name five. Keyed on the title those are
        two different alarms: a second task appears, and the first one - whose
        body now *understates* the fault and can never be rewritten, because no
        later poll will ever produce that title again - stays open forever
        reading like a final word.
        """
        wd = self._tool()
        first = {"members": {"mac-text": {"status": "stale", "behind": 51,
                                          "behind_for_s": 91200.0}}}
        second = {"members": {
            "mac-text": {"status": "stale", "behind": 51,
                         "behind_for_s": 95000.0},
            "mac-llm-8b-local": {"status": "stale", "behind": 51,
                                 "behind_for_s": 95000.0},
            "win-gemma-12b": {"status": "stale", "behind": 51,
                              "behind_for_s": 95000.0},
            "win-laya-int8": {"status": "stale", "behind": 51,
                              "behind_for_s": 95000.0},
            "win-yi-9b": {"status": "stale", "behind": 51,
                          "behind_for_s": 95000.0},
        }}
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            extra = ("--alarm-state-file", str(state),
                     "--alarm-refresh-seconds", "0")
            try:
                self._run(wd, tmp, self._probe_cmd(tmp, first, 2, "g1"),
                          alarm_url=url, token_file=token_file, extra=extra)
                self._run(wd, tmp, self._probe_cmd(tmp, second, 2, "g2"),
                          alarm_url=url, token_file=token_file, extra=extra)
                tasks = box.list(None)
                saved = json.loads(state.read_text())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(len(tasks), 1, [t["title"] for t in tasks])
            task = tasks[0]
            self.assertIn("mac-text", task["title"])
            # The title is a label and cannot be rewritten; the body is the
            # part every re-check refreshes, so it is the part that must be
            # present-tense.
            self.assertIn("win-gemma-12b", task["output"])
            self.assertIn("win-yi-9b", task["output"])
            self.assertEqual(list(saved), [wd.CODE_FAMILY],
                             "one fault, one state key")

    def test_an_alarm_opened_under_the_old_title_key_is_adopted(self):
        """Changing the key must not file a second task for a live fault.

        The key *was* the title. Any alarm already open carries a title key, so
        the first poll after the change has to recognise it as the same fault
        and re-key it - otherwise this fix creates exactly the duplicate it was
        written to prevent.
        """
        wd = self._tool()
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            url, token_file, box, server, thread = self._alarm_stub(tmp)
            state = Path(tmp) / "alarms.json"
            legacy = wd.alarm_title(wd.code_alarms(self.DRIFT))
            self.assertNotEqual(legacy, wd.CODE_FAMILY, "the test needs a change")
            state.write_text(json.dumps({legacy: {
                "task_id": "t-legacy", "issued_at": now - 3600.0,
                "last_append_at": now, "appends": 0}}), encoding="utf-8")
            extra = ("--alarm-state-file", str(state),
                     "--alarm-refresh-seconds", "21600")
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    self._run(wd, tmp, self._probe_cmd(tmp, self.DRIFT, 2, "lg"),
                              alarm_url=url, token_file=token_file, extra=extra)
                tasks = box.list(None)
                saved = json.loads(state.read_text())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

            self.assertEqual(tasks, [], "adoption must not file a second task")
            self.assertEqual(list(saved), [wd.CODE_FAMILY])
            self.assertEqual(saved[wd.CODE_FAMILY]["task_id"], "t-legacy")
            self.assertIn("adopting", out.getvalue())

    def test_every_recheck_says_which_set_it_is_describing(self):
        """A title that lags must not be read as the present tense."""
        wd = self._tool()
        block = wd.build_recheck_block(wd.code_alarms(self.DRIFT),
                                       1_800_000_000.0, 3600.0, False)
        self.assertIn(wd.CURRENT_SET_NOTE, block)

    def test_the_installed_timer_actually_carries_the_code_leg(self):
        """The lesson of this file, applied to this file's own wiring.

        vote_watchdog.py measured the win-laya wedge correctly for a whole day
        and nobody acted, because nothing called it. A code_drift_check.py that
        no timer invokes would repeat that exactly: correct, and paging nobody.
        """
        root = Path(__file__).resolve().parents[1]
        plist = (root / "tools" / "mac-launchd"
                 / "com.jewelroam.pandosi-think-watchdog.plist").read_text(
                     encoding="utf-8")
        self.assertIn("--code-cmd", plist)
        self.assertIn("code_drift_check.py", plist)
        self.assertIn("--code-timeout", plist)
        # Every host in the fleet, or a frozen node is simply invisible.
        self.assertIn("127.0.0.1:8765", plist)
        # pi5 sits behind the LAN that this job's responsible process may not
        # open, so it arrives as the mirror tools/mirror_ledger.py refreshes
        # over ssh. A LAN address here would look like coverage and be none:
        # the leg would report pi5 unreadable on every single poll.
        self.assertNotIn("192.168.31.22", plist)
        pi5 = re.search(r"--root-json pi5 (\S+)", plist)
        self.assertIsNotNone(pi5, "pi5 must still be a measured root")
        self.assertTrue(pi5.group(1).endswith("pi5-root.json"),
                        f"pi5 is read at {pi5.group(1)}")
        # An XML comment may not contain a doubled hyphen, and a plist that
        # does not parse is a timer that silently never runs - both of which
        # this file has already been through once.
        self.assertIsInstance(plistlib.loads(plist.encode("utf-8")), dict)
        # ...but the *address* win is read at matters as much as mentioning it.
        # A launchd agent whose program is the Homebrew interpreter is refused
        # on 192.168.31.x - Errno 65 for its own socket, curl rc=7 for a curl it
        # spawns, because the refusal follows the responsible process into its
        # children. The tailnet is not "local network" and answers that same
        # interpreter. Pinning the LAN address here would re-blind the leg that
        # this file exists to keep honest.
        win = re.search(r"--root win (\S+)", plist)
        self.assertIsNotNone(win, "win must still be a measured root")
        self.assertTrue(win.group(1).startswith("http://100."),
                        f"win is read at {win.group(1)}, not over the tailnet")
        # One shared state file, two alarm families - not a second file.
        self.assertIn("--alarm-state-file", plist)
        self.assertIn("think-watchdog-alarms.json", plist)
        self.assertNotIn("code-drift-alarms.json", plist)
        self.assertTrue((root / "tools" / "code_drift_check.py").exists())

    def test_the_installed_voter_judges_sleepable_macs_only_while_awake(self):
        """Mac may sleep, so its voters are gated on wakefulness, not deleted.

        A laptop that sleeps looks exactly like a silent voter on a green
        panel. Deleting the Mac members (--ignore-member) fixed the false
        alarm by also deleting the real one: a Mac that is awake and wedged is
        invisible. The split is --decision (must be voting) plus
        --sleep-gated-member (but only while this host is awake), so the
        always-on deployments page as before and an awake Mac pages too.
        """
        root = Path(__file__).resolve().parents[1]
        plist = (root / "tools" / "mac-launchd"
                 / "com.jewelroam.pandosi-think-watchdog.plist").read_text(
                     encoding="utf-8")
        args = plistlib.loads(plist.encode("utf-8"))["ProgramArguments"]
        vote_index = args.index("--vote-cmd")
        cmd = args[vote_index + 1]
        self.assertIn("--decision mac-text", cmd)
        self.assertIn("--decision mac-laya-int8", cmd)
        self.assertIn("--sleep-gated-member mac-text", cmd)
        self.assertIn("--sleep-gated-member mac-laya-int8", cmd)
        self.assertNotIn("--ignore-member mac-text", cmd)
        self.assertNotIn("--ignore-member mac-laya-int8", cmd)
        self.assertIn("--decision pandosi-pi5-x", cmd)
        self.assertIn("--decision pandosi-pi5-laya", cmd)
        self.assertIn("--decision win-laya-int8", cmd)

    def test_the_body_names_the_action_and_the_falsifier(self):
        """The cheap bounded fix, and what would prove the alarm wrong."""
        wd = self._tool()
        body = wd.build_alarm_body(wd.code_alarms(self.DRIFT), "{}",
                                   1_800_000_000.0)
        self.assertIn("RUNNING PROCESS booted", body)
        self.assertIn("never a model server", body)
        self.assertIn("Falsifier", body)
        # The rebuttal that is *not* a rebuttal is pre-empted in the body.
        self.assertIn("up-to-date tree under an old process", body)
        block = wd.build_recheck_block(wd.code_alarms(self.DRIFT),
                                       1_800_000_000.0, 7200.0, False)
        self.assertIn("still STALE", block)
        self.assertIn("2.0h", block)


class TestThinkStats(unittest.TestCase):
    """tools/think_stats.py - say only what the ledger can support.

    The proposer runs on more than one device now, so "is it evolving or just
    pacing?" is asked constantly. Every log tail looks the same either way.
    The contract this tool must keep: report the sample size and the 1-sigma
    band on every share, and never imply that diversity is progress. High
    entropy means varied; a fair coin is maximally entropic.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "think_stats.py"
        spec = importlib.util.spec_from_file_location("think_stats", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_a_share_never_reports_a_nan_band(self):
        ts = self._tool()
        # 0/12 repeated is a real observation, not a missing one.
        share, se = ts.share_with_se(0, 12)
        self.assertEqual(share, 0.0)
        self.assertEqual(se, 0.0)
        # 6/12 carries ~14.4pp of 1-sigma error; that is the point of the tool.
        share, se = ts.share_with_se(6, 12)
        self.assertAlmostEqual(share, 0.5, places=6)
        self.assertGreater(se, 0.14)
        self.assertLess(se, 0.15)

    def test_entropy_is_variedness_not_progress(self):
        ts = self._tool()
        from collections import Counter
        # One action forever is zero entropy, whatever it is.
        self.assertEqual(ts.shannon_bits(Counter({"seed.export": 10})), 0.0)
        # A fair coin between two actions is 1 bit - maximal for two outcomes,
        # and a state the colony must not call "better" than one action repeated.
        self.assertAlmostEqual(
            ts.shannon_bits(Counter({"a": 5, "b": 5})), 1.0, places=6)

    def test_a_round_that_produced_nothing_is_not_a_round_that_thought(self):
        ts = self._tool()
        rows = [
            {"action": "seed.export", "started_at": 1.0, "published": True},
            {"action": "", "started_at": 2.0, "published": False,
             "error": "held: could not reach the mailbox"},
        ]
        s = ts.summarise(rows)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["distinct_actions"], 1)  # "" is not an action
        self.assertEqual(s["silent_rounds"], 1)
        self.assertEqual(s["errored"]["count"], 1)
        self.assertAlmostEqual(s["published"]["share"], 0.5, places=6)

    def test_a_corrupt_ledger_line_is_skipped_not_fatal(self):
        ts = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            ledger.write_text(
                '{"action": "seed.export", "started_at": 1.0}\n'
                'not json at all\n'
                '\n'
                '{"action": "recruit.invite", "started_at": 2.0}\n',
                encoding="utf-8")
            rows = ts.load_ledger(ledger)
            self.assertEqual([r["action"] for r in rows],
                             ["seed.export", "recruit.invite"])

    def test_an_empty_ledger_says_empty_instead_of_dividing_by_zero(self):
        ts = self._tool()
        s = ts.summarise([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["rounds_per_hour"], 0.0)
        self.assertIn("empty ledger", ts.format_report("x", s))


