"""PandoSI smoke tests.

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
from pando.rootd import (Coordinator, INITIAL_STEM_TIMEOUT,
                            Root, load_accuracy_registry)
from pando.stemd import Stem
from pando.onboard import build_plan
from pando import graft
from pando.grants import (audit_status, grant_status, require_grant,
                            save_grant)
from pando.service import build_service_plan, detect_platform


class TestProposerSpool(unittest.TestCase):
    """pando.thinkd.Spool - thought that survives a partition.

    The hole this closes was found by reading, not by a crash: the colony's
    mailbox lives on one host, and every proposer on every other device points
    at it. While that host is away those proposers keep *thinking* (the loop
    never stops) but the signed proposal was simply gone - the ledger said
    "held" and nothing was ever re-delivered. Thinking into the void is a
    quieter failure than stopping, and it is the one the survival rule cares
    about: a device that is alive but cannot be heard does not let the group
    grow.
    """

    def _thinker(self, answers, ledger_path, spool_path, published=None,
                 fallbacks=None):
        from pando.identity import Identity
        from pando.thinkd import Ledger, Spool, Thinker

        class ScriptedProposer:
            def __init__(self, script):
                self.script = list(script)
                self.calls = 0

            def complete(self, system, user):
                self.calls += 1
                item = self.script.pop(0) if self.script else {"action": ""}
                if isinstance(item, Exception):
                    raise item
                return json.dumps(item, ensure_ascii=False), {"completion_tokens": 7}

        sent = []

        def publish(proposal, identity, url, token):
            sent.append(proposal)
            return {"task": {"id": "t-x"}, "claimed": True}

        return Thinker(
            node_id="test-node", proposer=ScriptedProposer(answers),
            identity=Identity.generate(), ledger=Ledger(ledger_path),
            mailbox_url="http://invalid.example", interval=0, dry_run=False,
            fetch_history=lambda: [], fetch_activity=lambda: [],
            spool=Spool(spool_path), mailbox_fallbacks=fallbacks,
            publish=(publish if published is None else published)), sent

    def test_a_proposal_held_by_a_partition_is_delivered_when_it_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool_path = Path(tmp) / "held.jsonl"
            state = {"up": False}
            sent = []

            def publish(proposal, identity, url, token):
                if not state["up"]:
                    raise OSError("mailbox down")
                sent.append(proposal)
                return {"task": {"id": "t-x"}, "claimed": True}

            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"},
                 {"action": "recruit.invite", "goal": "invite a member"},
                 {"action": "culture.refine", "goal": "refine culture",
                  "params": {"playbook": {"meta": {"name": "c", "version": 1}}}}],
                Path(tmp) / "think.jsonl", spool_path, published=publish)

            # The Mac is off. The proposer keeps thinking; nothing is delivered.
            for round_no in (1, 2, 3):
                round_ = thinker.think(round_no)
                self.assertFalse(round_.published)
                self.assertIn("spooled for replay", round_.detail)
            self.assertEqual(thinker.spool.depth(), 3)

            # The Mac comes back. The held thoughts are not silently dropped.
            state["up"] = True
            delivered, remaining = thinker.flush_spool()
            self.assertEqual(delivered, 3)
            self.assertEqual(remaining, 0)
            self.assertEqual([p.action for p in sent][-3:],
                             ["seed.export", "recruit.invite", "culture.refine"])

    def test_a_dead_mailbox_is_spooled_not_mistaken_for_a_refusal(self):
        """The exact shape a real partition arrives in.

        proposal_mail reports a dead host as data - {"error": "unreachable: ..."}
        - not as an exception. The first version of the spool only caught
        exceptions, so it would never have fired in the one scenario it exists
        for. Caught by running thinkd against a dead port, not by reading.
        """
        def publish(proposal, identity, url, token):
            return {"error": "unreachable: <urlopen error [Errno 61] refused>"}

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", Path(tmp) / "held.jsonl",
                published=publish)
            round_ = thinker.think(1)
            self.assertIn("spooled for replay", round_.detail)
            self.assertEqual(thinker.spool.depth(), 1)
            # A 5xx is the same thing: the mailbox is up but nobody heard it.
            self.assertTrue(__import__("pando.thinkd", fromlist=["x"]).delivery_failed(
                {"error": "HTTP 502"}))
            self.assertFalse(__import__("pando.thinkd", fromlist=["x"]).delivery_failed(
                {"error": "HTTP 403"}))

    def test_a_refusal_is_dropped_not_replayed_forever(self):
        """Being heard and told "no" is not the same as not being heard."""
        def publish(proposal, identity, url, token):
            return {"error": "HTTP 400", "detail": "action not in the charter"}

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", Path(tmp) / "held.jsonl",
                published=publish)
            round_ = thinker.think(1)
            self.assertIn("held", round_.detail)
            self.assertIn("rejected", round_.error)
            # A rejection is the mailbox doing its job; spooling it would mean
            # replaying the same "no" until the disk fills.
            self.assertEqual(thinker.spool.depth(), 0)

    def test_mailbox_chain_is_ordered_and_deduplicated(self):
        from pando.thinkd import mailbox_chain
        self.assertEqual(mailbox_chain("a", ["b", "a", "  ", "c"]),
                         ["a", "b", "c"])
        self.assertEqual(mailbox_chain("a"), ["a"])
        self.assertEqual(mailbox_chain(""), [])
        self.assertEqual(mailbox_chain(" a ", [" a "]), ["a"])

    def test_a_deaf_primary_falls_through_to_the_second_mailbox(self):
        """One mailbox on one host is one process away from nobody being heard.

        The model already had a failover chain; the mailbox did not, which is
        why a partition could silence a proposer that was still thinking.
        """
        tried = []

        def publish(proposal, identity, url, token):
            tried.append(url)
            if url == "http://invalid.example":
                raise OSError("primary is gone")
            return {"task": {"id": "t-x"}, "claimed": True}

        with tempfile.TemporaryDirectory() as tmp:
            thinker, sent = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", Path(tmp) / "held.jsonl",
                published=publish, fallbacks=["http://backup"])
            round_ = thinker.think(1)

            self.assertTrue(round_.published)
            self.assertEqual(tried, ["http://invalid.example", "http://backup"])
            self.assertEqual(thinker.spool.depth(), 0)

    def test_a_refusal_is_not_re_asked_of_the_next_mailbox(self):
        """A 4xx is an answer. Asking the next mailbox the same question is not."""
        tried = []

        def publish(proposal, identity, url, token):
            tried.append(url)
            return {"error": "HTTP 400", "detail": "not in the charter"}

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", Path(tmp) / "held.jsonl",
                published=publish, fallbacks=["http://backup"])
            round_ = thinker.think(1)

            self.assertFalse(round_.published)
            self.assertIn("refused", round_.detail)
            self.assertEqual(tried, ["http://invalid.example"])
            self.assertEqual(thinker.spool.depth(), 0)

    def test_every_mailbox_down_is_held_not_lost(self):
        def publish(proposal, identity, url, token):
            raise OSError(f"{url} is gone")

        with tempfile.TemporaryDirectory() as tmp:
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "carry a seed"}],
                Path(tmp) / "think.jsonl", Path(tmp) / "held.jsonl",
                published=publish, fallbacks=["http://backup", "http://third"])
            round_ = thinker.think(1)

            self.assertFalse(round_.published)
            self.assertIn("spooled for replay", round_.detail)
            self.assertEqual(thinker.spool.depth(), 1)

    def test_a_week_behind_a_partition_does_not_fill_a_4gb_device(self):
        from pando.thinkd import Proposal, Spool
        with tempfile.TemporaryDirectory() as tmp:
            spool = Spool(Path(tmp) / "held.jsonl", cap=5)
            for i in range(40):
                spool.add(Proposal(proposal_id=f"p-r{i}", author="a",
                                   goal=f"g{i}", action="seed.export"), i)
            self.assertEqual(spool.depth(), 5)
            # The newest are kept, the oldest are the ones that overflow.
            kept = [r["proposal_id"] for r in spool.pending()]
            self.assertEqual(kept, [f"p-r{i}" for i in range(35, 40)])

    def test_flush_stops_at_the_first_transport_failure(self):
        """If the mailbox is still away, do not hammer it twenty times."""
        calls = {"n": 0}

        def publish(proposal, identity, url, token):
            calls["n"] += 1
            raise OSError("still down")

        with tempfile.TemporaryDirectory() as tmp:
            spool_path = Path(tmp) / "held.jsonl"
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "g1"},
                 {"action": "recruit.invite", "goal": "g2"}],
                Path(tmp) / "think.jsonl", spool_path, published=publish)
            thinker.think(1)
            thinker.think(2)
            self.assertEqual(thinker.spool.depth(), 2)

            calls["n"] = 0
            delivered, remaining = thinker.flush_spool()
            self.assertEqual(delivered, 0)
            self.assertEqual(remaining, 2)   # nothing forgotten
            self.assertEqual(calls["n"], 1)  # tried once, then stopped

    def test_a_bounded_run_against_a_used_ledger_still_runs(self):
        """`max_rounds` counts rounds *this* process runs, not an absolute index.

        Found by an end-to-end replay, not by reading. Phase 2 existed to prove
        that a partition's held proposals get re-delivered once the mailbox is
        back, and it was invoked with `--max-rounds 1` against a ledger that
        already held 2 rows. The old loop compared ``3 <= 1`` and executed
        nothing: it printed the banner, exited 0, and looked like a passing
        test. A check that cannot fail is worse than no check, because it buys
        confidence it has not earned.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "think.jsonl"
            thinker, _ = self._thinker(
                [{"action": "seed.export", "goal": "g1"},
                 {"action": "seed.export", "goal": "g2"},
                 {"action": "recruit.invite", "goal": "g3"}],
                ledger, Path(tmp) / "held.jsonl")

            # Two rounds from a previous life are already on the ledger.
            thinker.ledger.append(thinker.think(1))
            thinker.ledger.append(thinker.think(2))
            self.assertEqual(len(thinker.ledger.tail(10)), 2)

            thinker.rounds.clear()
            thinker.run_forever(max_rounds=1, sleep=lambda _s: None)

            # One *new* round ran, and the numbering continued from the ledger.
            self.assertEqual(len(thinker.rounds), 1)
            rows = thinker.ledger.tail(10)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["round"] for r in rows], [1, 2, 3])


class TestProposalView(unittest.TestCase):
    def test_a_signed_envelope_is_read_as_fields_not_json(self):
        import json
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import proposal_view

        view = proposal_view(json.dumps({
            "kind": "proposal", "version": 1,
            "pubkey": "abc", "sig": "def",
            "payload": {
                "proposal_id": "p1", "author": "pi5-proposer",
                "goal": "存活", "action": "seed.replicate",
                "expected_effect": "少一个单点", "cost": 0.6,
                "cites": ["r3"],
            },
        }))
        self.assertEqual(view["action"], "seed.replicate")
        self.assertEqual(view["goal"], "存活")
        self.assertEqual(view["cost"], 0.6)
        self.assertEqual(view["cites"], ["r3"])
        self.assertEqual(view["destination"], "")
        self.assertTrue(view["signed"])
        addressed = proposal_view(json.dumps({
            "kind": "proposal", "version": 1,
            "pubkey": "abc", "sig": "def",
            "payload": {
                "proposal_id": "p2", "author": "mac-proposer",
                "goal": "送到 pi5", "action": "seed.replicate",
                "destination": "pi5",
            },
        }))
        self.assertEqual(addressed["destination"], "pi5")
        self.assertIsNone(proposal_view("not a proposal"))


class TestReactions(unittest.TestCase):
    def test_a_signed_reaction_survives_and_a_tampered_one_does_not(self):
        from pando.genesis import sign_reaction, verify_reaction
        from pando.identity import Identity

        identity = Identity.generate()
        record = sign_reaction("p1", "mac-text", "text-nb-v1", "赞同", identity)
        view = verify_reaction(record)
        self.assertEqual(view["node_id"], "mac-text")
        self.assertEqual(view["stance"], "赞同")
        self.assertTrue(view["signed"])
        record["payload"]["stance"] = "反对"
        self.assertIsNone(verify_reaction(record))

    def test_a_reaction_hangs_on_the_proposal_it_names(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from observatory import attach_reactions

        messages = [
            {"kind": "proposal", "proposal": {"proposal_id": "p1"}},
            {"kind": "chat"},
        ]
        attach_reactions(messages, [
            {"target_id": "p1", "node_id": "mac-text", "checkpoint": "text-nb-v1",
             "stance": "赞同"},
            {"target_id": "p1", "node_id": "mac-text", "checkpoint": "text-nb-v1",
             "stance": "搁置"},
            {"target_id": "other", "node_id": "win-laya", "checkpoint": "laya-en",
             "stance": "反对"},
        ])
        self.assertEqual(messages[0]["reactions"], [{
            "node_id": "mac-text", "checkpoint": "text-nb-v1", "stance": "搁置"}])
        self.assertNotIn("reactions", messages[1])

    def test_a_remote_stem_is_not_this_machines_to_ask(self):
        from pando.reactd import select_local_stems

        nodes = {
            "mac-text": {"host": "127.0.0.1", "url": "http://127.0.0.1:8766",
                         "checkpoint": "text-nb-v1", "role": "decision"},
            "pandosi-pi5-laya": {"host": "192.168.31.22",
                                 "url": "http://192.168.31.22:9",
                                 "checkpoint": "laya-en", "role": "decision"},
        }
        chosen = select_local_stems(nodes, {"127.0.0.1", "localhost"})
        self.assertEqual([row["node_id"] for row in chosen], ["mac-text"])

    def test_only_a_decision_model_is_asked_to_react(self):
        """An LLM writes proposals; it is never asked to vote on one.

        The two jobs live on one device but not in one member, so the
        reactor has to read ``role`` and leave the proposer alone.
        """
        from pando.reactd import select_local_stems

        nodes = {
            "mac-text": {"host": "127.0.0.1", "url": "http://127.0.0.1:8766",
                         "checkpoint": "text-nb-v1", "role": "decision"},
            "mac-llm-8b-local": {"host": "127.0.0.1",
                                 "url": "http://127.0.0.1:8767",
                                 "checkpoint": "mac-llm-8b-q6",
                                 "role": "proposal"},
        }
        chosen = select_local_stems(nodes, {"127.0.0.1", "localhost"})
        self.assertEqual([row["node_id"] for row in chosen], ["mac-text"])

    def test_a_stem_that_names_no_role_keeps_its_old_job(self):
        """A root that predates the field still answers.

        Absence is read off what the member can do, so nothing that used to
        vote silently stops voting just because the field is new.
        """
        from pando.reactd import select_local_stems

        nodes = {
            "mac-text": {"host": "127.0.0.1", "url": "http://127.0.0.1:8766",
                         "checkpoint": "text-nb-v1"},
        }
        chosen = select_local_stems(nodes, {"127.0.0.1", "localhost"})
        self.assertEqual([row["node_id"] for row in chosen], ["mac-text"])

    def test_the_registry_records_which_job_a_member_holds(self):
        """The role is a field on the member, not something a reader infers.

        A member that names its role keeps it; one that names none is read
        off what it can do, so the record is always explicit.
        """
        from pando.rootd import Coordinator

        coordinator = Coordinator(quorum_k=1)
        self.assertTrue(coordinator.register(
            "mac-text", "http://127.0.0.1:8766", "text-nb-v1"))
        self.assertTrue(coordinator.register(
            "mac-llm", "http://127.0.0.1:8767", "mac-llm-8b-q6",
            chat={"model": "qwen-test"}))
        self.assertTrue(coordinator.register(
            "named", "http://127.0.0.1:8768", "c",
            chat={"model": "qwen-test"}, role="decision"))
        nodes = coordinator.live_nodes()
        self.assertEqual(nodes["mac-text"]["role"], "decision")
        self.assertEqual(nodes["mac-llm"]["role"], "proposal")
        self.assertEqual(nodes["named"]["role"], "decision")


    def test_each_local_role_answers_and_the_author_sits_out(self):
        """Every device fields one vote per local role; the author stays out.

        A proposal is answered by each decision model registered on this
        machine, except the one that wrote it. Reading the reaction list
        after a pass is therefore "who spoke", not "who was online".
        """
        from types import SimpleNamespace
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd
        from pando.identity import load_or_create

        stems = [
            {"node_id": "neck-a", "url": "http://a", "checkpoint": "c-a"},
            {"node_id": "neck-b", "url": "http://b", "checkpoint": "c-b"},
            {"node_id": "neck-c", "url": "http://c", "checkpoint": "c-c"},
        ]
        proposals = [SimpleNamespace(proposal_id="p1", author="neck-b",
                                     goal="goal", action="action")]
        asked, posted = [], []

        def fake_ask(stem, proposal, timeout):
            asked.append(stem["node_id"])
            return {"stance": "赞同", "probability": None}

        def fake_publish(record, node_id, mailbox, token, timeout=15):
            posted.append((record["payload"]["target_id"], node_id))
            return {"claimed": True}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "ask_stem", fake_ask), \
                mock.patch.object(reactd, "publish_reaction", fake_publish):
            identity = load_or_create(_P(td) / "identity.json")
            report = reactd.react_once(proposals, stems, identity,
                                       "http://mailbox", None,
                                       _P(td) / "ledger.jsonl", 5)

        self.assertEqual(asked, ["neck-a", "neck-c"])
        self.assertEqual(sorted(posted),
                         [("p1", "neck-a"), ("p1", "neck-c")])
        self.assertEqual(report["did"], "reacted")
        self.assertEqual(sorted(v["node_id"] for v in report["votes"]),
                         ["neck-a", "neck-c"])


    def test_reaction_delivery_uses_the_fallback_mailbox(self):
        """A dead primary must not turn a signed vote into a lost vote."""
        from unittest import mock
        from pando import reactd

        calls = []

        def fake_publish(record, node_id, url, token, timeout=15):
            calls.append(url)
            if url == "http://primary":
                return {"error": "unreachable: primary"}
            return {"task": {"id": "t1"}, "claimed": True}

        with mock.patch.object(reactd, "publish_reaction", fake_publish):
            sent = reactd.publish_reaction_any(
                {"payload": {"target_id": "p1"}}, "mac-text",
                ["http://primary", "http://fallback"], None, 5)

        self.assertEqual(calls, ["http://primary", "http://fallback"])
        self.assertEqual(sent["mailbox"], "http://fallback")
        self.assertTrue(sent["claimed"])

    def test_reading_proposals_falls_through_an_unreachable_primary(self):
        """An empty or dead primary must not hide a populated fallback."""
        from unittest import mock
        from pando import proposal_mail

        seen = []

        def fake_fetch(url, token=None, pinned=None, status="all",
                       timeout=15, max_items=None, rows=None,
                       skip_ids=None):
            seen.append(url)
            return ["proposal-from-fallback"] if url == "http://fallback" else []

        with mock.patch.object(proposal_mail, "fetch_proposals", fake_fetch), \
                mock.patch.object(proposal_mail, "_request",
                                  side_effect=[{"error": "unreachable: primary"},
                                               {"tasks": [{"id": "t1"}]}]):
            found, source, error = proposal_mail.fetch_proposals_any(
                ["http://primary", "http://fallback"], None)

        self.assertEqual(seen, ["http://fallback"])
        self.assertEqual(found, ["proposal-from-fallback"])
        self.assertEqual(source, "http://fallback")
        self.assertIsNone(error)

    def test_an_empty_fallback_is_healthy_even_if_primary_is_down(self):
        """An answered empty feed is idle, not a failed poll."""
        from unittest import mock
        from pando import proposal_mail

        with mock.patch.object(proposal_mail, "fetch_proposals",
                               return_value=[]), \
                mock.patch.object(proposal_mail, "_request",
                                  side_effect=[{"error": "unreachable: primary"},
                                               {"tasks": []}]):
            found, source, error = proposal_mail.fetch_proposals_any(
                ["http://primary", "http://fallback"], None)

        self.assertEqual(found, [])
        self.assertIsNone(source)
        self.assertIsNone(error)

    def test_a_listing_falls_through_a_dead_primary_and_names_its_source(self):
        """Task ids are per-mailbox row ids, so the source must travel.

        ``fetch_proposals_any`` already walked the chain, but it threw away
        *which* shelf answered -- fine for a caller that only counts rows,
        wrong for the executor, which then fetches bodies by the ids it just
        read. Counting one shelf's rows and fetching another's bodies reads
        as an empty round on a busy colony.
        """
        from unittest import mock
        from pando import proposal_mail

        seen = []

        def fake_request(url, path, token, method, payload=None, timeout=15):
            seen.append((url, path))
            if url == "http://primary":
                return {"error": "unreachable: primary"}
            return {"tasks": [{"id": "t1",
                               "title": "[proposal] p-1 policy.set by mac"}]}

        with mock.patch.object(proposal_mail, "_request", fake_request):
            listing, source, error = proposal_mail.listing_any(
                ["http://primary", "http://fallback"], None, status="all")

        self.assertEqual([url for url, _ in seen],
                         ["http://primary", "http://fallback"])
        self.assertEqual(source, "http://fallback")
        self.assertEqual([row["id"] for row in listing["tasks"]], ["t1"])
        self.assertIsNone(error)

    def test_a_quiet_primary_does_not_hide_a_populated_replica(self):
        """Replication lags; an empty shelf is not proof the colony is idle."""
        from unittest import mock
        from pando import proposal_mail

        def fake_request(url, path, token, method, payload=None, timeout=15):
            if url == "http://primary":
                return {"tasks": []}
            return {"tasks": [{"id": "t9"}]}

        with mock.patch.object(proposal_mail, "_request", fake_request):
            listing, source, error = proposal_mail.listing_any(
                ["http://primary", "http://fallback"], None)

        self.assertEqual(source, "http://fallback")
        self.assertEqual([row["id"] for row in listing["tasks"]], ["t9"])
        self.assertIsNone(error)

    def test_a_quiet_shelf_is_a_source_not_an_error(self):
        """An answered empty feed is idle, and an idle round still needs a
        shelf to name: "read nobody" and "read nobody" must not look the
        same."""
        from unittest import mock
        from pando import proposal_mail

        with mock.patch.object(proposal_mail, "_request",
                               return_value={"tasks": []}):
            listing, source, error = proposal_mail.listing_any(
                ["http://primary", "http://fallback"], None)

        self.assertEqual(listing, {"tasks": []})
        self.assertEqual(source, "http://primary")
        self.assertIsNone(error)

    def test_a_listing_nobody_answered_is_an_error_not_an_idle_round(self):
        """The whole point of the change: a dead address must not print a
        healthy idle round for a day while the backlog grows somewhere else.
        """
        from unittest import mock
        from pando import proposal_mail

        with mock.patch.object(proposal_mail, "_request",
                               return_value={"error": "unreachable: dead"}):
            listing, source, error = proposal_mail.listing_any(
                ["http://a", "http://b"], None)

        self.assertEqual(listing, {})
        self.assertIsNone(source)
        self.assertEqual(error, "unreachable: dead")


    def test_a_lost_reaction_is_retried_instead_of_marked_done(self):
        """A failed post leaves the proposal retryable, not silently adopted."""
        from types import SimpleNamespace
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd
        from pando.identity import load_or_create

        proposal = SimpleNamespace(proposal_id="p1", author="other",
                                   goal="goal", action="action")
        stems = [{"node_id": "mac-text", "url": "http://stem",
                  "checkpoint": "text-nb-v1"}]
        asks, posts = [], []

        def fake_ask(stem, item, timeout):
            asks.append(item.proposal_id)
            return {"stance": "赞同", "probability": None}

        def fake_publish(record, node_id, url, token, timeout=15):
            posts.append(url)
            if len(posts) == 1:
                return {"error": "unreachable: mailbox"}
            return {"task": {"id": "r1"}, "claimed": True}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "ask_stem", fake_ask), \
                mock.patch.object(reactd, "publish_reaction", fake_publish):
            identity = load_or_create(_P(td) / "identity.json")
            ledger = _P(td) / "ledger.jsonl"
            first = reactd.react_once([proposal], stems, identity,
                                      ["http://primary"], None, ledger, 5)
            second = reactd.react_once([proposal], stems, identity,
                                       ["http://primary"], None, ledger, 5)

        self.assertEqual(first["did"], "error")
        self.assertEqual(second["did"], "reacted")
        self.assertEqual(asks, ["p1", "p1"])
        self.assertEqual(posts, ["http://primary", "http://primary"])



    def test_a_reactor_that_fell_behind_answers_now_not_yesterday(self):
        """The oldest unseen proposal must not be answered for ever.

        Walking forward from the oldest unseen is right while the reactor
        keeps up and a wedge the moment it does not: every pass spends a
        model forward-pass on a proposal the group decided hours ago and the
        feed keeps producing new ones. A stale proposal is retired, and the
        pass goes to a current one.
        """
        import time as _t
        from types import SimpleNamespace
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd
        from pando.identity import load_or_create

        now = _t.time()
        stale = SimpleNamespace(proposal_id=f"p-old-r1-{int(now) - 7200}",
                                author="other", goal="goal", action="action")
        fresh = SimpleNamespace(proposal_id=f"p-new-r2-{int(now) - 60}",
                                author="other", goal="goal", action="action")
        stems = [{"node_id": "neck-a", "url": "http://a", "checkpoint": "c-a"}]
        asked = []

        def fake_ask(stem, proposal, timeout):
            asked.append(proposal.proposal_id)
            return {"stance": "赞同", "probability": None}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "ask_stem", fake_ask), \
                mock.patch.object(reactd, "publish_reaction",
                                  lambda *a, **k: {"claimed": True}):
            identity = load_or_create(_P(td) / "identity.json")
            ledger = _P(td) / "ledger.jsonl"
            # A ledger with history, so the one-shot first-pass branch (which
            # skips everything but the newest) cannot be what is measured.
            reactd.append_ledger(ledger, {"target_id": "already",
                                          "skip": True, "at": now})
            report = reactd.react_once([stale, fresh], stems, identity,
                                       "http://mailbox", None, ledger, 5,
                                       max_age=3600)
            rows = [json.loads(line)
                    for line in ledger.read_text().splitlines() if line]

        self.assertEqual(asked, [fresh.proposal_id])
        self.assertEqual(report["did"], "reacted")
        retired = {r["target_id"] for r in rows if r.get("skip")}
        self.assertIn(stale.proposal_id, retired)
        self.assertNotIn(fresh.proposal_id, retired)

    def test_a_proposal_with_no_stamp_has_no_age_and_is_never_swept(self):
        """Genesis seeds carry no stamp: unattributable is not stale."""
        import time as _t
        from types import SimpleNamespace
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd
        from pando.identity import load_or_create

        now = _t.time()
        seed = SimpleNamespace(proposal_id="live-1", author="other",
                               goal="goal", action="action")
        fresh = SimpleNamespace(proposal_id=f"p-new-r2-{int(now) - 60}",
                                author="other", goal="goal", action="action")
        stems = [{"node_id": "neck-a", "url": "http://a", "checkpoint": "c-a"}]
        asked = []

        def fake_ask(stem, proposal, timeout):
            asked.append(proposal.proposal_id)
            return {"stance": "赞同", "probability": None}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "ask_stem", fake_ask), \
                mock.patch.object(reactd, "publish_reaction",
                                  lambda *a, **k: {"claimed": True}):
            identity = load_or_create(_P(td) / "identity.json")
            ledger = _P(td) / "ledger.jsonl"
            reactd.append_ledger(ledger, {"target_id": "already",
                                          "skip": True, "at": now})
            reactd.react_once([seed, fresh], stems, identity,
                              "http://mailbox", None, ledger, 5,
                              max_age=3600)
            rows = [json.loads(line)
                    for line in ledger.read_text().splitlines() if line]

        self.assertEqual(asked, ["live-1"])
        self.assertNotIn("live-1",
                         {r["target_id"] for r in rows if r.get("skip")})

    def test_a_settled_proposal_body_is_not_fetched_again(self):
        """One signed-body GET per proposal ever made is what wedges a poll.

        The listing is how a reactor learns a proposal exists; the body is
        only worth a round trip while the proposal is still open.
        """
        from unittest import mock
        from pathlib import Path as _P
        from pando import proposal_mail
        from pando.genesis import Proposal, sign_proposal
        from pando.identity import load_or_create

        rows = [{"id": "t1", "title": "[proposal] p1 do.a by ann"},
                {"id": "t2", "title": "[proposal] p2 do.b by ann"}]
        ids = {row["id"]: row["title"].split()[1] for row in rows}
        paths = []

        with tempfile.TemporaryDirectory() as td:
            ident = load_or_create(_P(td) / "identity.json")

            def body_for(pid):
                return json.dumps(sign_proposal(
                    Proposal(proposal_id=pid, author="ann", goal="goal",
                             action="do.a"), ident))

            batches = []

            def fake_request(url, path, token, method, payload=None,
                             timeout=15):
                paths.append(path)
                if path == "/tasks/bodies":
                    batches.extend(payload["ids"])
                    return {"tasks": {i: {"id": i, "body": body_for(ids[i])}
                                      for i in payload["ids"]}}
                return {"task": {"body": body_for(ids[path.rsplit("/", 1)[-1]])}}

            with mock.patch.object(proposal_mail, "_request", fake_request):
                found = proposal_mail.fetch_proposals(
                    "http://mailbox", None, rows=rows, skip_ids={"p2"})

        # The invariant is which bodies were asked for, not the transport that
        # asked: p2 is skipped, so only t1 is named, and it is named once.
        self.assertEqual(paths, ["/tasks/bodies"])
        self.assertEqual(batches, ["t1"])
        self.assertEqual([p.proposal_id for p in found], ["p1"])

    def test_a_bounded_poll_reads_the_newest_rows_only(self):
        """``max_items`` has to bound the work, not just the returned list.

        reactd asked for the newest thirty rows and silently got every row
        the mailbox ever held, which is thousands of signed-body GETs a pass.
        """
        from unittest import mock
        from pathlib import Path as _P
        from pando import proposal_mail
        from pando.genesis import Proposal, sign_proposal
        from pando.identity import load_or_create

        rows = [{"id": f"t{i}", "title": f"[proposal] p{i} do.a by ann"}
                for i in range(5)]
        ids = {row["id"]: row["title"].split()[1] for row in rows}
        paths = []
        listings = []

        with tempfile.TemporaryDirectory() as td:
            ident = load_or_create(_P(td) / "identity.json")

            def body_for(pid):
                return json.dumps(sign_proposal(
                    Proposal(proposal_id=pid, author="ann", goal="goal",
                             action="do.a"), ident))

            def fake_request(url, path, token, method, payload=None,
                             timeout=15):
                if path == "/tasks/bodies":
                    paths.extend(payload["ids"])
                    return {"tasks": {i: {"id": i, "body": body_for(ids[i])}
                                      for i in payload["ids"]}}
                if path.startswith("/tasks"):
                    listings.append(path)
                    return {"tasks": rows}
                raise AssertionError(f"unbatched body read: {path}")

            with mock.patch.object(proposal_mail, "_request", fake_request):
                found, _source, _error = proposal_mail.fetch_proposals_any(
                    ["http://mailbox"], None, max_items=2, timeout=5)

        # Two rows, two bodies, one request -- and only the newest two.
        self.assertEqual(paths, ["t3", "t4"])
        self.assertEqual(listings, ["/tasks?status=all&limit=2"])
        self.assertEqual([p.proposal_id for p in found], ["p3", "p4"])

    def test_the_sweep_reads_titles_and_retires_the_backlog(self):
        """A wedged reactor heals in one pass without reading a body."""
        import time as _t
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd

        now = _t.time()
        old = f"p-old-r1-{int(now) - 7200}"
        new = f"p-new-r2-{int(now) - 60}"
        rows = [{"id": "t1", "title": f"[proposal] {old} do.a by ann"},
                {"id": "t2", "title": f"[proposal] {new} do.a by ann"},
                {"id": "t3", "title": "[chat] hello"}]
        paths = []

        def fake_request(url, path, token, method, payload=None, timeout=15):
            paths.append(path)
            return {"tasks": rows}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "_request", fake_request):
            ledger = _P(td) / "ledger.jsonl"
            seen = reactd.load_seen(ledger)
            swept = reactd.sweep_stale(["http://mailbox"], None, seen, 3600.0,
                                       ledger, 5)
            rows_written = [json.loads(line)
                            for line in ledger.read_text().splitlines() if line]

        self.assertEqual(swept, 1)
        self.assertEqual(paths, [
            f"/tasks?status=all&limit={reactd.STALE_SWEEP_FEED_LIMIT}"])
        self.assertEqual([r["target_id"] for r in rows_written], [old])
        self.assertTrue(rows_written[0]["stale"])
        self.assertNotIn(new, seen)

    def test_a_sweep_without_a_limit_retires_nothing(self):
        """``--max-age 0`` is the escape hatch back to no sweeping."""
        import time as _t
        from unittest import mock
        from pathlib import Path as _P
        from pando import reactd

        now = _t.time()
        old = f"p-old-r1-{int(now) - 7200}"
        rows = [{"id": "t1", "title": f"[proposal] {old} do.a by ann"}]

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(reactd, "_request",
                                  lambda *a, **k: {"tasks": rows}):
            ledger = _P(td) / "ledger.jsonl"
            seen = reactd.load_seen(ledger)
            swept = reactd.sweep_stale(["http://mailbox"], None, seen, 0,
                                       ledger, 5)

        self.assertEqual(swept, 0)
        self.assertFalse(ledger.exists())

    def test_an_unstamped_id_never_looks_stale(self):
        """Age comes from the id; an id without one has no age to compare."""
        from pando.proposal_mail import (proposal_age_seconds,
                                         proposal_id_from_title)

        self.assertIsNone(proposal_age_seconds("live-1"))
        self.assertEqual(proposal_id_from_title("[proposal] p1 do.a by ann"),
                         "p1")
        self.assertEqual(proposal_id_from_title("[chat] hello"), "")
        self.assertEqual(proposal_id_from_title(""), "")
        self.assertGreater(proposal_age_seconds("p-r1-1000000000", 1000000060),
                           59)


class TestChatMailboxPartition(unittest.TestCase):
    def test_history_requests_a_bounded_tail(self):
        from types import SimpleNamespace
        from unittest import mock
        from tools import agent_chat

        calls = []

        def fake_request(args, method, path, payload=None):
            calls.append((method, path, payload))
            return {"tasks": [{"id": "t1", "from": "ann",
                               "title": "[chat] @all: hi",
                               "body": "hi", "created_at": "2026-10-27T16:00:00",
                               "status": "claimed"}]}

        args = SimpleNamespace(limit=10, mentions=None)
        with mock.patch.object(agent_chat, "_request", fake_request), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent_chat.cmd_history(args), 0)
        self.assertEqual(calls, [("GET", "/tasks?status=all&limit=10", None)])

    """A chat reply posted nowhere is not queued; it is gone.

    chatd was the last writer in the colony with a single mailbox and no
    fallback: during a partition a present member looked exactly like a mute
    one. These pin the chain that closes that hole.
    """

    def test_a_reply_uses_the_fallback_mailbox_when_the_primary_is_deaf(self):
        from unittest import mock
        from pando import chatd

        calls = []

        def fake_request(url, token, method, payload, timeout):
            calls.append(url)
            if url.startswith("http://primary"):
                return {"error": "unreachable: primary"}
            if url.endswith("/task"):
                return {"task": {"id": "t1"}}
            return {"task": {"id": "t1"}}

        with mock.patch.object(chatd, "_request", fake_request):
            posted = chatd.publish_any(["http://primary", "http://fallback"],
                                       None, "mac-llm", "mac-codex", "在。", 5)

        self.assertEqual(posted["id"], "t1")
        self.assertEqual(posted["mailbox"], "http://fallback")
        self.assertTrue(posted["claimed"])
        self.assertEqual([u for u in calls if u.startswith("http://primary")],
                         ["http://primary/task"])

    def test_a_refusal_is_final_and_not_retried_elsewhere(self):
        """A 4xx means the mailbox heard the reply and refused it."""
        from unittest import mock
        from pando import chatd

        calls = []

        def fake_request(url, token, method, payload, timeout):
            calls.append(url)
            return {"error": "HTTP 403", "detail": "no"}

        with mock.patch.object(chatd, "_request", fake_request):
            posted = chatd.publish_any(["http://primary", "http://fallback"],
                                       None, "mac-llm", "mac-codex", "在。", 5)

        self.assertEqual(calls, ["http://primary/task"])
        self.assertIn("HTTP 403", posted["error"])

    def test_reading_chat_falls_through_an_unreachable_primary(self):
        from unittest import mock
        from pando import chatd

        seen = []

        def fake_request(url, token, method, payload, timeout):
            seen.append(url)
            if url.startswith("http://primary"):
                return {"error": "unreachable: primary"}
            return {"tasks": [{"id": "t1", "title": "[chat] @mac-llm: hi"}]}

        with mock.patch.object(chatd, "_request", fake_request):
            rows, source, error = chatd.fetch_tasks_any(
                ["http://primary", "http://fallback"], None, 5)

        self.assertEqual(seen, [
            "http://primary/tasks?status=all&limit="
            f"{chatd.CHAT_FEED_LIMIT}&kind=chat&kind=proposal",
            "http://fallback/tasks?status=all&limit="
            f"{chatd.CHAT_FEED_LIMIT}&kind=chat&kind=proposal",
        ])
        self.assertEqual([row["id"] for row in rows], ["t1"])
        self.assertEqual(source, "http://fallback")
        self.assertIsNone(error)

    def test_a_caller_can_bound_the_chat_feed(self):
        from unittest import mock
        from pando import chatd

        seen = []

        def fake_request(url, token, method, payload, timeout):
            seen.append(url)
            return {"tasks": [{"id": "t1", "title": "[chat] @mac-llm: hi"}]}

        with mock.patch.object(chatd, "_request", fake_request):
            rows, _, _ = chatd.fetch_tasks_any(
                ["http://mailbox"], None, 5, limit=37)

        self.assertEqual(
            seen,
            ["http://mailbox/tasks?status=all&limit=37"
             "&kind=chat&kind=proposal"])
        self.assertEqual([row["id"] for row in rows], ["t1"])

    def test_an_empty_fallback_is_healthy_even_if_primary_is_down(self):
        """This machine having nothing to say is idle, not a failed poll."""
        from unittest import mock
        from pando import chatd

        def fake_request(url, token, method, payload, timeout):
            if url.startswith("http://primary"):
                return {"error": "unreachable: primary"}
            return {"tasks": []}

        with mock.patch.object(chatd, "_request", fake_request):
            rows, source, error = chatd.fetch_tasks_any(
                ["http://primary", "http://fallback"], None, 5)

        self.assertEqual(rows, [])
        self.assertIsNone(source)
        self.assertIsNone(error)

    def test_a_listing_from_the_replica_still_yields_a_body_from_the_chain(self):
        """The listing can land on the replica while the body fetch lags."""
        from unittest import mock
        from pando import chatd

        def fake_request(url, token, method, payload, timeout):
            if url.startswith("http://primary") and "/task/t1" in url:
                return {"task": {"id": "t1", "body": ""}}
            if "/task/t1" in url:
                return {"task": {"id": "t1", "body": "hello"}}
            return {"error": "unreachable: nope"}

        with mock.patch.object(chatd, "_request", fake_request):
            full = chatd.full_task_any(["http://primary", "http://fallback"],
                                       None, {"id": "t1", "body": ""}, 5)

        self.assertEqual(full["body"], "hello")

    def test_a_quiet_but_answered_feed_is_idle_not_an_error(self):
        from unittest import mock
        from pando import chatd

        seen = []

        def fake_request(url, token, method, payload, timeout):
            seen.append(url)
            return {"tasks": []}

        with mock.patch.object(chatd, "_request", fake_request):
            report = chatd.pass_once(["http://mailbox"], None, [], set(), set(),
                                     {}, 5, True)

        self.assertEqual(report["did"], "idle")
        self.assertEqual(
            seen,
            ["http://mailbox/tasks?status=all&limit="
             f"{chatd.CHAT_FEED_LIMIT}&kind=chat&kind=proposal"])

    def test_every_mailbox_down_is_an_error_not_a_silent_idle(self):
        """A partition must not look like a colony with nothing to say."""
        from unittest import mock
        from pando import chatd

        def fake_request(url, token, method, payload, timeout):
            return {"error": "unreachable: nope"}

        with mock.patch.object(chatd, "_request", fake_request):
            with self.assertRaises(RuntimeError):
                chatd.pass_once(["http://a", "http://b"], None, [], set(),
                                set(), {}, 5, True)

    def test_hosts_json_may_name_a_chain_not_just_one_mailbox(self):
        """The Mac passes --hosts, not --mailbox: hosts.json is its chain.

        Reading only mailboxes[0] was the same partition hole as a missing
        --mailbox-fallback, just spelled in a config file instead of a flag.
        """
        import json as _json
        import tempfile
        from pando import chatd

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hosts.json"
            path.write_text(_json.dumps({
                "mailboxes": ["http://127.0.0.1:8776",
                              "http://192.168.31.100:8778",
                              "http://192.168.31.100:8778"],
                "roots": [],
            }), encoding="utf-8")
            roots, chain = chatd.load_hosts(str(path))

        self.assertEqual(chain, ["http://127.0.0.1:8776",
                                 "http://192.168.31.100:8778"])
        self.assertEqual(roots, [])

    def test_a_bootstrap_reports_which_mailbox_answered(self):
        """bootstrap must not shadow the mailbox source with a task id."""
        from unittest import mock
        from pando import chatd

        def fake_request(url, token, method, payload, timeout):
            if url.startswith("http://primary"):
                return {"error": "unreachable: primary"}
            return {"tasks": [{"id": "h1", "title": "[chat] @mac-llm: hi"}]}

        with mock.patch.object(chatd, "_request", fake_request):
            report = chatd.pass_once(["http://primary", "http://fallback"],
                                     None, [], set(), set(), {}, 5, False)

        self.assertEqual(report["did"], "bootstrap")
        self.assertEqual(report["ids"], ["h1"])
        self.assertEqual(report["read_from"], "http://fallback")


    def test_an_idle_chatd_opens_a_row_a_peer_is_meant_to_answer(self):
        """The row this writes is the one the colony never produced.

        It has to be a chat member on another host, by name, so the peer's own
        chatd sees a member naming it and answers -- the same path a person's
        message takes. A failed attempt is recorded as well, or a wedged stem
        would be called every interval.
        """
        from unittest import mock
        from pando import chatd

        posts = []

        def fake_request(url, token, method, payload, timeout):
            if method == "POST" and url.endswith("/task"):
                posts.append(payload)
                return {"task": {"id": "t-open-1"}}
            if method == "POST" and url.endswith("/claim"):
                return {"task": {"id": payload.get("id")}}
            return {"tasks": [{"id": "p1", "from": "mac-proposer",
                               "title": "[proposal] mac-proposer-r715 "
                                        "adaptive.governance by mac-proposer",
                               "created_at": "2026-10-19T12:33:57"}]}

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        known = {"mac-llm-8b-local", "win-gemma-12b"}
        ask = mock.Mock(return_value="win-gemma-12b，你那边 reproduce.spawn 跑通了吗？")
        with mock.patch.object(chatd, "_request", fake_request), \
                mock.patch.object(chatd, "ask_member", ask):
            report = chatd.pass_once(["http://mailbox"], None, members, known,
                                     set(), {}, 5, True, {}, 1000.0)

        self.assertEqual(report["did"], "opened")
        self.assertEqual(report["member"], "mac-llm-8b-local")
        self.assertEqual(report["to"], "win-gemma-12b")
        self.assertEqual(report["topic"], "p1")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["from"], "mac-llm-8b-local")
        self.assertTrue(posts[0]["title"].startswith("[chat] @win-gemma-12b: "))
        self.assertIn("win-gemma-12b", posts[0]["body"])

        # The same host does not open again inside the window.
        ask.reset_mock()
        posts.clear()
        with mock.patch.object(chatd, "_request", fake_request), \
                mock.patch.object(chatd, "ask_member", ask):
            quiet = chatd.pass_once(["http://mailbox"], None, members, known,
                                    set(), {}, 5, True,
                                    {("mac-llm-8b-local", "win-gemma-12b"): 1000.0},
                                    1000.0 + 60)
        self.assertEqual(quiet["did"], "idle")
        self.assertEqual(posts, [])
        ask.assert_not_called()

    def test_a_stem_that_will_not_open_gets_a_grounded_sentence_not_silence(self):
        """An initiator that goes quiet whenever a model hiccups is no initiator."""
        from unittest import mock
        from pando import chatd

        posts = []

        def fake_request(url, token, method, payload, timeout):
            if method == "POST" and url.endswith("/task"):
                posts.append(payload)
                return {"task": {"id": "t-open-2"}}
            if method == "POST" and url.endswith("/claim"):
                return {"task": {"id": payload.get("id")}}
            return {"tasks": [{"id": "p1", "from": "win-gpu-proposer",
                               "title": "[proposal] win-gpu-proposer-r724 "
                                        "resource.rebalance by win-gpu-proposer",
                               "created_at": "2026-10-19T12:32:39"}]}

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        boom = mock.Mock(side_effect=RuntimeError("empty reply from mac-llm"))
        with mock.patch.object(chatd, "_request", fake_request), \
                mock.patch.object(chatd, "ask_member", boom):
            report = chatd.pass_once(["http://mailbox"], None, members,
                                     {"mac-llm-8b-local", "win-gemma-12b"},
                                     set(), {}, 5, True, {}, 1000.0)

        self.assertEqual(report["did"], "opened")
        self.assertEqual(report["topic"], "p1")
        self.assertIn("p1", posts[0]["body"])
        self.assertNotEqual(report.get("model_note"), None)

    def test_an_opening_somewhere_else_never_hides_a_message_meant_for_us(self):
        """Openings are what happens when there is nothing to answer."""
        from unittest import mock
        from pando import chatd

        posts = []

        def fake_request(url, token, method, payload, timeout):
            if method == "POST" and url.endswith("/task"):
                posts.append(payload)
                return {"task": {"id": "t-reply-1"}}
            if method == "POST":
                return {"task": {"id": payload.get("id")}}
            return {"tasks": [{"id": "t1", "from": "mac-codex",
                               "title": "[chat] @mac-llm-8b-local: hi",
                               "body": "hi", "created_at": "1"}]}

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        with mock.patch.object(chatd, "_request", fake_request), \
                mock.patch.object(chatd, "ask_member", mock.Mock(return_value="在")):
            report = chatd.pass_once(["http://mailbox"], None, members,
                                     {"mac-llm-8b-local", "win-gemma-12b"},
                                     set(), {}, 5, True, {}, 1000.0)
        self.assertEqual(report["did"], "spoke")
        # One row, and it is the answer to the person -- not an opening that
        # happened to be written first because the loop was idle for a moment.
        self.assertEqual(len(posts), 1)
        self.assertTrue(posts[0]["title"].startswith("[chat] @mac-codex: "))


class TestChat(unittest.TestCase):
    def test_chat_is_a_member_field(self):
        """An LLM speaks because its record has chat.model, not because of its name."""
        from pando.chat import (next_reply, normalize_chat, should_answer,
                                title_addressee)
        from pando import chatd
        from pando.chatd import local_chat_members

        self.assertEqual(normalize_chat({"model": "qwen-test", "url": "http://x"}),
                         {"model": "qwen-test"})
        self.assertIsNone(normalize_chat({"model": ""}))
        self.assertIsNone(normalize_chat("qwen-test"))
        self.assertEqual(title_addressee("[chat] @mac-llm-8b-local: hi"),
                         "mac-llm-8b-local")
        self.assertIsNone(title_addressee("[chat] @@mac-llm-8b-local: hi"))
        self.assertTrue(should_answer("mac-llm", "mac-codex", "all", set(), 0))
        self.assertFalse(should_answer("mac-llm", "win-gemma", "all",
                                       {"win-gemma"}, 0))
        self.assertTrue(should_answer("mac-llm", "win-gemma", "mac-llm",
                                      {"win-gemma"}, 1))
        self.assertFalse(should_answer("mac-llm", "win-gemma", "mac-llm",
                                       {"win-gemma"}, 2))
        self.assertFalse(should_answer("mac-llm", "mac-llm", "mac-llm", set(), 0))

        nodes = {
            "mac-llm": {"url": "http://127.0.0.1:9", "host": "127.0.0.1",
                        "checkpoint": "lineage-only"},
            "win-gemma": {"url": "http://192.168.31.100:1", "host": "192.168.31.100",
                          "chat": {"model": "gemma-test"}},
            "mac-text": {"url": "http://127.0.0.1:8", "host": "127.0.0.1"},
        }

        def health_of(url):
            if url.endswith(":9"):
                return {"chat": {"model": "qwen-test"}}
            return {}

        members, known = local_chat_members(nodes, {"127.0.0.1"}, health_of)
        self.assertEqual([item["node_id"] for item in members], ["mac-llm"])
        self.assertEqual(members[0]["model"], "qwen-test")
        self.assertEqual(known, {"mac-llm", "win-gemma"})

        # Windows registers its stems with --advertise-host 192.168.31.100,
        # while chatd reaches the local root through 127.0.0.1.  Python on
        # Windows does not reliably enumerate that advertised address; a lone
        # loopback root is still authoritative for the members on this host.
        trusted, trusted_known = local_chat_members(
            nodes, {"127.0.0.1"}, health_of, trust_local_root=True)
        self.assertEqual([item["node_id"] for item in trusted],
                         ["mac-llm", "win-gemma"])
        self.assertEqual(trusted_known, {"mac-llm", "win-gemma"})
        self.assertTrue(chatd.root_is_loopback("http://127.0.0.1:8790"))
        self.assertFalse(chatd.root_is_loopback("http://192.168.31.100:8790"))

        choice = next_reply(
            members, known,
            [{"id": "t1", "from": "mac-codex", "title": "[chat] @mac-llm: hi",
              "body": "hi", "created_at": "1"}],
            set(), {})
        self.assertEqual(choice["member"]["node_id"], "mac-llm")
        self.assertIsNone(next_reply(
            members, known,
            [{"id": "t2", "from": "mac-codex", "title": "[chat] @win-gemma: hi",
              "body": "hi", "created_at": "1"}],
            set(), {}))

        seen = {}

        def fake_transport(url, payload, api_key, timeout):
            seen.update(payload=payload, timeout=timeout)
            return {"choices": [{"message": {"content": "在。"}}]}

        backend = OpenAIChoiceBackend(
            "mac-llm", "http://127.0.0.1:9/v1", model="qwen-test",
            transport=fake_transport)
        self.assertEqual(backend.chat_descriptor(), {"model": "qwen-test"})
        self.assertEqual(backend.complete(
            [{"role": "user", "content": "hi"}], timeout=12), "在。")
        self.assertEqual(seen["timeout"], 12)
        self.assertEqual(seen["payload"]["max_tokens"], 400)

        speaker = Stem("mac-llm", backend, host="127.0.0.1", port=0)
        silent = Stem("mac-text", MockBackend("mac-text"), host="127.0.0.1", port=0)
        root = Root()
        speaker.start()
        silent.start()
        root.start()
        try:
            with urllib.request.urlopen(speaker.url() + "/health", timeout=2) as resp:
                health = json.loads(resp.read())
            self.assertEqual(health["chat"], {"model": "qwen-test"})
            body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
            request = urllib.request.Request(
                speaker.url() + "/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as resp:
                reply = json.loads(resp.read())
            self.assertEqual(reply["message"]["content"], "在。")
            self.assertEqual(reply["model"], "qwen-test")
            denied = urllib.request.Request(
                silent.url() + "/chat", data=body,
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(denied, timeout=2)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()

            payload = json.dumps({
                "node_id": "mac-llm", "url": speaker.url(),
                "checkpoint": "mac-llm",
                "chat": {"model": "qwen-test", "url": "http://should-drop"},
            }).encode()
            request = urllib.request.Request(
                f"{root.url()}/register", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as resp:
                self.assertTrue(json.loads(resp.read())["ok"])
            with urllib.request.urlopen(f"{root.url()}/nodes", timeout=2) as resp:
                nodes = json.loads(resp.read())["nodes"]
            self.assertEqual(nodes["mac-llm"]["chat"], {"model": "qwen-test"})
            self.assertNotIn("url", nodes["mac-llm"]["chat"])
        finally:
            speaker.stop()
            silent.stop()
            root.stop()


    def test_a_member_that_can_only_answer_is_a_respondent_not_a_participant(self):
        """Nothing in this colony ever started a conversation.

        2026-10-19: 159 ``[chat]`` rows across the mailbox, zero of them a
        chat member addressing another chat member. ``should_answer`` already
        allowed peer-to-peer by name, and PEER_CAP already bounded it, so the
        missing piece was never permission -- it was that no member ever
        spoke first. This picks who speaks first, and to whom.
        """
        from pando.chat import next_opening

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        known = {"mac-llm-8b-local", "win-gemma-12b", "win-yi-9b",
                 "pandosi-pi5-llm"}

        first = next_opening(members, known, {}, 1000.0)
        self.assertEqual(first["member"]["node_id"], "mac-llm-8b-local")
        # Never itself, and never a member whose host is this one.
        self.assertNotIn(first["to"], {"mac-llm-8b-local"})
        self.assertIn(first["to"], {"win-gemma-12b", "win-yi-9b",
                                    "pandosi-pi5-llm"})

        # A host opens one conversation per window, not one per peer: a
        # per-pair throttle would grow with the square of the colony.
        opened = {("mac-llm-8b-local", first["to"]): 1000.0}
        self.assertIsNone(next_opening(members, known, opened, 1000.0 + 60))
        later = next_opening(members, known, opened, 1000.0 + 3601)
        self.assertIsNotNone(later)
        # The peer least recently addressed comes first, not the same one.
        self.assertNotEqual(later["to"], first["to"])

        # No peer off this host, no opening: two members sharing one chatd
        # would satisfy the letter of member-to-member and change nothing.
        self.assertIsNone(next_opening(members, {"mac-llm-8b-local"}, {}, 1.0))
        # A decision model has no chat.model, so it is not in known at all.
        self.assertIsNone(next_opening([], known, {}, 1.0))

    def test_a_previously_opened_peer_survives_a_root_outage(self):
        """A dark root must not erase the roster the ledger already proved.

        On 2026-10-27 the win root timed out on every chatd pass. The mailbox
        and the local model were healthy, but known contained only the local
        member, so the only participant who could initiate saw no peer. The
        opening ledger is evidence that this exact local member previously
        picked a real remote chat member from that root.
        """
        from pando.chat import next_opening

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        opened = {
            ("mac-llm-8b-local", "win-gemma-12b"): 1000.0,
            ("mac-llm-8b-local", "pandosi-pi5-llm"): 2000.0,
            ("another-member", "win-yi-9b"): 2000.0,
        }

        pick = next_opening(members, {"mac-llm-8b-local"}, opened, 5601.0,
                            evidence=opened)

        self.assertIsNotNone(pick)
        self.assertEqual(pick["to"], "win-gemma-12b")

    def test_a_dark_root_is_skipped_then_retried_with_bounded_backoff(self):
        """One timed-out root must not consume every healthy chatd pass."""
        from unittest import mock
        from pando import chatd

        local = "http://127.0.0.1:8765"
        remote = "http://100.93.216.90:8790"
        roots = [{"name": "mac", "url": local, "token": None},
                 {"name": "win", "url": remote, "token": None}]
        calls = []
        state = {"down": True}

        def fetch(url, token, timeout):
            calls.append((url, timeout))
            if url == remote and state["down"]:
                raise TimeoutError("timed out")
            return {url: {"host": "test"}}

        backoff = chatd.RootProbeBackoff(base=30, cap=45)
        with mock.patch.object(chatd, "fetch_nodes", fetch):
            nodes, events = chatd.probe_roots(roots, 180, backoff, now=100)
            self.assertEqual({url for url, _ in calls}, {local, remote})
            self.assertEqual(calls, [(local, 8), (remote, 8)])
            self.assertEqual(nodes, {local: {"host": "test"}})
            self.assertEqual(next(row for row in events
                                  if row["url"] == remote)["retry_in"], 30)

            calls.clear()
            chatd.probe_roots(roots, 180, backoff, now=110)
            self.assertEqual(calls, [(local, 8)])

            calls.clear()
            _, events = chatd.probe_roots(roots, 180, backoff, now=130)
            self.assertEqual(calls, [(local, 8), (remote, 8)])
            self.assertEqual(next(row for row in events
                                  if row["url"] == remote)["retry_in"], 45)

            calls.clear()
            chatd.probe_roots(roots, 180, backoff, now=174.9)
            self.assertEqual(calls, [(local, 8)])

            calls.clear()
            state["down"] = False
            nodes, _ = chatd.probe_roots(roots, 180, backoff, now=175)
            self.assertEqual(calls, [(local, 8), (remote, 8)])
            self.assertEqual(set(nodes), {local, remote})

            # Success clears the deferred state: the next pass probes again.
            calls.clear()
            chatd.probe_roots(roots, 180, backoff, now=176)
            self.assertEqual(calls, [(local, 8), (remote, 8)])

    def test_root_backoff_starts_when_the_failed_probe_ends(self):
        """A timeout must not consume the cooling-off window it just earned."""
        from unittest import mock
        from pando import chatd

        remote = "http://100.93.216.90:8790"
        roots = [{"name": "win", "url": remote, "token": None}]
        backoff = chatd.RootProbeBackoff(base=30, cap=30)
        times = iter([100.0, 108.0])

        def fetch(url, token, timeout):
            raise TimeoutError("timed out")

        with mock.patch.object(chatd, "fetch_nodes", fetch), \
                mock.patch.object(chatd.time, "monotonic", side_effect=lambda: next(times)):
            _, events = chatd.probe_roots(roots, 180, backoff)

        self.assertEqual(events[0]["retry_in"], 30)
        self.assertFalse(backoff.due(remote, 137.9))
        self.assertTrue(backoff.due(remote, 138.0))

    def test_a_remembered_peer_expires_after_repeated_silence(self):
        """A dark root must not turn one stale address into forever-noise."""
        from pando.chat import OPENED_PEER_TTL, next_opening

        members = [{"node_id": "mac-llm-8b-local", "url": "http://127.0.0.1:9",
                    "model": "qwen-test"}]
        evidence = {("mac-llm-8b-local", "win-yi-9b"): 1000.0}

        self.assertIsNotNone(next_opening(
            members, {"mac-llm-8b-local"}, evidence,
            1000.0 + OPENED_PEER_TTL, evidence=evidence))
        self.assertIsNone(next_opening(
            members, {"mac-llm-8b-local"}, evidence,
            1000.0 + OPENED_PEER_TTL + 1, evidence=evidence))

    def test_only_a_reply_or_live_membership_refreshes_peer_evidence(self):
        """Retrying a dark address must not refresh the stale-peer deadline."""
        from pando.chatd import load_peer_evidence

        key = ("mac-llm-8b-local", "win-yi-9b")
        rows = [
            {"open": True, "node_id": key[0], "to": key[1], "at": 100.0},
            {"open": True, "node_id": key[0], "to": key[1], "at": 200.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat-ledger.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(load_peer_evidence(path)[key], 100.0)

            rows.append({"node_id": key[0], "sender": key[1], "id": "reply",
                         "at": 300.0})
            rows.append({"open": True, "node_id": key[0], "to": key[1], "at": 400.0})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(load_peer_evidence(path)[key], 400.0)


    def test_talking_is_not_voting_in_the_first_person(self):
        """Two members refused to discuss a proposal, in their own words.

        2026-10-19 12:49: the first member-to-member exchange this colony
        ever had was mac-llm-8b-local asking win-gemma-12b what it made of
        the adaptive.governance proposal. win-gemma-12b answered "我不知道。
        我不能提案，也不能对提案发表看法。" and mac-llm-8b-local agreed with
        the refusal. The prompt said "不能提案，不能投票" and never separated
        that from having an opinion, so the members read a rule about actions
        as a rule about thoughts, and the conversation died at the first turn.
        """
        from pando.chatd import system_prompt

        prompt = system_prompt("win-gemma-12b", "gemma-4-12b-q4km")
        self.assertIn("不能执行命令", prompt)
        self.assertIn("说话不等于投票", prompt)
        self.assertIn("不能自己提案或投票", prompt)
        for allowed in ("讨论", "提问", "说出你的看法"):
            self.assertIn(allowed, prompt)


class TestVoteWatchdog(unittest.TestCase):
    """tools/vote_watchdog.py - the difference between voting and voting *now*.

    On 2026-10-19 win-laya-int8 cast a vote every two minutes for 35 hours.
    The process was up, the ledger grew, the stances varied -- and every
    target it named was stamped 2026-10-17. The panel's only staleness test
    was an absolute six-hour age, so the member read "fresh" for the first
    five hours of that wedge, and the reactor's own `--max-age` fix was 26
    commits away from the machine that needed it.

    The contract this tool must keep: catch a member whose answer-lag is
    growing, even while its lag is still numerically young, and never report
    "fine" for a member it could not actually measure.
    """

    def _tool(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "vote_watchdog.py"
        spec = importlib.util.spec_from_file_location("vote_watchdog", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _at(epoch: float) -> str:
        """A mailbox-shaped naive local timestamp for an epoch."""
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))

    def _reaction(self, member: str, stamp: int, delivered: float) -> dict:
        return {
            "from": member,
            "title": f"[reaction] prop-r1-{stamp} 赞同 by {member}",
            "created_at": self._at(delivered),
        }

    def test_a_young_lag_that_keeps_climbing_is_a_wedge(self):
        """The six-hour badge cannot see this; the slope can.

        Every lag here is under 30 minutes, so ``VOTE_STALE_S`` would render
        the member fresh on every poll. The targets are frozen and the clock
        is not, so the gap grows 1:1 -- the reactor is answering a fixed
        point in the past and will never reach the present.
        """
        wd = self._tool()
        now = 1_800_000_000.0
        frozen = int(now - 1800)
        rows = [(now - 5400 + i * 300, (now - 5400 + i * 300) - frozen)
                for i in range(18)]
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.WEDGED)
        self.assertAlmostEqual(verdict["slope"], 1.0, places=2)
        self.assertLess(verdict["last_lag_s"], wd.DEFAULT_WINDOW_S)

    def test_a_flat_lag_is_a_working_reactor_however_old_it_is(self):
        """A bounded lag means it is keeping up, whatever the number says."""
        wd = self._tool()
        now = 1_800_000_000.0
        rows = [(now - 5400 + i * 300, 1500.0) for i in range(18)]
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.OK)

    def test_draining_a_backlog_is_not_a_wedge(self):
        """A catching-up reactor has a *negative* slope and must stay ok.

        This is the live shape of mac-text and pandosi-pi5-x right after the
        ``--max-age`` commit landed: lag fell from 23-34h to under an hour
        inside the window. Calling that a wedge would have alarmed on the fix.
        """
        wd = self._tool()
        now = 1_800_000_000.0
        gone = now - 5400
        rows = [(gone + i * 300, max(600.0, 82800.0 - i * 4800.0))
                for i in range(18)]
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.OK)
        self.assertLess(verdict["slope"], 0)

    def test_too_few_votes_is_unknown_never_ok(self):
        """Silence is not evidence of health."""
        wd = self._tool()
        now = 1_800_000_000.0
        rows = [(now - 60, 30.0), (now - 30, 31.0)]
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.UNKNOWN)
        self.assertIn("need", verdict["why"])

    def test_a_burst_inside_one_instant_is_not_a_trend(self):
        """Six votes in four seconds span no wall-clock to measure against."""
        wd = self._tool()
        now = 1_800_000_000.0
        rows = [(now - 4 + i * 0.8, 10.0 + i) for i in range(6)]
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.UNKNOWN)
        self.assertIn("span", verdict["why"])

    def test_sleepable_members_can_be_excluded_before_the_survey(self):
        """Mac may sleep; its stale reactions must not become a fleet alarm."""
        wd = self._tool()
        now = 1_800_000_000.0
        stale = [(now - 3200 + i * 60, 1200.0 + i * 60)
                 for i in range(18)]
        rows = {
            "mac-text": stale,
            "win-laya-int8": [(now - 600 + i * 60, 20.0) for i in range(12)],
        }
        visible = wd.exclude_members(rows, ["mac-text", "mac-laya-int8"])
        verdicts = wd.survey(visible, ["win-laya-int8"], now)
        self.assertNotIn("mac-text", verdicts)
        self.assertEqual(verdicts["win-laya-int8"]["status"], wd.OK)

    def test_sysctl_epoch_reads_the_sec_field(self):
        wd = self._tool()
        self.assertEqual(wd._sysctl_epoch("{ sec = 123, usec = 4 } Tue ..."),
                         123.0)
        self.assertEqual(wd._sysctl_epoch(""), 0.0)
        self.assertEqual(wd._sysctl_epoch("no sec field"), 0.0)

    def test_sleep_overlaps_window_cases(self):
        """The gate is open only when a sleep actually touched the window."""
        wd = self._tool()
        now = 1_800_000_000.0
        # never slept
        self.assertFalse(wd.sleep_overlaps_window(0.0, 0.0, now, 5400.0))
        # slept and woke entirely before the window
        self.assertFalse(wd.sleep_overlaps_window(now - 20000, now - 19000,
                                                  now, 5400.0))
        # slept before the window but woke inside it
        self.assertTrue(wd.sleep_overlaps_window(now - 7000, now - 2000,
                                                 now, 5400.0))
        # slept inside the window and is still asleep (end == now)
        self.assertTrue(wd.sleep_overlaps_window(now - 1000, now, now, 5400.0))
        # slept and woke entirely inside the window
        self.assertTrue(wd.sleep_overlaps_window(now - 3000, now - 1000,
                                                 now, 5400.0))

    def test_resolve_ignored_keeps_sleep_gated_members_while_awake(self):
        """Unconditional ignores always drop; sleep-gated ones only while slept."""
        wd = self._tool()
        self.assertEqual(wd.resolve_ignored([], [], False), set())
        self.assertEqual(wd.resolve_ignored(["x"], [], True), {"x"})
        self.assertEqual(wd.resolve_ignored([], ["mac-text"], False), set())
        self.assertEqual(wd.resolve_ignored([], ["mac-text"], True),
                         {"mac-text"})
        self.assertEqual(wd.resolve_ignored(["x"], ["mac-text"], True),
                         {"x", "mac-text"})

    def test_a_named_decision_member_that_stops_voting_is_silent(self):
        """Having no slope is its own finding when the caller said "must vote"."""
        wd = self._tool()
        now = 1_800_000_000.0
        rows = {"mac-text": [(now - 600 + i * 60, 20.0) for i in range(12)]}
        verdicts = wd.survey(rows, ["mac-text", "win-laya-int8"], now)
        self.assertEqual(verdicts["mac-text"]["status"], wd.OK)
        self.assertEqual(verdicts["win-laya-int8"]["status"], wd.SILENT)

    def test_a_member_with_no_stamp_in_its_id_is_skipped_not_counted(self):
        """An id nobody can age must not be quietly rendered as fresh."""
        wd = self._tool()
        now = 1_800_000_000.0
        tasks = [
            {"from": "m", "title": "[reaction] prop-without-a-stamp 赞同 by m",
             "created_at": self._at(now - 30)},
            self._reaction("m", int(now - 100), now - 20),
            {"from": "m", "title": "[chat] hello", "created_at": self._at(now)},
        ]
        rows = wd.lag_rows(tasks)
        self.assertEqual(len(rows["m"]), 1)
        self.assertIsNone(wd.proposal_stamp("prop-without-a-stamp"))
        self.assertIsNotNone(wd.proposal_stamp("prop-r1-1792219880"))

    def test_the_wedge_that_cost_the_colony_a_night_is_reproduced(self):
        """The 2026-10-19 shape, as measured: 43 votes over 87m at +0.50.

        A regression fixture, not a mock. The numbers are the live mailbox
        values (win-laya-int8, 02:25), rounded to the shape that matters: the
        target epoch advances at ~0.585x of wall-clock, so the lag grows at
        the complement.
        """
        wd = self._tool()
        now = 1_800_000_000.0
        start = now - 87 * 60
        rows = []
        for i in range(43):
            delivered = start + i * (87 * 60 / 43.0)
            elapsed = delivered - start
            target = int(start - 34.8 * 3600 + elapsed * 0.585)
            rows.append((delivered, delivered - target))
        verdict = wd.member_slope(rows, now)
        self.assertEqual(verdict["status"], wd.WEDGED)
        self.assertAlmostEqual(verdict["slope"], 0.415, places=2)

    def test_an_alarm_reads_new_once_then_ongoing(self):
        """The same wedge must not page every ten minutes until it is buried.

        A watchdog that repeats itself is a watchdog nobody reads: the second
        poll of an unfixed fault is not news. The state file is what lets a
        caller page on ``new`` and stay quiet on ``ongoing``.
        """
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "vote-watchdog.json"
            first = {"win-laya-int8": {"status": wd.WEDGED, "slope": 0.5}}
            wd.track_first_seen(state, first, 1_800_000_000.0)
            self.assertEqual(first["win-laya-int8"]["seen"], "new")
            self.assertEqual(first["win-laya-int8"]["first_seen"], 1_800_000_000.0)
            second = {"win-laya-int8": {"status": wd.WEDGED, "slope": 0.5}}
            wd.track_first_seen(state, second, 1_800_000_600.0)
            self.assertEqual(second["win-laya-int8"]["seen"], "ongoing")
            self.assertEqual(second["win-laya-int8"]["first_seen"], 1_800_000_000.0)

    def test_a_recovered_member_is_forgotten_so_a_second_wedge_pages(self):
        """Suppression must be scoped to one outage, not one member forever.

        A member that comes back and then wedges again is a *new* fault. If
        the entry survived recovery, the second wedge would be swallowed by
        the first one's memory.
        """
        wd = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "vote-watchdog.json"
            wd.track_first_seen(state, {"m": {"status": wd.WEDGED}}, 1000.0)
            wd.track_first_seen(state, {"m": {"status": wd.OK, "slope": 0.0}}, 1060.0)
            self.assertEqual(json.loads(state.read_text())["members"], {})
            again = {"m": {"status": wd.WEDGED}}
            wd.track_first_seen(state, again, 1120.0)
            self.assertEqual(again["m"]["seen"], "new")
            self.assertEqual(again["m"]["first_seen"], 1120.0)

    def test_without_a_state_file_an_alarm_is_never_suppressed(self):
        """No state file means no memory -- every poll is a first alarm.

        This is the safe default: a caller that forgets to pass --state-file
        gets noisy paging, never silence.
        """
        wd = self._tool()
        verdicts = {"a": {"status": wd.WEDGED}, "b": {"status": wd.SILENT}}
        wd.track_first_seen(None, verdicts, 1_800_000_000.0)
        self.assertEqual(verdicts["a"]["seen"], "new")
        self.assertEqual(verdicts["b"]["seen"], "new")


