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


class TestMailboxCodeBundle(unittest.TestCase):
    """The mailbox is a portable code peer, not only a task queue.

    A node can usually reach the mailbox even when every other port on that
    host is blocked or moved.  Serving the HEAD-keyed git bundle from the
    authenticated mailbox therefore removes a second listener and a firewall
    rule from the bootstrap path.  These are real HTTP and real git bundle
    checks; a mocked byte response would not prove that the receiving updater
    can consume the file.
    """

    def _module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_mailbox_under_test_code_bundle",
            Path(__file__).resolve().parents[1] / "tools" / "agent_mailbox.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _git(self, root: Path):
        env = os.environ.copy()
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                    "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
            env.pop(key, None)
        env.update({
            "GIT_AUTHOR_NAME": "bundle-test",
            "GIT_AUTHOR_EMAIL": "bundle@example.com",
            "GIT_COMMITTER_NAME": "bundle-test",
            "GIT_COMMITTER_EMAIL": "bundle@example.com",
        })

        def run(*args):
            return subprocess.run(list(args), check=True, capture_output=True,
                                  text=True, env=env)

        repo = root / "repo"
        run("git", "init", "-b", "main", str(repo))
        (repo / "README").write_text("one\n", encoding="utf-8")
        run("git", "-C", str(repo), "add", "README")
        run("git", "-C", str(repo), "commit", "-m", "one")
        return env, repo, run

    @staticmethod
    def _serve(module, box, **kwargs):
        server_class = getattr(module, "_MailboxServer",
                               module.ThreadingHTTPServer)
        server = server_class(
            ("127.0.0.1", 0),
            module.build_handler(box, "tok", **kwargs))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f"http://127.0.0.1:{server.server_address[1]}"

    @staticmethod
    def _fetch(url: str, token: str | None):
        request = urllib.request.Request(url)
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read(), response.headers.get("Content-Type")

    def test_the_route_is_authenticated_and_explicitly_optional(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            box = module.Mailbox(Path(tmp) / "box")
            server, thread, base = self._serve(module, box)
            try:
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    self._fetch(base + "/code.bundle", None)
                self.assertEqual(denied.exception.code, 403)
                denied.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as disabled:
                    self._fetch(base + "/code.bundle", "tok")
                self.assertEqual(disabled.exception.code, 404)
                disabled.exception.close()
                self.assertEqual(box.list("all"), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_it_follows_head_and_never_creates_a_task(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, repo, run = self._git(root)
            box = module.Mailbox(root / "box")
            cache = root / "cache" / "code.bundle"
            server, thread, base = self._serve(
                module, box, code_repo=repo, code_cache=cache)
            try:
                first, content_type = self._fetch(base + "/code.bundle", "tok")
                self.assertEqual(content_type, "application/octet-stream")
                first_head = run("git", "-C", str(repo), "rev-parse",
                                 "HEAD").stdout.strip()

                (repo / "README").write_text("two\n", encoding="utf-8")
                run("git", "-C", str(repo), "add", "README")
                run("git", "-C", str(repo), "commit", "-m", "two")
                second_head = run("git", "-C", str(repo), "rev-parse",
                                  "HEAD").stdout.strip()
                second, _ = self._fetch(base + "/code.bundle", "tok")

                self.assertNotEqual(first, second)
                self.assertNotEqual(first_head, second_head)
                received = root / "received.bundle"
                received.write_bytes(second)
                listed = run("git", "bundle", "list-heads",
                             str(received)).stdout
                self.assertIn(second_head, listed)
                self.assertEqual(cache.with_name("code.bundle.head")
                                 .read_text(encoding="utf-8").strip(),
                                 second_head)
                # A download is a read.  In particular it must not appear as
                # a [chat] or [proposal] row that an executor could claim.
                self.assertEqual(box.list("all"), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_concurrent_readers_never_see_a_torn_bundle(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, repo, run = self._git(root)
            box = module.Mailbox(root / "box")
            cache = root / "cache" / "code.bundle"
            server, thread, base = self._serve(
                module, box, code_repo=repo, code_cache=cache)
            results, errors = [], []

            def fetch_one():
                try:
                    results.append(self._fetch(base + "/code.bundle", "tok")[0])
                except Exception as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)

            workers = [threading.Thread(target=fetch_one) for _ in range(8)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=20)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 8)
            self.assertEqual(len(set(results)), 1)
            received = root / "concurrent.bundle"
            received.write_bytes(results[0])
            head = run("git", "-C", str(repo), "rev-parse",
                       "HEAD").stdout.strip()
            self.assertIn(head, run("git", "bundle", "list-heads",
                                    str(received)).stdout)
            self.assertEqual(list(cache.parent.glob("*.partial")), [])
            self.assertEqual(box.list("all"), [])


class TestCodeSync(unittest.TestCase):
    """A checkout may fast-forward. It may not clobber edits or restart a model."""

    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "code_sync_under_test",
            Path(__file__).resolve().parents[1] / "tools" / "code_sync.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _env(self):
        env = os.environ.copy()
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        env["GIT_AUTHOR_NAME"] = "sync-test"
        env["GIT_AUTHOR_EMAIL"] = "sync@example.com"
        env["GIT_COMMITTER_NAME"] = "sync-test"
        env["GIT_COMMITTER_EMAIL"] = "sync@example.com"
        return env

    def _run(self, env, *args):
        subprocess.run(list(args), check=True, env=env, capture_output=True, text=True)

    def _world(self, root: Path):
        env = self._env()
        src = root / "src"
        bare = root / "origin.git"
        client = root / "client"
        self._run(env, "git", "init", "-b", "main", str(src))
        (src / "README").write_text("one\n")
        self._run(env, "git", "-C", str(src), "add", "README")
        self._run(env, "git", "-C", str(src), "commit", "-m", "one")
        self._run(env, "git", "clone", "--bare", str(src), str(bare))
        self._run(env, "git", "clone", str(bare), str(client))
        return env, src, bare, client

    def test_a_fast_forward_moves_and_a_dirty_tree_does_not(self):
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            before = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env).stdout.strip()
            (src / "README").write_text("two\n")
            self._run(env, "git", "-C", str(src), "add", "README")
            self._run(env, "git", "-C", str(src), "commit", "-m", "two")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            moved = sync.sync_repo(client)
            self.assertTrue(moved["updated"])
            self.assertEqual(moved["reason"], "fast-forward")
            self.assertNotEqual(moved["after"], before)

            (client / "README").write_text("local edit\n")
            held = sync.sync_repo(client)
            self.assertFalse(held["updated"])
            self.assertEqual(held["reason"], "dirty")
            head = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(head, moved["after"])
            self.assertEqual((client / "README").read_text(), "local edit\n")

    def test_an_untracked_file_does_not_wedge_the_updater(self):
        """A stray artifact must not freeze every later round, and must not
        be silently clobbered either.

        Regression against `_dirty` using plain `git status --porcelain`,
        which lists untracked files. `node_code_watch` calls this guard every
        five minutes on a node with no operator, and its own bootstrap writes
        `code.bundle` into the checkout root -- so counting untracked files
        would freeze the Windows node permanently while its ledger read
        `dirty`, which looks like a healthy, working updater.

        Two halves, because "ignore untracked" must not become "clobber
        untracked": an untracked file that no commit mentions is left alone
        and does not block; the same path appearing in the incoming commit is
        refused by git and reported as `not-fast-forward`, with stderr kept.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            stray = client / "code.bundle"
            stray.write_bytes(b"not a git object\n")
            (src / "README").write_text("two\n")
            self._run(env, "git", "-C", str(src), "add", "README")
            self._run(env, "git", "-C", str(src), "commit", "-m", "two")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")

            moved = sync.sync_repo(client)
            self.assertTrue(moved["updated"])
            self.assertEqual(moved["reason"], "fast-forward")
            self.assertEqual(stray.read_bytes(), b"not a git object\n")

            (src / "code.bundle").write_text("now tracked\n", encoding="utf-8")
            self._run(env, "git", "-C", str(src), "add", "code.bundle")
            self._run(env, "git", "-C", str(src), "commit", "-m", "track it")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")

            blocked = sync.sync_repo(client)
            self.assertFalse(blocked["updated"])
            self.assertEqual(blocked["reason"], "not-fast-forward")
            self.assertTrue(blocked.get("error"), "git's refusal must be kept")
            self.assertEqual(stray.read_bytes(), b"not a git object\n")

    def test_a_fork_is_not_merged_and_a_model_unit_is_not_restarted(self):
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            (client / "README").write_text("client\n")
            self._run(env, "git", "-C", str(client), "add", "README")
            self._run(env, "git", "-C", str(client), "commit", "-m", "client")
            client_head = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env).stdout.strip()
            (src / "README").write_text("origin\n")
            self._run(env, "git", "-C", str(src), "add", "README")
            self._run(env, "git", "-C", str(src), "commit", "-m", "origin")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            forked = sync.sync_repo(client)
            self.assertFalse(forked["updated"])
            self.assertEqual(forked["reason"], "not-fast-forward")
            head = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(head, client_head)

            refused = sync.restart_units(["pandosi-llama.service"])
            self.assertEqual(refused["reason"], "refused")
            self.assertEqual(refused["restarted"], [])
            # The voter is a client: a fast-forward must be able to restart it,
            # or pi5 keeps running the previous module from memory.
            self.assertIn("pandosi-react.service", sync.SAFE_UNITS)
            self.assertNotIn("pandosi-llama.service", sync.SAFE_UNITS)
            # A stem is the client that *calls* capability(); it is not the
            # model server. Leaving it out is how pi5 kept reporting
            # code_head=None while its checkout was already at the tip.
            self.assertIn("pandosi-stem-laya.service", sync.SAFE_UNITS)
            self.assertIn("pandosi-stem-text.service", sync.SAFE_UNITS)
            self.assertIn("pandosi-stem-llm.service", sync.SAFE_UNITS)
            # coordinator holds the live member registry; it is not auto-restarted.
            self.assertNotIn("pandosi-root.service", sync.SAFE_UNITS)
            peer = sync.sync_repo(root / "missing", peer=str(bare))
            self.assertEqual(peer["reason"], "no-git")

    def test_a_peer_is_used_only_when_origin_cannot_be_fetched(self):
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            before = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env).stdout.strip()
            (src / "README").write_text("peer\n")
            self._run(env, "git", "-C", str(src), "add", "README")
            self._run(env, "git", "-C", str(src), "commit", "-m", "peer")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            self._run(env, "git", "-C", str(client), "remote", "set-url",
                      "origin", str(root / "missing.git"))
            missed = sync.sync_repo(client)
            self.assertEqual(missed["reason"], "fetch-failed")
            self.assertEqual(missed["after"], before)
            fetched = sync.sync_repo(client, peer=str(bare))
            self.assertTrue(fetched["updated"])
            self.assertEqual(fetched["via"], "peer")
            self.assertEqual((client / "README").read_text(), "peer\n")

    def test_a_sources_file_names_the_fallbacks_and_a_typo_cannot_wedge_it(self):
        """The list lives in a file so a peer that moves is a file edit.

        This is the whole repair for 2026-10-19: every node had exactly one
        hardcoded peer URL, the Mac left the home LAN, and each node's updater
        went on reporting a live, healthy process against a dead address. The
        file is what makes one codebase adapt to all three platforms without
        a branch -- a Windows path, a LaunchAgent and a systemd unit all pass
        the same `--peer-file` and read the same format.

        The unreadable-file case is the one that matters most: a sources file
        is a *fallback*, so a missing or mistyped one must leave the primary
        fetch alone rather than becoming a new way to stop the updater.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "code-sources.txt"
            sources.write_text(
                "# the Mac, tailnet first\n"
                "\n"
                "http://100.92.184.4:8478/code.bundle  # tailnet\n"
                "   \n"
                "http://192.168.31.41:8478/code.bundle\n"
                "http://192.168.31.41:8478/code.bundle\n",  # a repeat
                encoding="utf-8")
            # The reader is literal: comments and blanks drop, a repeat is
            # kept as written. Dedupe belongs to `peer_sources`, which is the
            # one place that decides the order actually dialed.
            self.assertEqual(sync.source_list(str(sources)), [
                "http://100.92.184.4:8478/code.bundle",
                "http://192.168.31.41:8478/code.bundle",
                "http://192.168.31.41:8478/code.bundle",
            ])
            # A path nobody wrote is not an error, it is an empty fallback set.
            self.assertEqual(sync.source_list(str(root / "absent.txt")), [])
            # Flags come first, then the file, and a repeat is dropped: a node
            # that names one bundle twice does not pay for the same failure
            # twice, and the order a human wrote is the order tried.
            self.assertEqual(
                sync.peer_sources("", ["http://a", "http://b", "http://a"],
                                  str(sources)),
                ["http://a", "http://b",
                 "http://100.92.184.4:8478/code.bundle",
                 "http://192.168.31.41:8478/code.bundle"])
            # The single-value spelling the first callers used still works.
            self.assertEqual(sync.peer_sources("http://only"), ["http://only"])

    def test_one_resolver_finds_each_platforms_file_without_a_flag(self):
        """No flag, same code, three state directories: the file is the adapter.

        The point of resolving by probe instead of by branch is that a unit
        definition, a LaunchAgent and a supervisor entry can pass the same
        empty flag and each host still finds its own file -- ``~/.pandosi`` on
        the Mac, ``~/.pando`` on the Pi, ``D:\\PandoRSI\\models`` on Windows. A
        host with none keeps the old behaviour exactly, and an explicit flag
        keeps the last word over the probe.
        """
        sync = self._load()
        real = sync.SOURCE_FILES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                first, second = root / "first.txt", root / "second.txt"
                sync.SOURCE_FILES = {
                    "code": (str(first), str(second)),
                    "seed": (str(first), str(second)),
                }
                # None exists: no file, and that is not an error.
                self.assertEqual(sync.source_file(), "")
                self.assertEqual(sync.source_list(sync.source_file()), [])
                second.write_text("http://second/code.bundle\n", encoding="utf-8")
                self.assertEqual(sync.source_file(), str(second))
                # The first that exists wins, not the last: order is the
                # contract a human reads off the list.
                first.write_text("http://first/code.bundle\n", encoding="utf-8")
                self.assertEqual(sync.source_file(), str(first))
                self.assertEqual(
                    sync.source_list(sync.source_file()),
                    ["http://first/code.bundle"])
                # An explicit path beats the probe, even to a file that is not
                # there -- an operator who names a file gets that file's
                # answer, including "empty", not a surprise fallback.
                self.assertEqual(
                    sync.source_file(str(root / "absent.txt")),
                    str(root / "absent.txt"))
                self.assertEqual(
                    sync.source_list(sync.source_file(str(root / "absent.txt"))),
                    [])
            finally:
                sync.SOURCE_FILES = real

    def test_a_dead_fallback_costs_one_fetch_and_the_first_live_one_wins(self):
        """`origin` first, then the list in order, and one winner per round.

        The order is the contract: the cheap path stays first, and a dead
        fallback costs exactly one failed fetch rather than the round. When
        everything fails the ledger has to name every address it dialed --
        "fetch-failed" on its own cannot tell a node whose peer moved from a
        node with no network, and that ambiguity is what let a stale address
        sit unnoticed for hours.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            self._advance_origin(env, src, bare, "peer\n")
            dead = root / "dead.git"          # never created; fetch fails
            self._run(env, "git", "-C", str(client), "remote", "set-url",
                      "origin", str(root / "missing.git"))
            # Order matters: a dead entry before the live one is survived.
            fetched = sync.sync_repo(client, peers=[str(dead), str(bare)])
            self.assertTrue(fetched["updated"])
            self.assertEqual(fetched["via"], "peer")
            self.assertEqual(fetched["peer"], str(bare))
            self.assertEqual((client / "README").read_text(), "peer\n")
            # Nothing answers: every address is named, and the tree does not
            # move. This is the row an operator reads to see the address moved.
            self._advance_origin(env, src, bare, "three\n")
            stranded = sync.sync_repo(client, peers=[str(dead), "http://127.0.0.1:1/x.bundle"])
            self.assertEqual(stranded["reason"], "fetch-failed")
            self.assertFalse(stranded["updated"])
            self.assertEqual(stranded["tried"], [
                "origin", str(dead), "http://127.0.0.1:1/x.bundle"])
            self.assertEqual((client / "README").read_text(), "peer\n")

    def _load_watch(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "node_code_watch_under_test",
            Path(__file__).resolve().parents[1] / "tools" / "node_code_watch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_outbox(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seed_outbox_under_test",
            Path(__file__).resolve().parents[1] / "tools" / "seed_outbox.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _advance_origin(self, env, src, bare, text="two\n"):
        (src / "README").write_text(text)
        self._run(env, "git", "-C", str(src), "add", "README")
        self._run(env, "git", "-C", str(src), "commit", "-m", text.strip())
        self._run(env, "git", "-C", str(src), "push", str(bare), "main")

    def test_a_fast_forward_stops_clients_but_a_dirty_tree_does_not(self):
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"
            self._advance_origin(env, src, bare)

            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 1}

            row = watch.watch_round(repo=client, on_update="clients",
                                    client_cmd="stop-clients",
                                    node_cmd="full-restart", ledger=ledger)
            self.assertTrue(row["updated"])
            self.assertEqual(row["action"], "client-restart")
            self.assertEqual(seen, ["stop-clients"])
            # A model is never reloaded by a plain fast-forward.

            # The ledger is a peer-readable record, one JSON object per line.
            lines = [json.loads(line)
                     for line in ledger.read_text().splitlines()]
            self.assertEqual(lines[-1]["action"], "client-restart")

            (client / "README").write_text("local edit\n")
            quiet = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertFalse(quiet["updated"])
            self.assertEqual(quiet["reason"], "dirty")
            self.assertEqual(quiet["action"], "none")
            self.assertEqual(seen, ["stop-clients"])

    def test_a_roster_change_forces_a_node_restart(self):
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"
            # The roster lives inside the fast-forwarded checkout, so the file
            # on disk is old before the merge and new after it. Here README
            # stands in for tools/win_node_supervisor.ps1.
            roster = client / "README"
            self._advance_origin(env, src, bare)

            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 2}

            # A running supervisor cannot learn a new service name, so a change
            # in its roster must escalate to the full restart even though
            # --on-update asked only for the client restart.
            row = watch.watch_round(repo=client, on_update="clients",
                                    client_cmd="stop-clients",
                                    node_cmd="full-restart",
                                    supervisor=roster, ledger=ledger)
            self.assertTrue(row["roster_changed"])
            self.assertEqual(row["action"], "node-restart")
            self.assertEqual(seen, ["full-restart"])

    def test_an_unchanged_tree_stays_quiet(self):
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 3}
            row = watch.watch_round(repo=client, on_update="clients",
                                    client_cmd="stop-clients")
            self.assertFalse(row["updated"])
            self.assertEqual(row["reason"], "already")
            self.assertEqual(row["action"], "none")
            self.assertEqual(seen, [])

    def test_the_mac_updater_is_a_oneshot_so_it_can_update_itself(self):
        """A resident updater keeps running the updater code it replaced.

        The loop has to be re-executed to pick up a new ``node_code_watch``,
        so the shipped agent is a oneshot with an interval and the job stays
        outside CLIENT_UNITS -- it never restarts itself, launchd does.
        """
        src = (Path(__file__).resolve().parents[1]
               / "tools" / "mac-launchd" / "com.jewelroam.pandosi-code-watch.plist")
        text = src.read_text(encoding="utf-8")
        self.assertIn("<string>--once</string>", text)
        self.assertIn("<key>StartInterval</key>", text)
        self.assertIn("<key>RunAtLoad</key>", text)
        self.assertNotIn("<key>KeepAlive</key>", text)
        self.assertNotIn("<string>--interval</string>", text)
        sync = self._load()
        self.assertEqual(
            [name for name, row in sync.CLIENT_UNITS.items()
             if "pandosi-code-watch" in str(row.get("darwin", ""))], [])

    def test_a_head_that_moved_without_a_fetch_still_restarts_clients(self):
        """The Mac commits in its own checkout, so HEAD moves with no fetch.

        ``git merge --ff-only`` then reports "already" and a watcher that
        acted only on ``updated`` would leave every client on old code for
        ever. The 2026-10-19 Mac stems were exactly that: the checkout was
        at HEAD and mac-text/mac-llm-8b-local still reported older heads.
        """
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"
            def head_of():
                return subprocess.run(
                    ["git", "-C", str(client), "rev-parse", "HEAD"],
                    check=True, capture_output=True, text=True,
                    env=env).stdout.strip()

            head = head_of()
            ledger.write_text(json.dumps({"acted_head": head}) + "\n")

            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 4}
            # A commit made in this checkout: origin never moves, so the
            # merge is a no-op and only the recorded head can tell the
            # watcher that the clients are behind.
            (client / "README").write_text("committed in place\n")
            self._run(env, "git", "-C", str(client), "add", "README")
            self._run(env, "git", "-C", str(client), "commit", "-m", "in place")
            moved = head_of()
            self.assertNotEqual(moved, head)

            row = watch.watch_round(repo=client, on_update="clients",
                                    client_cmd="stop-clients", ledger=ledger)
            self.assertFalse(row["updated"])
            self.assertEqual(row["reason"], "already")
            self.assertEqual(row["action"], "client-restart")
            self.assertEqual(seen, ["stop-clients"])
            self.assertEqual(row["acted_head"], moved)

            # And it converges: the next round is quiet, not a restart loop.
            again = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertEqual(again["action"], "none")
            self.assertEqual(seen, ["stop-clients"])

    def test_the_first_clean_round_takes_responsibility_for_head(self):
        """A watcher with no memory of acting must not assume clients are fresh.

        This is the one-time convergence: an installed watcher finds a
        checkout at some head, no record that anything was restarted onto
        it, so it restarts the clients once and writes the head down.
        """
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"

            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 5}
            first = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertFalse(first["updated"])
            self.assertEqual(first["action"], "client-restart")
            self.assertEqual(seen, ["stop-clients"])
            self.assertEqual(first["acted_head"], first["after"])

            second = watch.watch_round(repo=client, on_update="clients",
                                       client_cmd="stop-clients", ledger=ledger)
            self.assertEqual(second["action"], "none")
            self.assertEqual(seen, ["stop-clients"])

            # A dirty tree never acts, and never claims to have converged.
            (client / "README").write_text("edit\n")
            dirty = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertEqual(dirty["reason"], "dirty")
            self.assertEqual(dirty["action"], "none")
            self.assertEqual(seen, ["stop-clients"])

    def test_a_tree_that_does_not_parse_is_never_restarted_onto(self):
        """A commit that cannot even be compiled must not reach a node.

        The Windows box has nobody watching it: the logon supervisor would
        respawn a client that dies on import, every ten seconds, and nothing
        on that box could read the traceback. Refusing keeps the code that
        works in place -- and it must not record an ``acted_head`` either, or
        the watcher would decide it had already converged and stop trying.
        """
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"
            (src / "broken.py").write_text("def (:  # not python\n")
            self._run(env, "git", "-C", str(src), "add", "broken.py")
            self._advance_origin(env, src, bare)

            seen = []
            watch._spawn = lambda command: seen.append(command) or {
                "ran": True, "pid": 7}
            row = watch.watch_round(repo=client, on_update="clients",
                                    client_cmd="stop-clients", ledger=ledger)
            self.assertTrue(row["updated"], "the fetch itself still happens")
            self.assertEqual(row["action"], "refused")
            self.assertEqual(row["blocked_by"], "does-not-parse")
            self.assertTrue(row["error"], "the compiler's words are kept")
            self.assertEqual(seen, [])
            self.assertNotIn("acted_head", row)

            # No acted_head: the next round tries again rather than deciding
            # that a refused restart was a completed one.
            again = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertEqual(again["action"], "refused")
            self.assertEqual(seen, [])

            # A later commit that fixes the file is acted on normally.
            (src / "broken.py").write_text("ok = 1\n")
            self._run(env, "git", "-C", str(src), "add", "broken.py")
            self._run(env, "git", "-C", str(src), "commit", "-m", "fix")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            fixed = watch.watch_round(repo=client, on_update="clients",
                                      client_cmd="stop-clients", ledger=ledger)
            self.assertEqual(fixed["action"], "client-restart")
            self.assertEqual(seen, ["stop-clients"])

    def test_the_windows_driver_detaches_when_the_stop_would_hit_itself(self):
        """The updater is on the Windows client list, so it cannot wait.

        Waiting for the stop means being stopped in the middle of the wait,
        and the ledger line that explains why every client just vanished is
        never written. That is why Windows first used a PowerShell script
        that fired and returned -- the behaviour belonged in the shared
        driver, not in a script that also carried a copy of the roster.
        """
        sync = self._load()
        spawned = []
        sync._spawn_detached = lambda command: spawned.append(command) or {
            "ran": True, "pid": 4242}

        def run(command, **kwargs):
            self.assertIn("Get-CimInstance", " ".join(command),
                          "only the listing may be waited for")
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps([{"ProcessId": os.getpid(),
                             "CommandLine": "python tools/node_code_watch.py"},
                            {"ProcessId": 11,
                             "CommandLine": "python -m pando.worker"}]), "")

        out = sync._restart_windows(["proc"], run=run)
        self.assertEqual(out["reason"], "restarted")
        self.assertEqual(out["killed"], [os.getpid(), 11])
        self.assertTrue(out["detached"]["ran"])
        self.assertEqual(len(spawned), 1)
        self.assertIn("Stop-Process -Id", spawned[0])
        self.assertIn(str(os.getpid()), spawned[0])

    def test_restart_clients_is_a_cli_mode_with_no_fetch_and_no_repo(self):
        """The Windows shim runs *after* the fetch that decided to act.

        ``tools/win_restart_clients.ps1`` is the one entry point the logon
        supervisor calls by name, so it has to stay a file with that name.
        What it does is one delegation to the shared roster; a fetch or a
        model reload inside it would be a second, silent policy.
        """
        sync = self._load()
        seen = []
        sync.restart_units = lambda units, platform="": (
            seen.append((sorted(units), platform))
            or {"restarted": ["proc"], "reason": "restarted"})
        code = sync.main(["--restart-clients"])
        self.assertEqual(code, 0)
        self.assertEqual(seen, [(sorted(sync.CLIENT_UNITS), "")])

    def test_a_failed_restart_is_not_recorded_as_convergence(self):
        """``nothing-running`` is not the word "restarted".

        A pattern that matches nothing is how a roster goes wrong quietly.
        Recording it as "the clients are already on this head" is the same
        silence that left the Mac stems on ``dbe78c33`` for a day.
        """
        watch = self._load_watch()
        self.assertFalse(watch._restart_happened({"reason": "nothing-running"}))
        self.assertFalse(watch._restart_happened({"reason": "restart-failed"}))
        self.assertFalse(watch._restart_happened({}))
        self.assertTrue(watch._restart_happened({"reason": "restarted"}))

    def test_code_bundle_follows_head_not_the_cache_age(self):
        outbox = self._load_outbox()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, _bare, _client = self._world(root)
            dest = root / "cache" / "code.bundle"

            # A pre-push hook exports GIT_DIR. That must not redirect the
            # bundle away from the repository handed to code_bundle().
            previous = os.environ.get("GIT_DIR")
            os.environ["GIT_DIR"] = str(Path(__file__).resolve().parents[1] / ".git")
            try:
                first = outbox.code_bundle(src, dest, max_age=3600)
            finally:
                if previous is None:
                    os.environ.pop("GIT_DIR", None)
                else:
                    os.environ["GIT_DIR"] = previous
            self.assertEqual(first, dest)
            first_head = subprocess.run(
                ["git", "bundle", "list-heads", str(dest)],
                check=True, capture_output=True, text=True, env=env
            ).stdout.split()[0]
            stamp = root / "cache" / "code.bundle.head"
            self.assertEqual(stamp.read_text(encoding="utf-8").strip(), first_head)

            # A new commit is younger than max_age, so an mtime-only cache would
            # keep serving the previous commit. The bundle cache must key on HEAD.
            (src / "README").write_text("new head\n")
            self._run(env, "git", "-C", str(src), "add", "README")
            self._run(env, "git", "-C", str(src), "commit", "-m", "new head")
            second = outbox.code_bundle(src, dest, max_age=3600)
            self.assertEqual(second, dest)
            second_head = subprocess.run(
                ["git", "bundle", "list-heads", str(dest)],
                check=True, capture_output=True, text=True, env=env
            ).stdout.split()[0]
            self.assertNotEqual(second_head, first_head)
            self.assertEqual(stamp.read_text(encoding="utf-8").strip(), second_head)

    def test_code_watch_refreshes_the_peer_bundle_on_a_clean_head(self):
        """The Mac must serve the commit its clients were just restarted onto.

        A local commit advanced the checkout while the fallback bundle stayed
        on the previous HEAD. Local clients then reported the new commit and
        code-drift correctly called them ahead of the only pullable tip. The
        updater now refreshes its own HEAD-keyed cache during the same clean
        round, while an unchanged cache remains a no-op.
        """
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _src, _bare, client = self._world(root)
            cache = root / "cache" / "code.bundle"
            stamp = cache.with_name(cache.name + ".head")

            first = watch.watch_round(repo=client, bundle_cache=str(cache))
            client_head = subprocess.run(
                ["git", "-C", str(client), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, env=env
            ).stdout.strip()
            self.assertEqual(first["bundle"]["head"], client_head)
            self.assertTrue(first["bundle"]["rebuilt"])
            self.assertEqual(stamp.read_text(encoding="utf-8").strip(),
                             client_head)

            second = watch.watch_round(repo=client, bundle_cache=str(cache))
            self.assertEqual(second["bundle"]["head"], client_head)
            self.assertFalse(second["bundle"]["rebuilt"])

    def test_the_deployed_updater_adopts_a_newer_copy_only_if_it_compiles(self):
        """The copy that runs on a headless node has to receive its own fixes.

        2026-10-19: pi5's ten-minute job ran /home/admin/.pandosi/code_sync.py,
        a file *outside* the checkout it fast-forwards. It was current only
        because someone had copied it there by hand -- same sha256 as
        tools/code_sync.py, mtime 00:53 -- which means a later fix to the
        updater would land in the repo and never take effect on the node that
        needed it. Same one-level-up trap that left win's supervisor without
        its ``code-watch`` service.

        The guard is the other direction: a candidate that does not compile must
        never replace a working copy, or one bad commit leaves a node with no
        updater and no operator.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pandosi-src" / "tools" / "code_sync.py"
            target = root / ".pandosi" / "code_sync.py"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)

            source.write_text("print('one')\n", encoding="utf-8")
            target.write_text("print('one')\n", encoding="utf-8")
            same = sync.refresh_self(source, target)
            self.assertFalse(same["adopted"])
            self.assertEqual(same["reason"], "same")

            source.write_text("print('two')\n", encoding="utf-8")
            fresh = sync.refresh_self(source, target)
            self.assertTrue(fresh["adopted"])
            self.assertEqual(fresh["reason"], "adopted")
            self.assertEqual(target.read_text(encoding="utf-8"), "print('two')\n")

            # A candidate that cannot even be parsed leaves the working one.
            source.write_text("def broken(:\n", encoding="utf-8")
            refused = sync.refresh_self(source, target)
            self.assertFalse(refused["adopted"])
            self.assertEqual(refused["reason"], "candidate-does-not-compile")
            self.assertEqual(target.read_text(encoding="utf-8"), "print('two')\n")

            # A missing source is not an adoption either.
            source.unlink()
            gone = sync.refresh_self(source, target)
            self.assertEqual(gone["reason"], "no-source")

    def test_a_checkout_that_did_not_move_is_not_authority_to_rewrite_us(self):
        """Adopt the checkout's updater only from a checkout we confirmed.

        ``refresh_self`` overwrites the copy of this file that is *running*.
        That is safe exactly when the round just established where the
        checkout stands. A fetch that failed, a tree a human left dirty, or a
        history that will not fast-forward can all sit at an **older** commit,
        and adopting from one of those rolls the updater backwards -- deleting
        the very ``--self-source``/``--self-target`` flags this line needs,
        until the node is left with no updater at all. Standing still is not
        one of those cases: "already at the tip" is a confirmation.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            # The updater lives inside the checkout; the copy that *runs* is
            # the one outside it.
            (src / "tools").mkdir()
            (src / "tools" / "code_sync.py").write_text(
                "print('checkout')\n", encoding="utf-8")
            self._run(env, "git", "-C", str(src), "add", "tools/code_sync.py")
            self._run(env, "git", "-C", str(src), "commit", "-m", "updater")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            self._run(env, "git", "-C", str(client), "pull", "--ff-only",
                      str(bare), "main")
            source = client / "tools" / "code_sync.py"
            target = root / "deployed" / "code_sync.py"
            target.parent.mkdir(parents=True)
            target.write_text("print('deployed')\n", encoding="utf-8")

            def run():
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    sync.main(["--repo", str(client),
                               "--self-source", str(source),
                               "--self-target", str(target)])
                return json.loads(out.getvalue())

            # At the tip with nothing to fetch: still the best copy we have.
            already = run()
            self.assertEqual(already["reason"], "already")
            self.assertTrue(already["self"]["adopted"])
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "print('checkout')\n")
            self.assertEqual(already["self"]["sha256"],
                             hashlib.sha256(
                                 b"print('checkout')\n").hexdigest()[:12])

            # A dirty tree: the guard must not read "we looked" as "we know".
            target.write_text("print('deployed')\n", encoding="utf-8")
            (client / "README").write_text("local edit\n", encoding="utf-8")
            dirty = run()
            self.assertEqual(dirty["reason"], "dirty")
            self.assertFalse(dirty["self"]["adopted"])
            self.assertEqual(dirty["self"]["reason"], "checkout-not-confirmed")
            self.assertEqual(dirty["self"]["checkout"], "dirty")
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "print('deployed')\n")

            # A fetch that failed leaves the checkout behind the tip -- this is
            # the case that would delete the self-adopt flags themselves.
            self._run(env, "git", "-C", str(client), "checkout", "--", "README")
            self._run(env, "git", "-C", str(client), "remote", "set-url",
                      "origin", str(root / "missing.git"))
            missed = run()
            self.assertEqual(missed["reason"], "fetch-failed")
            self.assertFalse(missed["self"]["adopted"])
            self.assertEqual(missed["self"]["reason"], "checkout-not-confirmed")
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "print('deployed')\n")

            # A real fast-forward is the case the flag exists for.
            self._run(env, "git", "-C", str(client), "remote", "set-url",
                      "origin", str(bare))
            self._advance_origin(env, src, bare)
            moved = run()
            self.assertEqual(moved["reason"], "fast-forward")
            self.assertTrue(moved["self"]["adopted"])
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "print('checkout')\n")

    @staticmethod
    def _systemctl_stub(calls, rc=0):
        """Answer ``systemctl`` and let every other subprocess run for real."""
        real = subprocess.run

        def fake(argv, **kwargs):
            if isinstance(argv, (list, tuple)) and argv and argv[0] == "systemctl":
                calls.append(list(argv))
                return subprocess.CompletedProcess(argv, rc, "", "")
            return real(argv, **kwargs)

        return fake

    def test_a_unit_template_outside_the_checkout_is_installed_and_backed_up(self):
        """The units live outside the checkout, so nothing rewrote them either.

        2026-10-19: pi5's ``pandosi-code-sync.service`` on disk carried
        ``--self-source``/``--self-target`` while the unit *systemd had loaded*
        did not. The file had been overwritten with ``cat >`` and
        ``systemctl --user daemon-reload`` was never run, so the ten-minute
        timer kept running the old ExecStart and the round left no trace -- the
        host has no journal at all. "The file is right" and "systemd loaded the
        file" are different facts, and only the second one runs.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkout" / "tools" / "pi5-systemd"
            target = root / ".config" / "systemd" / "user"
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            (source / "pandosi-code-sync.service").write_text("ExecStart=/new\n")
            (target / "pandosi-code-sync.service").write_text("ExecStart=/live\n")
            # Anything this project does not ship stays where it is: a wrong
            # prefix, a wrong suffix, a nested path, and a stale backup.
            (source / "notes.md").write_text("notes\n")
            (source / "pandosi-notes.txt").write_text("notes\n")
            (source / "pandosi-old.service.bak").write_text("stale\n")
            (source / "nested").mkdir()
            (source / "nested" / "pandosi-evil.service").write_text("evil\n")

            calls = []
            real = subprocess.run
            subprocess.run = self._systemctl_stub(calls)
            try:
                # ``platform`` is pinned because this asserts the *systemd*
                # branch: the same call on a laptop has to reach launchd, and
                # a test that quietly followed the host would stop covering
                # the branch it was written for the day it ran on a Mac.
                first = sync.install_units(source, target, platform="linux")
                second = sync.install_units(source, target, platform="linux")
            finally:
                subprocess.run = real

            self.assertEqual(first["changed"], ["pandosi-code-sync.service"])
            self.assertEqual(first["reason"], "installed")
            self.assertTrue(first["reloaded"])
            self.assertEqual((target / "pandosi-code-sync.service").read_text(),
                             "ExecStart=/new\n")
            self.assertEqual(
                (target / "pandosi-code-sync.service.bak").read_text(),
                "ExecStart=/live\n")
            for junk in ("notes.md", "pandosi-notes.txt", "nested",
                         "pandosi-old.service.bak"):
                self.assertFalse((target / junk).exists())

            # Nothing left to rewrite -- and the reload still runs, because
            # "the file matches" is indistinguishable from "a previous round
            # wrote it and systemd never read it".
            self.assertEqual(second["changed"], [])
            self.assertEqual(second["reason"], "same")
            self.assertTrue(second["reloaded"])
            self.assertEqual(calls,
                             [["systemctl", "--user", "daemon-reload"]] * 2)

            # A reload that fails is reported, and the bytes that were written
            # are named, so a peer can tell "not adopted" from "not written".
            calls.clear()
            (source / "pandosi-code-sync.service").write_text("ExecStart=/newer\n")
            subprocess.run = self._systemctl_stub(calls, rc=1)
            try:
                failed = sync.install_units(source, target, platform="linux")
            finally:
                subprocess.run = real
            self.assertEqual(failed["reason"], "reload-failed")
            self.assertFalse(failed["reloaded"])
            self.assertEqual(failed["changed"], ["pandosi-code-sync.service"])
            self.assertEqual((target / "pandosi-code-sync.service").read_text(),
                             "ExecStart=/newer\n")

    def test_a_dirty_checkout_cannot_rewrite_the_units_either(self):
        """The confirmation that gates the updater gates the units.

        A tree a human left dirty can sit at an older commit. Adopting its
        templates would push yesterday's ``ExecStart`` onto a node running
        today's -- the same rollback the ``--self-*`` guard exists to stop.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            units = src / "tools" / "pi5-systemd"
            units.mkdir(parents=True)
            (units / "pandosi-code-sync.service").write_text("ExecStart=/new\n")
            self._run(env, "git", "-C", str(src), "add", "-A")
            self._run(env, "git", "-C", str(src), "commit", "-m", "units")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            self._run(env, "git", "-C", str(client), "pull", "--ff-only",
                      str(bare), "main")
            target = root / "deployed-units"
            target.mkdir()
            (target / "pandosi-code-sync.service").write_text("ExecStart=/live\n")

            def run():
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    sync.main(["--repo", str(client),
                               "--platform", "linux",
                               "--install-units",
                               str(client / "tools" / "pi5-systemd"),
                               "--install-units-target", str(target)])
                return json.loads(out.getvalue())

            calls = []
            real = subprocess.run
            subprocess.run = self._systemctl_stub(calls)
            try:
                (client / "README").write_text("local edit\n")
                dirty = run()
                self.assertEqual(dirty["reason"], "dirty")
                self.assertEqual(dirty["units"]["reason"],
                                 "checkout-not-confirmed")
                self.assertEqual(dirty["units"]["checkout"], "dirty")
                self.assertEqual(dirty["units"]["changed"], [])
                self.assertFalse(dirty["units"]["reloaded"])
                self.assertEqual(calls, [])
                self.assertEqual(
                    (target / "pandosi-code-sync.service").read_text(),
                    "ExecStart=/live\n")

                # At the tip: the templates may now be adopted.
                self._run(env, "git", "-C", str(client), "checkout", "--",
                          "README")
                moved = run()
                self.assertEqual(moved["reason"], "already")
                self.assertEqual(moved["units"]["reason"], "installed")
                self.assertEqual(moved["units"]["changed"],
                                 ["pandosi-code-sync.service"])
                self.assertTrue(moved["units"]["reloaded"])
                self.assertEqual(
                    (target / "pandosi-code-sync.service").read_text(),
                    "ExecStart=/new\n")
            finally:
                subprocess.run = real

    def test_units_are_reloaded_before_the_restart_that_uses_them(self):
        """The process must start the bytes that this round just installed.

        2026-10-20: pi5 fast-forwarded to the commit adding ``--grant-url`` and
        wrote the matching unit, but restarted the old unit first. The live
        executor kept its pre-grant argv until another commit arrived. A
        restart before ``daemon-reload`` looks successful in both ledgers yet
        runs yesterday's configuration.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            units = src / "tools" / "pi5-systemd"
            units.mkdir(parents=True)
            (units / "pandosi-execute.service").write_text(
                "ExecStart=/new --grant-url http://example/operator-grant\n")
            self._run(env, "git", "-C", str(src), "add", "-A")
            self._run(env, "git", "-C", str(src), "commit", "-m", "units")
            self._run(env, "git", "-C", str(src), "push", str(bare), "main")
            self._run(env, "git", "-C", str(client), "pull", "--ff-only",
                      str(bare), "main")
            target = root / "deployed-units"
            target.mkdir()
            (target / "pandosi-execute.service").write_text(
                "ExecStart=/old\n")

            calls = []
            real = subprocess.run
            subprocess.run = self._systemctl_stub(calls)
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    sync.main(["--repo", str(client),
                               "--platform", "linux",
                               "--install-units", str(units),
                               "--install-units-target", str(target),
                               "--restart", "pandosi-execute.service"])
                result = json.loads(out.getvalue())
            finally:
                subprocess.run = real

            self.assertEqual(result["units"]["changed"],
                             ["pandosi-execute.service"])
            self.assertTrue(result["units"]["reloaded"])
            self.assertEqual(result["restarted"],
                             ["pandosi-execute.service"])
            self.assertEqual(calls, [
                ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "restart",
                 "pandosi-execute.service"],
            ])
            self.assertTrue(result["restart_state"]["written"])

            # The stamp describes a restart that actually happened. Once it is
            # written, the ten-minute timer re-reads the same units but does
            # not churn every client on every round. A future head or unit-byte
            # change alters the fingerprint and re-arms the restart.
            calls.clear()
            subprocess.run = self._systemctl_stub(calls)
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    sync.main(["--repo", str(client),
                               "--platform", "linux",
                               "--install-units", str(units),
                               "--install-units-target", str(target),
                               "--restart", "pandosi-execute.service"])
                steady = json.loads(out.getvalue())
            finally:
                subprocess.run = real
            self.assertEqual(steady["restart"], "already-recorded")
            self.assertEqual(steady["restarted"], [])
            self.assertEqual(calls, [
                ["systemctl", "--user", "daemon-reload"],
            ])

    def test_a_missing_restart_stamp_heals_the_already_stale_generation(self):
        """The fix must arrive without a second human restart.

        The pi5 failure crossed an updater version boundary: the old updater
        restarted first and wrote the unit second. By the time the fixed
        updater runs, the checkout and deployed unit already match, so there
        is no fast-forward or unit diff left to trigger on. The absent stamp is
        the remaining evidence that the installed configuration has not yet
        been through a successful restart.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _src, _bare, client = self._world(root)
            units = client / "tools" / "pi5-systemd"
            target = root / "deployed-units"
            units.mkdir(parents=True)
            target.mkdir()
            text = "ExecStart=/new --grant-url http://example/operator-grant\n"
            (units / "pandosi-execute.service").write_text(text)
            (target / "pandosi-execute.service").write_text(text)

            calls = []
            real = subprocess.run
            subprocess.run = self._systemctl_stub(calls)
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    sync.main(["--repo", str(client),
                               "--platform", "linux",
                               "--install-units", str(units),
                               "--install-units-target", str(target),
                               "--restart", "pandosi-execute.service"])
                result = json.loads(out.getvalue())
            finally:
                subprocess.run = real

            self.assertEqual(result["reason"], "already")
            self.assertEqual(result["units"]["changed"], [])
            self.assertEqual(result["restarted"],
                             ["pandosi-execute.service"])
            self.assertEqual(calls, [
                ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "restart",
                 "pandosi-execute.service"],
            ])

    def test_one_client_list_maps_onto_three_service_managers(self):
        """The clients are named once, so a new host is a row and not a fork.

        Until 2026-10-19 each host carried its own copy of the deployment:
        pi5 in ``tools/pi5-systemd``, Windows inside
        ``win_node_supervisor.ps1``, the Mac in ``~/Library/LaunchAgents``
        where the repo could not see it at all. That is three places for the
        same client list to drift, and it drifted: the Mac's live think agent
        carried ``--mailbox-fallback`` addresses its checkout copy did not,
        and the checkout copy of ``pandosi-code-sync.service`` carried
        ``--self-*`` flags the loaded unit did not.
        """
        sync = self._load()
        repo = Path(__file__).resolve().parents[1]

        # SAFE_UNITS is not a second list, it is the Linux column of this one.
        self.assertEqual(sync.SAFE_UNITS,
                         frozenset(row["linux"] for row
                                   in sync.CLIENT_UNITS.values()))
        self.assertIn("think", sync.CLIENT_UNITS)

        # Every client resolves on every platform, and a platform that does
        # not run one says so with an empty handle rather than by vanishing.
        for name, row in sync.CLIENT_UNITS.items():
            self.assertTrue(row.get("linux"), name)
            self.assertNotEqual(row.get("windows", None), None, name)
            self.assertIn("darwin", row, name)

        # The Windows client restart delegates instead of carrying a second
        # roster. Equality between two copies was never the property that
        # mattered: the copy that actually runs is the one on the node, and
        # no test in this repo can read it.
        ps1 = (repo / "tools" / "win_restart_clients.ps1").read_text()
        self.assertIn("code_sync.py", ps1)
        self.assertIn("--restart-clients", ps1)
        self.assertNotRegex(ps1, r"\$pattern\s*=")
        self.assertIn("node_code_watch", sync.WINDOWS_CLIENT_PATTERN)
        self.assertNotIn("llama", sync.WINDOWS_CLIENT_PATTERN)

        # A model is not a client on any of the three, and the refusal is
        # spelled the same way from either dialect.
        for model in ("pandosi-llama.service", "pandosi-root.service",
                      "com.jewelroam.pandosi-llm8b", "llama-server.exe"):
            for platform in ("linux", "darwin", "windows"):
                handles, refused = sync.resolve_units([model], platform)
                self.assertEqual(handles, [], (model, platform))
                self.assertEqual(refused, [model], (model, platform))
        for row in sync.CLIENT_UNITS.values():
            for handle in row.values():
                self.assertNotIn("llama", handle)
                self.assertNotIn("coordinator", handle)

        # The dialects are interchangeable *on purpose*: pi5's unit file
        # names must work when the same code runs on the Mac, and vice versa.
        handles, refused = sync.resolve_units(
            ["pandosi-think.service", "think", "pandosi-stem-laya.service"],
            "darwin")
        self.assertEqual(refused, [])
        self.assertEqual(handles, ["launchd:com.jewelroam.pandosi-think",
                                   "proc:pando.worker"])
        self.assertEqual(sync.resolve_units(["pandosi-stem-laya.service"],
                                            "linux")[0],
                         ["pandosi-stem-laya.service"])

    def test_a_restart_is_the_hosts_own_verb_not_a_shell_string(self):
        """Same request, three verbs: systemctl, launchctl, Stop-Process.

        ``--client-cmd`` is a string a human wrote for one host, which is why
        the Mac could not have one: it has no supervisor script and no
        systemd. The names are the portable half.
        """
        sync = self._load()

        darwin_calls = []

        def darwin_runner(argv, **kwargs):
            darwin_calls.append(list(argv))
            if argv[0] == "pgrep":
                return subprocess.CompletedProcess(argv, 0, "4242\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        got = sync.restart_units(["think", "stem-laya"], platform="darwin",
                                 runner=darwin_runner)
        self.assertEqual(got["reason"], "restarted")
        self.assertEqual(got["driver"], "launchd")
        self.assertIn("com.jewelroam.pandosi-think", got["restarted"])
        self.assertEqual(got["killed"], [4242])
        self.assertIn(["kill", "-TERM", "4242"], darwin_calls)
        self.assertTrue(any(call[0] == "launchctl" and call[1] == "kickstart"
                            for call in darwin_calls))

        listing = json.dumps([
            {"ProcessId": 5, "CommandLine": "python.exe -m pando.proposer --x"},
            {"ProcessId": 6, "CommandLine": "llama-server.exe --model m.gguf"},
            {"ProcessId": 7, "CommandLine": "python.exe -m pando.coordinator"},
            {"ProcessId": 8, "CommandLine": "python.exe tools/other.py"},
        ])
        windows_calls = []

        def windows_runner(argv, **kwargs):
            windows_calls.append(list(argv))
            if "ConvertTo-Json" in argv[-1]:
                return subprocess.CompletedProcess(argv, 0, listing, "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        got = sync.restart_units(["think"], platform="windows",
                                 runner=windows_runner)
        self.assertEqual(got["driver"], "windows")
        # A model and a bystander are not matched; the clients are.
        self.assertEqual(got["killed"], [5, 7])
        stop = windows_calls[-1][-1]
        self.assertIn("Stop-Process -Id 5,7", stop)

        # And the guard is all-or-nothing before any driver runs.
        linux_calls = []
        refused = sync.restart_units(
            ["think", "pandosi-llama.service"], platform="linux",
            runner=lambda argv, **kw: linux_calls.append(argv))
        self.assertEqual(refused["reason"], "refused")
        self.assertEqual(linux_calls, [])

    def test_a_mac_agent_is_written_only_when_it_changed(self):
        """On macOS a reload is an outage, so it is not unconditional.

        Linux can afford ``daemon-reload`` every round: it re-reads text and
        touches nothing. ``bootout``/``bootstrap`` stops and starts the agent,
        and one of the Mac's agents is the mailbox listener every other Mac
        process talks to. So an unchanged plist is *checked* with
        ``launchctl print`` and reloaded only when launchd does not have it --
        which is the pi5 trap (a file written, never loaded) closed without
        turning every five-minute round into a restart of everything.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkout" / "tools" / "mac-launchd"
            target = root / "Library" / "LaunchAgents"
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            chat = ("<plist><dict><key>Label</key>"
                    "<string>com.jewelroam.pandosi-chat</string>"
                    "<key>RunAtLoad</key><true/></dict></plist>\n")
            (source / "com.jewelroam.pandosi-chat.plist").write_text(chat)
            (target / "com.jewelroam.pandosi-chat.plist").write_text("old\n")
            # A model agent, a stray name, and a stale backup are not adopted.
            (source / "com.jewelroam.pandosi-llm8b.plist").write_text(
                "<plist><dict><key>Label</key><string>"
                "com.jewelroam.pandosi-llm8b</string></dict></plist>\n")
            (source / "notes.plist").write_text(chat)
            (source / "com.pandosi.stem.laya.plist.bak").write_text("stale\n")
            # A plist whose Label and filename disagree is refused, because
            # the ledger would name a job launchd does not have.
            (source / "com.pandosi.stem.llm8b.plist").write_text(
                "<plist><dict><key>Label</key><string>wrong</string>"
                "</dict></plist>\n")

            calls = []

            def runner(argv, **kwargs):
                calls.append(list(argv))
                if argv[0] == "plutil":
                    return subprocess.CompletedProcess(argv, 0, "OK\n", "")
                if argv[0] == "launchctl" and argv[1] == "print":
                    return subprocess.CompletedProcess(argv, 1, "", "no job")
                return subprocess.CompletedProcess(argv, 0, "", "")

            first = sync.install_units(source, target, platform="darwin",
                                       runner=runner)
            self.assertEqual(first["changed"],
                             ["com.jewelroam.pandosi-chat.plist"])
            self.assertEqual(first["reason"], "installed")
            self.assertEqual(first["booted"], ["com.jewelroam.pandosi-chat"])
            self.assertEqual((target / "com.jewelroam.pandosi-chat.plist")
                             .read_text(), chat)
            self.assertEqual((target / "com.jewelroam.pandosi-chat.plist.bak")
                             .read_text(), "old\n")
            self.assertFalse(
                (target / "com.jewelroam.pandosi-llm8b.plist").exists())
            self.assertFalse((target / "notes.plist").exists())
            self.assertFalse(
                (target / "com.pandosi.stem.laya.plist.bak").exists())
            self.assertFalse((target / "com.pandosi.stem.llm8b.plist").exists())
            # The skip is a ledger fact, not a silence: one broken template
            # must not stop the round, and it must not disappear either.
            self.assertEqual([row["unit"] for row in first["skipped"]],
                             ["com.pandosi.stem.llm8b.plist"])
            self.assertEqual(first["skipped"][0]["reason"], "label-mismatch")

            # Second round: the bytes match, launchd has the job, so nothing is
            # stopped and started again.
            calls.clear()

            def loaded_runner(argv, **kwargs):
                calls.append(list(argv))
                return subprocess.CompletedProcess(argv, 0, "", "")

            second = sync.install_units(source, target, platform="darwin",
                                        runner=loaded_runner)
            self.assertEqual(second["changed"], [])
            self.assertEqual(second["booted"], [])
            self.assertEqual(second["reason"], "same")
            self.assertEqual([call for call in calls
                              if "bootstrap" in call or "bootout" in call], [])

            # ... but a plist on disk that launchd never loaded is bootstrapped
            # without being rewritten. That is the half Linux does with an
            # unconditional ``daemon-reload``.
            self.assertEqual(
                sync.install_units(source, target, platform="darwin",
                                   runner=runner)["booted"],
                ["com.jewelroam.pandosi-chat"])

    def test_windows_keeps_its_roster_inside_the_checkout(self):
        """There is no ``~/Library/LaunchAgents`` to write on Windows.

        The Windows supervisor's roster *is* a file in the checkout, so the
        fast-forward already carries it and ``node_code_watch`` escalates to a
        node restart when its hash changes. Pretending to install units there
        would be a function that reports success and did nothing, so it
        reports the truth instead.
        """
        sync = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkout"
            target = root / "deployed"
            source.mkdir()
            (source / "pandosi-think.service").write_text("x\n")
            got = sync.install_units(source, target, platform="windows")
            self.assertFalse(got["installed"])
            self.assertEqual(got["changed"], [])
            self.assertEqual(got["reason"], "roster-inside-checkout")
            self.assertFalse(target.exists())

    def test_an_update_without_a_command_uses_the_platform_layer(self):
        """The Mac has no supervisor script to name, so the watcher may not
        need one either. ``--on-update clients`` with no ``--client-cmd``
        hands the client *names* down to ``restart_units``.
        """
        watch = self._load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, src, bare, client = self._world(root)
            ledger = root / "code-sync-ledger.jsonl"
            self._advance_origin(env, src, bare)

            seen = []
            watch.restart_units = lambda units, platform="": seen.append(
                (list(units), platform)) or {"restarted": ["think"],
                                             "reason": "restarted",
                                             "driver": "launchd"}
            watch._spawn = lambda command: self.fail(
                "no shell command should be spawned: %r" % command)

            row = watch.watch_round(repo=client, on_update="clients",
                                    ledger=ledger, platform="darwin")
            self.assertTrue(row["updated"])
            self.assertEqual(row["action"], "client-restart")
            self.assertEqual(row["restart"]["driver"], "launchd")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0][1], "darwin")
            self.assertIn("think", seen[0][0])
            self.assertIn("stem-text", seen[0][0])
            lines = [json.loads(line)
                     for line in ledger.read_text().splitlines()]
            self.assertEqual(lines[-1]["restart"]["reason"], "restarted")

            # An update that changes nothing still records nothing twice.
            (client / "README").write_text("local edit\n")
            quiet = watch.watch_round(repo=client, on_update="clients",
                                      ledger=ledger, platform="darwin")
            self.assertFalse(quiet["updated"])
            self.assertEqual(len(seen), 1)


class TestCodeReceipts(unittest.TestCase):
    """A remote updater must be able to prove what it observed.

    A bundle cached on the Mac and a local ledger on Windows are two copies of
    the same claim. This contract makes the remote watcher push a signed
    observation over the authenticated mailbox route, so the receiver has
    evidence minted on the other side of the network.
    """

    @staticmethod
    def _identity():
        from pando.identity import Identity
        return Identity.generate()

    def test_round_trip_pins_the_first_key_and_refuses_tampering(self):
        from pando import code_receipt
        identity = self._identity()
        envelope = code_receipt.build_envelope(
            identity, node="win", member="win-root", head="a" * 40,
            action="client-restart", platform="windows", acted=True,
            reason="fast-forward", sent_at=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            receiver = code_receipt.CodeReceiptReceiver(tmp)
            answer = receiver.receive(envelope, expected_node="win", now=20.0)
            document = receiver.load("win")
            self.assertTrue(answer["ok"])
            self.assertEqual(document["received_at"], 20.0)
            self.assertEqual(document["sent_at"], 10.0)
            self.assertEqual(document["head"], "a" * 40)
            self.assertEqual(document["pubkey"], identity.public_key)

            changed = json.loads(json.dumps(envelope))
            changed["payload"]["head"] = "b" * 40
            with self.assertRaises(code_receipt.CodeReceiptSignatureError):
                receiver.receive(changed, expected_node="win")
            self.assertEqual(receiver.load("win")["head"], "a" * 40)

            replacement = code_receipt.build_envelope(
                self._identity(), node="win", member="win-root",
                head="c" * 40, action="none")
            with self.assertRaises(code_receipt.CodeReceiptSignatureError):
                receiver.receive(replacement, expected_node="win")
            with self.assertRaises(code_receipt.CodeReceiptError):
                receiver.receive(envelope, expected_node="pi5")

    def test_http_receipt_is_authenticated_and_is_not_a_mailbox_task(self):
        from pando import code_receipt
        from tools import agent_mailbox, node_code_watch

        identity = self._identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity_file = root / "identity.json"
            token_file = root / "mailbox.token"
            identity_file.write_text(json.dumps(identity.to_json()),
                                     encoding="utf-8")
            token_file.write_text("mail-token", encoding="utf-8")
            box = agent_mailbox.Mailbox(root / "mailbox")
            receiver = code_receipt.CodeReceiptReceiver(root / "receipts")
            server = agent_mailbox.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                agent_mailbox.build_handler(
                    box, "mail-token", code_receipts=receiver))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            endpoint = base + "/receipts/code/win"
            try:
                unauthorized = urllib.request.Request(
                    endpoint, data=b"{}",
                    headers={"Authorization": "Bearer wrong",
                             "Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(unauthorized, timeout=3)
                self.assertEqual(caught.exception.code, 403)
                caught.exception.close()

                sent = node_code_watch.send_code_receipt(
                    {"head": "d" * 40, "action": "client-restart",
                     "acted": True, "reason": "fast-forward"},
                    url=endpoint, token_file=str(token_file),
                    identity_file=str(identity_file), node="win",
                    member="win-root", platform="windows", timeout=3)
                self.assertTrue(sent["sent"], sent)

                read = urllib.request.Request(
                    endpoint, headers={"Authorization": "Bearer mail-token"})
                with urllib.request.urlopen(read, timeout=3) as response:
                    document = json.loads(response.read())["receipt"]
                self.assertEqual(document["head"], "d" * 40)
                self.assertEqual(document["action"], "client-restart")
                self.assertEqual(document["platform"], "windows")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(box.list("open"), [],
                             "a receipt is observation, not work")

    def test_an_unreachable_receiver_never_raises_into_the_updater(self):
        from tools import node_code_watch
        result = node_code_watch.send_code_receipt(
            {"head": "e" * 40, "action": "none"},
            url="http://127.0.0.1:1/receipts/code/win",
            token_file="/definitely/missing/token", identity_file="missing",
            node="win", timeout=0.1)
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "receipt-failed")


class TestPi5Telemetry(unittest.TestCase):
    """Pi5 must be observable over the same direction it already talks.

    A root table the Mac cannot dial is not evidence that Pi5 stopped. The
    sender therefore proves its identity with the durable node key, and the
    receiver stamps its *own* clock. These tests pin both halves of that
    contract, including the failures that must leave the previous snapshot
    untouched.
    """

    def _modules(self):
        from pando import telemetry
        from tools import agent_mailbox, code_sync
        return telemetry, agent_mailbox, code_sync

    @staticmethod
    def _identity():
        from pando.identity import Identity
        return Identity.generate()

    def test_the_receiver_accepts_the_pinned_key_and_stamps_its_clock(self):
        telemetry, _, _ = self._modules()
        identity = self._identity()
        envelope = telemetry.build_envelope(
            identity, nodes={"pandosi-pi5-x": {"host": "192.168.31.22"}},
            code_head="a" * 40, think_round=42, sent_at=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pi5-root.json"
            receiver = telemetry.TelemetryReceiver(
                destination, expected_pubkey=identity.public_key)
            answer = receiver.receive(envelope, now=2000.0)
            document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertTrue(answer["ok"])
        self.assertEqual(document["mirrored_at"], 2000.0)
        self.assertEqual(document["sent_at"], 1.0,
                         "the sender clock is diagnostic, not freshness")
        self.assertEqual(document["telemetry"]["host"], "pi5")
        self.assertEqual(document["code_head"], "a" * 40)

    def test_any_payload_change_or_unknown_field_breaks_the_signature(self):
        telemetry, _, _ = self._modules()
        identity = self._identity()
        envelope = telemetry.build_envelope(
            identity, nodes={"pandosi-pi5-x": {"host": "192.168.31.22"}},
            code_head="a" * 40, think_round=42)
        changed = json.loads(json.dumps(envelope))
        changed["payload"]["nodes"]["pandosi-pi5-x"]["host"] = "10.0.0.1"
        with self.assertRaises(telemetry.TelemetrySignatureError):
            telemetry.validate_envelope(changed, identity.public_key)

        unknown = json.loads(json.dumps(envelope))
        unknown["payload"]["surprise"] = True
        with self.assertRaises(telemetry.TelemetryError):
            telemetry.validate_envelope(unknown, identity.public_key)
        extra_top = json.loads(json.dumps(envelope))
        extra_top["surprise"] = True
        with self.assertRaises(telemetry.TelemetryError):
            telemetry.validate_envelope(extra_top, identity.public_key)

    def test_wrong_signer_and_missing_signature_are_refused(self):
        telemetry, _, _ = self._modules()
        identity = self._identity()
        other = self._identity()
        envelope = telemetry.build_envelope(
            identity, nodes={"pandosi-pi5-x": {}}, code_head="a" * 40,
            think_round=1)
        with self.assertRaises(telemetry.TelemetrySignatureError):
            telemetry.validate_envelope(envelope, other.public_key)
        envelope["sig"] = ""
        with self.assertRaises(telemetry.TelemetrySignatureError):
            telemetry.validate_envelope(envelope, identity.public_key)

    def test_a_failed_replace_leaves_the_last_valid_mirror_intact(self):
        telemetry, _, _ = self._modules()
        from unittest import mock
        identity = self._identity()
        envelope = telemetry.build_envelope(
            identity, nodes={"pandosi-pi5-x": {}}, code_head="b" * 40,
            think_round=9)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pi5-root.json"
            destination.write_text('{"mirrored_at": 10}\n', encoding="utf-8")
            receiver = telemetry.TelemetryReceiver(
                destination, expected_pubkey=identity.public_key)
            with mock.patch.object(telemetry.os, "replace",
                                   side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    receiver.receive(envelope, now=2000.0)
            self.assertEqual(destination.read_text(encoding="utf-8"),
                             '{"mirrored_at": 10}\n')

    def test_the_http_endpoint_requires_both_tokens_and_creates_no_task(self):
        telemetry, mailbox_tool, _ = self._modules()
        identity = self._identity()
        envelope = telemetry.build_envelope(
            identity, nodes={"pandosi-pi5-x": {}}, code_head="c" * 40,
            think_round=7)
        with tempfile.TemporaryDirectory() as tmp:
            box = mailbox_tool.Mailbox(Path(tmp) / "mailbox")
            receiver = telemetry.TelemetryReceiver(
                Path(tmp) / "pi5-root.json",
                expected_pubkey=identity.public_key)
            server = mailbox_tool.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                mailbox_tool.build_handler(box, "mail-token", receiver))
            endpoint = (f"http://127.0.0.1:{server.server_address[1]}"
                        "/telemetry/pi5")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                body = json.dumps(envelope).encode("utf-8")
                wrong = urllib.request.Request(
                    endpoint, data=body,
                    headers={"Authorization": "Bearer wrong",
                             "Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(wrong, timeout=3)
                self.assertEqual(caught.exception.code, 403)
                caught.exception.close()

                right = urllib.request.Request(
                    endpoint, data=body,
                    headers={"Authorization": "Bearer mail-token",
                             "Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(right, timeout=3) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(box.list("open"), [], "telemetry is not work")
            self.assertTrue((Path(tmp) / "pi5-root.json").is_file())

    def test_the_sender_signs_the_live_table_and_largest_ledger_round(self):
        telemetry, mailbox_tool, sync = self._modules()
        from unittest import mock
        identity = self._identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            identity_file = root / "identity.json"
            identity_file.write_text(json.dumps(identity.to_json()),
                                     encoding="utf-8")
            think_ledger = root / "think-ledger.jsonl"
            think_ledger.write_text(
                '{"round": 4}\nnot-json\n{"round": 19}\n', encoding="utf-8")
            root_token = root / "root.token"
            root_token.write_text("root-token", encoding="utf-8")
            mail_token = root / "mail.token"
            mail_token.write_text("mail-token", encoding="utf-8")
            destination = root / "received.json"
            receiver = telemetry.TelemetryReceiver(
                destination, expected_pubkey=identity.public_key)
            state: dict = {}

            class Handler(mailbox_tool.BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def _send(self, code, payload):
                    data = json.dumps(payload).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self):  # noqa: N802
                    if self.headers.get("X-Pairing-Token") != "root-token":
                        self._send(403, {"error": "pairing"})
                    elif self.path == "/nodes":
                        self._send(200, {"nodes": {
                            "pandosi-pi5-x": {"host": "192.168.31.22"},
                            "pandosi-pi5-laya": {"host": "192.168.31.22"},
                        }})
                    elif self.path == "/status":
                        self._send(200, {"members": 2, "quorum_k": 1})
                    else:
                        self._send(404, {"error": "not found"})

                def do_POST(self):  # noqa: N802
                    if self.headers.get("Authorization") != "Bearer mail-token":
                        self._send(403, {"error": "bearer"})
                        return
                    length = int(self.headers.get("Content-Length", 0))
                    state["envelope"] = json.loads(
                        self.rfile.read(length).decode("utf-8"))
                    try:
                        answer = receiver.receive(state["envelope"])
                    except telemetry.TelemetryError as exc:
                        self._send(403, {"error": str(exc)})
                        return
                    self._send(200, answer)

            server = mailbox_tool.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                endpoint = (f"http://127.0.0.1:{server.server_address[1]}"
                            "/telemetry/pi5")
                root_url = f"http://127.0.0.1:{server.server_address[1]}"
                with mock.patch.object(sync, "_head", return_value="d" * 40):
                    result = sync.send_telemetry(
                        repo, url=endpoint, token_file=str(mail_token),
                        root_url=root_url,
                        root_token_file=str(root_token),
                        identity_file=str(identity_file),
                        think_ledger=str(think_ledger), timeout=3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertTrue(result["sent"], result)
            self.assertEqual(result["think_round"], 19)
            self.assertEqual(state["envelope"]["payload"]["status"],
                             {"members": 2, "quorum_k": 1})
            self.assertEqual(destination.read_text(encoding="utf-8").count(
                "pandosi-pi5-x"), 1)

    def test_a_telemetry_failure_never_changes_the_updater_exit_code(self):
        _, _, sync = self._modules()
        from unittest import mock
        failed = {"sent": False, "status": "failed",
                  "reason": "telemetry-failed", "error": "offline"}
        argv = ["--repo", ".", "--telemetry-url", "http://127.0.0.1:1/x",
                "--telemetry-token-file", "m", "--telemetry-root-url", "r",
                "--telemetry-root-token-file", "rt",
                "--telemetry-identity", "i", "--telemetry-think-ledger", "l"]
        output = io.StringIO()
        with mock.patch.object(sync, "sync_repo", return_value={
                "reason": "already", "updated": False, "before": "a",
                "after": "a", "restarted": []}), \
                mock.patch.object(sync, "send_telemetry", return_value=failed), \
                contextlib.redirect_stdout(output):
            code = sync.main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["telemetry"], failed)


class TestCodeIdentity(unittest.TestCase):
    """A member that cannot say which commit it runs is a member nobody can audit.

    The 2026-10-18 finding was that the Windows node sat four commits behind and
    the only way to learn that was to ask it. These tests pin the two facts that
    make the check possible from any peer: every registration carries the head,
    and that head is the code the process actually loaded.
    """

    def _env(self) -> dict:
        env = os.environ.copy()
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(key, None)
        env.update({
            "GIT_AUTHOR_NAME": "code-id-test",
            "GIT_AUTHOR_EMAIL": "code-id@example.com",
            "GIT_COMMITTER_NAME": "code-id-test",
            "GIT_COMMITTER_EMAIL": "code-id@example.com",
        })
        return env

    def _repo(self, path: Path, text: str = "one\n") -> tuple[dict, str]:
        env = self._env()
        subprocess.run(["git", "init", "-q", "-b", "main", str(path)],
                       check=True, env=env, capture_output=True, text=True)
        (path / "README").write_text(text)
        subprocess.run(["git", "-C", str(path), "add", "README"],
                       check=True, env=env, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(path), "commit", "-qm", "one"],
                       check=True, env=env, capture_output=True, text=True)
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, env=env, capture_output=True,
                              text=True).stdout.strip()
        return env, head

    def test_a_checkout_can_name_its_own_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _env, head = self._repo(root / "one")
            self.assertEqual(graft.code_head(root / "one"), head)

    def test_something_that_is_not_a_checkout_says_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(graft.code_head(Path(tmp)), "")

    def test_a_hijacked_git_dir_does_not_answer_for_us(self):
        # The pre-push gate exports GIT_DIR. A probe that honoured it would
        # report this checkout's commit while claiming to describe another.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, mine = self._repo(root / "mine")
            # Different bytes on purpose: two identical trees would share a
            # commit sha and the test would pass without proving anything.
            _other_env, other = self._repo(root / "other", "other\n")
            self.assertNotEqual(mine, other)
            previous = os.environ.get("GIT_DIR")
            os.environ["GIT_DIR"] = str(root / "other" / ".git")
            try:
                self.assertEqual(graft.code_head(root / "mine"), mine)
            finally:
                if previous is None:
                    os.environ.pop("GIT_DIR", None)
                else:
                    os.environ["GIT_DIR"] = previous

    def test_capability_carries_the_head_it_is_running(self):
        cap = graft.capability()
        self.assertIn("code_head", cap)
        self.assertEqual(cap["code_head"], graft.code_head())

    def test_a_live_process_reports_what_it_loaded_not_what_arrived_since(self):
        # Memoized on purpose: a stem that was fast-forwarded underneath itself
        # still runs the old bytes until it restarts, and must say so.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            env, first = self._repo(repo)
            saved_root, saved_head = graft.ROOT, graft._CODE_HEAD
            graft.ROOT, graft._CODE_HEAD = repo, None
            try:
                self.assertEqual(graft.running_commit(), first)
                (repo / "README").write_text("two\n")
                subprocess.run(["git", "-C", str(repo), "add", "README"],
                               check=True, env=env, capture_output=True, text=True)
                subprocess.run(["git", "-C", str(repo), "commit", "-qm", "two"],
                               check=True, env=env, capture_output=True, text=True)
                second = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                    env=env, capture_output=True, text=True).stdout.strip()
                self.assertNotEqual(first, second)
                self.assertEqual(graft.code_head(repo), second)   # disk moved
                self.assertEqual(graft.running_commit(), first)   # process did not
            finally:
                graft.ROOT, graft._CODE_HEAD = saved_root, saved_head

    def test_an_installed_copy_answers_with_its_stamp(self):
        # pip copies the tree and never copies .git, so git cannot answer for
        # an installed member. Measured 2026-10-19: every pi5 stem reported an
        # empty code_head while its venv was in fact current -- "cannot say"
        # and "out of date" looked identical from every peer.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package / graft.CODE_HEAD_STAMP).write_text("abc123\n",
                                                         encoding="utf-8")
            self.assertEqual(graft.installed_code_head(package), "abc123")

    def test_a_missing_stamp_says_nothing_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(graft.installed_code_head(Path(tmp)), "")

    def test_the_installer_stamps_the_copy_it_just_made(self):
        from tools import code_sync

        # The writer lives in a file that runs outside the checkout, so it
        # cannot import the reader. Pin the two literals instead of trusting
        # that they were kept in step.
        self.assertEqual(code_sync.INSTALLED_HEAD_STAMP, graft.CODE_HEAD_STAMP)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _env, head = self._repo(root / "repo")
            package = root / "site-packages" / "pando"
            package.mkdir(parents=True)
            self.assertEqual(
                code_sync.stamp_installed_head(sys.executable, root / "repo",
                                               package), head)
            self.assertEqual(graft.installed_code_head(package), head)

    def test_a_stamp_is_never_written_for_a_repo_with_no_commit(self):
        from tools import code_sync

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "site-packages" / "pando"
            package.mkdir(parents=True)
            (root / "repo").mkdir()
            self.assertEqual(
                code_sync.stamp_installed_head(sys.executable, root / "repo",
                                               package), "")
            self.assertEqual(list(package.iterdir()), [])

    def _load_observatory(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "observatory_under_test",
            Path(__file__).resolve().parents[1] / "tools" / "observatory.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_page_marks_who_is_running_old_code(self):
        obs = self._load_observatory()

        def row(name: str, head: str) -> dict:
            return {"name": name, "roots": ["mac"], "lineage": "laya-en",
                    "host": "127.0.0.1", "capability": {"code_head": head},
                    "activity": None, "seen": None, "stats": {}, "role": "decision",
                    "chat": None, "pubkey": ""}

        rows = [row("mac-laya-int8", "a" * 40), row("pi5-laya", "a" * 40),
                row("win-laya-int8", "b" * 40), row("win-gemma-12b", "")]
        shown = {item["name"]: item for item in obs.present_people(rows)}
        # The fleet's commit is what most members report.
        self.assertEqual(shown["mac-laya-int8"]["code"], "a" * 7)
        self.assertFalse(shown["mac-laya-int8"]["code_stale"])
        # A member naming another commit is behind, and says so on the page.
        self.assertEqual(shown["win-laya-int8"]["code"], "b" * 7)
        self.assertTrue(shown["win-laya-int8"]["code_stale"])
        # A member that cannot name one predates the field: unknown, not fine.
        self.assertEqual(shown["win-gemma-12b"]["code"], "未知")
        self.assertTrue(shown["win-gemma-12b"]["code_unknown"])

    def test_the_page_shows_how_old_each_member_s_latest_vote_is(self):
        """A reactor that is running can still be answering yesterday.

        2026-10-19: win-laya-int8 appended a reaction every two minutes while
        every target it named was 35 hours old. Its process was alive and its
        vote was worthless, and nothing on the page said so. The *newest*
        target a member answered is the only record that separates "voting"
        from "voting on yesterday", so it gets a column of its own, with the
        unit written into every value the way the other columns do it.
        """
        obs = self._load_observatory()
        now = 1_800_000_000.0

        def member(name: str, role: str) -> dict:
            return {"node_id": name, "roots": ["mac"], "lineage": "laya-en",
                    "host": "127.0.0.1", "capability": {}, "stats": {},
                    "role": role}

        def vote(node: str, age_s: float) -> dict:
            stamp = int(now - age_s)
            return {"node_id": node, "target_id": "mac-proposer-r1-%d" % stamp,
                    "checkpoint": "laya-en", "stance": "赞同"}

        rows = obs.build_people(
            [member("mac-laya-int8", "decision"),
             member("win-laya-int8", "decision"),
             member("pi5-laya", "decision"),
             member("mac-llm-8b-local", "proposal")],
            [], [],
            [vote("mac-laya-int8", 1800.0),        # 0.5 h: answering today
             vote("win-laya-int8", 35.4 * 3600.0),  # behind, like the real node
             vote("win-laya-int8", 40.0 * 3600.0)],  # older: newest must win
            now=now)
        shown = {row["name"]: row for row in obs.present_people(rows)}
        self.assertEqual(shown["mac-laya-int8"]["fresh"], "0.5 小时")
        self.assertFalse(shown["mac-laya-int8"]["fresh_stale"])
        # A member answering yesterday is behind, and says so on the page.
        self.assertEqual(shown["win-laya-int8"]["fresh"], "35.4 小时")
        self.assertTrue(shown["win-laya-int8"]["fresh_stale"])
        # A decision member with no vote at all is a finding, not a blank.
        self.assertEqual(shown["pi5-laya"]["fresh"], "无票")
        self.assertTrue(shown["pi5-laya"]["fresh_stale"])
        # A proposer is not expected to vote, so it stays unnamed, not bad.
        self.assertEqual(shown["mac-llm-8b-local"]["fresh"], "—")
        self.assertFalse(shown["mac-llm-8b-local"]["fresh_stale"])
        # Every value carries its unit, so the column can be audited.
        self.assertEqual(obs.audit_view({"people": obs.present_people(rows),
                                         "messages": []}), [])


class TestPi5UnitTemplates(unittest.TestCase):
    """The pi5 roster must live in the repo, not only on the SD card.

    2026-10-19: the live pi5 ran four user units that had no template here --
    chat, seed intake, and the ten-minute code fast-forward among them. A node
    rebuilt from this repo would have looked healthy and been unable to talk,
    to receive a seed, or to update itself: exactly the parts that let it keep
    running when no cloud agent is watching. A missing template stays invisible
    until the rebuild, so the roster is checked here instead.
    """

    #: Units a rebuilt pi5 must end up with. Adding a resident role to the node
    #: without adding its template here is the bug this list exists to catch.
    ROSTER = (
        "pandosi-llama.service",
        "pandosi-root.service",
        "pandosi-stem-text.service",
        "pandosi-stem-laya.service",
        "pandosi-stem-llm.service",
        "pandosi-react.service",
        "pandosi-think.service",
        "pandosi-think-watchdog.service",
        "pandosi-chat.service",
        "pandosi-seed-pull.service",
        "pandosi-code-sync.service",
        "pandosi-execute.service",
    )

    def _units(self) -> dict:
        root = Path(__file__).resolve().parents[1] / "tools" / "pi5-systemd"
        return {path.name: path.read_text(encoding="utf-8")
                for path in sorted(root.iterdir())
                if path.suffix in (".service", ".timer")}

    def test_the_roster_is_complete_and_every_template_is_a_unit(self):
        units = self._units()
        for name in self.ROSTER:
            self.assertIn(name, units, f"no template for {name}")
        for name, text in units.items():
            self.assertIn("[Unit]", text, name)
            if name.endswith(".service"):
                self.assertIn("[Service]", text, name)
                self.assertIn("ExecStart=", text, name)
            else:
                self.assertIn("[Timer]", text, name)
                self.assertIn("OnUnitActiveSec=", text, name)

    def test_a_writer_never_loses_a_message_to_a_mailbox_partition(self):
        """Every pi5 unit that posts to the colony mailbox needs a fallback.

        2026-10-19: the pi5 react unit was the last writer with no fallback.
        Win's supervisor and the Mac's launchd both had one, and pi5's own
        proposer had one -- so this was a single missed flag, found by reading
        ``systemctl show pandosi-react.service`` on the live board, not by a
        test. A reaction posted to the fallback replicates back to the primary
        within one replication pass; that is what "queued" means here. A
        reaction posted nowhere is gone, and pi5 hosts a decision model whose
        vote counts toward quorum.
        """
        units = self._units()
        writers = [name for name, text in units.items()
                   if "pando.reactor" in text or "pando.proposer" in text
                   or "pando.messenger" in text]
        self.assertTrue(writers, "no pi5 writer units at all")
        for name in writers:
            self.assertIn("--mailbox-fallback", units[name],
                          f"{name} can lose a message to a partition")

    def test_a_timer_cannot_run_a_job_that_is_not_a_template(self):
        units = self._units()
        timers = [name for name in units if name.endswith(".timer")]
        self.assertTrue(timers, "no timers at all")
        missing = []
        for name in timers:
            for unit in re.findall(r"^Unit=(\S+)$", units[name], re.M):
                if unit not in units:
                    missing.append(f"{name} -> {unit}")
        self.assertEqual(missing, [])

    def test_the_fast_forward_only_restarts_units_the_guard_allows(self):
        """A ten-minute job that may restart a model would OOM a 4 GB pi5."""
        from tools import code_sync

        units = self._units()
        restarted = set(re.findall(
            r"--restart\s+(\S+)", units["pandosi-code-sync.service"]))
        self.assertTrue(restarted, "the fast-forward restarts nothing")
        self.assertEqual(restarted - set(code_sync.SAFE_UNITS), set())
        # And what it does restart must be a unit this repo can rebuild.
        self.assertEqual(restarted - set(units), set())

    def test_every_voter_stem_is_restarted_by_the_fast_forward(self):
        """The allow-list and the deployed unit must not drift apart.

        ``SAFE_UNITS`` is the guard and ``--restart`` is the order. A unit
        added to one and not the other fails silently in opposite directions:
        in the guard only, it is never restarted; in the order only,
        ``restart_units`` refuses the whole batch and *no* unit restarts.

        Measured 2026-10-19: all three pi5 stems reported ``code_head=None``
        with the checkout at ``9e69dea``, while their unit files had not been
        touched since Oct 17. The stems are the voters -- restarting them is
        a re-registration, and is charged at zero model memory.
        """
        from tools import code_sync

        restarted = set(re.findall(
            r"--restart\s+(\S+)", self._units()["pandosi-code-sync.service"]))
        wants = {name for name in code_sync.SAFE_UNITS
                 if name.startswith("pandosi-stem-")}
        self.assertEqual(wants, {
            "pandosi-stem-text.service",
            "pandosi-stem-laya.service",
            "pandosi-stem-llm.service",
        })
        self.assertEqual(wants - restarted, set(),
                         "a stem the guard allows is never actually restarted")
        self.assertNotIn("pandosi-llama.service", restarted,
                         "the fast-forward must never reload a model")


    def test_the_updater_running_on_pi5_can_receive_its_own_fixes(self):
        """The ten-minute job must not run a hand-copied, frozen updater.

        It runs a copy outside the checkout (it has to: the checkout is what
        gets rewritten), so the copy is told where its successor lives. Source
        inside the checkout, target outside it -- and adopt only what compiles,
        which ``refresh_self`` enforces.
        """
        units = self._units()
        line = units["pandosi-code-sync.service"]
        source = re.search(r"--self-source\s+(\S+)", line)
        target = re.search(r"--self-target\s+(\S+)", line)
        self.assertIsNotNone(source, "the updater is never told about its successor")
        self.assertIsNotNone(target, "no deployed copy to refresh")
        self.assertTrue(source.group(1).startswith("/home/admin/pandosi-src/"))
        self.assertTrue(target.group(1).startswith("/home/admin/.pandosi/"))
        # And the file it adopts is the one this repo can rebuild.
        self.assertEqual(Path(source.group(1)).name, "code_sync.py")


class TestCodeDriftCheck(unittest.TestCase):
    """tools/code_drift_check.py - "was the code that thought the code we run?"

    On 2026-10-19 win-laya-int8 answered stale proposals for ~39 hours. Its
    process was up, its ledger grew, and ``/nodes`` already carried
    ``capability.code_head`` - the commit it booted from. Nothing read it. A
    green column nobody is told to look at is a docstring, and that is the
    defect this tool closes.

    The contract it must keep: being behind is *not* the signal, because one
    push makes the whole fleet behind for a watcher interval. Being behind and
    *staying* behind is. A member reporting a commit this checkout has never
    seen is ``diverged``, never "behind" - that would be a lookup failure
    wearing a measurement's clothes.

    A root that reaches this tool as a file (``--root-json``) carries the same
    weight as one it dialled, but only while the copy is fresh. A mirror nobody
    refreshed is this machine's blind spot; filed as a member fault it would
    page a human about a host that may be perfectly current.
    """

    def _tool(self):
        import importlib.util
        path = (Path(__file__).resolve().parents[1]
                / "tools" / "code_drift_check.py")
        spec = importlib.util.spec_from_file_location("code_drift_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    #: The checkout these tests measure is one this class builds, not the one
    #: the developer happens to be sitting in. Exercising "behind by N" needs
    #: real ancestry -- ``HEAD~1`` -- and that name does not exist in a
    #: shallow clone: ``actions/checkout@v7.0.1`` fetches at depth 1 by default.
    #: On 2026-10-19 this class passed on the machine that pushed it and
    #: errored on the runner with ``rev-parse HEAD~1`` exit 128, ten tests at
    #: a time, for twenty consecutive pushes. Three fixture commits have the
    #: ancestry wherever it runs, and no fact about the repository under test
    #: can turn them green.
    _fixture_dir: "tempfile.TemporaryDirectory[str] | None" = None
    _fixture: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_dir = tempfile.TemporaryDirectory(prefix="code-drift-repo-")
        cls._fixture = Path(cls._fixture_dir.name) / "checkout"
        cls._fixture.mkdir()
        cls._fixture_git("init", "-b", "main")
        for index in range(3):
            (cls._fixture / "README").write_text(
                f"fixture {index}\n", encoding="utf-8")
            cls._fixture_git("add", "README")
            cls._fixture_git("commit", "-m", f"fixture {index}")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fixture_dir is not None:
            cls._fixture_dir.cleanup()
            cls._fixture_dir = None
        cls._fixture = None

    @classmethod
    def _fixture_git(cls, *args: str) -> None:
        """One commit chain that owes nothing to the host's identity or env."""
        _git_in(cls._fixture, "-c", "user.name=fixture",
                "-c", "user.email=fixture@localhost",
                "-c", "commit.gpgsign=false", *args)

    def _repo(self) -> Path:
        """The fixture checkout. Never this working tree; see ``_fixture``."""
        assert self._fixture is not None, "setUpClass did not build a fixture"
        return self._fixture

    def test_the_measured_checkout_is_a_fixture_not_this_working_tree(self):
        """The defect stated as a test: history depth must not decide it.

        The runner clones at depth 1 and this machine has the whole history,
        so a test that measures the tree it is running in cannot mean the same
        thing in both places. Pinned here so an edit that reaches back for
        ``Path(__file__).resolve().parents[1]`` fails on the developer's
        machine, which is the only place it can still be caught cheaply.
        """
        here = Path(__file__).resolve().parents[1].resolve()
        self.assertNotEqual(self._repo().resolve(), here)

    def test_the_fixture_ignores_a_git_dir_the_environment_leaks(self):
        """A pre-push hook exports GIT_DIR; ``-C`` must still mean ``-C``.

        On 2026-10-19 this fixture inherited that variable and committed its
        three revisions into this very worktree, because ``git -C`` obeys
        ``GIT_DIR`` when it is set. Naming the repo in argv and stripping the
        environment is the fix, so this poisons ``GIT_DIR`` at the outer
        checkout and asserts the temp repo is the one that grows.
        """
        outer = Path(__file__).resolve().parents[1]
        probe = ["git", "-C", str(outer), "rev-parse", "HEAD"]
        before = subprocess.run(probe, capture_output=True, text=True,
                                env=_git_env()).stdout.strip()
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "elsewhere"
            other.mkdir()
            saved = os.environ.get("GIT_DIR")
            os.environ["GIT_DIR"] = str(outer / ".git")
            try:
                _git_in(other, "init", "-b", "main")
                (other / "README").write_text("elsewhere\n", encoding="utf-8")
                _git_in(other, "add", "README")
                _git_in(other, "-c", "user.name=t", "-c", "user.email=t@x",
                        "commit", "-m", "elsewhere")
            finally:
                if saved is None:
                    os.environ.pop("GIT_DIR", None)
                else:
                    os.environ["GIT_DIR"] = saved
            self.assertTrue((other / ".git").exists(),
                            "the temp repo must be the one that got a repo")
        after = subprocess.run(probe, capture_output=True, text=True,
                               env=_git_env()).stdout.strip()
        self.assertEqual(after, before, "the outer checkout must not move")

    def _rev(self, ref: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self._repo()), "rev-parse", f"{ref}^{{commit}}"],
            capture_output=True, text=True, check=True, env=_git_env())
        return proc.stdout.strip()

    @staticmethod
    def _run(mod, argv):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return mod.main(argv)

    def test_a_commit_this_checkout_never_saw_is_diverged_not_behind(self):
        """An unknown commit is a stranger, not a laggard."""
        mod = self._tool()
        repo = self._repo()
        status, behind, why = mod.classify(repo, mod.head_of(repo), "0" * 40)
        self.assertEqual(status, mod.DIVERGED)
        self.assertIsNone(behind)
        self.assertIn("does not know", why)

    def test_a_member_that_reports_no_head_is_unknown(self):
        """A process too old to stamp its commit must not read as "current"."""
        mod = self._tool()
        repo = self._repo()
        status, behind, why = mod.classify(repo, mod.head_of(repo), "")
        self.assertEqual(status, mod.UNKNOWN)
        self.assertIsNone(behind)
        self.assertIn("no code_head", why)

    def test_a_member_on_a_real_ancestor_is_behind_by_the_count(self):
        mod = self._tool()
        repo = self._repo()
        status, behind, _ = mod.classify(repo, mod.head_of(repo),
                                         self._rev("HEAD~1"))
        self.assertEqual(status, mod.BEHIND)
        self.assertGreaterEqual(behind, 1)

    def test_a_member_ahead_of_this_checkout_is_diverged(self):
        """A comparison base behind its own member must not call it late."""
        mod = self._tool()
        repo = self._repo()
        older_tip = self._rev("HEAD~1")
        status, behind, why = mod.classify(repo, older_tip, mod.head_of(repo))
        self.assertEqual(status, mod.DIVERGED)
        self.assertIsNone(behind)
        self.assertIn("ahead", why)

    def test_unpushed_local_commits_do_not_start_the_fleet_grace_clock(self):
        """Only a published commit can make a member late.

        A local commit starts the clock before anyone outside this checkout
        can pull it. On 2026-10-19 that produced a real Windows false alarm:
        the alarm fired after fifteen minutes, and Windows converged to the
        newly pushed tip two minutes later. The comparison base must be the
        branch upstream; HEAD remains the fallback for a repo with no remote.
        """
        mod = self._tool()
        with tempfile.TemporaryDirectory(prefix="code-drift-upstream-") as tmp:
            root = Path(tmp)
            bare, seed = root / "origin.git", root / "seed"
            bare.mkdir()
            seed.mkdir()
            _git_in(bare, "init", "--bare")
            _git_in(seed, "init", "-b", "main")
            (seed / "README").write_text("published\n", encoding="utf-8")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "add", "README")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-m", "published")
            _git_in(seed, "remote", "add", "origin", str(bare))
            _git_in(seed, "push", "-u", "origin", "main")
            published = mod.head_of(seed)

            (seed / "README").write_text("local only\n", encoding="utf-8")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "add", "README")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-m", "local")

            self.assertNotEqual(mod.head_of(seed), published)
            self.assertEqual(mod.deploy_tip(seed), published)

    def test_a_peer_bundle_head_can_be_the_deployable_tip(self):
        """Peer deployment is a real path, not an unpublished private commit.

        The 2026-10-21 alarm stayed open for days because the Mac fleet ran a
        local commit already served as ``/code.bundle`` while GitHub's
        ``origin/main`` was two commits behind. Every Mac member was then
        classified as permanently *ahead*. A tip file makes the served bundle
        part of the comparison base, while the upstream-only test above keeps
        a genuinely private commit from starting the grace clock.
        """
        mod = self._tool()
        with tempfile.TemporaryDirectory(prefix="code-drift-tip-file-") as tmp:
            root = Path(tmp)
            bare, seed = root / "origin.git", root / "seed"
            bare.mkdir()
            seed.mkdir()
            _git_in(bare, "init", "--bare")
            _git_in(seed, "init", "-b", "main")
            (seed / "README").write_text("published\n", encoding="utf-8")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "add", "README")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-m", "published")
            _git_in(seed, "remote", "add", "origin", str(bare))
            _git_in(seed, "push", "-u", "origin", "main")

            (seed / "README").write_text("served locally\n", encoding="utf-8")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "add", "README")
            _git_in(seed, "-c", "user.name=fixture",
                    "-c", "user.email=fixture@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-m", "served")
            served = mod.head_of(seed)
            tip_file = root / "code.bundle.head"
            tip_file.write_text(served + "\n", encoding="utf-8")

            self.assertEqual(mod.deploy_tip(seed, [tip_file]), served)

    def test_a_fresh_drift_lags_and_only_persisting_drift_pages(self):
        """The grace window, end to end through main().

        A push makes every member behind for one interval; paging on that
        would page the fleet on every commit. The first poll is a lag, the
        same reported commit still behind after the grace is the alarm.
        """
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")
        mod.fetch_nodes = lambda url, token, timeout=mod.DEFAULT_TIMEOUT_S: {
            "nodes": {"mac-text": {"capability": {"code_head": older}}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo), "--root", "mac", "http://x", "",
                    "--min-persist-seconds", "900",
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", str(Path(tmp) / "snap.json"),
                    "--json"]
            self.assertEqual(self._run(mod, argv + ["--now", "1000.0"]), 0)
            self.assertEqual(self._run(mod, argv + ["--now", "1901.0"]), 2)

    def test_a_lag_inside_the_grace_still_exits_zero(self):
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")
        mod.fetch_nodes = lambda url, token, timeout=mod.DEFAULT_TIMEOUT_S: {
            "nodes": {"mac-text": {"capability": {"code_head": older}}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo), "--root", "mac", "http://x", "",
                    "--min-persist-seconds", "900",
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, argv + ["--now", "1000.0"]), 0)
            self.assertEqual(self._run(mod, argv + ["--now", "1899.0"]), 0)

    def test_moving_to_a_different_commit_restarts_the_grace_clock(self):
        """Moving is progress; an inherited clock would punish it.

        A member that upgrades from twenty commits behind to one must not keep
        serving the twenty-commit fault it just fixed.
        """
        mod = self._tool()
        a, b = self._rev("HEAD~1"), self._rev("HEAD~2")
        now = 100_000.0
        state = {"mac-text": {"first_seen": now - 10_000, "head": a}}
        rows = {"mac-text": {"member": "mac-text", "reported": b,
                             "raw": mod.BEHIND, "behind": 2, "why": "2 behind"}}
        mod.track(state, rows, now, 900)
        self.assertEqual(rows["mac-text"]["status"], mod.LAG)
        self.assertEqual(state["mac-text"]["first_seen"], now)
        self.assertEqual(state["mac-text"]["head"], b)

    def test_catching_up_forgets_the_member_so_a_later_drift_pages_again(self):
        """Memory is scoped to one drift, never to one member forever."""
        mod = self._tool()
        older, tip = self._rev("HEAD~1"), mod.head_of(self._repo())
        now = 100_000.0
        state = {"mac-text": {"first_seen": 0.0, "head": older}}
        rows = {"mac-text": {"member": "mac-text", "reported": tip,
                             "raw": mod.CURRENT, "why": "on the tip"}}
        mod.track(state, rows, now, 900)
        self.assertEqual(rows["mac-text"]["status"], mod.OK)
        self.assertNotIn("mac-text", state)
        again = {"mac-text": {"member": "mac-text", "reported": older,
                              "raw": mod.BEHIND, "behind": 1,
                              "why": "1 behind"}}
        mod.track(state, again, now + 10, 900)
        self.assertEqual(again["mac-text"]["status"], mod.LAG)
        self.assertEqual(state["mac-text"]["first_seen"], now + 10)

    def test_a_first_seen_of_zero_is_a_timestamp_not_a_missing_value(self):
        """``x or default`` would re-date a valid epoch on every poll.

        The state file is written by machines and read back by machines, so a
        falsy-but-present stamp must not be read as absent: that resets the
        grace clock forever, which is a watchdog that can never page.
        """
        mod = self._tool()
        state = {"m": {"first_seen": 0.0, "head": "abc"}}
        rows = {"m": {"member": "m", "reported": "abc", "raw": mod.BEHIND,
                      "behind": 5, "why": "5 behind"}}
        mod.track(state, rows, 10_000.0, 900)
        self.assertEqual(rows["m"]["status"], mod.STALE)

    def test_a_root_we_cannot_read_is_rc1_not_a_member_fault(self):
        """Our blind spot is not somebody's lag, and must not be filed as one."""
        mod = self._tool()
        repo = self._repo()
        tip = mod.head_of(repo)

        def fetch(url, token, timeout=mod.DEFAULT_TIMEOUT_S):
            if "ghost" in url:
                raise urllib.error.URLError("no route to host")
            return {"nodes": {"mac-text": {"capability": {"code_head": tip}}}}

        mod.fetch_nodes = fetch
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--repo", str(repo),
                    "--root", "mac", "http://mac", "",
                    "--root", "ghost", "http://ghost", "",
                    "--retry-delay", "0",
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, base + ["--now", "1000.0"]), 1)

    def test_a_stale_member_outranks_an_unreadable_root(self):
        """Members first: a real fault pages even while a peer is down."""
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")

        def fetch(url, token, timeout=mod.DEFAULT_TIMEOUT_S):
            if "ghost" in url:
                raise urllib.error.URLError("no route to host")
            return {"nodes": {"mac-text": {"capability": {"code_head": older}}}}

        mod.fetch_nodes = fetch
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            state.write_text(json.dumps(
                {"mac-text": {"first_seen": 9_000.0, "head": older}}),
                encoding="utf-8")
            base = ["--repo", str(repo),
                    "--root", "mac", "http://mac", "",
                    "--root", "ghost", "http://ghost", "",
                    "--retry-delay", "0",
                    "--min-persist-seconds", "900",
                    "--state-file", str(state),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, base + ["--now", "10000.0"]), 2)


    def test_a_dropped_packet_is_retried_and_an_http_error_is_not(self):
        """A transport blip must not delete a host from the page.

        On 2026-10-19 the first drift page named only the Mac's own two
        members. Win and pi5 dropped a packet at that instant, so the three
        frozen members the tool exists to name were the three missing from it.
        "We could not see them" and "they are gone" are different claims and
        only one of them is what a retry buys back. A 403 is an answer, not a
        dropped packet, so it is never retried.
        """
        mod = self._tool()
        calls = []
        slept = []

        def flaky(url, token, timeout=mod.DEFAULT_TIMEOUT_S):
            calls.append(url)
            if len(calls) < 3:
                raise urllib.error.URLError("no route to host")
            return {"nodes": {}}

        real_sleep = mod.time.sleep
        try:
            mod.fetch_nodes = flaky
            mod.time.sleep = slept.append
            got = mod.fetch_nodes_retrying("http://x/nodes", None,
                                           retries=2, delay=0.5)
        finally:
            mod.time.sleep = real_sleep
        self.assertEqual(got, {"nodes": {}})
        self.assertEqual(len(calls), 3, "two retries after the first failure")
        self.assertEqual(slept, [0.5, 0.5])

        calls.clear(); slept.clear()

        def forbidden(url, token, timeout=mod.DEFAULT_TIMEOUT_S):
            calls.append(url)
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

        try:
            mod.fetch_nodes = forbidden
            mod.time.sleep = slept.append
            with self.assertRaises(urllib.error.HTTPError) as caught:
                mod.fetch_nodes_retrying("http://x/nodes", None,
                                         retries=2, delay=0.5)
        finally:
            mod.time.sleep = real_sleep
        caught.exception.close()  # HTTPError is file-like; close it, don't leak
        self.assertEqual(len(calls), 1, "a 403 is a definitive answer")
        self.assertEqual(slept, [])

    def test_one_unreadable_moment_does_not_hide_a_frozen_member(self):
        """End to end: the page names the frozen member, not the wifi."""
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")
        attempts = {"n": 0}

        def fetch(url, token, timeout=mod.DEFAULT_TIMEOUT_S):
            if "8790" in url:
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise urllib.error.URLError("no route to host")
                return {"nodes": {"win-laya-int8": {
                    "capability": {"code_head": older}}}}
            return {"nodes": {"mac-text": {
                "capability": {"code_head": mod.head_of(repo)}}}}

        real_sleep = mod.time.sleep
        try:
            mod.fetch_nodes = fetch
            mod.time.sleep = lambda _s: None
            rows, unreadable = mod.survey(
                repo, mod.head_of(repo),
                [("mac", "http://127.0.0.1:8765", ""),
                 ("win", "http://192.168.31.100:8790", "")],
                retries=1, retry_delay=0.0)
        finally:
            mod.time.sleep = real_sleep
        self.assertEqual(unreadable, [], "one blip is not a blind spot")
        self.assertEqual(attempts["n"], 2, "the second attempt answered")
        self.assertEqual(rows["win-laya-int8"]["raw"], mod.BEHIND)

    def _mirror(self, tmp, members, mirrored_at=None) -> Path:
        """A root table as the copier leaves it: /nodes plus when it was read."""
        payload: dict = {"nodes": members}
        if mirrored_at is not None:
            payload["mirrored_at"] = mirrored_at
        path = Path(tmp) / "pi5-root.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_fresh_mirror_measures_the_member_behind_it(self):
        """A copied answer is an answer: pi5 gets rows, not an exemption."""
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")
        members = {"pandosi-pi5-x": {"capability": {"code_head": older}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo),
                    "--root-json", "pi5", str(self._mirror(tmp, members, 1100.0)),
                    "--min-persist-seconds", "900",
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            # Inside the grace window a push is a lag, not a fault.
            self.assertEqual(self._run(mod, argv + ["--now", "1100.0"]), 0)
            # The copier refreshes the file; the member is still behind, so now
            # it has *stayed* behind and the timer may page.
            self._mirror(tmp, members, 2900.0)
            self.assertEqual(self._run(mod, argv + ["--now", "2900.0"]), 2)

    def test_a_mirror_nobody_refreshed_is_our_blind_spot(self):
        """A dead copier must never be filed as the peer falling behind.

        rc=1 is the whole point: the host is invisible to us, which is our
        fault to report, while rc=2 would name a member that may be current.
        """
        mod = self._tool()
        repo = self._repo()
        older = self._rev("HEAD~1")
        members = {"pandosi-pi5-x": {"capability": {"code_head": older}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo),
                    "--root-json", "pi5", str(self._mirror(tmp, members, 1000.0)),
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, argv + ["--now", "2000.0"]), 1)

    def test_a_mirror_that_cannot_say_its_age_is_not_evidence(self):
        """Fail closed: an undated copy could be last week's answer."""
        mod = self._tool()
        repo = self._repo()
        tip = mod.head_of(repo)
        members = {"pandosi-pi5-x": {"capability": {"code_head": tip}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo),
                    "--root-json", "pi5", str(self._mirror(tmp, members)),
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, argv + ["--now", "2000.0"]), 1)

    def test_a_hand_made_table_can_opt_out_of_the_freshness_claim(self):
        """The escape hatch, so a one-off file is not permanently unreadable."""
        mod = self._tool()
        repo = self._repo()
        tip = mod.head_of(repo)
        members = {"pandosi-pi5-x": {"capability": {"code_head": tip}}}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--repo", str(repo),
                    "--root-json", "pi5", str(self._mirror(tmp, members)),
                    "--root-json-max-age", "0",
                    "--state-file", str(Path(tmp) / "state.json"),
                    "--snapshot-file", "", "--json"]
            self.assertEqual(self._run(mod, argv + ["--now", "2000.0"]), 0)


