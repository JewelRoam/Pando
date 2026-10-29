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


class TestLocalCi(unittest.TestCase):
    def _module(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "local_ci.py"
        spec = importlib.util.spec_from_file_location("local_ci_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_local_gate_names_every_check_the_workflow_runs(self):
        module = self._module()
        self.assertEqual(module.drift(module.WORKFLOW.read_text(encoding="utf-8")), [])

    def test_the_gate_mirrors_the_swarm_workflow_too(self):
        """``swarm.yml`` fires on pushes touching coordinator/worker; mirror it.

        Until 2026-10-19 this gate mirrored only ``ci.yml``. A push touching
        ``pando/coordinator.py`` starts nine swarm checks on the runner and none
        here, so the gate could pass and the user still got a red run. A gate
        that mirrors one of two workflows is not a gate, it is a coin flip.
        """
        module = self._module()
        self.assertTrue(module.SWARM_WORKFLOW.is_file())
        problems = module.drift(
            module.SWARM_WORKFLOW.read_text(encoding="utf-8"),
            module.SWARM_CHECKS, "swarm.yml")
        self.assertEqual(problems, [])
        self.assertGreaterEqual(len(module.SWARM_CHECKS), 9)

    def test_the_gate_says_out_loud_what_it_cannot_run_here(self):
        """A skipped workflow must be printed, never silently counted as green.

        ``service.yml`` needs a Linux user systemd bus. macOS has launchd, so
        this machine cannot run it. The honest options are to skip it loudly
        or to lie; the gate prints the reason on every run.
        """
        module = self._module()
        self.assertIn("service.yml", module.PLATFORM_SKIPS)
        reason = module.PLATFORM_SKIPS["service.yml"].lower()
        self.assertIn("systemd", reason)
        self.assertIn("launchd", reason)

    def test_the_gate_also_runs_where_the_history_is_missing(self):
        """The mirror that would have caught 2026-10-19 before the push.

        On 2026-10-19 ten drift tests reached for ``HEAD~1`` and errored on
        the runner, twenty pushes running, while this machine stayed green --
        full history here, one commit there. A gate that only ever ran here
        cannot tell the two apart, so it must run the suite in a depth-1
        checkout too, and it must be wired into ``main`` to matter.
        """
        module = self._module()
        self.assertTrue(callable(getattr(module, "shallow_parity", None)))
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("shallow_parity(python)", source)

    def test_the_gate_uses_the_ci_interpreter_not_the_shell_default(self):
        """A green 3.14 run must not stand in for the 3.12 runner.

        The workflows pin 3.12. Before this check the gate launched every
        child with ``sys.executable``, so a developer shell on 3.14 could pass
        while 3.12 failed on syntax or a stdlib difference. The configured
        interpreter must win and every child argv must name it.
        """
        from unittest import mock

        module = self._module()
        with mock.patch.dict(os.environ,
                             {"PANDO_CI_PYTHON": "/ci/python312"}, clear=False), \
                mock.patch.object(module, "_python_version", return_value="3.12"), \
                mock.patch.object(module.shutil, "which", return_value=None):
            self.assertEqual(module._ci_python(), "/ci/python312")

        self.assertEqual(
            module._argv("unittest", ["-m", "unittest"], "/ci/python312")[0],
            "/ci/python312")
        self.assertEqual(
            module._argv("py_compile", ["-m", "py_compile"], "/ci/python312")[0],
            "/ci/python312")
        self.assertEqual(
            module._argv("import numpy", [], "/ci/python312")[0],
            "/ci/python312")

    def test_the_gate_refuses_a_non_312_interpreter(self):
        """A wrong configured interpreter is a setup error, not a fallback.

        Falling back to ``sys.executable`` would recreate the false-green this
        gate exists to prevent. The failure must name the expected version.
        """
        from unittest import mock

        module = self._module()
        with mock.patch.dict(os.environ,
                             {"PANDO_CI_PYTHON": "/wrong/python"}, clear=False), \
                mock.patch.object(module, "_python_version", return_value="3.14"), \
                mock.patch.object(module.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "3.12"):
                module._ci_python()

    def test_depth_one_parity_uses_the_ci_interpreter(self):
        """The depth-1 suite must run under the same Python as the runner."""
        from types import SimpleNamespace
        from unittest import mock

        module = self._module()
        with mock.patch.object(module, "_shallow_checkout"), \
                mock.patch.object(module.subprocess, "run",
                                  return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(module.shallow_parity("/ci/python312"), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[:3],
                         ["/ci/python312", "-m", "unittest"])

    def test_workflow_python_version_matches_the_gate(self):
        """The workflow pin and local gate must name one Python version."""
        module = self._module()
        workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        versions = set()
        for path in workflows.glob("*.yml"):
            for line in path.read_text(encoding="utf-8").splitlines():
                match = re.match(r'\s*python-version:\s*"([^"]+)"\s*$', line)
                if match:
                    versions.add(match.group(1))
        self.assertEqual(versions, {module.CI_PYTHON})

    def test_the_gate_runs_children_without_an_inherited_repository(self):
        """The gate is a hook, so it inherits GIT_DIR; its children must not.

        A linked worktree exports ``GIT_DIR`` to its hooks (verified on this
        machine). With ``GIT_DIR`` *and* ``GIT_WORK_TREE`` set, even
        ``git clone`` refuses to run -- "working tree already exists" -- so
        the depth-1 parity check would die before testing anything. CI has
        neither variable; the gate strips them so it measures CI's
        environment rather than whatever shell happened to invoke it.
        """
        module = self._module()
        poison = {
            "GIT_DIR": "/nonexistent/repo.git",
            "GIT_WORK_TREE": "/nonexistent/tree",
            "GIT_INDEX_FILE": "/nonexistent/index",
            "GIT_OBJECT_DIRECTORY": "/nonexistent/objects",
            "GIT_COMMON_DIR": "/nonexistent/common",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/nonexistent/alt",
        }
        saved = {key: os.environ.get(key) for key in poison}
        try:
            os.environ.update(poison)
            clean = module._ci_env()
            for key in poison:
                self.assertNotIn(key, clean)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_ci_dependency_pins_are_exact_and_used_by_every_workflow(self):
        """An unpinned install is a different test every morning.

        On 2026-10-19 the runner resolved exactly the same versions as this
        machine, but only by luck. A new transitive release could turn an
        untouched commit red. The constraints file is the one place the
        tested set lives; this test refuses a range, a missing pin, or a
        workflow that installs packages without pointing pip at it.
        """
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / "ci_constraints.py"
        spec = importlib.util.spec_from_file_location("ci_constraints_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pins = module.read_constraints()
        self.assertEqual(pins["onnx"], "1.23.2")
        self.assertEqual(pins["onnxruntime"], "1.31.0")
        self.assertEqual(module.installed_problems(pins), [])
        self.assertEqual(module.workflow_problems(), [])

    def test_runner_and_action_versions_are_pinned(self):
        """The runner image and action code are inputs, so name them exactly.

        On 2026-10-19 GitHub announced that ``ubuntu-latest`` was moving to
        Ubuntu 26. A green commit can turn red when that alias changes. The
        same goes for a floating action tag: it silently changes the code that
        checks out the tree and installs Python. Require an explicit runner
        and a full commit SHA for every action used by these workflows.
        """
        workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        for path in sorted(workflows.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("-latest", text, path.name)
            for line in text.splitlines():
                if "runs-on:" in line:
                    self.assertRegex(line, r"^\s*runs-on:\s+[A-Za-z0-9_.-]+\s*$",
                                     path.name)
            for line in text.splitlines():
                if "uses:" not in line or line.lstrip().startswith("#"):
                    continue
                match = re.search(r"uses:\s+([^\s#]+)", line)
                self.assertIsNotNone(match, f"{path.name}: malformed uses line: {line}")
                action = match.group(1)
                self.assertRegex(
                    action,
                    r"^[^@\s]+@[0-9a-f]{40}$",
                    f"{path.name}: action must be pinned to a full commit SHA: {action}",
                )

    def test_pre_push_runs_the_local_gate(self):
        hook = Path(__file__).resolve().parents[1] / "tools" / "githooks" / "pre-push"
        text = hook.read_text(encoding="utf-8")
        self.assertIn("tools/local_ci.py", text)
        # The hook must prefer the interpreter that mirrors CI. It also has to
        # work when the checkout has no .venv at all, because a stale or
        # missing developer venv must not be the thing that blocks a push.
        self.assertIn(".venv-ci312/bin/python", text)
        self.assertIn(".venv-ci312/Scripts/python.exe", text)
        self.assertIn("PANDO_CI_PYTHON", text)
        self.assertIn("python3.12", text)
        self.assertNotIn("no project venv", text)
        if os.name != "nt":
            # Windows has no POSIX exec bit (stat never reports 0o111 on
            # NTFS); git runs hooks through sh there, so the bit is moot.
            self.assertTrue(hook.stat().st_mode & 0o111)


