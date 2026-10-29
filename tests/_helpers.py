"""Shared test helpers extracted from the old monolithic test_smoke.py."""

import os
import subprocess


def _git_env() -> dict:
    """An environment that names no repository of its own.

    ``git`` hands a pre-push hook ``GIT_DIR``, and a ``git -C`` command that
    inherits it obeys the variable instead of ``-C``. On 2026-10-19 the drift
    fixture below inherited it and wrote its three README commits into this
    worktree instead of the temp dir, and the same variable broke the hook's
    own probe. Test git calls therefore name their repo in argv and inherit
    nothing that points somewhere else.
    """
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return env


def _git_in(repo, *args: str) -> None:
    """One git command in ``repo``, with the host's GIT_* variables removed."""
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True, env=_git_env())
