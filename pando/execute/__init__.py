"""The proposal executor: judge adopted proposals, carry out the few it owns.

Split by responsibility:

- :mod:`pando.execute.consts` — stances and the action boundary
- :mod:`pando.execute.judge` — tally, judge, adopted, retry bookkeeping
- :mod:`pando.execute.next` — per-action oldest-adopted selectors
- :mod:`pando.execute.selectors` — which mailbox rows a round may see
- :mod:`pando.execute.actuators` — the stager table for installed verbs
- :mod:`pando.execute.operators_exec` — deploy attempts + catalogue auto-deploy
- :mod:`pando.execute.reconcile` — hosts.json views (no sockets)
- :mod:`pando.execute.seed_export` — sealing + offer bookkeeping
- :mod:`pando.execute.main` — the round itself, CLI entry

Public names are re-exported eagerly for ``from pando.execute import X``.
Private helpers (``_load_reactions`` etc.) are resolved lazily via
``__getattr__`` so tests that patch them on the package keep working without
this package freezing every submodule at import time.
"""
from .consts import ACTUATORS, ADOPT, DECLINE, HOLD, STANCES
from .actuators import INSTALLED_STAGERS
from .judge import (MAX_ATTEMPTS, adopted, deferred_reasons, failure_counts,
                    judge, settle_after_failure, tally)
from .next import (foreign_culture_refines, foreign_meta_proposes,
                   foreign_rebalances, foreign_recruit_invites,
                   foreign_reproduce_spawns, next_culture_refine,
                   next_installed, next_meta_propose, next_policy_set,
                   next_rebalance, next_recruit_invite, next_reproduce_spawn,
                   next_seed_export)
from .operators_exec import auto_deploy_profiles
from .reconcile import known_hosts, pull_destinations
from .seed_export import (addressed_replications, done_ids, export_seed,
                          foreign_adopted_seeds, foreign_policy_sets,
                          foreign_replications, latest_sealed, left_ids,
                          local_replications, offer_replication, received_acks,
                          replication_skip, summarize, write_status)
from .selectors import (DEFAULT_MAILBOXES, DEFAULT_PER_ACTION, DEFAULT_STATE_DIR,
                        STATE_PATHS, execution_selectors, executor_mailboxes,
                        host_mailboxes, local_authors, rows_for_execution)
from .main import (build_parser, dry_run, local_unacted, main, resolve_state,
                   run_once)

__all__ = [name for name in dir() if not name.startswith('_')]

_LAZY = (
    '_load_reactions', '_append_deployment_attempt', '_hosts_table',
    '_seed_module',
)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        for mod in ('main', 'judge', 'operators_exec', 'reconcile',
                    'seed_export', 'selectors'):
            m = importlib.import_module(f'.{mod}', __name__)
            if hasattr(m, name):
                globals()[name] = getattr(m, name)
                return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
