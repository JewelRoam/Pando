"""The stager table for installed verbs.

``INSTALLED_STAGERS`` is the only place that knows how to carry out a verb
nobody wrote code for: it binds the verb to the actuator it reuses.  Kept
equal to ``pando.operators.BINDABLE`` by a test, so a verb can never be
installable and unstaged at once.

``ACTUATORS`` — the whole action boundary — lives in
:mod:`pando.execute.consts`; this module re-exports it for convenience.
"""
from ..culture import DEFAULT_OUTBOX as DEFAULT_CULTURE_OUTBOX
from ..culture import stage as stage_culture
from ..invite import DEFAULT_OUTBOX as DEFAULT_INVITE_OUTBOX
from ..invite import stage as stage_invite
from ..operators import DEFAULT_OUTBOX as DEFAULT_OPERATOR_OUTBOX
from ..operators import OperatorError
from ..operators import stage as stage_operator
from ..rebalance import DEFAULT_OUTBOX as DEFAULT_REBALANCE_OUTBOX
from ..rebalance import stage as stage_rebalance
from ..spawn import DEFAULT_OUTBOX as DEFAULT_SPAWN_OUTBOX
from ..spawn import stage as stage_spawn

from .consts import ACTUATORS  # noqa: F401  (re-exported)

#: How an installed verb is staged: the actuator it binds to, the outbox that
#: actuator writes, and the flag that relocates it. A new verb has no stager of
#: its own -- reusing the one it binds to is the whole point -- so this table is
#: the only place that knows how to carry out a verb nobody wrote code for.
#: Kept equal to ``pando.operators.BINDABLE`` by a test, so a verb can never be
#: installable and unstaged at once.
INSTALLED_STAGERS: dict[str, tuple] = {
    "culture.refine": (stage_culture, DEFAULT_CULTURE_OUTBOX, "culture_outbox"),
    "recruit.invite": (stage_invite, DEFAULT_INVITE_OUTBOX, "invite_outbox"),
    "reproduce.spawn": (stage_spawn, DEFAULT_SPAWN_OUTBOX, "spawn_outbox"),
    "resource.rebalance": (stage_rebalance, DEFAULT_REBALANCE_OUTBOX,
                           "rebalance_outbox"),
    "meta.propose": (stage_operator, DEFAULT_OPERATOR_OUTBOX,
                     "operator_outbox"),
}


def stage_bound(proposal, args, identity) -> dict:
    """Stage an installed verb through the actuator its record binds to."""
    entry = INSTALLED_STAGERS.get(proposal.action)
    if entry is None:
        raise OperatorError(
            f"no executor stages {proposal.action}; an installed verb may only "
            "bind to one of " + ", ".join(sorted(INSTALLED_STAGERS)))
    stager, default_outbox, arg_name = entry
    return stager(proposal, identity=identity,
                  outbox=getattr(args, arg_name, default_outbox))
