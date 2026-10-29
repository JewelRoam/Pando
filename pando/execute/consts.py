"""Decide whether a proposal may be executed, and carry out the few it owns.

The colony can already propose and vote. Handing every adopted proposal
straight to an actuator would run almost every one of them: the decision
models currently answer 赞同 far more often than they refuse. So an action
runs only when three things are true — the action is in the action space,
enough other decision models adopted it under the policy in force, and a
tested actuator for that exact action exists.

``ACTUATORS`` is that boundary, and it is short on purpose. As of today:

- ``seed.export`` — seal this host's identity into a bundle. Wired.
- ``seed.replicate`` — stage that bundle for a host that pulls it. The offer
  waits in the local outbox; there is no push transport, and naming a host is
  not permission to open a socket to it.
- ``policy.set`` — move a knob in ``pando/policy.py``. Wired, bounded, and the
  only actuator that changes what the gate does next round.
- ``culture.refine`` — stage a signed, validated playbook *candidate* into a
  local outbox. Wired up to a review queue, not to a hot swap: nothing loads
  the candidate, so adopting a culture change can never rewrite the rules in
  the same breath as proposing it. Applying stays a human/gate step.
- ``recruit.invite`` — mint one signed, *inactive* pairing invitation into a
  local outbox. Wired up to a review queue, not to an admission: nothing writes
  the token into a running root, and the manifest is ``applied: false``. A
  staged invitation is a credential, so the artifact is written 0600 and never
  published to the mailbox or the ledger.
- ``reproduce.spawn`` — stage a signed reproduction *plan* into a local
  outbox: what would be copied, where, and every reason the parent still
  refuses. Same review-queue boundary as culture.refine, and wider: the
  proposal may carry labels only, ``sandbox_confirmed`` is always false, and
  nothing here calls ``pando.reproduce.reproduce``. A plan cannot be an
  approval.
- ``resource.rebalance`` — draft a signed *move* plan into a local outbox:
  which member, onto which host, and every precondition that is still unmet.
  Labels only (``node_id``, ``target_host``); a ``path`` or ``command`` is
  refused, so a proposal cannot name the state directory or the stop/start
  order. Nothing stops a member, starts one, or copies a byte.
- ``meta.propose`` — draft a signed *operator spec* into a local outbox: the
  name of a new proposal operator and the prose asking for it. This is the one
  action whose payload is code in the end, so the artifact carries labels only
  and ``code``/``source``/``module``/``path``/``command`` are refused rather
  than ignored. Nothing imports a module or defines a class.

That closes the action space: all eight declared actions now have an actuator,
and six of the eight stop at a signed draft in a local outbox. The two that
touch the world on their own are ``seed.export`` (writes a sealed bundle under
this host's own ``executions/``) and ``policy.set`` (moves one bounded knob).
Neither is reachable from a proposal alone.

Nothing here starts itself, and nothing here opens a socket. A dry run reads
the mailbox and prints counts. A ``--once`` round seals at most one seed,
applies at most one rule change, stages at most one culture candidate, mints at
most one invitation, and drafts at most one plan for each of reproduction,
member move, and a new operator — each recorded in an append-only ledger.
"""
from __future__ import annotations

ADOPT = "赞同"
HOLD = "搁置"
DECLINE = "反对"
STANCES = (ADOPT, HOLD, DECLINE)

#: The actions this program can actually carry out. Everything else in
#: ``ACTION_SPACE`` is a declared intent: adopted, never executed. Kept as
#: one list so the boundary is readable from outside and cannot drift


ACTUATORS: tuple[str, ...] = (
    "seed.export",
    "seed.replicate",
    "policy.set",
    "culture.refine",
    "recruit.invite",
    "reproduce.spawn",
    "resource.rebalance",
    "meta.propose",
)



#: How an installed verb is staged: the actuator it binds to, the outbox that
#: actuator writes, and the flag that relocates it. A new verb has no stager of
#: its own -- reusing the one it binds to is the whole point -- so this table is
#: the only place that knows how to carry out a verb nobody wrote code for.
