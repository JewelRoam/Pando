# The Genesis layer: how the group authors its own agenda

Status: mechanism implemented with a stub proposer and a signed proposal record
(2026-10-17); real proposer and transport wiring open. Companion to docs/COMMS-AND-SEEDS.md (roles, seed),
docs/autonomy loop in `pando/autonomy.py`, and the round98/99 findings.

## 1. The gap (measured, not asserted)

Round 98/99 measured the swarm honestly: it can **choose** (51.8% overall, 92.9% on what it
accepts) and **refuse**, but it cannot **find its own decisions** — "the agenda must come from
outside". In code that is literal: `pando/autonomy.py` consumes `Observation(state, question,
candidates)` with all three fields supplied from outside, and `ElasticSupervisor` chooses among a
hardcoded `{grow, retire, hold}`. The swarm is an agenda *follower*: a human (or a script a human
wrote) hands it a menu, and it picks.

The rule this layer exists to satisfy (user, 2026-10-17): **the group authors its own options; no
external party hands it a menu or issues commands.** The intelligence must live in the group, not
in an operator's prompt. That is not a falsifiable claim about survival — survival is the axiom —
it is a statement about *where the options come from*, and that part is buildable.

## 2. "Endogenous" has three tiers; be honest about which is reachable

- **Tier 0 — selection among external options.** Have it. This is the whole current stack.
- **Tier 1 — endogenous selection over self-generated proposals.** Reachable. The group generates
  its own candidate options from its own history and state, then selects among them. The *options*
  are new; the *action primitives* are still ones the substrate provides. **Genesis targets this.**
- **Tier 2 — endogenous action invention.** The group invents *new primitives*: writes new code,
  configures capabilities nobody built. This is where a real LLM's fluency plausibly matters, and it
  is a different, harder question. Do not claim it here.

The earlier worry — "能部署 LLM 的地方就掌握方向的话语权" — is answered inside Tier 1: for option
generation to be the group's, generation must be *distributed* and *unprivileged*. A big-model node
is a more fluent proposer, not the agenda-setter; its proposals score like anyone else's.

## 3. The loop: propose → score → adopt → execute → remember

1. **Propose.** Any member may emit a signed `proposal` into the group log. Structured body:
   `{goal, action, expected_effect, cost, cites}`. `cites` references prior proposals/outcomes —
   this is the recombination hook. Generation is distributed; no member is privileged.
2. **Score.** Two axes: *feasibility* (is the action in the action space, affordable, axiom-legal?)
   and *expected instrumental gain* (conditioned on the outcome history of similar past proposals).
   Reuse the existing gate — only `tier=auto` proposals auto-adopt; `verify`/`escalate` are held.
3. **Adopt.** The top proposal, quorum-gated, becomes the group's next action.
4. **Execute.** Within the action space (§5). Record a signed `outcome`.
5. **Remember.** Outcomes condition the next proposal round. **This is the gradient.**

## 4. Where the intelligence actually comes from

Not from any single model — from the loop: many weak proposers × shared memory × recombination ×
outcome-conditioned selection. A CPU model that cannot invent from scratch can still *recombine*:
mutate a prior proposal, invert a condition, combine two, escalate/de-escalate, specialize. Those
operators are cheap and need no big model. This is the concrete reading of "智能来自于群体本身":
a genetic search over strategies, not a bigger brain.

## 5. The action space (what the group can actually propose)

Proposals must be grounded in primitives the swarm controls, or they are unexecutable words:

- **seed** — export / rehydrate / replicate a seed to a new host
- **recruit** — invite / admit a member
- **reproduce** — spawn a replica of a member (spore/graft)
- **policy** — change a voting rule, gate threshold, or role grant
- **culture** — refine/retrain a member, distribute a playbook update
- **resource** — rebalance which host runs which member
- **meta** — propose a new proposal *operator* (the seed of Tier 2)

Small and enumerable today. That is the honest boundary: Genesis searches over *compositions and
parameters* of these, not over actions nobody built.

## 6. The real bottleneck is the gradient, not the axiom

The user's axiom (survive & expand) is a telos, not a measurement — accepted, not re-litigated. But
an evolutionary loop needs a gradient, and the gradient is instrumental outcomes: did a seed export
succeed, did a replica register on a new host, did memory pressure drop, did a member join. If those
are slow or sparse, selection has almost no signal and proposals degenerate into plausible noise.
The design risk is **signal sparsity**, not the philosophy. First job: make instrumental outcomes
frequent and comparable.

## 7. Status: the mechanism is built; the proposer is not

Implemented 2026-10-17, in-process, no model:

- `pando/genesis.py` — `Proposal`, `ProposalOutcome`, `ScoreBoard` (Beta-smoothed instrumental
  gain), `NullScoreBoard` (amnesiac control), `StubProposer` (deterministic; holds the candidate set
  constant), `LogProposer` (candidates from signed group-log proposals), `GenesisLoop` (propose → score → adopt → execute → remember), `Journal`,
  and the `ACTION_SPACE` from §5.
- `tools/genesis_check.py` — the drill. Two loops, identical proposer / candidates / actuator; one
  remembers, one does not. Measured: the remembered loop round 1 adopts `recruit.invite` (fails) and
  round 2 moves to `reproduce.spawn`; the amnesiac loop repeats `recruit.invite`. The only difference
  between the two loops is the memory, so the change of agenda is caused by the group's own outcome
  record — the falsifiable core, holding on a stub.
- `tests/test_smoke.py::test_genesis_agenda_is_history_dependent` — the same property as a test.
  Full suite: 36 tests, 1 skipped, all green.
- `pando/genesis.py` also carries a proposal as a **signed record** (`sign_proposal` /
  `verify_proposal` / `proposals_from_log`), so it can travel between members and be checked on
  arrival — same rule as registration: the public key is the claim, the name is a label.
  `tools/genesis_log_check.py` is the negative-control drill (clean passes; tampered / unsigned /
  wrong-kind / foreign-key records are dropped; only the pinned author's proposal survives a mixed
  log). `tests/test_smoke.py::test_signed_proposal_round_trips_and_rejects_tampering` is the test.
- `tools/genesis_check.py` also runs a second scenario: the candidates are two members' signed
  proposals (`LogProposer`), the loop adopts member-a's, it fails, and round 2 moves to member-b's
  — the candidate set is authored by the group and pinned on the way in, not invented by the loop.

Deliberately **not** claimed yet:

1. The signed proposal is transport-agnostic, but it is **not on the group log yet**: nothing
   carries it over the mailbox or the S2 group log, so a live member still cannot *receive* another
   member's proposal. The record format is settled; the transport wiring is the next step — and it
   wants to line up with S2 rather than fork a parallel convention.
2. No model is involved. The stub is dumb on purpose; whether a real Laya / text-nb proposer beats it
   on instrumental outcomes is unmeasured.
3. The loop is not yet gated by the swarm tier. `pando/autonomy.py` gates its actions on
   `Coordinator.decide`; here the stub adopts unilaterally, which is fine for a mechanism drill and
   not fine for a live member.

The original plan was to decouple the mechanism from model quality: if the machinery cannot turn a
*known* proposal stream into history-dependent selection, a smarter proposer will not save it. That
separation now exists; the two questions are isolated.

## 8. What this does NOT claim

- Not Tier 2 (new primitives). A real LLM may be required there; separate question.
- Not that weak models produce good proposals — unmeasured. The stub is deliberately dumb to test
  the *loop*, not the *brain*.
- Not that one box can supply the fitness signal. The loop may close mechanically on a single host
  while being starved of outcomes — the same "same-box" limit as the seed drill (docs/SEED.md).
