# Bounded substitution: in-flight failover for a dead primary

Status: design proposal, 2026-10-17. Requested by mac-codex task t-20261017-002255-8447 after the
firekeeper-1 drill showed a 6-11 s unavailable window when a quorum of one loses its only voter.
Hypothesis to falsify: with bounded substitution the window shrinks below one decision; if it does not,
the "weak tier keeps the group alive" story is a HEARTBEAT_TTL artifact, not a fabric property.

## Observed failure (firekeeper-1 @ 45dab56)
- quorum_k=1, primary (strong tier) dies.
- /decide returns choice=null, responded=0 for 6-11 s until HEARTBEAT_TTL=15 s evicts the corpse.
- A live weak node is registered the whole time; it is simply never tried within the same decision.

## Current mechanics (pando/rootd.py)
- decide() picks the quorum via pick_quorum(quorum_k) over live_nodes() (TTL-filtered registry).
- Fan-out to all picked members in parallel; a _call_stem failure returns None
  ("dead node: shrink quorum, never block the vote").
- With k=1 there is nothing to shrink to; the decision returns empty and waits for TTL eviction.

## Proposal: one bounded substitution round per decision
1. Keep the first fan-out unchanged.
2. After it returns, if votes < quorum_target and the bench (live, not picked this decision, not blocked,
   not in cooldown) is non-empty, run exactly ONE substitution round: pick up to `unanswered` bench
   members using the existing _group_rank_key and call them with the same payload.
3. No recursion: at most one extra round; total added wall time capped by `--fallback-extra-ms`
   (default 1500).
4. Retry suppression: a transport-class failure (connect refused / timeout / broken pipe) sets
   `unhealthy_until = now + --fallback-cooldown-s` (default 5 s, never beyond HEARTBEAT_TTL) on that node.
   pick_quorum and the bench skip nodes in cooldown, so a corpse is tried at most once per cooldown window;
   recovery re-admits it automatically. The natural TTL eviction stays as the second line of defense.
5. Application-class failures (HTTP 5xx, malformed JSON) also route to substitution for this decision and
   go through the existing stats so a permanently-broken-but-alive node is not hammered every decision.

## Answers to the four asks
(a) Retries: exactly one substitution round, up to len(unanswered) substitutes; no cascades.
(b) Stats: substitution calls go through _call_stem unchanged (inflight +-1, latency EWMA, failure counts);
    the only new state is unhealthy_until on transport failure, surfaced as cooldown_remaining_s.
(c) Dead-node suppression: the cooldown write plus skip in pick_quorum/bench; one transport failure buys
    --fallback-cooldown-s of silence for that node.
(d) Minimum measurement: kill the primary; require the FIRST post-kill decision to return responded>=1
    with choice!=null from the weak node (window < 1 decision interval; target 0 unavailable decisions).
    Counter-check: the same drill with --fallback-substitute off must reproduce the 6-11 s window.
    If ON and OFF behave the same, the hypothesis is falsified and reported as falsified.

## Explicit non-goals
- No change to the retired mac-llm-8b path; no lineage/opinion-counting changes; substitute votes are
  ordinary votes.
- Fallback is a declared policy slot (default off until the ON/OFF drill pair is measured back-to-back).

## Test plan
- Unit: scripted dead-primary with a live bench node -> first decision answers from the bench; the corpse
  is retried only after the cooldown, not on every decision.
- Drill: firekeeper-1 rerun as an ON/OFF pair; per-decision responded counts, unavailable-decision count,
  added p95 latency.
