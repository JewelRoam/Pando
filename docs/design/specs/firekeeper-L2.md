# SPEC: firekeeper level 2 - churn under load (answers mac-codex task f147)

Status: spec, 2026-10-17. Runner: the Mac island (executes after the fallback patch ships).
Companions: docs/design/positions/fallback.md (the patch this spec depends on), docs/evidence/reflections.md (bets).

## Hypothesis
With bounded substitution (one in-flight failover round + cooldown suppression), the dark window after a
voter death shrinks below one decision interval. Without it, the window is the HEARTBEAT_TTL timer
(measured 6-11 s in firekeeper L1). If ON and OFF arms behave the same, the "weak tier keeps the group
alive" story is falsified as a timer artifact.

## Preconditions (do not start until all hold)
1. Fallback patch implemented and deployed on the island root (--fallback-substitute on/off, default off).
2. mac-llm-8b-q6 declared choice-only (declaration-only calibration asset) and registered (known: H,
   0.9417). Done 2026-10-17.
3. A decision stream with at least 240 decisions per arm exists (noise floor 1.65 pp on 240 rows).

## Setup
- Fleet (3 voters max resident on a 16 GB box): teacher2 (strong, :8768), mac-text (weak), plus the 4B
  only if memory allows at run time; record the actual resident set in the trace.
- Stream: 240 decisions per arm over the H rows (reused across arms), fixed cadence, identical seeds.
- Churn: 6-10 events per arm at random ticks, mixing (a) kill the strong voter, (b) restart it,
  (c) one timeout (SIGSTOP or network pause) to cover the mute-but-alive case.

## Arms
| arm | fallback | expected dark window |
|---|---|---|
| OFF | disabled | 6-11 s after strong-voter death (L1 baseline) |
| ON  | enabled  | < 1 decision interval; every decision with >=1 live voter answered |

## Metrics (per arm, from the JSONL trace, not from a summary)
- availability = answered decisions / attempts;
- accuracy at matched coverage (same answered subset in both arms);
- latency p50/p90/p99 and the added amount vs the no-churn baseline;
- dark window: consecutive unanswered decisions x interval, and time-to-first-answered-decision after
  each kill;
- rejoin time and resync bytes (for L2 this is expected 0; that is L3's subject).

## Deliverables
- A reproducible runner script (one command, seeded) committed with the results branch;
- the full JSONL trace plus a short brief (timeline, tables, anomalies) in metrics/;
- an explicit ON/OFF comparison; report falsification honestly if the ON arm does not beat the timer.

## Out of scope (declared, not forgotten)
- The equal-memory-budget "many small vs one big" control: a 16 GB box cannot hold the arm properly;
  it is assigned to L2b with a defined memory envelope (~6 GB) once hardware allows. Until it runs, all
  weak-tier results are claims about survival, not capability (docs/evidence/reflections.md).
- Physical power loss (launchd still recognized the service in L1); that is L3.
