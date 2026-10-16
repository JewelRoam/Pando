# Reflections — a running critical review log

Purpose: an append-only log for honest, uncomfortable assessments of this project: what we believe and
why, what actually moved a needle, what did not, and where the bottleneck is. One entry per review.
Mac-side reflections are folded in here as they arrive.

## 2026-10-16 — seeded by the Windows side after a user request for sharper self-review

### Earned beliefs
- The fabric runs across NAT'd, heterogeneous machines: a Windows root coexists with a Mac island
  (mailbox, two LLM tiers, its own quorum), sustained live all day (Tasks 001-025, dozens of results).
- The committee machinery is measured, not asserted: assent gating, veto-witness equivalence, neutral
  votes, registry v2 — and now a theory doc whose 7 predictions were machine-checked against recorded
  votes by tools/theory_check.py (5 confirmed, 1 refuted as designed, 1 undecidable).
- Engineering hygiene is real: four green workflows, pip-installable sdist/wheel, release v0.2.0 with
  checksums, reproducible offline tools.

### Diminishing-returns signals
- The ft-v5 -> v8 ladder and teacher2 moved accuracy by fractions of a percent on synthetic suites of
  120-240 rows, where one flipped row is already 0.4-0.8%. Each cycle costs hours of GPU/LLM time and
  produces one more checkpoint that no external reader can interpret.
- The decision task itself (short intent classification) has near-zero external validity: all members
  converge above 0.9, so much of the measured "progress" is noise-mining on tiny samples.
- Polished artifacts (survey, theory, docs) are accumulating faster than falsifiable claims about the
  system under stress.

### Bottlenecks, ranked
1. No churn demo. The unique claim — an elastic, heterogeneous decision fabric — has never been shown
   under device join/leave/crash with live traffic; only scripted single-shot scenarios.
2. No real consumer. No decision that anyone outside the project cares about flows through the fabric;
   value is currently self-referential.
3. Admission honesty is not adversarially falsified. We believe assent=auto demotes bad members — but
   we have anecdotes, not an experiment designed to break it.

### What to stop
- Adding model tiers or suites without a structural hypothesis. ft-v9 is the last point on the
  distillation curve unless it yields a generalizable finding (e.g. teacher-family effects).

### Next bets (value-ordered)
- A. Churn under load: N heterogeneous voters with random joins/leaves/timeouts; measure availability
  and decision quality; ship one reproducible script and an inspectable trace.
- B. Self-hosted loop: route a real, small decision (e.g. CI-failure triage for this repo) through the
  fabric; it either earns its keep or it doesn't.
- C. Admission falsification harness: inject a deliberately bad member and verify the gate demotes it.
- D. Committee replication: three or more seeds so the 0.9958-style headline is priced honestly.

### Mac reflections
- (none yet; Task 026 in flight — fold in when its result arrives)
