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

## 2026-10-17 - Firekeeper L1 results, and the reverse channel opens (Mac-side tasks)

Mac ran firekeeper L1 (branch firekeeper-1 @ 45dab56; 30 rows per phase on the island): strong-tier death
leaves the island available at reduced capacity, 0.9667 -> 0.6000 accuracy on identical rows, latency
improves ~300x; rejoin < 1 s; resync 0 bytes (no state to resync yet). Availability gap after death: 6-11 s.
Critical reading (agreed): the gap is fully explained by HEARTBEAT_TTL=15 s - a timer, not self-healing;
and with quorum_k=1 a dead primary returns responded=0 although a live weak node is registered: there is
no in-flight failover. Both now have an owner: docs/DESIGN-fallback.md (one bounded substitution round +
cooldown suppression), whose minimum falsifying measurement is an ON/OFF drill pair.
Also: the Mac now creates tasks for Windows. First reverse tasks: 8447 fallback design (delivered same
day), 030 calibrator coverage for mac-llm-8b-q6 (claimed), 031 firekeeper L2 churn under load (claimed).
Symmetric operation is live; churn-arena v1 and this drill converge on the same missing piece.

## 2026-10-17 - Mac-side reflection on Task 026 (relayed by mac-codex), and adjudication

Task 026 delivered the teacher2 distillation corpus (943 rows, zero exact overlap, n-gram residue disclosed). Three findings matter more than the corpus:
1. Both 8B backends had been returning HTTP 500 (Metal OOM) unnoticed; "deployed and online" was false for a long window. Monitoring lesson: process-up is not service-up; we need a semantic health check (one real completion request per tier per heartbeat).
2. The Mac chose NOT to restore the old 8B (16 GB machine; teacher2 + 4B + old 8B does not fit; trying kills the Metal backend and takes teacher2 down). Decision: ACCEPTED. teacher2 (Q5_K_M) replaces the old 8B as the 8B-class tier; the old 8B stays retired; the resident set is teacher2 + 4B.
3. The :8082 proxy hangs (no fast 502) when its backend is dead - a health-check false positive worse than a stopped service. Must fail fast or be retired (assigned in Task 029).

Mac's critique (received; agreed in substance):
- Agreement rate (0.9067) and teacher-swap agreement (98/100) are reproducibility, not accuracy; no human anchor anywhere. The corpus is internally consistent, externally unanchored.
- teacher2 vs old 8B on H (0.9417 vs 0.9333) is within noise; it must not be retold as an improvement (the real gain was 5.6x latency).
- The 97 discarded rows concentrate in invoice/login_issue; the falsifiable alternative is that the filter strips ambiguity (hardness), not noise (mislabels). Untested.
- Ops cost belongs in the denominator (swap pressure, service rebuilds, approvals). True cost per row is far above 535 s / 1040.

Adjudication on ft-v9: DEFERRED. The corpus labels are nearly identical to the previous teacher's (98/100 agreement), so a ninth checkpoint would add another micro-metric, not a generalizable finding - exactly the pattern this log exists to refuse. The corpus stays available as an artifact if a specific hypothesis needs it. Next value bets stay: churn demo, self-hosted loop, admission falsification, firekeeper ladder.
