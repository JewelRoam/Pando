# Power and Competence: on "smarter" and "dumber" members in an open fabric

Status: design position, 2026-10-17. Written after the question: an open fabric will admit members of very
different competence — should it decentralize, or let the smartest lead? This note states what our own
measurements already say, the position we adopt, and the experiments that could falsify it.

## The two questions that get conflated
- Structural power: who can admit, exclude, define policy, hold keys, survive others' exits.
- Epistemic weight: how much a given answer counts for a given decision, right now.

PandoRSI decentralizes the first and *measures* the second. Confusing them produces the two classic traps:
"every voice is equal" (epistemic relativism, measurably worse) and "the best member decides" (an epistemic
monarchy that is structurally centralizing and fails open on the king's blind spots).

## What our measurements already show
- Under assent + veto fallback, the committee *is* the witness: decided rows ≡ witness accuracy
  (D 0.9917, E 0.9917, F 0.9833, H 0.9958). "Smartest leads" is not hypothetical — it is already the
  fallback seat.
- On split H (cross-family members, mean phi = 0.0148) plain majority over six heterogeneous members
  reaches 1.0000 at k >= 5 — better than the best single member (0.9958) and better than the witness
  fallback. Decentralized counting wins *when errors are decorrelated*.
- On split F (domain shift, mean phi = 0.3575) plain majority collapses to 0.88-0.93 while the best solo
  member scores 0.9833 — weak *and correlated* members actively harm. The assent gate rescues the auto
  tier to 0.9904.
- The "best" member is domain-specific: ft-v3 leads on known wording (0.9667) and the 8B leads on
  general/novel wording (0.9833); the 4B flips the same way. There is no universal king — an argument
  against a permanent leader, not against weighting.
- Copies of the same model add latency, not information (rounds 95-97). The resource that matters is
  disagreement between distinct lineages: the v7/v8 two-seed committee agreed on 94.2% of H rows and
  every agreed row (226/226) was correct.

## Position
- Decentralization is about power, not epistemology. No permanent master; cheap entrance and exit; rules
  that bind everyone; anyone replaceable. It does NOT mean pretending all answers deserve equal weight.
- Weight is measured, per domain, revocable, and decaying — yesterday's expert is today's liability.
- The enemy is correlation, not weakness. Weak-but-independent members improved H to 1.0000;
  strong-but-copied members add nothing. A committee from one lineage is a monarchy wearing a lab coat.
- The smartest seat must be a *seat*: fillable, domain-scoped, renewable, and losable by measurement.
  That is the line between meritocracy and centralization.

## Rules we adopt
1. The assent gate stays; the auto tier may require cross-lineage agreement (lineage quorum >= 2) when
   stakes are high. A single-witness veto fallback is a declared, time-boxed policy — never a default identity.
2. Diversity is a maintained resource: keep a lineage census; reserve anchor seats for independent
   lineages (e.g. non-distilled members); treat a rising mean phi as a first-class alarm.
3. Separation of powers over data: the lineage that trains must not own the evaluation. Fresh evaluators
   stay outside the teacher line.
4. The strong member is continuously challengeable: fresh suites recruit challengers, and a challenger
   that loses does not get to keep a crown either.

## Experiments that could falsify this position
- King test: degrade the strongest member mid-stream; committee damage must stay bounded (no collapse to
  witness accuracy). If it collapses, our fallbacks are king-dependent.
- Echo test: inject k correlated copies of the incumbent; the gate/quorum must not let them fake a
  supermajority. If they can, correlation is a cheat vector.
- Flip test: move the workload to a distribution where the current witness is weak; the registry must
  re-elect within N decisions (measured), not persist.

These three tests join the admission-falsification harness in docs/evidence/reflections.md (next bets A-D).
