# Research Round 57c supplement — 14 disagreement cases classified (2026-10-14)

From the round-57 live divergence data (all 40 tasks, 3 lineages).

## Classification

### Category A: majority CORRECT (5/14 = 36%)
These are genuinely hard inputs where one lineage dissenting is expected.
The majority correctly identifies the answer.
- en03, td03: noul urgent/order_missing → en+td agree true, ml dissents
- ml06: Arabic invoice question → en+ml agree invoice, td dissents
- en11: charging cable smoke → en+td agree critical, ml says negligible
- td09: 50 failed logins → ml+td agree critical, en says high (close)

### Category B: ADJACENT-LEVEL risk disagreement (5/14 = 36%)
Score-type risk questions where lineages disagree on the DEGREE not the
direction. All are 5-level choice questions (converted from score).
- en04: printer blinking → en=low, ml+td=critical (expected low)
- ml04: Korean employee explosion → all underestimate (expected high)
- en12: opening hours → en+td=moderate, ml=negligible (expected negligible)
- ml10b: Arabic hours question → en+td=moderate, ml=negligible
- td10: user prefers email → all say moderate, expected negligible

### Category C: GENUINE ERRORS (4/14 = 29%)
These are the RLCD training targets — the majority is clearly wrong.
- en02: unknown sender asks for password → 2/3 say "open", expected "block"
  SECURITY RELEVANT: the swarm would open a phishing email!
- ml11b: Hindi "order never arrived" → 2/3 say false, expected true
  LANGUAGE GAP: Hindi e-commerce complaint not understood
- ml12b: Italian "product broken after two days" → 2/3 say refund,
  expected complaint (intent confusion: broken product = complaint not refund)
- td11: bank_transfer vs original_payment → workflow routing error
  FINE-GRAIN CONFUSION: refund method distinction not learned

## RLCD training targets (from Category C)
1. Security: phishing email detection (en02) — highest priority
2. Hindi e-commerce: order status complaints (ml11b)
3. Italian intent: broken product = complaint not refund (ml12b)
4. Workflow: refund method selection (td11)
