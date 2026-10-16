# Task 013 result — the breadth axis: new-intent-family suite F + LLM votes

Status: done

## What was built

- `assets/heldout5_test.jsonl` — 120 rows, six **new** intent families the
  four older suites never contain, 20 rows each:
  `reschedule`, `billing_clarify`, `permission_escalation`, `data_export`,
  `content_appeal`, `device_triage`.
- Each row: `task_id` (`big7-####`), `state`, one-sentence `question`,
  `candidates` (3–4, expected always present; 26 rows have 3, 94 have 4),
  `expected`, `qtype="choice"`, `checkpoint="general"`. All 120 states are
  unique.
- Builder + overlap auditor: `tools/build_new_intent_suite.py`
  (regenerate with `.venv/bin/python tools/build_new_intent_suite.py`).

## Zero-overlap verification (measured, not asserted)

Audited against all 960 rows of `heldout_test` / `heldout2_test` /
`heldout3_test` / `big_test`:

| Check | Result |
|---|---:|
| label intersection (new 6 vs old 8) | `[]` (0) |
| `expected` label intersection | `[]` (0) |
| cross-suite cue↔state shared word-run ≥ 3 | 0 |
| max cross-suite cue↔state shared run | 1 word |
| max internal run between two different new families' cues | 1 word |
| filler 3-gram leaks into an old state | 0 |

So the two label sets are disjoint and no cue or filler reuses a 3-word run
from any old-suite row. By construction each new row carries exactly one
family cue and `expected` is that family (asserted in `validate_by_construction`).

## LLM votes (variant B prompt, temperature 0, max_tokens 64)

Both models answer candidate-only (no JSON/confidence). Variant B carries no
confidence, so `probability` is recorded as **0.5** on every row for both.

| Metric | 4B (`qwen3-4b-instruct-2507`) | 8B (`qwen3-8b-instruct`) |
|---|---:|---:|
| accuracy on F (120 rows) | **0.9333** (112/120) | **0.9833** (118/120) |
| parse errors | 1 | 0 |
| regex fallbacks | 119 | 120 |
| avg latency | 307.0 ms | 428.4 ms |
| p50 / p95 latency | 292.7 / 404.2 ms | 384.0 / 608.7 ms |
| avg tok/s | 25.60 | 18.99 |

Per-class accuracy:

| class | 4B | 8B |
|---|---:|---:|
| reschedule | 1.000 (20/20) | 1.000 (20/20) |
| billing_clarify | 0.800 (16/20) | 0.900 (18/20) |
| permission_escalation | 1.000 (20/20) | 1.000 (20/20) |
| data_export | 0.900 (18/20) | 1.000 (20/20) |
| content_appeal | 0.900 (18/20) | 1.000 (20/20) |
| device_triage | 1.000 (20/20) | 1.000 (20/20) |

Note: the only mistakes both models share are two deliberately terse
`billing_clarify` rows (`big7-0013` "these two charges look the same",
`big7-0115` "something on here is unexplained") that they route to
`content_appeal`; the ambiguity is in the subject of "these/here", not in the
label set. 8B also misses two more billing rows; 4B's other misses are
`content_appeal→permission_escalation`, `data_export→device_triage`, and one
no-parse row — a "breadth" tax concentrated on the billing family.

## Services

- 4B: `127.0.0.1:8083` internal / `*:8082` proxy, model `qwen3-4b-instruct-2507`
- 8B: internal / proxy per existing setup, model `qwen3-8b-instruct`
- Both left online after the run. mac-text stem (8766), rootd (8765) and the
  mailbox service (8776) were not touched. No tokens written to the repo.

## Artifacts

- `assets/heldout5_test.jsonl`
- `metrics/llm-votes-f-4b.json`
- `metrics/llm-votes-f-8b.json`
- `metrics/llm-votes-f-report.md` (this file)
- `tools/build_new_intent_suite.py`
- detailed summaries: `~/.pandosi/task013/heldout5-{4b,8b}-summary.json`,
  aggregate `~/.pandosi/task013/heldout5-breadth.json`

## Reproduce

```bash
.venv/bin/python tools/build_new_intent_suite.py
.venv/bin/python tools/llm_decision_check.py --variant b \
  --suite assets/heldout5_test.jsonl \
  --base-url http://127.0.0.1:8082/v1 \
  --api-key-file ~/.pandosi/llm-api-8b.token \
  --model qwen3-8b-instruct \
  --out metrics/llm-votes-f-8b.json
.venv/bin/python tools/llm_decision_check.py --variant b \
  --suite assets/heldout5_test.jsonl \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key-file ~/.pandosi/llm-api.token \
  --model qwen3-4b-instruct-2507 \
  --out metrics/llm-votes-f-4b.json
```
