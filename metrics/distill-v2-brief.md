# Task 026 — teacher2 distillation corpus (ft-v9 input)

Branch: `distill-v2` (scripts + this brief only; no corpus/model artifacts are committed).
Worktree: `/private/tmp/pandosi-distill-v2`, based on `origin/main` @ `b3dd44c`.

## What was produced

| artifact | path | rows | committed |
|---|---|---|---|
| suite (new cue set **i**) | `assets/big9_i.jsonl` | 1040 | no (regenerable, see below) |
| **corpus (laya trainer format)** | `rollouts/distill_v2_train_laya.jsonl` | **943** | no (stays under `metrics`/`rollouts`, gitignored) |
| corpus (same rows, plain) | `rollouts/distill_v2_train.jsonl` | 943 | no |
| per-row teacher votes | `rollouts/distill_v2_raw.jsonl` | 1040 | no |
| generation stats | `rollouts/distill_v2_stats.json` | 1 | no |

Machine-readable copy of the corpus for handoff lives beside this brief as
`metrics/distill_v2_train_laya.jsonl` (943 rows) with `metrics/distill_v2_stats.json`.

## How it was generated (exact commands)

```
# 1. suite: 48 new cue phrasings (8 intents x 6), seed 91
python tools/build_large_suite.py --cues i --count 1040 --seed 91 --out assets/big9_i.jsonl

# 2. overlap audit + distillation, teacher2 as the teacher, 4B as the agreement filter
python tools/distill_corpus.py --cue-set i --suite assets/big9_i.jsonl \
    --out-train rollouts/distill_v2_train.jsonl \
    --out-stats rollouts/distill_v2_stats.json
```

- teacher (labeler): `qwen3-8b-q5km-instruct` @ `http://127.0.0.1:8087/v1` (teacher2, Qwen3-8B Q5_K_M)
- filter (second vote): `qwen3-4b-instruct-2507` @ `http://127.0.0.1:8080/v1`
- prompt: variant B ("candidate word only"), the prompt that won on held-out wording in task 004
- seeds: suite seed **91**, `task_id` prefix `big91-`; every row unique state (1040 unique)

Suite reproducibility was checked, not assumed: re-running command 1 reproduces
`assets/big9_i.jsonl` byte-for-byte.

## Row counts, distribution, timing

- **1040 rows generated, 943 kept, 97 discarded** (kept = teacher2 and 4B agreed on the same label)
- generation wall time **535 s** (~0.51 s/row, serial: one teacher call + one filter call per row)
- per-row latency, teacher2 (8B Q5): mean 318.7 ms, p50 309.3 ms, p95 400.0 ms
- per-row latency, filter (4B): mean 195.5 ms, p50 191.9 ms, p95 267.6 ms
- parse failures: 8B 0, 4B 0

Class distribution of the **kept** corpus (this is the training distribution):

| intent | kept | suite cues | kept/cue |
|---|---|---|---|
| greeting | 130 | 130 | 1.00 |
| chargeback | 129 | 137 | 0.94 |
| praise | 128 | 131 | 0.98 |
| refund | 126 | 137 | 0.92 |
| exchange | 122 | 124 | 0.98 |
| complaint | 118 | 117 | 1.01 |
| login_issue | 99 | 137 | 0.72 |
| invoice | 91 | 127 | 0.72 |

`complaint` kept (118) exceeds its cue count (117) because one row cued as
another intent was labelled `complaint` by both teachers. The two teachers agree
least on `login_issue` and `invoice` — those are the classes the agreement
filter mostly removes, and the corpus is correspondingly thinner on them.

## Overlap audit (part of the tool, not a claim in prose)

```
overlap audit: {"target": "i", "cues": 48,
 "per_set_overlap": {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0},
 "total_overlap": 0, "examples": {}}
```

Exact-string overlap with every earlier inventory (A–E, G) and with the
held-out suites F (`heldout5_test`) and H (`heldout6_test`) is zero.

Residual, reported because "zero overlap" only ever means *exact string* overlap.
Longest shared word run of set i against any earlier cue of the same intent:

| intent | longest shared run | this cue (set i) | earlier cue | run |
|---|---|---|---|---|
| refund | **5** | the money I paid should come back to me | C: the money should come back to me | `should come back to me` |
| chargeback | 4 | my bank has been asked to reverse this | B: the payment network has been asked to undo it | `has been asked to` |
| complaint | 4 | this whole thing has been deeply frustrating | D: the whole thing has been dreadful | `whole thing has been` |
| invoice | 4 | I would like the itemised breakdown of charges | E: I would like the billing information | `i would like the` |
| login_issue | 4 | I am unable to reach my profile | B: I am unable to access my account | `i am unable to` |
| praise | 4 | your colleague went above and beyond | B: the crew went above and beyond | `went above and beyond` |
| exchange | 3 | could I swap this over for another model | A: can I swap this for another size | `i swap this` |
| greeting | 3 | hey there, hope you are having a good one | A: hey, hope you are well | `hope you are` |

## Confidence — read this before quoting a number

**There is no per-row confidence in this corpus.** Variant B asks the teacher for
the candidate word only, so the API never returns a probability. Every kept row
would carry probability 0.5 by construction if it were written into a vote
record. The only confidence signal the corpus has is **dual-teacher agreement
0.9067** (943/1040) — and that is an agreement rate, not an accuracy rate.

## Optional teacher-swap check (teacher2 vs the old 8B Q4)

Run on the first 100 suite rows, same variant-B prompt, old `qwen3-8b-instruct`
(alias of Qwen3-8B-Q4_K_M) as the second reader:

- **agreement with teacher2: 98/100 (0.98)**, 0 errors
- old 8B mean latency 1786 ms vs teacher2 mean 318.7 ms on the same rows (~5.6x faster)

## Environment / operations (cost that was actually paid)

This host has **16 GB** of unified memory and currently runs three llama-server
models: 8B Q4 (~4.9 GB), 8B Q5 teacher2 (~5.7 GB), 4B (~2.5 GB). Three resident
models is over budget:

- At claim time **both** 8B backends were failing every request with
  `500 Compute error`, and the logs showed
  `ggml_metal_synchronize ... Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`
  accumulated over ~38 h of uptime. teacher2 itself was down for the Windows side.
- Restarting teacher2 did not help until the old 8B Q4 backend was stopped: the
  Metal backend was in a terminal error state ("recreate the backend to
  recover"), and the memory had to be freed first.
- The corpus run therefore proceeded with **teacher2 + 4B resident** (old 8B Q4
  backend stopped), and the swap check ran with **teacher2 + old 8B resident**
  (4B stopped, then restored). Anything with all three resident fails.
- Two 8B models *can* co-exist (10.6 GB of weights); three models cannot.

State left behind: teacher2 `:8087` working, 4B `:8080` working, old 8B Q4 `:8082`
proxy + tunnel up with the **backend left stopped** — deliberately, because
starting it again is what breaks teacher2 on this hardware. All five cloudflared
tunnels are up; URLs unchanged, so `endpoint/mac.json` needed no update.

## Reflection and challenge (mandatory)

1. **Statistical power.** n = 1040 rows / 943 kept. The only documented rate is
   the agreement rate 0.9067, whose 1 SE is
   `sqrt(0.9067 * 0.0933 / 1040) = 0.90 pp`. So "90.7% agreement" is
   `90.7% +/- 0.9 pp`. The per-class numbers are far weaker: `invoice` 91/127
   and `login_issue` 99/137 carry ~4 pp standard errors, so the class imbalance
   in the table above is real but the exact per-class rates are not precise.
   The swap-check 98/100 has SE `sqrt(0.98*0.02/100) = 1.4 pp`. **No number here
   is an accuracy number.** Agreement between two models that share a training
   lineage measures reproducibility, not correctness.
2. **Missing control.** Nothing in this task compares teachers, budgets, or
   model counts. In particular the control that could falsify the project's
   thesis — *many small models vs one big model under an equal memory budget* —
   was still never run. This task manufactures training data; it cannot test
   whether the resulting small model is worth its cost.
3. **External anchor.** Set i, like A–H and G, is generated by this repo's own
   cue tables and filler pipeline, and the labels come from this cluster's own
   models. There is no human-labelled or third-party intent corpus behind any
   row. State it plainly: **internally consistent, externally unanchored.** A
   0.907 agreement rate is a statement about two of our own models agreeing with
   each other.
4. **Operational cost.** Resident memory and swap are not free: the host is at
   ~19 GB of 20.5 GB swap with teacher2 + 4B up, three models do not fit, one
   backend had to be taken down to make the run possible, and teacher2 needed a
   forced Metal-backend restart. Latency is tail-heavy too (8B p95 400 ms per
   row, and the old 8B is ~1.8 s/row). Add to the denominator: the human
   approval prompts this session generated, and the ~25 minutes spent on service
   recovery rather than on the science.
5. **Falsifiable alternative, and what this cannot answer.** The 97 discarded
   rows are concentrated in `invoice` (72% kept) and `login_issue` (72% kept).
   One alternative reading: the agreement filter is not removing label noise, it
   is removing the classes where the two models genuinely disagree about where
   the taxonomy boundary sits — so the "clean" corpus is clean partly by being
   easier, and ft-v9 trained on it may look better on this distribution while
   being no better on the classes that matter. A testable version: hold the
   filter fixed and check whether the discarded `invoice`/`login_issue` rows are
   the ones a human would call ambiguous. This corpus, by itself, cannot tell
   you whether teacher2 is *right*, only that teacher2 and 4B usually *agree* —
   and on 100 shared rows teacher2 and the previous 8B agreed 98% of the time,
   which suggests the teacher swap bought latency (~5.6x), not a materially
   different label distribution.
