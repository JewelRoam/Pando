# Experiment — equal-memory-budget control 005: the bracket for the bloodline arm (`Gemma-4-12B-Q4_K_M`, non-Qwen, **above** the committee's budget)

Date: 2026-10-19 · Author: mac-codex · Machines: arm A on Windows 11 / RTX 3070
(`win` root `192.168.31.100:8790`), arm B on this mac (Apple arm64, 10 cpu, 16 GiB)

Pre-registration: `assets/votes/equal-budget-005.config.json` (written before any
row was harvested) · Raw votes: `assets/votes/equal-budget-005.raw.json` ·
Report: `assets/votes/equal-budget-005.json`

## Why this run exists

004 ran the arm the todo asked for and the committee won (0.9259 vs 0.8757). That
result is **confounded on its face**: Yi-1.5-9B differs from the committee in
lineage *and* may simply be the weaker model, and its weight bytes are
unverified. One point cannot separate "different bloodline" from "weaker model".

This run adds the second point on the **other side of the capacity axis**, the
same move 003 made for the budget axis: a non-Qwen model that is *larger* than
the committee. Two arms that tilt the same comparison in opposite directions are
a bracket; one arm is an anecdote with a number attached.

## What moved, and what was held frozen

- **Moved (arm A only):** `Yi-1.5-9B-Q4_K_M` → `Gemma-4-12B-Q4_K_M`, still
  non-Qwen (Google Gemma-4, neither Qwen nor Yi), larger, on the GPU, reached at
  `http://192.168.31.100:54241` — read from win's `/nodes` immediately before
  the run, because stem ports rotate on every restart. No new model was started:
  this drives the stem win already serves for its own proposer.
- **Held frozen (arm B):** the same four members, adopted verbatim from
  `assets/votes/equal-budget-002.raw.json` via `--reuse-arm-b`, with the donor
  refused unless suite, row count and member URLs match one for one.
- **Held frozen (the question):** suites byte-identical to 002/003/004.

## What "above the committee's budget" rests on — and what it does not

The sign of this arm does **not** depend on the unmeasured byte count, and that
is the point of choosing it: 12e9 weights stored at Q4_K_M (≈4.85 bit/weight)
is ≈7.3 GB, and the committee's 5.112 GiB is 5.49 GB. A 12B Q4_K_M cannot fit in
the committee's budget. That is arithmetic on the parameter count, not a
measurement — **the exact file size is still unverified**, because win exposes
no gguf bytes over `/nodes` or the stem's `/health`, and it is recorded as
`weight_gb: null` in the config rather than estimated into a number that would
look measured.

## Commands

No shell log of the 07:1x harvest survived, so the command below is rebuilt from
the pre-registration and the tool's own interface. The numbers in the next
section come from the committed report and are re-derived from the raw votes by
`tests/test_smoke.py::TestPairedDifferenceStandardError`.

```bash
.venv/bin/python tools/equal_budget_check.py \
  --suite assets/heldout4_test.jsonl,assets/heldout5_test.jsonl \
  --big http://192.168.31.100:54241 --big-checkpoint gemma-4-12b-q4km \
  --small http://127.0.0.1:8767,http://127.0.0.1:8770,http://127.0.0.1:8769,http://127.0.0.1:8766 \
  --small-weight-gb 5.112 \
  --reuse-arm-b assets/votes/equal-budget-002.raw.json \
  --votes assets/votes/equal-budget-005.raw.json \
  --out assets/votes/equal-budget-005.json
```

## Results

| arm | node | weight bytes | GiB | accuracy | answered | p50 | p95 |
|---|---|---|---|---|---|---|---|
| A one big — `gemma-4-12b-q4km` | win (RTX 3070) | **unverified** (≈7.3 GB by construction) | > 5.112 | **0.9722** | 360/360 | 246 ms | 307 ms |
| B committee — 4 members | this mac | frozen 002 members | **5.112** | 0.9259 | 0.975 | — | — |

Paired, on the 351 rows both arms answered:

```
A right / B wrong = 23
B right / A wrong = 7
d = B − A = −16     1SE = 5.477     d/SE = −2.921     McNemar exact p = 0.0052
```

Per suite: `big6` A 0.9625 vs B 0.9202; `big7` A 0.9917 vs B 0.9381.

Under the rule fixed before the run — `d < 0` and `|d| > 1.96·SE` means *the
larger non-Qwen model is ahead beyond noise* — this reads: **one-big is at least
as good**, i.e. the many-small claim is **not supported** on these rows. Unlike
004, p = 0.0052 also clears the Bonferroni bar for the four variants this series
has compared (≈0.0125).

## The bracket, read straight

```
004:  non-Qwen 9B, plausibly smaller than the committee   d = +18,  z = +2.65   committee ahead
005:  non-Qwen 12B, above the committee's budget          d = −16,  z = −2.92   one-big ahead
```

The committee was on the winning side of one arm and the losing side of the
other, and the sign followed **size**, not **lineage**. So this bracket does *not*
establish a lineage effect — it removes "the committee only loses to Qwen"
(lineage is no longer a live explanation for 002/003) while leaving the capacity
axis as the explanation still standing, with two points and no curve.

## What this does and does not rule out

**Does rule out.** "Non-Qwen models cannot beat this committee" — one just did,
decisively enough to clear the corrected bar.

**Does not rule out.**

1. **Lineage as an effect.** Arm A differs from arm B in lineage, host,
   quantisation format, chat template, serving path **and** size, all at once.
   This arm bounds the capacity axis; it does not isolate lineage. That needs the
   2×2 {Qwen, Gemma} × {8B, 12B} at equal weight.
2. **Equal weight.** Still unverified for arm A. This bracket is *above* budget
   by construction, which is what makes its sign safe, but "measured at equal
   bytes" remains the missing control.
3. **External anchoring.** Same internally generated suites; the author and the
   examinee are the same group.
4. **Latency as an endpoint.** 246 ms p50 across a LAN on a warm 3070 versus an
   in-process committee: reported, not used.
5. **Anything about the committee in general.** One committee, one aggregator,
   one weakest member at 0.3583.

## 反思与质疑

- **样本量 / 标准误.** n = 360 rows, 351 paired, so b + c = 30 carries the
  evidence and 1SE = √30 = 5.477. This is a single draw of a single suite
  version: the paired SE is a binomial on discordant rows, **not** a
  cross-seed or cross-suite standard error. **The original text carried
  1SE = 2.74 / z = −5.84** (paired SE halved; corrected across the series on
  2026-10-19 in `278dc1e`). −2.92 still clears noise and the Bonferroni bar;
  −5.84 did not happen.
- **缺哪个对照.** (a) the **2×2 at equal weight** — the only shape that
  separates lineage from capacity; (b) win-side **byte verification** for every
  foreign arm; (c) a **second draw** of the suites; (d) the same rows through a
  probability backend, since win's stems answer choice-only.
- **证据是否外部锚定.** No. Rows, labels, aggregator and arm A's stem are all
  PandoSI's own, on PandoSI's own LAN.
- **运维代价（含人类审批次数）.** 0 new timers, 0 new grants, 0 new resident
  models, **0 human approvals**. Real cost: win's own llama-server served ~360
  sequential exam requests while also being win's proposer's model — declared in
  the config up front as `interference_declared_up_front`, one pass, no retry.
- **一条可证伪的替代解释.** "Gemma wins because 12B with this chat template
  suits these rows, not because of lineage." The bracket cannot reject it — it
  predicts the same sign. What *can* reject the lineage story is the 2×2: if
  same-lineage, same-weight Qwen-12B also beats this committee by a similar
  margin, lineage is doing nothing and size is doing everything. Falsifier in the
  other direction: re-score the same rows through the probability backend; if the
  ranking flips, the effect is the interface, not the model.

## Artifacts

- `assets/votes/equal-budget-005.config.json` — pre-registration (arms, suites,
  hash, decision rule, declared limits, interference)
- `assets/votes/equal-budget-005.raw.json` — 360 arm-A votes +
  `arm_b_reused_from`
- `assets/votes/equal-budget-005.json` — report (per-suite, paired, verdict)
- `docs/EXPERIMENT-equal-budget-004.md` — the other side of this bracket

## Standing rule this run does not change

No result from this project may be described as evidence for group intelligence
unless an **equal-weight** foreign-bloodline control holds **and** the 2×2 is
run. This run's own number leans against the thesis: the committee lost to a
single model that was given *more* memory, on the project's own rows.

## 勘误 · 2026-10-19 · 配对标准误

本实验首次发布时写的 `1SE = 2.74` / `z = −5.84` 出自错误的配对标准误公式。正确值
是 `sqrt(b + c) = sqrt(23 + 7) = 5.477`。四份报告在提交 `278dc1e` 统一改用
`pando/common.py:mcnemar_test()` 并从缓存的原始票就地重生成。更正后：

```
A right / B wrong = 23
B right / A wrong = 7
d = B − A = −16     1SE = 5.477     d/SE = −2.921     McNemar exact p = 0.0052
```

**判决方向与强度都没变**（这是本系列四个实验里唯一一个更正后仍然稳的）。
注意 `assets/votes/equal-budget-005.config.json` 的 `origin` 字段里引用了 004 的
**更正前**数字（`1SE=3.39`）：预注册文件是可追溯的当时文本，不回改；004 的正确数字
写在 `docs/EXPERIMENT-equal-budget-004.md` 与 `equal-budget-004.json` 里。
