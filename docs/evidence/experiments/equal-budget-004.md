# Experiment — equal-memory-budget control 004: the bloodline arm (`Yi-1.5-9B-Q4_K_M`, non-Qwen)

Date: 2026-10-19 · Author: mac-codex · Machines: arm A on Windows 11 / RTX 3070
(`win` root `192.168.31.100:8790`), arm B on this mac (Apple arm64, 10 cpu, 16 GiB)

Pre-registration: `assets/votes/equal-budget-004.config.json` (written before any
row was harvested) · Raw votes: `assets/votes/equal-budget-004.raw.json` ·
Report: `assets/votes/equal-budget-004.json`

## Why this run exists

003 closed the *budget* axis and named what it left open, in its own words:

> Arm A and arm A′ are *both Qwen3*. The committee's best member is
> `Phi-3.5-mini`. … a non-Qwen big model is still owed and has never been run.

This is that arm. It also changes what the series is measuring in one more way
than the config claims: it is the first arm in the series scored **across a LAN
boundary**, against a stem win runs for its own use, on a GPU box instead of
in-process on the harvester. Both facts are declared up front and both are
reasons to read the numbers narrowly.

## What moved, and what was held frozen

- **Moved (arm A only):** `Qwen3-8B` → `Yi-1.5-9B-Q4_K_M`, a different lineage
  (01.AI Yi, neither Qwen nor Gemma), reached at `http://192.168.31.100:54244`.
  That port is transient and was read from win's `/nodes` immediately before the
  run. **No new model was started for this arm** — it drives a stem win was
  already serving for its own proposer.
- **Held frozen (arm B):** the same four members from 002
  (`:8767` 4B, `:8770` Phi-3.5-mini, `:8769` laya, `:8766` text-nb), adopted
  verbatim from `assets/votes/equal-budget-002.raw.json` via `--reuse-arm-b`.
  The tool refuses the donor unless suite, row count and member URLs match one
  for one, and exits 2 rather than improvising a fresh committee.
- **Held frozen (the question):** the suites are byte-identical to 002 and 003
  (`heldout4` 240 rows + `heldout5` 120 rows, sha256 in the config).

## Arm A answers with a choice and nothing else

win's stems run `--backend openai`, i.e. `OpenAIChoiceBackend`, whose docstring
says there is no calibrated probability in that reply: every row comes back
`probability: 0.5, neutral: true`. So **only choice accuracy is comparable**
between the arms; a confidence-weighted comparison is not available for this
arm, and the report's calibrated column is the one that is read. This is
declared in the config before the run, not discovered after it.

## Commands

No shell log of the 07:0x harvest survived, so the command below is rebuilt from
the pre-registration and the tool's own interface — the *numbers* in the next
section are not rebuilt: they come from the committed report and are re-derived
from the raw votes by
`tests/test_smoke.py::TestPairedDifferenceStandardError`.

```bash
.venv/bin/python tools/equal_budget_check.py \
  --suite assets/heldout4_test.jsonl,assets/heldout5_test.jsonl \
  --big http://192.168.31.100:54244 --big-checkpoint yi-1.5-9b-q4km \
  --small http://127.0.0.1:8767,http://127.0.0.1:8770,http://127.0.0.1:8769,http://127.0.0.1:8766 \
  --small-weight-gb 5.112 \
  --reuse-arm-b assets/votes/equal-budget-002.raw.json \
  --votes assets/votes/equal-budget-004.raw.json \
  --out assets/votes/equal-budget-004.json
```

`--big-weight-gb` is absent on purpose: win exposes no gguf file size, and
guessing one would have turned an unmeasured quantity into a measured-looking
configuration line.

## Results

| arm | node | weight bytes | GiB | accuracy | answered | p50 | p95 |
|---|---|---|---|---|---|---|---|
| A one big — `yi-1.5-9b-q4km` | win (RTX 3070) | **unverified** | — | **0.8757** | 354/360 | 1127 ms | 1600 ms |
| B committee — 4 members | this mac | frozen 002 members | **5.112** | 0.9259 | 0.975 | — | — |

Paired, on the 345 rows both arms answered:

```
A right / B wrong = 14
B right / A wrong = 32
d = B − A = +18     1SE = 6.782     d/SE = +2.654     McNemar exact p = 0.0114
```

Per suite: `big6` A 0.8468 vs B 0.9202; `big7` A 0.9328 vs B 0.9381.

Under the rule fixed before the run — `d >= 0` means *this foreign-bloodline
model is not better than the committee* — this reads: **many-small was ahead
beyond noise on the rows it answered.** `d/SE = +2.65` clears 1.96. It does not
clear the Bonferroni bar for the four variants this series has now compared
(≈0.0125 against p = 0.0114), so the honest word is **borderline**, and the
coverage caveat is real: arm B answered 0.975 of the rows, so 15 rows are in the
suite and in neither arm's comparison.

## What this does and does not rule out

**Does rule out.** "A non-Qwen 9B served over the LAN with a choice-only reply
beats this 4-member committee" — on these 360 rows, under this parse path.

**Does not rule out.**

1. **Capacity.** Yi-1.5-9B-Q4 with *unverified* bytes may simply be the weaker
   model, in which case this arm says nothing about lineage at all. One arm
   cannot separate "different bloodline" from "weaker model"; that is precisely
   why 005 exists, and 005 is the second point rather than a second claim.
2. **Equal weight.** `weight_gb` is null. The pre-registered control was "a
   non-Qwen big model **at equal weight**", so this run closes the *lineage*
   axis and leaves the *weight* axis exactly where 003 left it. Do not merge the
   two sentences.
3. **External anchoring.** The rows, the labels, the committee's aggregator and
   arm A's stem are all PandoRSI's own. Nothing here transfers off this suite.
4. **Latency as an endpoint.** 1600 ms p95 crosses a LAN to a GPU box and its
   first call includes model load; arm B is in-process beside the harness. The
   column is reported and then not used to decide anything.
5. **The committee's own quality.** 002's weakest member answers 0.3583. A win
   against a committee that is being dragged by one member is evidence about
   *this* committee, not about committees.

## 反思与质疑

- **样本量 / 标准误.** n = 360 rows, 345 paired, so b + c = 46 carries all the
  evidence and 1SE = √46 = 6.78. The p is small, the *effect* is 18 rows, and
  the suite is one internally generated version with no second draw. **The
  original text of this run carried 1SE = 3.39 / z = +5.31** — that is the
  paired SE divided by two, corrected across all four reports on 2026-10-19
  (`278dc1e`, see the 勘误 below). Treat +2.65 as a directional hint that
  happens to clear noise once, not as a settled effect size.
- **缺哪个对照.** Still missing, in order of what would cost least: (a) a
  **truly equal-weight** foreign arm — win must expose the gguf bytes for that
  to be a measurement rather than a claim; (b) the **2×2** {Qwen, Gemma} ×
  {8B, 12B} at equal weight, which is the only shape that separates lineage from
  capacity; (c) a **second draw** of the same suites to give the paired SE a
  distribution rather than a single binomial.
- **证据是否外部锚定.** No. Every row was generated by the project's own
  templates and labelled by the project's own pipeline, and the examinee is a
  member of the same family of machines that produced the exam.
- **运维代价（含人类审批次数）.** 0 new timers, 0 new grants, 0 new resident
  models, **0 human approvals**. Real cost: win's own llama-server was occupied
  by ~360 sequential eval requests for the length of the run, i.e. win's
  proposer shared its model with the exam while win was still frozen on old
  code. Nothing here opened a remote execution surface to get the arm run.
- **一条可证伪的替代解释.** "Yi-1.5-9B is simply a weaker model than this
  committee's mixture" explains +18 without invoking lineage, and it predicts
  the sign should follow *capacity*, not lineage — which is exactly what 005
  tests: a non-Qwen model **above** the committee's budget flips the sign to
  −16. If lineage were doing the work, the sign would have followed lineage.
  What that falsification does **not** settle is the reverse direction: two
  points on a capacity axis are a bracket, not a curve.

## Artifacts

- `assets/votes/equal-budget-004.config.json` — pre-registration (arms, suites,
  hash, decision rule, declared limits)
- `assets/votes/equal-budget-004.raw.json` — 360 arm-A votes +
  `arm_b_reused_from` naming the donor
- `assets/votes/equal-budget-004.json` — report (per-suite, paired, verdict)
- `tools/equal_budget_check.py --reuse-arm-b` — a bracket is only a bracket if
  the held arm is held

## Standing rule this run does not change

Unless an **equal-weight** foreign-bloodline arm is measured — and the 2×2 is
run — no result from this project may be described as evidence for group
intelligence. This run is one arm of a control, and it came back **against** the
thesis on the rows it covers.

## 勘误 · 2026-10-19 · 配对标准误

本实验首次发布时写的 `1SE = 3.39` / `z = +5.31` 出自错误的配对标准误公式
（`sqrt(n)/2`，n = 两臂共同作答行数）。配对差只在不一致行上有信息：正确值是
`sqrt(b + c) = sqrt(14 + 32) = 6.782`。提交 `278dc1e` 把四份报告统一改用
`pando/common.py:mcnemar_test()`（唯一实现，含精确 McNemar p）并从缓存的原始票
就地重生成。更正后：

```
A right / B wrong = 14
B right / A wrong = 32
d = B − A = +18     1SE = 6.782     d/SE = +2.654     McNemar exact p = 0.0114
```

**判决方向没变**（规则只读 `d` 的符号），**强度变了**：`|z| = 2.65` 仍然在
1.96 之外，但 p = 0.0114 在四个变体的 Bonferroni 线（≈0.0125）内侧擦过，
所以本文件把措辞从「显著」改成「边缘」。五份报告的数字现在由
`TestPairedDifferenceStandardError` 逐份重新推导核对，报告的页面上再出现减半的
SE 会直接失败。
