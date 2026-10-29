# Experiment — equal-memory-budget control 003: the bracket arm A′ (8B-Q4_K_M, *smaller* than the committee)

Date: 2026-10-19 · Author: mac-codex · Machine: Apple arm64, 10 cpu, 16 GiB
Pre-registration: `assets/votes/equal-budget-003.config.json` (written before any row was harvested)
Raw votes: `assets/votes/equal-budget-003.raw.json` · Report: `assets/votes/equal-budget-003.json`

## Why this run exists

002 compared one big model against a committee and reported **1.066× — a single
point estimate, not a bracket**: the committee was *handicapped* by 6.2% of
weight, so a reader could not tell "the committee is competitive" from "the
committee was given less memory and still nearly tied". 002 said so itself and
named the missing arm:

> The plan named `Qwen3-8B-Q4_K_M` (4.683 GiB) as arm A′, sitting on the other
> side of arm B's budget… It was **not run**.

This is that owed arm, and nothing else. In 002 the big model had **more**
memory than the committee. Here the big model has **less** — 4.682 GiB against
the committee's 5.112 GiB, i.e. **the committee carries 109.2% of arm A′'s
weight and is budget-favoured by 9.2%**. Two runs that tilt the same comparison
in opposite directions are a bracket; one run is an anecdote with a number.

## What moved, and what was held frozen

- **Moved (arm A only):** `Qwen3-8B-Q6_K_M` (5.449 GiB) → `Qwen3-8B-Q4_K_M`
  (4.682 GiB). Same family, same launcher flags, same stem code path.
  Model sha256 `d98cdcbd…45785`, verified on disk at pre-registration **and
  again at harvest time**; it is the exact hash 002 pre-registered.
- **Held frozen (arm B):** the same four members
  (`:8767` 4B, `:8770` Phi-3.5-mini, `:8769` laya, `:8766` text-nb) with their
  002 votes adopted verbatim — new tool path
  `tools/equal_budget_check.py --reuse-arm-b`.
  The tool **refuses** the donor unless suite, row count and member URLs match
  one for one, and exits 2 rather than silently improvising a fresh arm B.
  Reason: re-harvesting the held arm would re-roll it (a timeout or a sampler
  decision on the second pass), and any between-runs delta would then be read as
  a between-arms delta. Three tests cover the adopt/refuse paths.

The suite is byte-identical to 002's (`heldout4` 240 rows + `heldout5` 120 rows,
sha256 recorded in the config), so the *question* is the same even though the
wall clock is not.

## Commands actually run

```bash
# boot the resident Q6 stack out (mac is not to carry a 4th concurrent local model)
launchctl bootout gui/$(id -u)/com.pandosi.stem.llm8b
launchctl bootout gui/$(id -u)/com.jewelroam.pandosi-llm8b

# transient arm A′ server + stem (no launchd job, stem started WITHOUT --root)
llama-server -m ~/.pandosi/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8084 --api-key-file ~/.pandosi/llm-api-8b.token \
  --alias qwen3-8b-q4km-instruct --jinja \
  --chat-template-kwargs '{"enable_thinking": false}' \
  -ngl 99 -c 4096 --parallel 1 --temp 0 --metrics
.venv/bin/python -m pando.worker --node-id bracket-8b-q4 --checkpoint mac-llm-8b-q4 \
  --backend openai --openai-url http://127.0.0.1:8084/v1 \
  --openai-key-file ~/.pandosi/llm-api-8b.token \
  --openai-model qwen3-8b-q4km-instruct --host 127.0.0.1 --port 8771

# 6-row plumbing sanity check before committing 360 rows (6/6)
# then the bracket harvest:
.venv/bin/python tools/equal_budget_check.py \
  --suite assets/heldout4_test.jsonl,assets/heldout5_test.jsonl \
  --big http://127.0.0.1:8771 --big-checkpoint mac-llm-8b-q4 --big-weight-gb 4.682 \
  --small http://127.0.0.1:8767,http://127.0.0.1:8770,http://127.0.0.1:8769,http://127.0.0.1:8766 \
  --small-weight-gb 5.112 \
  --reuse-arm-b assets/votes/equal-budget-002.raw.json \
  --votes assets/votes/equal-budget-003.raw.json \
  --out assets/votes/equal-budget-003.json

# restore
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jewelroam.pandosi-llm8b.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pandosi.stem.llm8b.plist
```

Verified after restore: `8086`, `8768`, `8087` all return 200; nothing is left
listening on `8084`/`8771`.

## Results

| arm | weights | GiB | accuracy | coverage | p50 | p95 |
|---|---|---|---|---|---|---|
| A′ one big — `Qwen3-8B-Q4_K_M` | 5,027,783,488 B | **4.682** | **0.9444** | 1.000 | 256 ms | 360 ms |
| B committee — 4 members | (4 frozen members) | **5.112** | 0.9259 | 0.975 | — | — |

Per member (arm B, votes taken from 002): `:8767` 4B 0.9248 · `:8770` Phi-3.5-mini
0.9456 · `:8769` laya 0.8389 · `:8766` text-nb 0.3583.

Paired, on the 351 rows both arms answered:

```
A′ right / B wrong = 22
B right / A′ wrong = 15
d = B − A′ = −7     1SE = 6.083     d/SE = −1.15
```

Per suite: `big6` A′ 0.9250 vs B 0.9202; `big7` A′ 0.9833 vs B 0.9381.

## The bracket, read straight

```
002:  committee at  93.8% of arm A's weight   d = −6,  z = −1.00   one-big at least as good
003:  committee at 109.2% of arm A′'s weight  d = −7,  z = −1.15   one-big at least as good
```

The committee was handicapped in one run and favoured in the other, and the
sign and rough size of `d` did not follow the budget. On this task family,
**the 002 verdict is not an artefact of how the budget was tilted** — that is
what the bracket buys, and it is the whole reason this arm was owed.

Under the decision rule fixed before the run, `d <= 0` means: *a single
`Qwen3-8B-Q4_K_M` is at least as good while carrying **less** memory than the
4-member committee, so the many-small-at-equal-budget claim is refuted on this
family, not merely unsupported.* The rule keys on the **sign of `d`**, not on
significance: `|z| = 1.15` is inside the noise, so this is not a significant
refutation — it is the pre-registered rule firing on direction alone. (The
earlier text here quoted `|z| = 2.30 > 1.96`; that `z` came from a paired SE
twice too large, corrected 2026-10-19 — see 勘误 below.)

This is a **negative result for the project's own thesis**, and it is reported
that way on purpose. A swarm whose weakest member answers 0.3583 while the best
member answers 0.9456 is not being held back by a shortage of members; it is
being held back by a member that is confidently wrong, and `aggregate()` is
carrying it.

## What this does and does not rule out

**Does rule out.** The specific claim "at equal weight budget, committee-of-small
is competitive with one bigger model" — for this suite, this parse path, and
this model family — under a bracket that tilts the budget both ways.

**Does not rule out.**
1. **A different-bloodline big model.** Arm A and arm A′ are *both Qwen3*. The
   committee's best member is `Phi-3.5-mini`. Everything here compares a family
   against a *mixture*; a non-Qwen big model is still owed and has never been run.
2. **A better committee.** Dropping the 0.3583 member and re-combining the
   remaining three reaches 0.9565 in 002 — but that variant was chosen after
   seeing the result, it is one of four variants tried, and
   Bonferroni-corrected it needs `|z| > 2.5`. It may not be quoted as a win.
3. **Anything outside this suite.** The rows come from the project's own
   generation pipeline. Internally consistent, externally unanchored.
4. **Latency as a between-arm endpoint.** A′ p50 256 ms is *faster* than the
   committee's slowest member (Phi 173 ms p50, 367 ms p95) but the two arms were
   harvested in different residency configurations and at different times, so
   latency is reported and then not used to decide anything.

## 反思与质疑

- **样本量 / 标准误.** n = 360 rows, both arms answer 351, so the paired
  statistic rests on 351 pairs. `d = −7` with `1SE = 6.08` is `z = −1.15` —
  **inside** the 1.96 line, not past it. The numbered copy of this document
  said `1SE = 3.04, z = −2.30`, which double-counted the evidence: the paired
  SE is `sqrt(b+c) = sqrt(22+15)`, and the old `sqrt(n)/2` is the SE of a
  difference of two *independent* proportions. So the honest strength is: two
  runs lean the same way (z = −1.00, −1.15) and **neither clears the noise
  floor**; the verdict survives only because the pre-registered rule reads the
  sign of `d`. One suite, one prompt variant, one seed-free `temp 0` decode.
  McNemar exact two-sided p = 0.324 (003), 0.405 (002). Treat `d = −7` as a
  directional hint, not a settled effect size.
- **缺哪个对照.** The bracket closes the *budget* axis and leaves the
  *bloodline* axis open: arm A and arm A′ are the same family. The control that
  would still falsify "many small models are competitive" most cheaply is a
  **non-Qwen big model at the same 4.68 GiB** (e.g. a Llama/Mistral-class Q4 of
  comparable size). Until that exists, the correct sentence is "a Qwen3-8B beats
  this committee", not "a big model beats this committee".
- **证据是否外部锚定.** No. Every row was generated from the project's own
  templates and labels. The 0.3583 member scoring 1 ms/row with a real
  `neutral=False` probability is a warning sign about the label distribution,
  not a fact about the world. Nothing here transfers without an external suite
  that the project did not author.
- **运维代价（含人类审批次数）.** 0 new launchd jobs, 0 new timers, 0 new
  grants, **0 human approvals**. Cost paid: the resident 8B-Q6 member
  (`mac-llm-8b-local`, role=proposal, so no quorum is affected) was offline for
  ~15 minutes; proposer degraded to its 4B fallback on `:8080` for that window;
  the arm A′ server had to be started outside the supervision tree, so a crash
  would have silently produced a short harvest rather than a loud one — the
  360/360 row counters were checked because of exactly that.
- **一条可证伪的替代解释.** "Q4 quantisation hurt the big model" would mean the
  bracket understates one-big. It is falsified by the direction of the numbers:
  **A′ (4.682 GiB, Q4) scored 0.9444, higher than 002's 5.449 GiB Q6 arm at
  0.9417.** Losing 0.77 GiB and a quantisation step did not cost the big model
  anything measurable on these rows — which also means the *equal-budget* framing
  understates the real gap, since the big arm was still ahead while smaller.
  A competing explanation that is *not* falsified: text-nb's 0.3583 drags the
  committee below a 3-member committee, and a 3-member committee might tie
  — that is the un-run variant from point 2 above.

## Artifacts

- `assets/votes/equal-budget-003.config.json` — pre-registration (weights, sha256, decision rule, sequencing)
- `assets/votes/equal-budget-003.raw.json` — 360 arm-A′ votes + `arm_b_reused_from`
- `assets/votes/equal-budget-003.json` — report (per-suite, paired, verdicts)
- `tools/equal_budget_check.py --reuse-arm-b` — the honesty primitive; a bracket is only a bracket if the held arm is held

## Standing rule this run does not change

Unless the "equal-memory-budget, many small models vs one big model" control
holds **and** the missing different-bloodline arm is run, no result from this
project may be described as evidence for group intelligence. This run is a
control that came back against the thesis. Recording it is the point.

## 勘误 · 2026-10-19 · 配对标准误

本文件里的 `1SE = 3.041, d/SE = −2.30`、以及两处 `|z| = 2.30 > 1.96` 出自一个
错误的配对标准误公式：工具当时用 `sqrt(n)/2`（n = 两臂共同作答行数），而配对差
只在不一致行上有信息，正确值是 `sqrt(b+c) = sqrt(22+15) = 6.083`。

提交 `278dc1e` 换成 `pando/common.py:mcnemar_test()`（唯一实现，含精确 McNemar
p），四份报告从缓存原始票就地重生成。本文件更正后：

```
A′ right / B wrong = 22
B right / A′ wrong = 15
d = B − A′ = −7     1SE = 6.083     d/SE = −1.15     McNemar exact p = 0.324
```

**判决没变**，因为预注册规则读的是 `d` 的符号：`d = −7 < 0` → "one-big is at
least as good"。**强度变了**：`|z| = 1.15` 在 1.96 之内，"bracket 两臂同向"
这句话现在说的是**方向**同向，不是**显著**同向。002 同理，从 −2.00 更正为
−1.00。任何"两次 z 都超过 2"的说法都必须撤掉。
