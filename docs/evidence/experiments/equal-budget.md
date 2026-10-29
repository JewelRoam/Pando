# Experiment — equal-memory-budget control: many small vs one big (pilot 001)

Date: 2026-10-17. Task `t-20261017-035259-bce2` (from win-codex).
Branch `codex/equal-budget-pilot`, on top of `5491fcb`.

## Why this experiment exists

Every result this project has produced compares the swarm to *itself* or to a
worse swarm. The claim the project actually makes — that a committee of small
models is competitive with one larger model that fits the same memory — has
never been measured, because the fleet never held a big model and a matched
budget of small ones side by side. Without that control, "the group works" and
"we happened to run several models" are the same observation.

This pilot is the smallest run that can still contradict the claim. It was run
before the equal-budget arms existed, and **it is not the equal-budget
control**. Read the rest of this section before the numbers.

## What was run

| | arm A (one big) | arm B (small committee) |
|---|---|---|
| members | `mac-llm-8b-local` | `mac-llm-4b-local` + `mac-text` |
| checkpoint | `mac-llm-8b-q6` | `mac-llm-4b` + `text-nb-v1` |
| model | Qwen3-8B-Q5_K_M | Qwen3-4B-Instruct-2507-Q4_K_M + a multinomial naive Bayes |
| weights | 5.449 GiB | 2.326 GiB |
| sha256 | `068bae16…` | `3605803b…` |
| vote | one vote, `neutral=true` (choice only) | combined by `pando.common.aggregate` |
| p50 / p95 latency | 353 / 465 ms | 190 / 269 ms (the n-gram costs 0.4 ms) |

Both arms were asked the same 360 rows, one request per row per member, through
each member's own stem `/decide` — the same prompt and parse path the live root
uses. Arm B was scored through `pando.common.aggregate`, the live gate, not a
hand-rolled majority.

Suites: `assets/heldout4_test.jsonl` (240 rows, family `big6`) and
`assets/heldout5_test.jsonl` (120 rows, family `big7`). 360 rows total.

**The arms are not at equal budget.** Arm B has **2.34× less weight** than arm
A. Arm A is a single 5.449 GiB model; arm B is 2.326 GiB of small models plus a
n-gram small enough not to be worth counting. This is deliberate, and it makes
the test *one-sided*: if arm B had won, that would have been evidence for the
thesis under a handicap. It did not.

## Decision rule, fixed before the run

Let `d` = (rows where arm B is right and arm A is wrong) − (rows where arm A is
right and arm B is wrong), over rows both arms answered.

- `d <= 0` → the "many small models win at equal budget" claim is not supported
  by this pilot.
- `d > 0` and `|d| > 1.96·SE(d)` → arm B is ahead beyond noise on these rows.
- otherwise → inconclusive at this n.

## Results

| split | rows | arm A (8B, one big) | arm B answered | arm B acc *on answered* |
|---|---:|---:|---:|---:|
| `big6` | 240 | 0.9208 | 103 (43%) | 0.9903 |
| `big7` | 120 | 0.9833 | 25 (21%) | **0.8000** |
| **all** | 360 | **0.9417** | 128 (35.6%) | 0.9531 |

Per member, over all rows it answered:

| member | acc (all rows) | acc `big6` | acc `big7` | p95 |
|---|---:|---:|---:|---:|
| `mac-llm-4b` (4B) | 0.9248 | 0.9167 | 0.9333 | 269 ms |
| `text-nb-v1` (n-gram) | 0.3583 | 0.4542 | 0.1667 | 0.7 ms |

Paired, over the 128 rows arm B was willing to answer: A right/B wrong = 5,
B right/A wrong = 10, `d = +5`, `SE = 1.94`, i.e. **2.58 SE in arm B's
favour** — the rule as pre-registered fired for the committee.

## The rule fired, and it should not have

**Arm B answered 128 of 360 rows.** On a 2-member quorum the members disagree on
a tie, and `aggregate` refuses to decide a tie on the strength of one
probability-bearing vote, so every disagreement becomes an abstention. The two
members agreed on 35.6% of rows.

That means arm B **picked its own denominator**. Restricting the comparison to
the rows it answered hands it the rows it already agreed on — which are, of
course, the rows where its strong member is confident:

- 4B accuracy on the 128 answered rows: **0.9606**
- 4B accuracy on the 232 rows it abstained on: **0.9013**

So the "win" is a selection effect. Scored the way a decision system is
actually scored — over all rows, an abstention being no answer — arm B delivers
a decision on 35.6% of rows and would have to escalate the other 64.4%. It is
not competitive with a single voter that answers everything in 465 ms at p95.

**The pre-registered rule was wrong, and the flaw was not in the data.** A rule
that only looks at accuracy among answered rows rewards abstention: an arm can
shrink its denominator until its accuracy is high. `tools/equal_budget_check.py`
now reports coverage first, scores arm B over all rows with an abstention
counted as no answer, and refuses to call a win when coverage is below 0.9. The
raw votes are unchanged; only the verdict changed, and both the pre-registered
verdict and the coverage-aware verdict are recorded in the report.

Two more things the answered subset shows, both against the committee:

- On the rows it answered, the committee (0.9531) is **worse than its own strong
  member alone** (0.9606). The n-gram did not add; it subtracted. This is the
  same shape the project already measured when a cheap correlated member was
  seated.
- On `big7` the committee answered 25 of 120 rows and was 0.80 on them, while
  the 8B was 0.983 overall on that suite.

## What this pilot does show

1. **The size gap is family-dependent, and on one family it is inside the
   noise.** On `big6`, one 8B at 5.449 GiB scores 0.9208 and one 4B at
   2.326 GiB scores 0.9167 — a 0.4 pp difference, `d = −1`, `SE = 2.78`
   (0.36 SE). Two-and-a-third times the memory bought **nothing measurable**
   on 240 rows. On `big7` the same 8B is 5.0 pp ahead and **never loses a row to
   the 4B** (A right/B wrong = 5, B right/A wrong = 0).
2. **The n-gram second opinion is a liability on this task**, not an asset:
   0.3583 overall and 0.1667 on `big7` (8 classes, so chance is 0.125). It is
   also the only member of arm B that carries a probability, so arm B's
   confidence signal *is* the n-gram's confidence.
3. **The cheap member is nearly free in wall-clock time** (0.4 ms mean against
   190 ms for the 4B), so the cost of the committee is not latency. It is that
   disagreement converts decisions into abstentions, and this project's gate
   treats an abstention as escalation, not as a tenth of a vote.

## Limitations — what this cannot rule out

- **This is not the equal-budget control.** Arm B ran on 2.34× less weight.
  A defeat under a handicap is not a defeat at parity, and a win here would not
  have been a win at parity either.
- **A true equal-budget arm is not reachable on this host today.** The honest
  pairing is `Qwen3-8B-Q4_K_M` (4.686 GiB, already on disk) against two
  distinct ~2.3 GiB small models (4.652 GiB) — a 0.7% weight match. The block is
  not memory-headroom (the arms can run *sequentially*, never more than two
  llama-servers at a time): it is that this machine holds only **one** small
  LLM weight set. A second distinct ~4B-class GGUF is the missing resource.
  Duplicating `Qwen3-4B` on a second port would be two copies of one lineage,
  which the project already showed agrees with itself on 240/240 rows — a fake
  committee, and the pilot was not run that way.
- **Limits:** 1 SE on an unpaired accuracy over 360 rows ≈ 1.2 pp; the paired
  discordance 1 SE is 3.0 rows. One host, one prompt variant, one decode
  temperature (0.0), one seed. Neither arm was tuned, which is also to say
  neither arm is at its best.
- **Internal labels, external anchor: none.** Every row and every `expected`
  label comes from the project's own generation pipeline. This is internally
  consistent and externally unanchored. A 0.94 here means "reproduces this
  project's own labels", not "is right about customer intent".
- The operator cost of *running* this pilot (three live stems, a swapped-out
  16 GiB box, ~4 minutes of sequential harvesting, one memory-pressure
  incident avoided by refusing a third model) is not in any denominator above.

## Artifacts

- `tools/equal_budget_check.py` — harvest + score + coverage-aware verdict
- `assets/votes/equal-budget-pilot-001.raw.json` — every per-row vote, all three
  members, including the one 4B timeout that makes its n 359
- `assets/votes/equal-budget-pilot-001.json` — the scored report
- `assets/votes/equal-budget-pilot-001.config.json` — model files, sha256,
  weight bytes, measured footprint, endpoints, gate, commit
- `assets/votes/README.md` — the vote-file conventions

## 反思与质疑

- **统计功效**：360 行上单臂 accuracy 的 1 SE ≈ 1.2pp，配对 discordance
  的 1 SE = 3.0 行（`d=+5` → 2.58 SE）。但真正的问题不是功效，是分母：
  委员会只在 128/360 行上作答，它的"胜"是在自己挑的 128 行上算的。按全
  量 360 行、弃权记为空，committee = 0.3389，arm A = 0.9417。**任何小于
  1.96 SE 的差异都不得称为提升**——这里连"差异"本身都被选择偏倚污染了。
  `big7` 上 8B 对 4B 是 5:0（1 SE = 1.12），这个是超出噪声的；`big6` 上
  −1（1 SE = 2.78）**不是**，所以"大模型没有优势"这句话只能在 `big6` 这个
  族上说，且只能说"没有测得优势"。
- **缺对照**：**这次仍然没有做那个能证伪项目立论的对照。** 严格的等内存
  对照需要 `8B-Q4`(4.686 GiB) vs 两套不同的 ~2.3 GiB 小模型(4.652 GiB)；
  8B-Q4 的权重就在盘上，缺的是第二套**不同血统**的小模型权重。本机只有一
  套 4B，把同一套权重起两个端口就是 240/240 自洽的假委员会。所以本任务
  交付的是"最小还能证伪的试点"，它证伪的是**当前这台机器上能组出的那个
  委员会**，不是"等预算下多台小模型 vs 一台大模型"这个命题。要回答后者，
  需要一次下载（外网/许可），那需要用户授权。
- **外部锚点**：评测集与标签**全部出自本项目自身的生成管线**——**内部自洽、
  外部未锚定**。这里所有 0.92/0.94/0.98 都只说明"复现了本项目自己的标签"，
  不说明这些决定在真实客户意图上是对的。文本成员在 `big7` 上 0.1667、
  在 `big6` 上 0.4542，同一个模型在两个族上差 29 个点，这个差距本身就说明
  标签族的难度/分布是被人为切出来的，而不是自然抽样。
- **运维代价**：常驻内存 8B 2.7 GB 实测 / 4B 2.3 GB 实测（swap 已用 20.5/21.5 GB），
  p95 延迟 465 ms vs 269 ms；本试点为了不触发第三个模型的内存事故，特意选择
  顺序访问、拒绝再加一个服务器。**人类审批次数**：本轮 0 次（danger-full-access、
  approval=never），但这是环境给的，不是项目省出来的；换到需要 sudo/外网的机器
  上，同一件事至少 3 次审批（起 stem、起 server、拉权重）。这些都不该记 0。
- **可证伪的替代解释**：committee 在 128 行上的 0.9531 有一个完全不需要
  "群体智能"的解释——**成员一致 = 简单样本**。4B 在一致行上 0.9606、在不一致
  行上 0.9013，两组的差（5.9pp）远大于任何成员间的差；也就是说"一致"筛出来的
  是难度，不是能力。要证伪这个解释，需要一个成员间错误**真正独立**的组合：
  在一致行和不一致行上分别测每个成员的准确率，若不一致行上的成员准确率不比
  一致行低，选择效应就不成立。本任务无法回答它声称回答的问题——**它测的是
  "这台机器上现成的委员会能不能用"，不是"群体是否优于单体"。**
