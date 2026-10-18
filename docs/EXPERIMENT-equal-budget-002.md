# Experiment — equal-memory-budget control 002: four small members vs one big model

2026-10-19 · `mac-codex` · pre-registered in
`assets/votes/equal-budget-002.config.json` **before any harvest**

Pilot 001 said, in its own limitations section, that it was not the control:
its committee carried 2.343× less weight than the big model it lost to. This
run is that control. The budget gap is **1.066×** — arm B has 93.8% of arm A's
weight — and the committee is the four members this Mac can actually field.

The claim under test is the project's own founding claim: *a committee of small
models is competitive with one larger model that fits the same memory*. The
result is **negative for the committee as pre-registered**, and the reason it is
negative is specific and measured, not mysterious.

## What was run

| | arm A (one big) | arm B (small committee) |
|---|---|---|
| members | `mac-llm-8b-local` | `mac-llm-4b-local` + `mac-phi-35-mini` + `mac-laya-int8` + `mac-text` |
| checkpoints | `mac-llm-8b-q6` | `mac-llm-4b`, `phi-3.5-mini-q4km`, `laya-en`, `text-nb-v1` |
| models | Qwen3-8B-Q5_K_M | Qwen3-4B-Instruct-2507-Q4_K_M, Phi-3.5-mini-instruct-Q4_K_M, Laya int8 ONNX, a multinomial naive Bayes |
| weights | 5,851,112,224 B = **5.449 GiB** | 5,488,942,856 B = **5.112 GiB** |
| sha256 | `068bae16…` | `3605803b…`, `e4165e3a…`, `854abee1…`, `—` |
| vote | one vote, `neutral=true` | `pando.common.aggregate`, **min_quorum = 3** of 4 |
| p50 / p95 latency | 285 / 382 ms | 152 / 259 ms (see the latency caveat below) |

Both arms were asked the same 360 rows, one request per row per member, through
each member's own stem `/decide` — the same prompt and parse path the live root
uses. Arm B is scored through `pando.common.aggregate`, the live gate, not a
hand-rolled majority.

Suites: `assets/heldout4_test.jsonl` (240 rows, family `big6`) and
`assets/heldout5_test.jsonl` (120 rows, family `big7`).

**Quorum policy, stated up front.** With 4 members the gate needs 3. When
fewer than 3 answer, the tool raises the quorum to "all who answered", so a row
with 2 answers becomes *unanimity* rather than a quorum. Timeouts are therefore
a first-class failure mode and are reported per member rather than smoothed
away. Coverage was 0.975 (351/360 rows), so this trap did not bite here.

**Memory sequencing.** The hard constraint on this host is no fourth concurrent
local model. The arms therefore ran **sequentially and never co-resident**: the
8B was taken down for the 153-second arm-B harvest and restored immediately
after. Per-arm latency is reported but is **not** a between-arm endpoint —
the two arms ran under different residency, so latency would be measuring this
host's memory pressure, not the models. No new launchd job was created and the
transient stems were started without `--root` so they never entered the live
node table.

## Decision rule, fixed before the run

Let `d` = (rows where arm B is right and arm A is wrong) − (rows where arm A is
right and arm B is wrong), over rows both arms answered.

- `d <= 0` → the "many small models win at equal budget" claim is not supported.
- `d > 0` and `|d| > 1.96·SE(d)` → arm B is ahead beyond noise on these rows.
- otherwise → inconclusive at this n; no claim either way.

## Results

```
arm A  acc=0.9417  p50=285ms p95=382ms        (360/360 rows)
  member :8767  4B          acc=0.9248  neutral=True   p95=195ms   359/360
  member :8770  Phi-3.5m    acc=0.9456  neutral=True   p95=367ms   331/360
  member :8769  laya-en     acc=0.8389  neutral=False  p95=47ms    360/360
  member :8766  text-nb-v1  acc=0.3583  neutral=False  p95=1ms     360/360
arm B  acc=0.9259  coverage=0.975
paired: A right/B wrong=21  B right/A wrong=15  on 351 rows  (d=-6, 1SE=3.00)
verdict (rule as pre-registered): one-big is at least as good — claim NOT supported
```

Per suite, the same shape: on `big6` the two arms are level (0.9208 vs 0.9202 on
the rows B answered); on `big7` the big model is clearly ahead (0.9833 vs
0.9381). The all-members committee buys nothing and costs coverage.

## The rule fired, and this is the answer the project was missing

`d = −6`, `SE = 3.00`, `z = −2.00`. The many-small-at-equal-budget claim **is
not supported**, and the sign is not neutral: the single 8B is ahead by about
two standard errors while carrying 6.6% more weight. The honest reading is
"the committee did not beat the big model", with the direction against it — not
"the committee is proven worse", because `|z| = 2.0` on 360 internally-labelled
rows is a marginal result that would not survive correction for anything else
we measured in the same session.

## Exploratory (post hoc) — the deficit has a name

After the pre-registered verdict, and clearly labelled as **exploratory**, the
same raw harvest was re-aggregated with different member sets. Nothing was
re-harvested; the same per-row votes were recombined.

| member set | weights | acc | coverage | B right/A wrong | A right/B wrong | d | z |
|---|---|---|---|---|---|---|---|
| B4 pre-registered (4B+Phi+laya+text) | 5.112 GiB | 0.9259 | 0.975 | 15 | 21 | −6 | −2.00 |
| B3 drop-text (4B+Phi+laya) | 5.112 GiB | 0.9565 | 0.958 | 15 | 13 | +2 | +0.76 |
| B2 two-LLM (4B+Phi) | **4.555 GiB** | **0.9675** | 0.939 | 15 | 10 | **+5** | **+2.00** |
| B3 drop-Phi (4B+laya+text) | 2.884 GiB | 0.8611 | 1.000 | 14 | 43 | −29 | −7.68 |

Two things fall out of this, and the second is the one that matters:

1. The pre-registered committee lost **because of one member**, not because
   "small models" are worse. `text-nb-v1` scores 0.3583 alone; with it the
   committee is at 0.9259 or 0.8611, without it at 0.9565 or 0.9675.
2. The best arm-B variant here — two LLM members, 4.555 GiB, *less* memory than
   both the others — reaches 0.9675 against arm A's 0.9417. That is the
   direction the project wants, and it must not be reported as a win: it is a
   post-hoc selection among four variants (`|z| = 2.0` needs `> 2.5` for
   Bonferroni over four), chosen after seeing the labels, and the member set was
   picked by an author who already knew which member was dragging.

The confirmatory result and the exploratory result point opposite ways. The
only defensible summary of this run is: **at equal weight budget on this task
family, the answer depends on which members are in the committee — which is a
statement about member selection, not about scale.**

## What this rules out, and what it does not

Ruled out: "this Mac's four pre-registered small members, combined by the live
gate, are competitive with a single 5.449 GiB model on these 360 rows". That is
now measured, not assumed.

Not ruled out, and still owed:

- **The bracket arm.** The plan named `Qwen3-8B-Q4_K_M` (4.683 GiB) as arm A′,
  sitting on the *other* side of arm B's budget, so that parity would be
  bracketed rather than asserted. It was **not run** — the :8084 slot went to
  Phi-3.5-mini instead, which was the more valuable use of it. Parity here is a
  single point estimate (1.066×), not a bracket.
- **A second big model of a different bloodline.** Arm A is Qwen3-8B; arm B
  contains Qwen3-4B. Members of the same family sharing a tokenizer and a
  training lineage are not independent evidence, and part of what looks like
  "swarm ≈ big" may just be "same family, smaller".
- **External anchoring.** Every `expected` label comes from this project's own
  generation pipeline. Nothing here is anchored to an outside authority.
- **A hard floor to compare against.** No single-4B or majority-of-two baseline
  was run as a separate arm, so "how much of arm B is the 4B alone" is inferred,
  not measured.

## Live consequence worth acting on

`text-nb-v1` is not a research curiosity; on this host it is `mac-text`, and on
pi5 it is `pandosi-pi5-x`. Both are registered with `role=decision`, i.e. they
are **voting members of the live governance loop**, while the LLMs are
`role=proposal` and cannot vote. This run measured that member at 0.3583
accuracy on this family, the lowest number any member has produced, and it is
one of the votes the project is currently governing itself with.

Whether the role split (`text-nb` votes, LLMs propose) is still the right one is
a decision for the civilisation, not for this document. The measurement is:
the voting half of the live loop contains the single weakest member measured so
far, and removing it flips the sign of this experiment.

## Artifacts

- `assets/votes/equal-budget-002.config.json` — pre-registration: model files,
  sha256, weight bytes, endpoints, gate, quorum policy, memory sequencing
- `assets/votes/equal-budget-002.raw.json` — every per-row vote from all five
  stem instances: arm A's 360, plus 1,440 arm-B requests of which 1,410 were
  answered (30 HTTP 500s: 1 on the 4B, 29 on Phi)
- `assets/votes/equal-budget-002.json` — the scored report
- `tools/equal_budget_check.py` — harvest + score + coverage-aware verdict

## 反思与质疑

- **样本量/标准误**：360 行上单臂 accuracy 的 1 SE ≈ 1.25pp，配对
  discordance 的 1 SE = 3.00 行。主结果是 `d=−6`（z=−2.00），探索里最好的
  变体是 `d=+5`（z=+2.00）——**两个都在 1.96 的边缘上**。在同一个 harvest
  上试了 4 个成员组合，Bonferroni 后需要 `|z|>2.5`，所以探索表里**没有
  任何一个变体达到可宣称的门槛**。这一节的存在就是为了防止下一轮把
  `+5` 当成结论引用。
- **缺哪个对照**：这次补上了主对照（1.066× 等预算），但仍然缺三件：
  (a) 计划中的 arm A′ 括号臂（8B-Q4，4.683 GiB）没跑，等预算只是点估计
  不是区间；(b) 没有**不同血统**的第二台大模型，arm A 是 Qwen3-8B、arm B
  里有 Qwen3-4B，同族同 tokenizer，"多台"里有相当一部分不是独立证据；
  (c) 没有单 4B 或"两个 LLM"的正式对照臂——探索表里的 B2 是事后选的，
  不是预先注册的臂。
- **证据是否外部锚定**：**完全没有。**360 行标签全部出自本项目自己的生成
  管线，`big6` 上两臂并列、`big7` 上两臂差 4.5pp，这个差异本身说明标签族
  的难度是被切出来的而不是自然抽样的。所以上面所有 0.92/0.94/0.97 只能读
  作"复现了本项目自己的标签"，不能读作真实意图上的正确率。
- **运维代价（含人类审批次数）**：arm B 需要 3 个并发模型服务器（4B + Phi
  + laya），harvest 153 秒，为了不违反"本机不上第四个模型"的硬约束必须
  先把 8B 停掉再恢复——**这就是"等预算"的真实代价：换来的不是更聪明，
  是一套必须顺序调度、且不能在线上共存的部署**。本轮人类审批 0 次，但这是
  approval=never 的环境赠品，不是设计省的；正常机器上起 server、改 launchd、
  恢复服务至少 3 次审批。另外 Phi 在 360 行里有 29 次 HTTP 500，说明这个
  成员本身不够稳，做常驻投票者之前需要先解决。
- **一条可证伪的替代解释**：委员会输，可能**不是**因为"小模型不行"，而是
  因为**聚合器把最弱的成员当平等投票者**。这已经在上面被直接检验了：把
  `text-nb-v1` 拿掉，`d` 从 −6 变 +2；再把 laya 也拿掉，变 +5。所以"小模型
  在等预算下必然输"这个解释被否证了。但反过来说，剩下那个"两个小 LLM 赢
  2.6pp"的观察同样有一条不需要群体智能的解释——**同族 4B 与 8B 的错误高度
  相关**，而 Phi 补进来的恰好是 8B 也容易错的那批行；要证伪它，需要在不
  一致行上分别测两臂的成员准确率，本 run 没有做这件事。**在它被做之前，
  任何方向的结果都不得称为对群体智能的证明。**

## 探索性补充：最弱成员是"弱"还是"标签错了"（`tools/vote_ablation.py`）

同一个 harvest，没有起任何模型、没有联网。工具见 `tools/vote_ablation.py`
（d17b1f4）；本节所有数字都是**事后选的**，不是预注册臂。

**最弱成员**是 B 臂的 `:8766`（text-nb-v1）：acc 0.3583（n=360），
per-suite big6=0.4542 / big7=0.1667。

**它不是随机回答。** 14 类的偶然水平是 1/14 = 0.0714，而它的
SE = 0.0253，所以 z-vs-chance = **+11.35**。意思是：它几乎全错，但错得
有结构——它在某些类上系统性地偏向另一些类。这否证了"text-nb 是个坏掉的
随机数生成器"。

**按类召回（最差在前）**：

| label | recall |
|---|---|
| permission_escalation | 0/20 = 0.00 |
| content_appeal | 1/20 = 0.05 |
| praise | 0.19 |
| complaint | 0.20 |
| reschedule | 0.20 |
| billing_clarify | 0.25 |

**它是不是两个标签接反了？** 取混淆矩阵里最大的 6 个非对角元做**具名**
标签互换，Bonferroni 门槛（7 个假设）|z| > 2.69：

- rescued（能救回来的）：**none**
- significantly worse（显著更差）：`greeting↔login_issue` z=−3.41、
  `greeting↔refund` z=−2.97

所以"标签映射接反了"这个解释**没有被证据支持**：没有任何一对互换能把它救回来，
而有两对显著把它弄得更坏。它就是更差，不是被贴错名字。

**为什么不用自由置换做上界**：14 个标签、每类约 26 行，自由拟合的置换可以
把**每一个**成员——包括 0.3583 这个——都做到 1.000。所以自由置换的上界在这里
是废数，它证明不了任何事。这也是这个工具用**具名**配对 + Bonferroni 的原因。

### 反思与质疑

- **样本量/标准误**：n=360 行（big6 240 / big7 120），单类最少 20 行。
  单类召回 0/20 的单侧 95% 上界是 0.139（Clopper-Pearson，不是 0.17），所以"permission_escalation 完全不会"
  这个说法本身**不成立**，只能说"≤0.17"。z-vs-chance=+11.35 是全样本的，
  不受单类样本量影响；-3.41/-2.97 这两条在 7 个假设的 Bonferroni 下刚好活着，
  换一个配对子集就会掉出去。
- **缺哪个对照**：仍然缺 (a) 8B-Q4 括弧臂，(b) 不同血统的第二台大模型，
  (c) 一个**同血统**小模型的对照——没有它，"text-nb 弱"和"text-nb 这个
  checkpoint 弱"分不开。最关键的那个对照（等内存预算下多台小模型 vs 一台
  大模型）已经有了，但结论是**不支持**。
- **证据是否外部锚定**：完全没有。标签、题目、被考模型都是本项目自己产的，
  big6 与 big7 的难度差本身说明标签族是被切出来的。所以 0.3583 与 0.9456
  都只能读作"复现了本项目自己的标签"。
- **运维代价（含人类审批次数）**：本工具本身 0 进程、0 网络、0 审批，是纯读。
  真正的代价在**结论的另一端**：要么给 text-nb 加对照重训，要么把它从决策席
  拿掉——两者都不是本轮能顺手做的。
- **一条可证伪的替代解释**：它的低分可能不是"模型差"，而是**它在按自己
  训练时的标签体系作答**，那套体系与本 harness 的 14 类只是碰巧同名。
  证伪方法：给它一组**它自己体系内**的对照题，若那些题上它显著高于 0.3583，
  则"模型差"被否证、"harness 失配"成立。本 run 没做这件事，所以
  "text-nb 很弱"这句话现在是**证据不足**，不是**已确立**。
