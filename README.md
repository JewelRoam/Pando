# PandoSI

**The forest that is one tree.**

PandoSI is a research project building an elastically scalable, fully
heterogeneous, self-replicating **unified intelligence** — many cheap
devices, one mind.

- **Pando** — the largest single organism on Earth: an aspen colony of
  ~47,000 stems that looks like a forest but shares one root system.
  Latin: *I spread*.
- **SI** — Super Intelligence.

## What works today (all verified on real hardware/software)

- **Decision-model swarm**: 3 Laya lineages (421M/322M, Apache-2.0,
  non-autoregressive, single forward pass) vote with calibrated
  probabilities; quorum requires checkpoint diversity.
- **Heterogeneous execution**: one Intel AI PC = multiple ExecutionUnits.
  Live-verified: CPU (106 ms) + iGPU via GatherND graph surgery (25.6 ms,
  4x CPU). NPU blocked on op support (documented, 4 repair paths).
- **Non-OpenVINO accelerator path**: an RTX 3070 joins as an independent
  ONNX Runtime CUDA stem. Same graph/inputs: CUDA p50 15.0 ms vs CPU
  125.6 ms (8.36x), with 14/14 decision parity and <=0.0009 confidence
  delta across the English suite.
- **Four-runtime HTTP consensus**: CPU ORT + RTX 3070 CUDA + Intel iGPU
  OpenVINO + Intel NPU OpenVINO registered as four checkpoint-distinct
  workers and returned four votes per decision. Parallel quorum fan-out
  reduced steady-state wire p50 from 430.5 ms to 237.8 ms, and per-stem
  EWMA memory now gives unresponsive workers a 0.7-1.0 s learned budget
  instead of the 30 s global timeout.
- **Controlled slow-node isolation**: a 2 s slow worker was timed out,
  removed from the next 24 ms decision, and retried after cooldown.
- **LAN-interface protocol**: root and stems can use a real LAN address
  (`--host` / `--advertise-host`) instead of loopback; same-host latency
  showed no penalty, while true remote-network effects remain untested.
- **Remote-link degradation recovery**: a proxied worker degraded from
  100 ms to 1.5 s, was timed out and skipped in a 16 ms decision, then
  rejoined after cooldown when the link recovered.
- **Runtime elastic membership**: one running root moved from 3 to 5 workers
  by live registration, then back to 4 after heartbeat expiry, with a
  successful decision at every size.
- **Live CUDA hot-plug**: Intel CPU/iGPU/NPU served the root first; an RTX
  3070 joined the running swarm after model load and expanded the quorum
  checkpoints from 3 to 4 without a restart.
- **Hierarchical two-group consensus**: a bridge represents one remote
  group as a single checkpoint; when that group was partitioned, the local
  three-worker quorum continued and skipped the bridge after cooldown.
- **CUDA RLCD training**: a real `laya.train.finetune` run on the RTX 3070
  used 1.74 GB peak memory, produced a reloadable CUDA checkpoint, and
  preserved 14/14 English decision labels while changing confidence.
- **Multi-step CUDA evolution**: ten exported live-lineage contrast pairs
  produced nine optimizer updates on the 3070; decision labels stayed
  stable (14/14), exposing the schema conflict fixed in round 76.
- **Canonical decision schemas**: question identity now includes a hash of
  the option set, so en/ml/td rollouts with incompatible label spaces can
  no longer be silently merged during RLCD export.
- **Held-out RLCD check**: after removing 9 training-overlap states, 31
  held-out tasks kept 31/31 labels but showed no accuracy gain, which is
  the honest current limit of ten contrast pairs and one epoch.
- **Confidence overfitting detected**: with a deterministic 34/6 split and
  20 contrast rows, 72 CUDA updates left held-out accuracy unchanged while
  increasing confidence, including on one wrong answer. The small-data
  training branch is stopped.
- **Expanded RLCD pool**: 78 new multi-lingual prompts yielded 45 contrast
  rows and a 41-update CUDA candidate. Held-out accuracy stayed unchanged,
  but confidence drift fell versus the over-trained small-data run.
- **Live A/B promotion gate**: the v7 candidate was exported to ONNX and
  placed beside the baseline in the real root. It agreed 6/6 on holdout but
  produced no accuracy gain and became 0.9996 confident on a wrong answer,
  so it was not promoted.
- **Post-training calibration**: a temperature fit on 22 labeled items
  reduced v7's wrong-answer confidence from 0.9996 to 0.9316, but holdout
  accuracy and calibration evidence remained insufficient for promotion.
- **Spore parent provenance**: three-hop reproduction now carries monotonic
  generation and parent-manifest hashes; a forged lineage link is rejected.
- **Spore reproduction DAG**: branching child lineages are validated for
  roots, missing parents, generation gaps, and optional single-chain mode.
- **Cached parent lineage**: received manifests are stored by hash, and
  `join` distinguishes verified lineage from missing-history evidence.
- **Peer-fetched lineage**: a node holding only a descendant pulls the
  missing parent manifest from its serving peer by content hash
  (`/spore/manifests/<hash>`), checks hash + Ed25519 signature, and upgrades
  lineage to verified — a wrong-hash or bad-signature manifest is refused.
- **Runtime-agnostic quorum**: a native PyTorch Laya stem (`--backend laya`,
  `torch-cuda` checkpoint) votes in the same consensus as the OpenVINO/ONNX
  Runtime Intel stems and the ONNX Runtime CUDA stem — four workers, two
  ecosystems, one HTTP contract, tier `auto`. The contract itself is
  documented with no framework named in it (`docs/STEM-PROTOCOL.md`).
- **Pipeline staging, measured**: `tools/stage_split.py` cuts the Laya graph
  at a node boundary (shared external weights, verified numerically exact
  per provider) and `tools/stage_pipeline_check.py` chains the halves over
  real HTTP. The activation interface is only 0.15–1.2 MB and the loopback
  hop ~1–2 ms, but staging still does **not** pay on one shared CPU, and
  offloading half the model to a slower device is 0.47x. Replicate-and-vote
  stays the default; stage only on disjoint devices with comparable
  throughput (`docs/research/2026-10-14-round94.md`).
- **Ecosystem-independent votes**: the same Laya checkpoint run through ONNX
  Runtime (CPU) and native PyTorch (CUDA) agreed on **27/27** choice tasks
  with a 0.3% mean confidence gap and identical accuracy — heterogeneous
  replicas are interchangeable voters, and ensemble diversity must come from
  the checkpoint, not the runtime
  (`tools/cross_ecosystem_check.py`, `docs/research/2026-10-14-round95.md`).
- **Topology planner**: `pandosi/topology.py` decides replicate vs stage vs
  infeasible from measured device/link numbers, splitting work in proportion
  to device speed and enforcing `L < T`, memory fit, and a minimum gain over
  replication. `tools/topology_check.py` pins it to the project's own
  measurements (in CI). It predicts that a speed-proportional split beats a
  synchronous quorum 8.7x where a 50/50 node split lost 0.47x — a
  model-based prediction, flagged as such
  (`docs/research/2026-10-14-round96.md`).
- **Staging tested, and it loses**: the speed-proportional split measured
  **0.66x / 0.74x** against quorum-1 replication on the 3070 — the round-96
  prediction is falsified and its 8.7x figure retracted (it compared against
  a forced quorum of 2). The ceiling is only 1.17x for a 5.9x-slower partner,
  and the loopback hop is 1.9–11.1 ms under load, not the 1.2 ms measured
  idle. The planner now prices the loaded hop and evaluates smaller quorums
  as the replication baseline (`docs/research/2026-10-14-round97.md`).
- **Autonomy, honestly bounded**: Laya's built-in act head returns 1.0 for
  every state through *both* ecosystems, so the model cannot decide whether
  to engage — the observation and the question must come from outside.
  `pandosi/autonomy.py` drives observe → decide → act with a two-part gate
  (allowed tier **and** a minimum responder count), because the coordinator's
  `min(3, len(selected))` quorum lets a lone survivor report `auto` with
  agreement 1.0. `tools/autonomy_check.py` proves both behaviours end to end
  (`docs/research/2026-10-14-round98.md`).
- **The gate is worth having**: on the 27 labeled choice tasks the three
  lineages agree that well — `tier=auto` covers **51.8% of the work at 92.9%
  accuracy** against 51.8% undifferentiated, while cross-lineage splits are
  right only 16.7% of the time. Confidence separates cleanly (0.825 correct
  vs 0.427 wrong). The swarm knows when it is guessing, at no extra cost
  (`tools/gating_check.py`, `docs/research/2026-10-14-round99.md`).
- **Verified end to end**: with the production calibration applied, the gate
  covers **48.1% at 92.3%** locally — and a real `rootd` with three real
  lineage stems over HTTP reproduces that **exactly** (delta 0.0000), with
  the held set only 14.3% accurate (2/14). Threshold tuning
  (`tools/gate_tune.py`) finds the shipped `0.6/0.6` default already
  optimal. Two defects surfaced on the way: calibration is selected by the
  checkpoint *name* (a misnamed stem silently changes the gate), and a naive
  sweep would have shipped a threshold that acts on a minority vote
  (`docs/research/2026-10-14-round100.md`).
- **A bet judged on evidence, and a loss remembered**: a growth is now
  judged on the *median* of three latency samples rather than one, a rollback
  refuses to remove a member the swarm depends on (the question is "can the
  swarm serve without it", not "is it fast"), and a rolled-back change is
  remembered by `(host, lineage)` with a TTL so the loop stops re-proposing
  it. Running the loop long enough exposed the oscillation: rollback alone
  ended at 848.7 ms having grown into the same harm again; with the memory it
  ends at **144.1 ms** with 2 reproductions instead of 3
  (`docs/research/2026-10-14-round116.md`).
- **Growth that knows where it is going, and can undo itself**: the
  supervisor refuses to grow a lineage onto a host that already runs one
  (same compute, no new information), and treats a growth as a bet —
  recording the latency before it and rolling the change back if the bet
  loses, under the *same* reproduce grant that authorised it. Measured: the
  round-114 setup now needs **one** reproduction (363.0 → 126.5 ms) instead
  of two ending at 791.8 ms; and when the second growth is forced through, a
  rollback recovers from 882 ms to 124 ms within two rounds
  (`tools/elastic_growth_check.py`,
  `docs/research/2026-10-14-round115.md`).
- **Growth through real reproduction, and where it stops helping**: the
  supervisor's `grow` action is wired to the real spore path (served
  artifact, declared lineage and digest, attestation, `reproduce` grant), and
  it now has a cooldown so it does not stack changes. Measured with two slow
  members at an 80 ms target: the first reproduction took p50 from 449.7 ms
  to **119.3 ms**, the second — another ONNX worker on the *same host* — took
  it to **981 ms**. Growing across devices adds capacity; growing within one
  host adds voters, not cores, and the supervisor's own signal then reports
  "a per-decision-cost problem, not a capacity one"
  (`tools/elastic_growth_check.py`,
  `docs/research/2026-10-14-round114.md`).
- **The elasticity loop closes**: `GET /status` exposes the swarm's own
  signals (members, quorum size/floor/ceiling, latency target, decision
  EWMA) and `ElasticSupervisor` decides grow / retire / hold. It grows only
  when decisions are slow, the quorum is already at its floor and there are
  members to spare — anything else is a policy question, not a capacity one.
  Measured: 2 members at 92.8 ms → grow → 3 members at ~61 ms, then three
  rounds of hold. A playbook category must now resolve to a real gating
  table, so a typo fails at load instead of silently yielding the permissive
  floor (`tools/elastic_loop_check.py`,
  `docs/research/2026-10-14-round113.md`).
- **Category by declaration**: a playbook question declares
  `category = "security"`, so the quorum floor follows what the operator
  wrote down rather than a keyword scan of the question text. `risk_check`
  — whose name and instructions contain no security word — now holds k=4
  under load where the keyword rule would have given it k=2, at a measured
  cost of 28% throughput. An unclassified question in a categorised playbook
  fails conservative, and keywords can still only raise the floor
  (`docs/research/2026-10-14-round112.md`).
- **The swarm's edge over its best member is real but small — and the
  members matter 13x more than the combiner**: over 480 labeled tasks (two
  independent 240-task splits) the weighted vote scores **89.6%** against
  the best single lineage's **88.5%** — +1.0 point, 95% CI [-1.0, +3.1] —
  and +1.6 points at the coverage the shipped gate accepts, CI [0.00, +3.4].
  Round 117's headline (the gate beaten by "the best member", 92.3% vs
  100%) does not survive scale: that row was the *reference* lineage, here
  the weakest of the three, and its 100% was 13 tasks. What does survive is
  structural: the three admitted lineages span **13.5 accuracy points**
  while every combiner in a six-way sweep spans 1, their errors arrive
  together (phi +0.23 to +0.50, both-wrong 2.0-4.3x the independence rate),
  and the answer sits in the trio on **91.7%** of the 157 disagreements
  while the best rule captures **74.5%**. That 17-point gap is the headroom,
  and it needs more diverse members rather than more tasks: at the observed
  5.6% discordance, resolving a 1-point difference takes ~3,900 tasks
  (`tools/build_large_suite.py`, `tools/ensemble_significance.py`,
  `docs/research/2026-10-15-round118.md`).
- **Replication buys capacity, not accuracy — and an uncounted replica is
  worse than neutral**: the repository's one trained checkpoint
  (`artifacts/laya-rlcd-v7.onnx`, different weights by hash) makes
  **240/240 identical decisions** to the `english` base it was fine-tuned
  from — disagreement 0.0%, error correlation **phi = +1.000**. Admitting it
  as a third voter drops the weighted vote from 90.4% to **85.4%**, which is
  exactly `english`'s own accuracy to four decimals: when two of three
  members always agree, the majority *is* the duplicated member, so a quorum
  with a duplicate is one model with extra steps. `Coordinator.pick_quorum`
  already round-robins across checkpoint groups, but when k exceeds the
  number of distinct checkpoints — the state an elastically replicated swarm
  converges to — the duplicate votes at full weight. Real diversity has to
  come from a different base or tokenizer, not from a descendant
  (`docs/research/2026-10-15-round119.md`).
- **Neither a better combiner nor a narrower ballot reaches the headroom**:
  re-asking the trio with only the disputed candidates on the ballot — what
  the verify tier would do in production — lifts the weighted vote on
  disagreements from 75.6% to **76.9%**, leaves the answer unchanged on 66 of
  78 rows, and where it manufactures unanimity (31 rows) the accuracy is
  *identical* to the first ballot (83.9%). Both routes to the 93.6% the trio
  holds on those rows are now closed by measurement; it is a members problem
  (`docs/research/2026-10-15-round120.md`).
- **The ensemble buys decorrelation, not accuracy**: adding a fourth voter
  from outside the neural family — a zero-shot keyword matcher at 95.0%, a
  naive Bayes at 93.75%, and a control that is 98.75% accurate but whose
  errors are aligned with an existing member — ranks the resulting
  ensembles in the *inverse* order of accuracy and in the same order as
  error correlation. The decorrelated outsider takes the weighted vote to
  **100% on the 187 rows the shipped gate accepts** (trio: 96.26%) and to
  95.0% at full coverage (trio: 88.75%); the most accurate voter of the
  three makes it *worse*. The premise needs members chosen for
  disagreement, not for solo accuracy (`tools/foreign_voter_check.py`,
  `docs/research/2026-10-15-round122.md`).
- **On wording it has never seen, the swarm still beats its best member — and
  the diversity rule still holds**: the generated suite's two seeds turned
  out to share all 53 of their phrasings, so a genuinely held-out suite was
  built (`assets/heldout_test.jsonl`, 48 new phrasings, 0 seen before). Every
  lineage loses 10-15 points on it (`english` 85.0→75.4, `typed-decisions`
  87.1→72.5 — the previous "best member" is the one that loses most), the
  gate trades coverage for accuracy (172/240 at 90.1%), and the ensemble's
  matched-coverage edge survives at +1.17 points. On that instrument the
  round-122 outsiders collapse to 57.5% and 17.5% — they were recognising
  vocabulary — but the mechanism behind them holds with accuracies that now
  *bracket* the neural members: a 57.5%-accurate decorrelated voter is worth
  **+4.1 points** at matched coverage, while a 76.25%-accurate voter whose
  errors were aligned with an existing member is worth −2.5
  (`docs/research/2026-10-15-round123.md`).
- **Corroboration is now a policy you can require**: the lineage rule was
  verified over real HTTP — a 2:1 stem majority of one lineage over another
  is an `auto` today and a `verify` with the flag on — and the same harness
  exposed the failure growth would otherwise hide: **three copies of one
  lineage reach `auto` with agreement 1.00**, one opinion corroborated by
  nothing. `[gating] corroboration_floor` refuses `auto` below that many
  distinct lineages (default 1, so a single device still works alone), and
  the decision response now reports `lineages` and the floor
  (`tools/lineage_quorum_e2e_check.py`,
  `docs/research/2026-10-15-round124.md`).
- **A device cannot mint independence from the model it holds**: reversing
  the ballot flips **0 of 240** answers, and the quantised sibling the
  project would ship to an NPU moves 20% of answers yet keeps the family's
  errors (phi +0.41, no ensemble gain). The measured spectrum is monotone in
  correlation — a fine-tuned descendant and a reversed ballot sit at
  phi +1.000, a quantised sibling at +0.41, a different prompt rendering at
  +0.45, and only a voter from outside the family at +0.07, which is the one
  that pays (+4.1 points). On wording none of them has seen the trio's own
  phi *rises* to +0.36..+0.82, so the family moves together exactly where
  the task gets hard (`tools/same_family_voter_check.py`,
  `docs/research/2026-10-15-round125.md`).
- **The independent member is now shipped, and it found a bug in the
  aggregation rule**: `pandosi/textmodel.py` is a 19 KB naive Bayes
  (`stemd --backend text`) that any device can serve, calibrated through the
  project's own mechanism. Added as a fourth live HTTP stem it made the swarm
  *worse* — because three members cannot tie and four can, and the shipped
  gate answered a 2-2 split with silence, discarding exactly the information
  the new member brought. A tied count is now broken by confidence mass at
  the `verify` tier (a tie still can never auto-execute): the trio goes
  73.75% → **77.08%** at full coverage and the four-member swarm
  72.92% → **78.33%**, with the `auto` tier untouched
  (`docs/research/2026-10-15-round126.md`, `…round127.md`).
- **Scaling is a membership problem, not a head-count**: adding a *second*
  cheap independent member stops paying — four of them leave the swarm
  **11.6 points worse** than the trio (paired bootstrap, ahead in 0.1% of
  2000 resamples) — because the cheap members resemble each other
  (chars × mixed **phi +0.898**, words × centroid +0.687) while each is
  decorrelated from the neural family. A synthetic control with *truly*
  independent voters is monotone in count at every accuracy down to 50%
  (six 50% members: +13 points; six 60% members: +17). So the fleet's
  constraint is error correlation, not member strength or member count: a
  device is worth a seat only if it diverges from the members already seated
  (`tools/member_portfolio_check.py`, `tools/scaling_curve_check.py`,
  `docs/research/2026-10-15-round128.md`).
- **Count the opinions, not the devices**: a playbook now declares
  `[lineages] blocs`, groups whose errors move together, and each group
  votes once (`aggregate(blocs=)`; floors and `corroboration_floor` count
  opinions). Four correlated members counted by name cost the swarm 11.6
  points; declared as one bloc, the same seven members score **77.92%**
  against the trio's 77.08% — the family's information kept and its vote
  count not. A fleet can therefore grow with same-family devices for
  throughput at no cost to the decision, while growth for *intelligence*
  still needs a new opinion
  (`docs/research/2026-10-15-round129.md`).
- **One device, several compute units, one opinion — measured again under the
  new rules**: the same weights on the CPU (ONNX Runtime) and on the Intel
  iGPU (OpenVINO) give **identical choices on 240/240** held-out tasks with a
  0.0002 confidence gap, at 12.3 vs **18.0 decisions per second** (79 vs
  53 ms median). Registered as two stems, the quorum reports **3 opinions for
  4 stems** — the lineage rule collapsing the hardware pair — and the second
  unit therefore buys capacity, not information. The NPU is still out: the
  current OpenVINO vpux compiler rejects `node_MatMul_215` with unbounded
  dimensions, the op-level class of blocker rounds 52-54 solved twice
  (`tools/multi_eu_swarm_check.py`,
  `docs/research/2026-10-15-round130.md`).
- **The NPU was a graph problem, not a hardware problem**: the current
  OpenVINO vpux compiler was rejecting `gnr1_gather_el` — a node produced by
  this repository's own GatherND repair for the *Intel GPU*. Handing it
  `artifacts/laya-en-static-nonisnan.onnx` (both repairs, 20 nodes fewer)
  compiles in 39 s and votes. Three units on one machine, same weights, 240
  held-out tasks: CPU / iGPU / NPU give **identical choices on 240/240** with
  confidence gaps of 0.0002 and 0.0003, at **17.6 / 10.0 / 8.7 decisions per
  second** (iGPU / CPU / NPU — the NPU is not a throughput win on this model,
  its case is efficiency per watt). Five stems, three opinions: the extra
  silicon buys capacity, not information
  (`tools/multi_eu_swarm_check.py`,
  `docs/research/2026-10-15-round131.md`).
- **Corroboration is paid for in throughput**: with one lineage on the three
  units (so every configuration answers identically), a units-by-quorum
  matrix under four concurrent clients gives 1u-k1 14.7 dec/s, **2u-k1
  27.1/s**, 3u-k1 24.1/s, 3u-k2 14.4/s, **3u-k3 10.5/s** (p50 262/133/140/
  245/374 ms). Two units at k=1 reach exactly the sum of their solo rates
  (10.0 + 17.6 = 27.6/s) — but the third adds little and sometimes subtracts,
  because **three units on one host are not three servers**: they share one
  CPU for tokenization, data movement and orchestration. Dropping k from 3 to
  1 multiplies throughput 2.3x and halves p50 latency, and the answer does
  not change. Corroboration is a dial with a measured price
  (`tools/hetero_capacity_check.py`,
  `docs/research/2026-10-15-round132.md`).
- **The host is the ceiling, and one host is noisy**: re-running the capacity
  matrix with the stems started the way production starts them — separate
  `stemd` processes, own interpreter and session — changes nothing: 1 unit
  26.5 / 2 units 25.0 / 3 units 26.0 dec/s at k=1, while **k=3 costs 2-3x**
  (9.6 dec/s, p50 383 ms). Adding accelerators buys nothing because a single
  stem already saturates the host's per-request work (tokenisation, JSON,
  HTTP). Under four concurrent clients the units' *ranking* also flips —
  the NPU, slowest sequentially (8.7 dec/s), is fastest under load
  (33.2 /s) — and the same cell measured 16.8 and 34.1 /s in two runs, so
  single-host numbers carry 30-50% noise. The next doubling is another
  machine or less host work per request, not another unit
  (`tools/process_fleet_capacity_check.py`,
  `docs/research/2026-10-15-round133.md`).
- **The CPU is the wrong silicon to vote with**: measured with each unit in
  true isolation and a host that is somebody's computer (eight cores burned by
  unrelated work), the CPU stem serves 16.8 decisions/s idle and **6.0/s
  loaded** while costing the machine **791→1647 ms of host CPU per decision**;
  the Intel iGPU does 171→341/s at ~6 ms and the **NPU 268→296/s at
  1.4-1.8 ms** — a factor of 500-1000 in what the unit costs the host. The
  accelerators are not just idle capacity, they are *load-invariant*: they
  keep serving while the CPU stem loses 64% of its throughput. The value of a
  heterogeneous fleet on a personal computer is availability, not peak
  throughput. (Two harness defects were found and fixed on the way:
  `Coordinator.blocked` does not isolate a unit — only `retire()` does — and
  the loader had been repeating one task; per-unit attributions from rounds
  132-133 are withdrawn.)
  (`tools/idle_silicon_check.py`,
  `docs/research/2026-10-15-round134.md`).
- **The accelerators drop out under concurrency — and the response now says
  so**: stems report their measured host cost per decision on every
  heartbeat, `pick_quorum` prefers the cheapest ten-fold cost band inside a
  lineage group (the CPU stem's share of served decisions fell from a third
  to zero), and the decision response gained an **`unanswered`** field naming
  members that were picked and said nothing. That field immediately showed
  the real problem: at four concurrent clients the OpenVINO iGPU and NPU
  stems fail roughly half their calls while the ONNX CPU stem never does
  (sequentially the same NPU stem answers 10/10). Round 134's accelerator
  throughput and cost figures are **withdrawn** — they were dropouts counting
  as speed. The next fix is what `MockBackend.serialize` already models: a
  device should serve one decision at a time
  (`docs/research/2026-10-15-round135.md`).
- **A device serves one decision at a time — and the numbers come back**:
  stems now serialise `backend.decide` by default (`stemd --parallel` to opt
  out). The accelerator dropouts went from 49-56 silent decisions per 120 to
  **zero**, and the fleet now serves entirely from the accelerators, leaving
  the CPU for the user. Re-measured service times on a loaded host: CPU
  94 → **309 ms** at **2856 ms of host CPU per decision**, Intel iGPU
  53 → 35 ms at 36 ms, Intel NPU 112 → 109 ms at **4 ms** — a corrected cost
  ratio of 80-700x, and the accelerators do not degrade at all while the CPU
  stem loses 3.3x
  (`docs/research/2026-10-15-round136.md`).
- **Independence is the admission test; the rest is noise**: ten cheap
  members (three feature views × three smoothing strengths, plus a
  nearest-centroid classifier) were measured for accuracy, mean error
  correlation with the neural trio, rescues, harms and the value each adds to
  the swarm. All ten are decorrelated (mean phi −0.06…+0.08) and nine add a
  positive amount — the family-level effect of rounds 122-129, reproduced ten
  times. But *within* that family nothing predicts the value: rank
  correlation against the measured gain is **−0.55 for accuracy** (the most
  accurate members are not the most valuable), −0.02 for phi and −0.55 for
  net rescues, with the whole spread inside 240-row sampling error. A fleet
  should admit on decorrelation and measure the rest after the fact
  (`tools/member_value_check.py`,
  `docs/research/2026-10-15-round137.md`).
- **Reproduction costs about half a minute and buys nothing**: measured end
  to end against a running root, a new member's birth is **9.4 s to load**
  the artifact + **0.2 s to register** + **5.0 s to its first vote** (14.6 s,
  transfer excluded; ~29 s with round 106's 14.2 s whole-artifact transfer).
  What it buys is nothing: the second member of the same lineage is the same
  opinion on the same silicon, and putting it in the quorum takes the fleet
  from 26.1 to 13.5 decisions/s exactly as k predicts. A self-replicating
  swarm is cheap to grow and hard to make smarter, and both halves are now
  numbers (`tools/device_birth_check.py`,
  `docs/research/2026-10-15-round138.md`).
- **A learned combiner beats every single member, and locates the rest of the
  gap**: treating the quorum's votes and calibrated confidences as features
  and fitting a regularised logistic regression on the dev split (l2 chosen
  by five-fold CV on dev, evaluated once on the 240 wording-held-out rows)
  gives **80.42% against the best single member's 75.42%** — the first clear
  ensemble margin in this project. On the rows the gate accepts it sits 0.6
  points from the oracle (90.70 vs 91.28); on the 68 rows the gate *declines*
  it reaches **54.4% against the best member's 41.2%** with an oracle of
  67.7% — the remaining headroom lives exactly where the swarm refuses to
  answer, and a fitted rule recovers about three-quarters of it there
  (`tools/learned_combiner_check.py`,
  `docs/research/2026-10-15-round139.md`).
- **The combiner is now a shipped artifact, and it is advisory by
  construction**: `tools/train_combiner.py` writes a **4 KB**
  `assets/combiner.json` (fitted on the dev votes, l2 cross-validated on dev
  alone) and `rootd --combiner <path>` attaches its answer as
  `suggested_choice` **only where the gate does not auto-execute** — replayed
  over the held-out rows it reproduces 80.42% full coverage and **54.41% on
  the 68 rows the gate declines**, with 0 of its 68 suggestions landing on an
  auto row. Wiring it live caught a contract violation the offline replay
  could not: the model suggested a label outside the request's candidates, so
  a suggestion is now masked to the question's options like any member's vote
  (`pandosi/combiner.py`, `tools/combiner_artifact_check.py`,
  `docs/research/2026-10-15-round140.md`).
- **Three disjoint wording sets, and a combiner that beats every member on
  unseen wording**: the generator now carries a third inventory
  (`--cues c`, 0 cue overlap with the other two), which separates the roles —
  combiner fitted on set A, text model trained on set B, **evaluated once on
  set C**. On C the plain majority beats the best single member by 3.3
  points, and a combiner over the members' votes plus the text model's
  posterior reaches **82.92% against the best member's 73.75%**; on the 77
  rows the gate declines it answers **62.34%** against the member's 40.26%
  and an oracle of 71.43%. Votes alone carry most of it (58.44%), the text
  adds +3.9 points there
  (`tools/combiner_text_check.py`, `assets/heldout2_test.jsonl`,
  `docs/research/2026-10-15-round141.md`).
- **The combiner artifact now carries its own text model (spec 2)**: the
  feature block appends the text model's posterior over the candidates and
  the artifact embeds that model, so one signed **35 KB** file ships both
  (`--reports` for the votes, `--text-train` for the wording, both recorded).
  Replayed on split C — unseen by the combiner and by the embedded text model
  — it reproduces **82.92% full coverage** (gate 77.08%, best member 73.75%)
  and **62.34% on the 77 rows the gate declines**, with 0 of its 77
  suggestions landing on an auto row
  (`pandosi/combiner.py`, `tools/combiner_artifact_check.py`,
  `docs/research/2026-10-15-round142.md`).
- **The quorum is now invariant under replication**: a same-lineage vote is
  folded into its lineage (`aggregate(count_lineages=True)`, `rootd
  --lineage-quorum`), so counting stems no longer lets a copied lineage
  outvote a distinct one. Replaying the recorded decisions with replicas
  added: the shipped rule moves by up to **14.6 points** of accuracy
  depending only on which lineage was copied — replicating the *weakest*
  member drags the quorum to 74.6%, replicating the *best* inflates it to
  90.0% — while the lineage-counted rule moves by **0.000** and is byte-identical
  on a quorum with no duplicates
  (`tools/lineage_quorum_check.py`, `docs/research/2026-10-15-round121.md`).
- **Per-category quorum floors and retirement**: a playbook sets a
  `quorum_floor` per category, so under load an ordinary question gave up
  voters down to 2 while a security question stopped at 4 — twice the
  independent voters for 22% less throughput, with both keeping the full
  quorum while idle. The floors are binding, so an unreachable latency
  target means "grow the swarm", not "weaken the check". And the other half
  of elasticity now exists: a slow member can be retired through the grant
  ledger, with the root blocking its re-registration so the removal is
  durable rather than a five-second deletion
  (`tools/retirement_check.py`, `docs/research/2026-10-14-round111.md`).
- **A quorum that knows how busy it is**: `pick_quorum` now orders voters by
  in-flight load (then latency band, then rotation), and
  `rootd --adaptive-quorum --latency-target-ms` lets the quorum size shrink
  under latency pressure and recover when idle. Load-aware selection beat
  blind rotation by 16–24% with lower latency; the adaptive root held full
  agreement while idle (k=5) and settled at k=3 under load, hitting its
  120 ms target at 71 req/s where static k=5 gave 23.5 req/s at 306 ms
  (`tools/scaling_law_check.py`,
  `docs/research/2026-10-14-round110.md`).
- **What scales is the quorum, not the replica**: with a device-like
  serialized replica, going 1 → 6 replicas bought only 9% throughput while
  every voter was used on every decision. With `--quorum-k 2` fixed and
  member rotation, the same 6 replicas reached **1.9x** (37.4 vs 19.6 req/s
  at concurrency 4), bounded by `N/k = 3` parallel decisions. Capacity is
  bought from agreement diversity, so "reproduce more" and "decide faster"
  are different instructions (`tools/scaling_law_check.py`,
  `docs/research/2026-10-14-round109.md`).
- **Delta replication, end to end**: `rootd --chunking cdc` serves
  content-defined manifests and `join --chunk-store` reuses what a replica
  already holds. Two real roots and two real joiners: the second device
  replicated a different lineage for **105 MB instead of 1610 MB (15.3x)**,
  measured from the receiving side. The chunk store is packed (one
  append-only blob), which removed the wall-clock regression from the first
  cut: the delta leg is now 5.84 s versus 10.49 s for a full transfer
  (`tools/delta_replication_check.py`,
  `docs/research/2026-10-14-round108.md`).
- **Delta replication**: content-defined chunking plus a content-addressed
  chunk store, so a replica fetches only the chunks it lacks. Replicating a
  fine-tune that shares 93.5% of its weights with its parent moved **105 MB
  instead of 1610 MB (15.3x)** and spent **0.37 s transferring instead of
  5.56 s**, with the reconstruction verified byte-identical. The wall clock
  is held back by the on-disk chunk store (4,870 small reads), which is
  recorded as the next fix (`tools/spore_delta_check.py`,
  `docs/research/2026-10-14-round107.md`).
- **Batched replication**: spore chunks transfer 64 at a time over one
  keep-alive connection (`/spore/chunks/<name>/<start>/<count>`) while every
  chunk is still hash-verified, with a per-chunk fallback for a bad chunk.
  The 1.68 GB artifact went from ~125 s to **14.2 s** (8.8x, 113 MB/s,
  byte-identical); end-to-end replication from ~4.5 min to **59 s**
  (`tools/spore_transfer_check.py`,
  `docs/research/2026-10-14-round106.md`).
- **Self-replication loop**: `pandosi/reproduce.py` orders the existing
  pieces — Spore transfer, artifact attestation, lineage admission, and an
  operator-approved `reproduce` grant — and refuses before anything moves.
  Verified end to end: the child pulled a 1.68 GB payload over the Spore
  endpoints, attested it, passed admission, and grew the quorum 1 → 2 with a
  signed lineage record. It also found four bugs that no check covered, the
  worst being that `pandosi.join` could not start at all
  (`tools/reproduce_check.py`, `docs/research/2026-10-14-round105.md`).
- **Signed playbooks**: a playbook's canonical bytes are covered by a
  detached Ed25519 signature, which includes the admitted lineages and their
  artifact digests — so the whole declaration chain is attributable.
  `rootd --playbook-key <hex>` pins the signer, and a missing or invalid
  signature is fatal at startup; without a pin the root logs that writing the
  file is enough to change what it admits. Tampering with `admit` invalidates
  the signature and the root refuses to start
  (`tools/sign_playbook.py`, `docs/research/2026-10-14-round104.md`).
- **Artifact attestation**: a stem reports the SHA-256 of the artifact it
  actually loaded — the graph *and* its external-data weight blob — and a
  playbook that declares `[lineages.artifacts]` refuses any stem that cannot
  match it. Verified end to end: the same honest stem registers when the
  declared digest is right and is refused when it is not
  (`tools/attestation_check.py`, `docs/research/2026-10-14-round103.md`).
- **Declared lineage identity**: a playbook declares `[lineages].admit`,
  and the root refuses at registration any stem whose checkpoint is not
  declared — an id is admitted only if it is declared *and* ships a
  calibration table. Identity, admission and calibration now come from one
  declaration instead of three conventions, while a root with no
  declaration stays permissive for synthetic harnesses. A declaration is a
  claim, not proof: attestation against the Spore manifest is the next step
  (`docs/research/2026-10-14-round102.md`).
- **Calibration is enforced where the risk is**: `rootd` warns at
  registration and reports `uncalibrated_checkpoints` in every decision;
  `AutonomyLoop(require_calibration=True)` refuses to act on them. Placing
  the hard rule in the coordinator instead broke every synthetic harness and
  two CI checks, which is recorded — the coordinator is shared by production
  and test rigs, so the rule belongs in the component that acts
  (`docs/research/2026-10-14-round101.md`).
- **Real lineage identity over HTTP**: en/ml/td ONNX stems now register as
  distinct checkpoints; twelve shared choice tasks produced 11 cross-lineage
  splits, making model diversity visible to the live quorum.
- **Signed self-replication (sandboxed)**: Spore payload = weights +
  playbook + harness. Ed25519 manifest, 64KiB chunk Merkle tree,
  per-chunk hash-on-arrival, selective re-fetch, TOFU key pinning,
  sandbox hard gate. Full SPROUT/VERIFY/LIVE chain run with real 571MB
  weights.
- **Playbook governance**: TOML strategy layer (authored offline) drives
  consensus gating; hash-chained version history; structural policy
  enforcement (score heads are informational-only — enforced).
- **Local-first onboarding**: `python -m pandosi.join` starts a loopback
  root+stem with zero privilege; LAN discovery/listening requires an
  explicit, expiring grant.


## Architecture

\\mermaid
graph TB
    subgraph playbook[Playbook - strategy layer]
        PB[playbook.toml<br/>authored offline]
        PH[pb_history.jsonl<br/>hash chain]
    end

    subgraph root[rootd - governance]
        QV[question enforcement]
        GT[confidence gating<br/>playbook thresholds]
        RC[RolloutRecorder<br/>full vote matrix]
        SD[/shutdown kill-switch/]
    end

    subgraph stems[Execution stems - heterogeneous]
        S1[stem CPU<br/>ORT fp32<br/>en lineage]
        S2[stem iGPU<br/>OpenVINO<br/>en lineage]
        S3[stem NPU<br/>OpenVINO IsNaN-free<br/>en lineage]
        S4[stem ml<br/>ORT fp32<br/>ml lineage]
    end

    subgraph spore[Spore - self-replication]
        MS[manifest<br/>Ed25519 signed]
        CH[chunk Merkle tree<br/>64KiB verify-on-arrival]
        SB[sandbox gate<br/>THREAT-MODEL enforced]
    end

    PB --> PH --> QV
    QV --> GT
    GT --> RC
    root -->|fan-out| S1 & S2 & S3 & S4
    S1 & S2 & S3 & S4 -->|votes| RC
    MS --> CH --> SB
    SB -.->|lab only| stems
\
### Data flow

1. **Playbook** (TOML) defines: question schemas, confidence gating, action bindings
2. **rootd** validates: question in playbook, thresholds from [gating]
3. **Fan-out**: request sent to k stems with checkpoint diversity
4. **Votes collected**: each stem runs Laya inference, returns {choice, probability}
5. **Aggregation**: majority vote + tier assignment (auto/verify/escalate)
6. **Rollout recorded**: full vote matrix + playbook_hash for A/B attribution
7. **Evolution**: disagreement pairs exported to RLCD trainer → new lineage

## Quickstart

```bash
pip install -e ".[runtime]"                  # source checkout
python -m pandosi doctor --plan              # safe onboarding plan
python -m pandosi onboard --dry-run          # no install/elevation
python -m pandosi join                       # local root + stem (loopback)
python -m pandosi doctor                     # capability report
python tools/dashboard.py                    # render local metrics
```

Installed-package form:

```bash
pip install "pandosi[runtime]"
pandosi doctor --plan
```

Directly from GitHub before the first PyPI release:

```bash
pip install "pandosi[runtime] @ git+https://github.com/JewelRoam/PandoSI.git@main"
pandosi doctor --plan
```

The wheel includes playbook examples, small graft suites, and calibration
JSON files. Model weights and generated ONNX graphs remain separate.
CI builds and inspects the wheel archive so packaged entry points and
runtime assets cannot silently disappear.
Source distributions are checked too; a `v*` tag creates a GitHub release
with wheel, sdist, and `SHA256SUMS`.

LAN is a separate explicit grant:

```bash
python -m pandosi grants grant lan_listen \
  --resource 0.0.0.0:8080 --hours 24 --yes
python -m pandosi join --lan
```

Grant creation, revocation, allowed use, and denial are recorded in a local
hash chain; `python -m pandosi.grants audit` verifies it.

Persistent services are plan-only and require a separate autostart grant:

```bash
python -m pandosi.grants grant service_autostart   --resource user-agent/pandosi --hours 24 --yes
python -m pandosi.service --role root --json
python -m pandosi.service --role root
```

On a native host, verify the full lifecycle without leaving artifacts:

```bash
python tools/service_check.py --yes
```

CI also runs this check for Linux `systemd --user`; Termux still needs a
real Android host.

Adapters cover Windows user Startup files, Linux `systemd --user`, and
Termux `sv`. Tokens are read from a file, never embedded in argv; the planner
does not install, start, or uninstall anything.

Real-model mode: install `laya`, export ONNX (tools/vendor/export_onnx.py),
then `python -m pandosi.join --backend onnx --onnx-path artifacts/laya-en.int8.onnx`.

LAN roots generate a pairing token and protect registration, `/nodes`, `/decide`, playbook, and root Spore endpoints.
For an additional stem, pass the token printed by the root:

```bash
python -m pandosi.stemd --node-id stem-2 --root http://ROOT_IP:8080 \
  --pairing-token TOKEN_FROM_ROOT
```

## Docs

- [Blueprint (zh-CN)](docs/BLUEPRINT.zh-CN.md) — architecture v0.7
- [ADOPTION](docs/ADOPTION.md) — three-ring capability model, survival metrics
- [Onboarding (zh-CN)](docs/ONBOARDING.zh-CN.md) — consent tiers, sandbox policy, product UX
- [THREAT-MODEL](THREAT-MODEL.md) — defensive-research framing, hard guarantees
- [METRICS](docs/METRICS.md) — dashboard design
- [ROADMAP](docs/ROADMAP.md) — phases with measurable gates
- [Research notes](docs/research/) — 28 rounds of evidence-backed findings
- [TERMUX.md](TERMUX.md) — Android TV box onboarding guide
- [Swarm check](docs/SWARM-CHECK.md) — multi-process tokenized consensus and shrinkage
- [RTX 3070 CUDA worker](docs/CUDA-3070.md) — non-OpenVINO HTTP integration
- [Slow-stem check](tools/slow_stem_check.py) — timeout, cooldown, and retry proof
- [Windows service check](docs/SERVICE-CHECK-WINDOWS.md) — real lifecycle validation record
- [Linux CI service check](docs/SERVICE-CHECK-LINUX-CI.md) — user-systemd lifecycle validation record

## Status

Research preview. Defensive framing (see THREAT-MODEL). Replication
experiments run only in sandboxes. 40-task acceptance suite measured; score
heads documented as informational-only (round 24).
