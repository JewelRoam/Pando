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
