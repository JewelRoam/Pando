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
pip install pynacl zeroconf                  # deps
python -m pandosi.doctor --plan              # safe onboarding plan
python -m pandosi.onboard --dry-run          # no install/elevation
python -m pandosi.join                       # local root + stem (loopback)
python -m pandosi.doctor                     # capability report
python tools/dashboard.py                    # render local metrics
```

LAN is a separate explicit grant:

```bash
python -m pandosi.grants grant lan_listen \
  --resource 0.0.0.0:8080 --hours 24 --yes
python -m pandosi.join --lan
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
