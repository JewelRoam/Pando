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

Adapters cover Windows scheduled tasks, Linux `systemd --user`, and Termux
`sv`. Tokens are read from a file, never embedded in argv; the planner does
not install, start, or uninstall anything.

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

## Status

Research preview. Defensive framing (see THREAT-MODEL). Replication
experiments run only in sandboxes. 40-task acceptance suite measured; score
heads documented as informational-only (round 24).
