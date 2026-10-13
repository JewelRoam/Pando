# PandoSI Research Summary — definitive findings (v0.1)

## The question

Can one device (or a cluster of cheap devices) do heterogeneous elastic
assembly — multiple execution units voting as one intelligence?

## The answer

**Yes, definitively verified on real hardware.** One Intel AI PC running
CPU + iGPU + NPU, each with a different backend and lineage, all voting
in the same consensus quorum, governed by the same signed playbook.

## Key findings (48 rounds, 57 research logs)

### What works (verified with real data)

| Finding | Evidence |
|---|---|
| Decision-model swarm consensus works | 3 lineages, checkpoint-diverse quorum, live smoke |
| Heterogeneous EU assembly on one device | CPU+iGPU+NPU all voting (round 56) |
| iGPU 4x faster than CPU via graph surgery | 25.6ms vs 106ms, 0/40 divergence |
| NPU works after 4-step enablement chain | 95.7ms, 40/40 decision match |
| Checkpoint-diverse quorum produces independent evidence | 35% split rate, 14 contrast pairs |
| Signed self-replication works end-to-end | 9132 chunks, byte-identical, tamper-rejected |
| Governance pipeline: 5 layers all code-enforced | playbook→chain→gate→enforce→gating |

### What doesn't work (honest negatives)

| Finding | Evidence |
|---|---|
| Backend numerics do NOT add diversity | 0/40 divergence between ORT-CPU and OV-iGPU |
| Score heads are not decision-grade | 27 measurements: content-blind (en/ml), domain-narrow (td) |
| 5-level risk classification exceeds model capability | adjacent-level errors dominate |
| Split-case majority voting is unreliable | 36% (5/14) — verify routing is necessary |

### Key design principles discovered

1. **Diversity is quantifiable and necessary**: 35% split rate across 3
   lineages proves independence; homogeneous replicas → 0% divergence →
   0 training signal → 0 evolutionary fuel.
2. **Calibration layer is mandatory**: Laya ships uncalibrated temperatures;
   all confidences must pass ECE validation before tier gating.
3. **Proxy immunity is critical**: Windows urllib registry proxy black-holes
   LAN traffic; all endpoints must set NO_PROXY.
4. **Static exports for GPU/NPU, dynamic for CPU**: the Laya graph's
   internal Reshape ops are only consistent for export-time shapes.
5. **Score questions are informational-only**: structurally enforced in
   the playbook validator, not just documented.

## Architecture summary

```
playbook.toml (strategy, authored offline)
  → pb_history chain (audit trail)
    → rootd (governance: question enforcement + gating + chain verification)
      → fan-out to 3+ stems (checkpoint-diverse, EU-diverse)
        → consensus (majority vote + tier gating)
          → rollout recording (playbook_hash attribution)
            → RLCD contrast pairs → trainer → new lineage → repeat
```

## EU reality table (Intel AI PC, Core Ultra 7 255H)

| EU | backend | decision p50 | status |
|---|---|---|---|
| CPU | ORT fp32 | 54-58ms | ✅ reference |
| iGPU (Arc 140T) | OpenVINO (surgery + multi-bucket) | 25-41ms | ✅ 4x CPU |
| NPU (AI Boost) | OpenVINO (IsNaN-free) | 95-102ms | ✅ verified |

## Remaining work

- NPU: deep graph analysis (IsNaN was removed; other ops may block)
- Termux/TV box: cross-device federation (guide written, needs hardware)
- RLCD training: run end-to-end (data layer verified, trainer compatible)
- Dashboard: lineage graph rendering
- Multi-bucket auto-discovery in join (partially done)
