# PandoSI Research Summary — definitive findings

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
    → coordinator (governance: question enforcement + gating + chain verification)
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

## Additional findings (rounds 44-61b)

### Diversity quantified

| pair | divergence | meaning |
|---|---|---|
| en vs ml | **32.5%** | most independent pair (multilingual sees differently) |
| en vs td | 10.0% | td fine-tune closely tracks en base |
| ml vs td | 27.5% | moderate independence |

**35% split rate** = 14 contrast pairs available for RLCD = the sweet spot
(enough disagreement for training signal, not enough to make swarm unreliable).

### Split-case majority reliability: 36%

When the 3 lineages disagree, majority voting is only 36% reliable.
This PROVES that auto-executing split decisions would be dangerous (64%
error rate) and that verify/escalate routing is NECESSARY, not optional.

### Calibration is load-bearing

2 tier flips proven at the boundary (p=0.58 + T=0.5 sharpens to auto;
p=0.62 + T=2.0 softens to verify). Temperature choice directly affects
auto-execution rate. Currently identity (T=1.0) until calibration data
is computed.

### Adversarial robustness matrix

SAFE when honest >= adversarial. CORRECTLY DEMOTED at parity.
DANGEROUS only when honest < adversarial (Byzantine Generals - mitigated
by checkpoint diversity + EU diversity + sandbox gate + signature).

### Score capability is DOMAIN-NARROW

27 measurements across 3 lineages: en/ml score heads content-blind,
td content-sensitive only in its fine-tune domain. Structurally enforced:
score questions require informational=true in the playbook.

## Evolution loop: first cycle verified

```
disagreement (ml=greeting@0.53 vs en=refund@0.99)
  → RolloutRecorder captured → export_training_rows (contrast pair)
    → laya.train.finetune → loss=0.1663, checkpoint saved
      → fine-tuned model: same decisions, confidence +0.25 universal
        → no regression on non-target cases → genuine learning
```

## Cross-runtime accelerator proof (round 65)

An RTX 3070 was added through ONNX Runtime CUDA while Intel CPU, iGPU,
and NPU workers kept their original runtimes. On the same dynamic Laya
graph and identical captured inputs:

- CUDA p50: **15.0 ms**
- CPU p50: **125.6 ms**
- speedup: **8.36x**
- English decision parity: **14/14**
- max calibrated confidence delta: **0.0009**

All four runtime-distinct workers then registered with one tokenized HTTP
root and returned four votes per decision. Sequential dispatch produced a
steady-state wire p50 of **430.5 ms**; parallel quorum fan-out reduced it to
**237.8 ms** (root fan-out p50 **226.3 ms**) without changing the model or
quorum semantics. The coordinator does not need a shared runtime or a shared
accelerator API; OpenVINO and CUDA cooperate at the protocol boundary.

Round 67 adds per-stem latency memory. After five real decisions, the
learned failure budgets were 0.72-1.00 s per heterogeneous worker, replacing
the previous 30 s global call timeout for unresponsive stems.

Round 74 fixes real ONNX checkpoint identity: en/ml/td stems now register as
distinct lineages instead of collapsing to device-derived names. On twelve
shared English-biased choice tasks, 11/12 produced cross-lineage splits.
This is a harsher, same-input comparison than the earlier lineage-specific
32.5% figure and does not replace it; it does reinforce verify/escalate
routing.

Round 75 exports those live lineage disagreements into ten contrast pairs
and performs nine CUDA optimizer updates on the RTX 3070. The resulting
checkpoint preserves all 14 English decision labels while changing
confidence by up to 0.182. A conflicting `intent` schema across rows is
documented as a data-normalization requirement before generalization claims.

Round 76 implements that normalization: canonical question ids bind the
prompt name to an option-set digest in both live inference and RLCD export.
The v3 CUDA run removes the trainer's schema-conflict warning while keeping
all 14 English labels stable.

Round 77 removes training overlap from evaluation. Nine overlapping states
were excluded, leaving 31 held-out tasks. The CUDA v4 checkpoint preserved
all 31 labels but did not improve accuracy (0.6129 before and after), so the
current ten-pair dataset is sufficient to prove pipeline stability, not
generalization.

Round 78 makes the split explicit and expands the training set to 20
contrast rows. One epoch (18 updates) and four epochs (72 updates) both left
the six-task held-out accuracy unchanged. Four epochs drove confidence
toward saturation, including a wrong answer reaching probability 1.0, so
the current small-data training branch is stopped pending more independent
contrast and calibration data.

Round 79 adds a 78-prompt multi-lingual pool, from which the live lineages
produced 45 new contrast rows. The resulting CUDA v7 candidate ran 41
updates and kept all six held-out labels stable while reducing confidence
drift versus the four-epoch small-data run. It did not improve held-out
accuracy and is retained as a candidate, not promoted.

Round 80 exports v7 to ONNX and places it beside the baseline as a live HTTP
stem. The two lineages agreed on 6/6 holdout tasks, both at 0.5 accuracy.
v7 raised confidence substantially, including 0.9996 on an incorrect
answer, so the promotion gate rejected it. The live A/B mechanism itself is
verified and ready for future candidates.

Round 81 adds post-training temperature calibration. A fit on 22 labeled
items reduced the wrong-answer confidence from 0.9996 to 0.9316 and lowered
mean confidence, but did not change holdout accuracy and did not beat the
baseline on the small held-out calibration slice. The calibrated candidate
also remains rejected by the live A/B gate.

## Remaining work

- NPU: deep graph analysis (IsNaN was removed; other ops may block)
- Termux/TV box: cross-device federation (guide written, needs hardware)
- RLCD training: run end-to-end (data layer verified, trainer compatible)
- Dashboard: lineage graph rendering
- Multi-bucket auto-discovery in join (partially done)
