# PandoSI v0.1 research preview — release notes (2026-10-14)

## What shipped

### Verified on real hardware (Core Ultra 7 255H)

| Capability | Evidence |
|---|---|
| Decision-model swarm consensus (3 lineages, checkpoint-diverse quorum) | round 9/10, live smoke |
| Heterogeneous dual-EU consensus (iGPU + CPU, different backends) | round 35, FINAL INTEGRATION OK |
| iGPU 4x speedup via GatherND graph surgery | round 34, 42ms vs 106ms |
| Signed self-replication (SPROUT/VERIFY/LIVE, 571MB real weights) | round 17/33, sandbox-gated |
| Playbook governance (TOML -> hash chain -> rootd gate -> consensus) | round 20/27/44 |
| RLCD data path (swarm -> rollouts -> contrast pairs -> trainer) | round 38/47, format-compatible |
| 40-task graft acceptance suite (3 lineages measured) | round 23/44, score anomaly found+closed |

### Honest negatives

- Backend numerics do NOT add decision diversity (0/40 divergence, refuted)
- Score heads: content-blind on en/ml; domain-narrow on td → informational-only
- 5-level risk classification: adjacent-level errors dominate → 3-level adopted
- NPU: blocked on op support (4 repair paths documented, not yet exhausted)
- iGPU variable-length serving: multi-bucket static exports (round 41); lazy
  per-shape compile is a ~8s-per-shape cliff for workloads with many shapes

### Known limitations

- Weights (571MB-1.6GB) must be pre-distributed or served; no P2P weight sync yet
- WAN federation (cross-city groups) is designed but not integration-tested
- RLCD training not yet run end-to-end (data layer verified compatible)
- No NPU execution (upstream OpenVINO op support required)
- Single-file stems (onnxruntime bundled) not yet packaged

## Verification

```bash
python -m pando.doctor              # 10/10 on the dev box
python -m unittest discover -s tests  # 2/2 smoke
python tools/demo.py                  # end-to-end consensus + dashboard
```

## Research archive

48 rounds: docs/research/ (round 1-48, 2026-10-13 to 2026-10-14).
Session backups: docs/session-logs/ (42 snapshots, sanitized).
