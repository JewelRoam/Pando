# PandoRSI Roadmap

## Achievements (2026-10-14)

| milestone | evidence |
|---|---|
| Triple-EU consensus (CPU+iGPU+NPU) | round 56, live smoke |
| iGPU 4x speedup (GatherND surgery) | round 34, 42ms vs 106ms |
| NPU unblocked (4-step enablement chain) | round 54, 95.7ms p50 |
| Multi-hop spore chain (3 hops, 0 failures) | round 62 |
| Evolution loop IMPROVES model | round 57e, confidence +0.25 |
| Divergence quantified (35% split = RLCD fuel) | round 57 |
| Adversarial robustness matrix | round 61 |
| Calibration layer (temperature scaling) | round 60, 2 tier flips proven |


## Phase 0 — Blueprint (current)
- [x] Vision, naming, protocols defined
- [x] Evidence base collected (see RESEARCH-NOTES.md)
- [x] Repo scaffolding, LICENSE

## Phase 1 ' + chr(8212) + ' Standstill Prototype (target: 2-3 weeks)

> Progress 2026-10-13: pando package scaffolded (common/backend/graft/worker/coordinator).
> 6/6 unittests pass incl. real-HTTP e2e; multi-process smoke OK (root + 2 stems,
> heartbeat registration, fan-out vote, conservative escalate when quorum < 3).
> MockBackend stands in for Laya; checkpoint diversity enforced in quorum picking.
> Progress (same day, later): pando/spore.py landed (Ed25519 manifest + 64KiB
> chunk Merkle tree, verify-on-arrival, tamper detection; CDC deferred to Phase 5).
> PyNaCl installed. One-shot crypto smoke passed (sign/verify roundtrip, tamper
> rejection). ssets/graft_suite.jsonl: 16-task acceptance set (choice/score/noul
> x 3 checkpoints), statically validated.
Goal: consensus decision service on heterogeneous devices, no replication yet.

- [x] `worker`: single-binary worker serving Laya via ONNX Runtime (x86 verified;
      ARM/Termux path documented, needs hardware)
- [x] `coordinator`: coordinator — fan-out, k-replica voting, confidence gating,
      calibration layer (blueprint 3.2.1), playbook question enforcement
- [x] `graft`: capability probe + EU enumeration + role assignment
- [x] Dashboard v2: static render with lineage votes, playbook attribution,
      disagreement log (live update mode queued)
- [x] Root quorum diversity: >=2 distinct Laya checkpoints per vote
- [ ] First 4GB TV-box Graft benchmark published (cold load / RSS / latency / 30-min thermal)
- **Gate**: 5+ devices (mixed x86/ARM), 50+ decisions/s, 100% availability with k-1 nodes

## Phase 2 (SPROUT/VERIFY/LIVE verified with real 571MB weights) — Spore Replication (target: +3 weeks, sandbox only)
Goal: self-replication in a firewalled sandbox.

> Progress 2026-10-14: transfer + integrity + selective re-fetch implemented
> (spore.py, tools/fix_gathernd.py separate); sandbox gate shipped as
> pando/lab.py with Dockerfile; SPROUT/VERIFY/LIVE chain verified with
> real 571MB weights; TOFU pinning + persistent root keypair live.

- [x] Spore payload builder + Ed25519 signature + per-chunk Merkle
- [x] SCAN/GRAFT/SPROUT/VERIFY state machine (multi-hop chain verified)
- [x] Selective re-fetch (3 attempts) + failure taxonomy
- **Gate**: >90% replication success rate on heterogeneous sandbox fleet, quantified timeline

## Phase 3 — Topology-Aware Scheduling (target: +2 weeks)
Goal: two-group WAN demo.

- [ ] RTT matrix + DP scheduling across groups
- [ ] Pipeline-parallel within group (for any future >memory model)
- [ ] Coordinator placement optimizer (leyten/shard insight)
- **Gate**: cross-group consensus works over WAN with quantified latency budget

## Phase 4 — Playbook Evolution (target: +4 weeks)
Goal: offline LLM advisor loop.

- [ ] Failure-log pipeline -> LLM analysis -> playbook v(n+1)
- [ ] Signed playbook rollout + rollback
- **Gate**: measurable improvement on playbook blind-spot tasks across 3 iterations

## Phase 5 — RLCD Evolution (data layer VERIFIED: trainer accepts rows)

> Progress: export_training_rows emits nested format;
> laya.train.items_from_rows accepts it (1 item, 0 skipped);
> first real fine-tune completed (loss=0.1663, confidence +0.25 universal).
> A/B attribution via playbook_hash in every rollout entry. — Distributed RLCD (research)
Goal: swarm self-trains its own checkpoints.

- [ ] Decision rollout collection at scale
- [ ] Federated RLCD training loop (Hivemind-style)
- [ ] A/B: old lineage vs new lineage consensus accuracy
- **Gate**: new lineage measurably outperforms parent on held-out sandbox tasks

## Open-source release checklist
- [ ] Defensive-research framing, sandbox-only guarantees
- [ ] Reproducible sandbox (Vagrant/Nix)
- [ ] Metrics dashboard as first-class artifact

## Testing policy (project decision 2026-10-13)

- Default loop = static checks (py_compile / ruff) + research. Do NOT add unit tests.
- One smoke file only: `tests/test_smoke.py` (2 cases). Run at milestones, after
  protocol changes, or before any release step. Never run per-edit.
- Mocks limited to decision backends (no model weights in CI path). Every other
  layer (HTTP, registry, quorum, aggregation) runs real code in-process or over
  real localhost sockets.
- Real-device validation replaces mocks whenever hardware is attached; benchmark
  results land in docs/research with the Graft acceptance protocol.

## Git workflow (project decision 2026-10-13)

- Remote: https://github.com/JewelRoam/PandoRSI.git (main branch).
- Commit + push at every milestone: after each research round that changes the
  blueprint, after each new module, and before/after any smoke-test milestone.
- Commit messages: imperative, one-line summary + short body listing artifacts.
- Never commit model weights, metrics/, or secrets (see .gitignore).
- Initial commit: e25d141 (blueprint + skeleton + release scaffolding).

## Session-log backup policy (2026-10-14)

- 	ools/backup_session.py extracts + sanitizes this session into
  docs/archive/session-logs/<timestamp>-session-backup.md; run at milestones and before
  pushing. Agent appends a curated summary to the Summary section per run.
