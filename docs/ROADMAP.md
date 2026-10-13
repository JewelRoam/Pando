# PandoSI Roadmap

## Phase 0 — Blueprint (current)
- [x] Vision, naming, protocols defined
- [x] Evidence base collected (see RESEARCH-NOTES.md)
- [ ] Repo scaffolding, LICENSE, threat-model statement

## Phase 1 ' + chr(8212) + ' Standstill Prototype (target: 2-3 weeks)

> Progress 2026-10-13: pandosi package scaffolded (common/backend/graft/stemd/rootd).
> 6/6 unittests pass incl. real-HTTP e2e; multi-process smoke OK (root + 2 stems,
> heartbeat registration, fan-out vote, conservative escalate when quorum < 3).
> MockBackend stands in for Laya; checkpoint diversity enforced in quorum picking.
> Progress (same day, later): pandosi/spore.py landed (Ed25519 manifest + 64KiB
> chunk Merkle tree, verify-on-arrival, tamper detection; CDC deferred to Phase 5).
> PyNaCl installed. One-shot crypto smoke passed (sign/verify roundtrip, tamper
> rejection). ssets/graft_suite.jsonl: 16-task acceptance set (choice/score/noul
> x 3 checkpoints), statically validated.
Goal: consensus decision service on heterogeneous devices, no replication yet.

- [ ] `stemd`: single-binary worker serving Laya via ONNX Runtime (x86 first, then ARM)
- [ ] `rootd`: coordinator — fan-out, k-replica voting, confidence gating
- [ ] `graft`: capability probe + role assignment (< 60s onboarding)
- [ ] Dashboard: decisions/s, consensus accuracy, per-node latency, device inventory
- [ ] Root quorum diversity: >=2 distinct Laya checkpoints per vote (correlated-error fix)
- [ ] First 4GB TV-box Graft benchmark published (cold load / RSS / latency / 30-min thermal)
- **Gate**: 5+ devices (mixed x86/ARM), 50+ decisions/s, 100% availability with k-1 nodes

## Phase 2 — Spore Replication (target: +3 weeks, sandbox only)
Goal: self-replication in a firewalled sandbox.

- [ ] Spore payload builder + signature verification
- [ ] SCAN/GRAFT/SPROUT/VERIFY state machine
- [ ] Failure taxonomy + retry policies
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

## Phase 5 — Distributed RLCD (research)
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
