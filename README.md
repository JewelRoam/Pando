# PandoSI

**The forest that is one tree.**

PandoSI is a research project building an elastically scalable, fully heterogeneous,
self-replicating **unified intelligence** — many cheap devices, one mind.

- **Pando** — the largest single organism on Earth: a quaking aspen colony of ~47,000 stems
  that looks like a forest but shares one root system. Latin: *I spread*.
- **SI** — Super Intelligence.

It runs a swarm of small, non-autoregressive **decision models** (System 1, e.g. Laya, 321-421M params,
single forward pass, calibrated probabilities) on arbitrary commodity hardware —
CPUs, TV boxes, phones, single-board computers — coordinated by lightweight protocols:

| Protocol | Role |
|---|---|
| **Stem** | A device joins by running a decision-model worker. No shard, no special hardware. |
| **Root** | Consensus layer: k replicas vote on calibrated probabilities (quorum). |
| **Spore** | Self-replication payload: weights + playbook + harness. Scan -> graft -> sprout -> verify. |
| **Graft** | Heterogeneous onboarding: capability probe, role assignment, zero reconfiguration. |

Topology-aware scheduling: **data parallel across groups (WAN), pipeline parallel within
groups (LAN)** — pipeline for capability, data-parallel for elasticity. A playbook
(written once by a human or a System-2 LLM) defines goals; the swarm executes it at
runtime with zero external model dependency.

> Research status: blueprint phase. All replication experiments are designed for
> **closed, firewalled sandboxes** and published strictly as defensive (blue-team) research.

## Docs

- [Blueprint (zh-CN)](docs/BLUEPRINT.zh-CN.md) — full architecture and method.
- [Roadmap](docs/ROADMAP.md) — phases, milestones, measurable targets.
- [Research Notes](docs/RESEARCH-NOTES.md) — evidence base with citations.

## Quickstart (Ring 0 - zero privilege)

```bash
pip install pynacl zeroconf          # deps (onnxruntime optional for real Laya)
python -m pandosi.join               # becomes root+stem if no swarm on LAN,
                                     # otherwise discovers and joins via mDNS
```

