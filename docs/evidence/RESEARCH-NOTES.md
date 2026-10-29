# Research Notes — evidence base

Snapshots taken 2026-10-13. Index-sourced items marked (idx); direct reads marked (live).

## Decision models (System 1)

- Laya: 421M English (ModernBERT-large, 512 ctx), 322M multilingual (mmBERT-base,
  1024 ctx up to 8k), ~33ms/T4, batch 7.2ms/q, Apache-2.0.
  https://huggingface.co/convaiinnovations/laya (live, idx)
- CPU ports: OpenVINO int8 (rupeshs/laya-ov-int8), LiteRT (litert-community/laya-LiteRT). (idx)
- Jev: TypeSafe AI, closed API, 2026-09-15, founder Diogo Almeida. (idx)
- Jev beat Pokemon Red in <1 week WITH Claude Opus 5 coaching. (idx, Tom's Hardware)
- Jev edge orchestration: -22.4% latency vs DeepSeek, -61.9% vs Gemini. arXiv:2609.23136 (idx)

## Distributed inference evidence

- Anthropic 2026-10-12: Gemma4-26B-A4B on 15x $55 Android TV boxes (4GB RAM),
  pipeline parallel, 0.39 tok/s; 32k-context agent run = 27-32h.
  Likens spread to "digital slime mold". https://www.anthropic.com/research/self-replicating-ai (live)
- leyten/shard: GLM-5.2 744B across 6 US states over public internet, PP + speculative
  decoding + async pipelining, ~30 tok/s. Plain PP over WAN: 1.87 tok/s (latency-bound).
  Coordinator placement: 174->102ms ring RTT. Mid-request node-kill heal demonstrated. (live)
- CrossPipe (USENIX ATC25): pipeline schedules for cross-datacenter training. (idx)
- EdgeShard (IEEE IoT J. 2024): DP-optimized device selection + model partition on edge. (idx)
- Galaxy arXiv:2405.17245: collaborative edge inference over idle trusted devices. (idx)
- Petals: swarm PP with DHT, nodes join/leave; upstream inactive since 2024-09;
  public swarm reported dead 2026-09. (idx)
- llama.cpp RPC: distributed memory pool, static addresses, PoC/fragile per README. (live)
- vLLM Elastic Expert Parallelism (2026-05): runtime worker scaling, GPU/Ray only. (idx)

## Physical metaphors

- Pando: ~47,000 aspen stems, one root system, ~6000t, possibly 80,000 years old.
  Latin "I spread". https://en.wikipedia.org/wiki/Pando_(tree) (live)
- Physarum polycephalum: single multinucleate cell, no cell walls, solved Tokyo subway
  network optimization. "Many-headed slime". (live, PMC4594612)

## Naming-space collisions checked

- Medusa: taken by FasterDecoding/Medusa + thustorage/Medusa (same field) -> excluded.
- Hydra: heavily used across ecosystems -> excluded despite iconic imagery.
- Ouroboros: Cardano consensus protocol -> excluded.
- Mycelium: Bitcoin wallet + arXiv project -> excluded.
- Rhizome: existing AI-agents project Rhizome-Project -> excluded.
- Pando: pando.com defunct media; namespace effectively clean -> chosen.

## CPU benchmarks (report v2, 2026-10-13) (idx)

- Pi 5 interactive threshold ~= decode >=10 tok/s, realistic cap <=1B.
  Gemma 3 270M int8: 23.7/17.7/11.8 tok/s @ ctx 1280/2048/4096.
  Llama 3.2 1B Q4: 10-15 tok/s. Gemma 4 E2B QAT: ~7. Qwen3 4B: 5-7.
- Phones: OnePlus 12 ExecuTorch 1B = 50.2 tok/s (TTFT 0.3s); Galaxy S25 0.6B = 12.9.
- Thermal: uncooled Pi 5 throttles ~85C, sustained throughput -~1/3; M1 Pro 3h batch nominal.
- 4GB TV box (Amlogic S905): ZERO public full benchmarks. Bandwidth est. 8-10GB/s
  -> 1B Q4 ceiling 10-12 tok/s, likely 4-8 actual; 3B+ likely OOM/swap. Estimates only.
- Min validation protocol: cold load, peak RSS, TTFT, decode, 30-min thermal curve, task quality.

## Consensus caveat (report v2) (idx)

- Identical-weight replica voting has highly correlated errors; ensemble gains require
  heterogeneous checkpoints / temperatures / schemas.
  PandoRSI Root must enforce >=2 distinct Laya checkpoints per quorum.

## Evolution feasibility (report v2) (idx)

- laya-rs ships an RLCD training loop; Flower supports Android/Pi/Jetson clients.
- Decision-model rollout = single forward pass (non-autoregressive): weak-device
  rollout collection + centralized update is far more feasible than LLM RL
  (INTELLECT-2's 4xRTX3090 threshold applies to LLM RL, not this route).
- Multi-agent orchestration costs ~15x tokens; coordinator context and bad-plan
  cascade are the real bottlenecks for hierarchical-agent route (route C).
