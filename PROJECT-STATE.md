# PandoSI Project State

*Last updated: 2026-10-14 (after ~3 hours of continuous development)*

## Where we are

PandoSI v0.1 research preview is **feature-complete for Phase 1-2** and
**Phase 5 data layer is verified**. The system runs on a single Intel AI PC
(Core Ultra 7 255H) with three heterogeneous execution units (CPU, iGPU,
NPU) all voting in the same consensus quorum.

## What exists (verified)

| Component | Files | Status |
|---|---|---|
| Decision swarm (rootd + stemd) | rootd.py, stemd.py, backend.py, common.py | 2/2 tests, live smokes |
| Capability probe + EU enumeration | graft.py | 3 EUs detected live |
| Calibration layer (temperature scaling) | calibrate.py | math verified |
| Signed self-replication (Spore) | spore.py | 3-hop chain, 0 failures |
| Sandbox gate (Ring 2) | lab.py | refusal + attested paths |
| Playbook governance (5 layers) | playbook.py + rootd integration | 6/6 E2E |
| RLCD evolution data path | rollout.py + export | trainer-compatible |
| 40-task graft suite | assets/graft_suite.jsonl + graft_eval.py | 3 lineages measured |
| NPU-compatible graph | postprocess_onnx.py + IsNaN removal | GPU compile OK |
| Multi-bucket iGPU serving | backend.py (_MultiBucketSession) | 25-40ms variable-length |
| Dashboard v2 | tools/dashboard.py | lineage + attribution + disagreements |
| CUDA worker | backend.py CUDA branch + tools/cuda_bench.py | RTX 3070 live |
| Heterogeneous HTTP check | tools/hetero_http_check.py | CPU+iGPU+NPU+CUDA live |
| PyTorch (non-OpenVINO) stem | backend.py TorchLayaBackend + stemd --backend laya | torch-cuda votes in the same quorum |
| Stem wire contract | docs/STEM-PROTOCOL.md | framework-free join spec (3 endpoints) |
| Stage graph splitter | tools/stage_split.py | node-boundary cut, shared external weights |
| Pipeline measurement | tools/stage_pipeline_check.py | real HTTP 2-stage chain, latency + throughput |
| Adaptive stem timeouts | rootd.py per-stem EWMA | 5s cold -> 0.7-1.0s learned |
| Slow-node isolation | tools/slow_stem_check.py | 4->3 quorum, 24ms cooldown skip |
| LAN-interface HTTP | tools/lan_http_check.py + stemd advertise host | same-host protocol path verified |
| Remote-link degradation | tools/remote_latency_check.py | timeout -> skip -> recovery verified |
| Runtime elastic membership | tools/swarm_check.py | live 3 -> 5 -> 4 transitions |
| CUDA hot-plug | tools/hetero_http_check.py --hotplug-cuda | Intel 3 -> +RTX 3070 |
| Hierarchical groups | pandosi/bridge.py + tools/two_group_check.py | remote aggregate + partition fallback |
| CUDA RLCD training | tools/rlcd_cuda_train.py + artifacts/rlcd-cuda-v6 | 72 updates overfit confidence, no held-out accuracy gain |
| Deterministic suite split | tools/build_suite_split.py | 34 train / 6 holdout tasks |
| Expanded prompt pool | tools/build_prompt_pool.py | 78 prompts, 45 contrast rows |
| CUDA candidate v7 | artifacts/rlcd-cuda-v7 | 41 updates, stable holdout, not promoted |
| Live lineage A/B | tools/lineage_ab_check.py | v7 6/6 agreement, no accuracy gain, rejected |
| Post-training calibration | tools/calibrate_checkpoint.py | T=2.67, confidence reduced, not promoted |
| Real lineage HTTP identity | backend/stemd checkpoint override + tools/real_lineage_http_check.py | en/ml/td distinct, 11/12 split |

## What is proven (key numbers)

| finding | value | source |
|---|---|---|
| iGPU graph p50 | 25.6ms (4x faster than CPU 106ms) | eu_bench, round 34 |
| NPU graph p50 | 95.7ms (IsNaN-free surgery graph) | round 54 |
| iGPU full suite accuracy | 65% (matches CPU exactly, 0/40 diverge) | round 48 |
| NPU full suite accuracy | 65% (matches CPU exactly, 0/40 diverge) | round 56b |
| Cross-lineage divergence (en vs ml) | 32.5% (most independent pair) | round 57 |
| Split-case majority reliability | 36% (verify routing is NECESSARY) | round 57b |
| RLCD fine-tune confidence delta | +0.25 universal, no regression | round 57e/f |
| Spore 3-hop chain | 97 chunks, 0 integrity failures | round 62 |
| Spore generation lineage | generation 0 -> 1 -> 2 + parent hash | round 82 |
| Reproduction DAG | tools/spore_dag_check.py | branch validation + invalid links rejected |
| Cached Spore lineage | pandosi/spore.py manifest cache | valid / parent-missing / invalid states |
| Root Spore HTTP surface | tools/spore_root_check.py | pubkey/manifest/by-hash + 404/403 gates |
| Peer-fetched parent manifest | /spore/manifests/<hash> + ensure_parent_lineage | missing parent recovered over HTTP, bad hash/signature refused |
| Unified CLI | pandosi/cli.py + `python -m pandosi` | doctor/join/stem/grants/service/lab entry |
| Installable wheel | tools/package_check.py + runtime extras | console script verified outside repo |
| Packaged assets | pandosi/resources.py + setuptools data-files | playbook + calibration found outside repo |
| Wheel integrity | tools/verify_wheel.py + CI build | 37 entries, required code/assets present |
| Source distribution | tools/verify_sdist.py + release workflow | sdist checked, tag release prepared |
| GitHub URL install | tools/package_check.py --source git | clone -> wheel -> console verified |
| CUDA vs CPU p50 | 15.0 ms vs 125.6 ms (8.36x) | round 65 |
| CUDA decision parity | 14/14 English tasks; max conf delta 0.0009 | round 65 |
| Four-runtime HTTP p50 | 430.5 ms sequential -> 237.8 ms parallel | rounds 65/66 |
| Two-ecosystem quorum | torch-cuda + onnx/onnx-cuda/onnx-gpu, tier auto, 4/4 | round 92 |
| Pipeline staging on one CPU | 0.95-1.13x throughput (no gain), +1.4-24.6 ms latency | round 94 |
| Cross-device staging (CUDA+CPU) | 0.47x throughput, +50 ms latency | round 94 |
| Activation interface size | 0.15-1.22 MB, loopback hop 1-2 ms | round 94 |

## What is NOT yet done

| item | blocker | effort estimate |
|---|---|---|
| NPU deep op analysis | IsNaN removed but other ops may block | 1-2 days |
| Termux/TV box real test | needs Android hardware | 0.5 day on device |
| Docker image build test | needs Docker environment | 0.5 day |
| RLCD scale-up beyond one contrast pair | needs more accumulated pairs | 1-2 days |
| Multi-hop with graph adaptation | origin -> B(NPU) -> C(CPU) format conversion | 1 day |
| Dashboard live update mode | design done, not coded | 0.5 day |
| Rootd calibration from held-out set | needs labeled eval data | 0.5 day |

## How to resume

1. Read RESEARCH-SUMMARY.md for the definitive findings
2. Run `python tools/demo.py` to verify everything still works
3. Pick any item from "What is NOT yet done" above
4. All code, docs, and research logs are in this repo and on GitHub

## Repository

https://github.com/JewelRoam/PandoSI (main branch, ~30 commits)
