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
| Cross-ecosystem votes | tools/cross_ecosystem_check.py | ONNX vs PyTorch, 27/27 agreement |
| Topology planner | pandosi/topology.py + tools/topology_check.py | replicate/stage/infeasible from measured numbers |
| Autonomy loop | pandosi/autonomy.py + tools/autonomy_check.py | observe-decide-act, two-part gate |
| Gate risk-coverage | tools/gating_check.py | auto 51.8% coverage at 92.9% accuracy |
| Gate threshold tuning | tools/gate_tune.py | shipped 0.6/0.6 already optimal |
| Gate live verification | tools/gate_e2e_check.py | live == offline, delta 0.0000 |
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
| Ecosystem effect on votes | ONNX vs PyTorch: 27/27 agreement, conf gap 0.0028 | round 95 |
| Ensemble vs best member (accuracy) | n=27: 51.9% vs 55.6-59.3% (superseded) | round 117 |
| Ensemble vs best member (n=480) | weighted vote 89.6% vs best member 88.5% | round 118 |
| Ensemble edge, paired bootstrap | +1.0 pt full CI [-1.0,+3.1]; +1.6 pt at gate coverage CI [0.00,+3.4] | round 118 |
| Membership vs combiner | lineages span 13.5 pt, six combiners span 1.0 pt | round 118 |
| Member error correlation | phi +0.23..+0.50; both-wrong 2.0-4.3x independence | round 118 |
| Majority vote vs independent errors | predicts 7.4% wrong, observed 11.7% | round 118 |
| Split-case reliability | 74.5% (best rule) of 157 splits; 91.7% available in trio | round 118 |
| Resolution cost | 1 pt at 5.6% discordance needs ~3,900 tasks | round 118 |
| Gate vs single-model confidence | 92.3% vs 100% was 13 of 27 tasks; not reproduced at n=480 | round 117/118 |
| Oracle available to the trio | 74.1% (n=27) -> 95.2% (n=480); 91.7% on splits | round 117/118 |
| Trained lineage vs its parent | rlcd-v7 = english on 240/240 decisions, phi +1.000 | round 119 |
| Cost of a duplicate member | weighted vote 90.4% -> 85.4%, equal to the duplicated member | round 119 |
| Quorum over lineages | pick_quorum round-robins checkpoint groups; a duplicate still votes at full weight | round 119 |
| Majority with a duplicate | accuracy == the duplicated member's accuracy (theorem, 4dp match) | round 119 |
| Re-ask on disagreements | weighted vote 75.6% -> 76.9%; answer moves on 12/78 rows | round 120 |
| Re-ask unanimity as a signal | 31/78 settle, accuracy identical to the first ballot (83.9%) | round 120 |
| Routes to the split headroom | combiner <=1.0 pt, narrower ballot +1.3 pt, of 18 available | round 118/120 |
| Lineage-counted quorum | `aggregate(count_lineages=)` + `rootd --lineage-quorum` | round 121 |
| Invariance under replication | shipped drifts up to 14.6 pt by copy pattern; collapsed drifts 0.000 | round 121 |
| Replica of the weakest member | quorum falls toward that member (0.746 vs 0.892) | round 121 |
| Near-duplicate under a new name | still admitted at full weight - needs an admission-time divergence test | round 121 |
| Ensemble buys decorrelation, not accuracy | 98.75%-accurate aligned voter helps least; 95.00% decorrelated helps most | round 122 |
| Matched-coverage edge with a decorrelated member | 96.26% (trio) -> 100% on the 187 rows the gate accepts | round 122 |
| Same-family member (rlcd-v7) | phi +1.000, -5.0 points | round 119 |
| Suite phrasing leak | dev and test share all 53 cues; every test row reuses a dev cue | round 122 |
| Phrasing-held-out suite | assets/heldout_test.jsonl, 48 cues seen in no earlier split | round 122 |
| Accuracy on unseen wording | english 85.0->75.4, multilingual 74.2->64.2, typed-decisions 87.1->72.5 | round 123 |
| Member ranking is wording-dependent | typed-decisions best on shared wording, english on new wording | round 123 |
| Outsiders on unseen wording | bayes 93.75->57.50, rules 95.0->17.5 but 42/42 precise | round 123 |
| Decorrelated weak voter still wins | 57.5% member: +4.1 pt matched, +3.8 pt full; 76.25% aligned member: -2.5 pt full | round 123 |
| Lineage rule over real HTTP | 2:1 stem majority auto -> lineage tie verify; 4 scenarios pass | round 124 |
| Corroboration floor | `[gating] corroboration_floor`: replicas no longer auto-execute at floor 2 | round 124 |
| Decision response observability | `lineages` + `corroboration_floor` reported when observable | round 124 |
| Ballot order | reversed candidates flip 0/240 answers; phi +1.000 | round 125 |
| Quantised sibling (int8 vs fp32 english) | 48/240 flips, but phi +0.413; no ensemble gain | round 125 |
| Same-family diversity spectrum | monotone in phi: 1.000 (descendant/ballot) -> 0.41 (int8) -> 0.07 (outside family, +4.1 pt) | round 125 |
| Trio correlation on unseen wording | phi +0.36..+0.82 (dev: +0.21..+0.53) - harder wording correlates them more | round 125 |
| Text member (naive Bayes) | pandosi/textmodel.py, 19 KB artifact, stemd --backend text | round 126 |
| 4-member swarm, over real HTTP | trio 73.75% / +text 72.92% at full coverage - a loss | round 126 |
| Heartbeat TTL | an un-heartbeating stem is dropped after 15 s; a check must heartbeat | round 126 |
| Tied quorum answers | 2-2 splits ranked by confidence at verify, never auto | round 127 |
| Tie-break measured | trio 73.75 -> 77.08; +text 72.92 -> 78.33; auto tier unchanged | round 127 |
| Member portfolio | best single addition is the weakest member (centroid 42.9%): +3.39 pt, 100% of resamples | round 128 |
| Four cheap additions | -11.62 pt vs the trio; two reweightings do not fix it | round 128 |
| Cheap members resemble each other | chars x mixed phi +0.898, words x centroid +0.687; each ~0 vs the trio | round 128 |
| Synthetic independent voters | monotone in count at every accuracy >= 0.5; six 50% members = +13 pt | round 128 |
| Bloc override | on 152 trio-unanimous rows the 4 cheap members override 37 times, wrong 32 | round 128 |
| Error concentration | most common wrong answer: 27-39% of a neural member's errors (uniform would be 14%) | round 128 |
| Bloc collapse | `[lineages] blocs` + `aggregate(blocs=)` ; opinions counted, not stems | round 129 |
| Bloc collapse measured | 4 correlated members: 65.42% (names) -> 77.92% (one bloc), +12.5 pt | round 129 |
| Growth with replicas | same-family devices cost nothing: 77.92% vs a trio at 77.08% | round 129 |
| Bloc representation | mass consensus loses 2.48 pt to the best member (7.3% of resamples) - open | round 129 |
| CPU vs Intel iGPU (same weights) | identical choices 240/240, confidence gap 0.0002 | round 130 |
| Heterogeneous unit speed | CPU 12.3 dec/s (79 ms), iGPU 18.0 dec/s (53 ms) | round 130 |
| Second unit in the quorum | 4 stems, 3 opinions; the device buys capacity, not information | round 130 |
| NPU, current OpenVINO | blocked: vpux rejects node_MatMul_215 (unbounded dims) | round 130 |
| GPU graph fix | artifacts/laya-en-static-fixed.onnx compiles on the iGPU in 12-15 s | round 130 |
| NPU unblocked | laya-en-static-nonisnan.onnx compiles on the NPU in 39 s | round 131 |
| Three units, one voter | CPU / iGPU / NPU: identical choices 240/240, gaps 0.0002/0.0003 | round 131 |
| Unit speeds (this model) | iGPU 17.6 dec/s, CPU 10.0, NPU 8.7 (53/94/112 ms median) | round 131 |
| Cross-unit graph dialects | GPU reject: GatherND; NPU reject: the GPU's GatherElements repair | round 131 |
| Corroboration costs throughput | 3 units: k=1 24.1/s, k=2 14.4/s, k=3 10.5/s (p50 140/245/374 ms) | round 132 |
| Units add capacity at k=1 | 1 unit 14.7/s -> 2 units 27.1/s = the sum of their solo rates | round 132 |
| One host, not three servers | the 3rd unit (NPU) adds little or subtracts; all stems share the host CPU | round 132 |
| Elastic dial, measured | k=3 -> k=1 at 3 units: 2.3x throughput, half the p50 latency | round 132 |
| Process-shaped fleet | separate stemd processes behave the same: 1u 26.5, 2u 25.0, 3u 26.0 /s | round 133 |
| k is reproducible, units are not | k=3 costs 2-3x in both process shapes; units add nothing | round 133 |
| Unit rank depends on load | at 4 clients: NPU 33.2/s > CPU 24.9 > iGPU 24.0 (sequential: NPU last) | round 133 |
| Single-host variance | the same cell measured 16.8/s and 34.1/s in two runs - 30-50% noise | round 133 |
| `blocked` does not isolate | pick_quorum ignores it; only retire() removes a node | round 134 |
| Per-unit numbers (isolated) | CPU 16.8/s idle, 6.0/s loaded; iGPU 171/341; NPU 268/296 | round 134 |
| Host CPU per decision | CPU 791->1647 ms; iGPU 6.6->5.1; **NPU 1.4->1.8** (500-1000x) | round 134 |
| Load invariance | accelerators keep serving while 8 cores are busy; the CPU stem loses 64% | round 134 |
| Host cost reported by stems | measured `host_ms_per_decision`, refreshed every heartbeat | round 135 |
| Cost-band quorum picking | ten-fold bands: the expensive unit is not woken while a cheap one is free | round 135 |
| `unanswered` in the response | names the members that were picked and said nothing | round 135 |
| Accelerator dropout | under 4 concurrent clients the iGPU/NPU stems fail ~half their calls; the CPU stem never does | round 135 |
| Round 134 accelerator numbers | **withdrawn** - the dropouts masqueraded as throughput | round 135 |
| Device should serialise | the `serialize` model (one decision at a time) is the next fix | round 135 |
| Stems serialise by default | lock around `backend.decide`; `stemd --parallel` to opt out | round 136 |
| Dropouts fixed | silent decisions 49-56/120 -> **0**; the fleet routes to the accelerators | round 136 |
| Unit table (serialised, loaded host) | CPU service 94->309 ms at 2856 ms host CPU/decision; iGPU 53->35 ms at 36; NPU 112->109 ms at **4** | round 136 |
| Cost ratio, corrected | 80-700x between the NPU and the CPU per decision | round 136 |
| Member portfolio (10 cheap members) | all decorrelated (mean phi -0.06..+0.08); 9 of 10 add value | round 137 |
| What predicts a member's value | nothing does at 240 rows: accuracy rank -0.55 (inverted), phi -0.02, net rescues -0.55 | round 137 |
| Admission rule, bounded | admit on decorrelation; measure the rest after the fact | round 137 |
| Birth latency (transfer excluded) | load 9.4 s + register 0.2 s + first vote 5.0 s = 14.6 s | round 138 |
| Birth latency (whole artifact) | ~29 s, composed with round 106's 14.2 s transfer | round 138 |
| What the newborn buys | nothing: k=1 26.1/s -> k=2 13.5/s (its 4 votes are the quorum working) | round 138 |
| Spore pull gate | `sandbox_confirmed` refuses by default; the check does not assert it | round 138 |
| Speed-proportional split (measured) | 0.66x / 0.74x vs quorum-1 replication; prediction falsified | round 97 |
| Hop cost under load (loopback) | 1.9-11.1 ms vs 1.2 ms idle | round 97 |
| Staging ceiling (5.9x slower partner) | 1.17x theoretical maximum | round 97 |
| Laya act head | saturated at 1.0 on 8 states, both ecosystems | round 98 |
| Lone-survivor quorum | tier=auto with 1 responder; needs a responder policy | round 98 |
| Gate value (3 lineages) | auto 92.9% vs 51.8% overall; splits 16.7% | round 99 |
| Gate with production calibration | 48.1% coverage at 92.3%; held set 14.3% accurate | round 100 |
| Live vs offline gate | identical to 4dp over real HTTP stems | round 100 |
| Calibration keying defect | temperature selected by checkpoint name string | round 100 |
| Calibration enforcement | report in rootd, enforce in AutonomyLoop | round 101 |
| Declared lineage admission | playbook [lineages].admit + /register 403 | round 102 |
| Artifact attestation | artifact_digest + [lineages.artifacts] + 403 | round 103 |
| Signed playbooks | sign_playbook.py + --playbook-key pin | round 104 |
| Self-replication loop | pandosi/reproduce.py + tools/reproduce_check.py | quorum 1 -> 2, signed lineage record |
| Batched spore transfer | /spore/chunks/<name>/<start>/<count> + keep-alive | 1.68 GB in 14.2 s (8.8x) |
| Delta replication | build_manifest_cdc + ChunkStore + /spore/blobs | 1610 MB -> 105 MB (15.3x) |
| Packed chunk store | ChunkStore (chunks.bin + index) | delta leg 5.84 s vs 10.49 s cold |
| Delta in the live path | rootd --chunking cdc + join --chunk-store | end-to-end 105 MB vs 1610 MB |
| rlcd-v7 calibration | assets/calibration/rlcd-v7.json | T=2.6704 from round 81 |
| Scaling law of the quorum | tools/scaling_law_check.py | replicas alone +9%; fixed k + rotation 1.9x |
| Configurable quorum size | rootd --quorum-k + per-decision rotation | scaling unit is the quorum |
| Load-aware quorum picking | Coordinator in-flight ordering | 16-24% over blind rotation |
| Adaptive quorum size | --adaptive-quorum/--latency-target-ms | holds k=5 idle, k=3 under load |
| Per-category quorum floors | playbook gating quorum_floor | security k=4 vs ordinary k=2 under load |
| Retirement | pandosi/elastic.py + POST /retire + blocklist | slow member removed, stays removed |
| Declared categories | playbook question category + conservative default | risk_check holds k=4, not k=2 |
| Category validation | load rejects an unknown category | typo fails at load, not at decision time |
| Elastic supervisor | GET /status + ElasticSupervisor | grew 2 -> 3 members, p50 92.8 -> 61 ms |
| Growth via real reproduction | grow wired to reproduce() + cooldown | 1st growth 449.7 -> 119.3 ms, 2nd 138.7 -> 981 ms |
| Cooldown/executor rule | executed actions only | an unexecuted decision starts no cooldown |
| Host-aware growth | Member.host + max_per_host | 1 reproduction (363->126.5 ms) vs 2 (->791.8 ms) |
| Growth rollback | bet tracked, undone under the reproduce grant | 882 -> 124 ms in two rounds |
| Bet judged on a median | rollback_samples/rollback_window_s | both verdicts correct on one mechanism |
| Negative memory | rolled_back[(host,checkpoint)] + TTL | no repeat: 848.7 -> 144.1 ms |
| Capacity accounting | rollback_allowed | refuses if the swarm depends on the node |
| Reproduce grant | grants.py action "reproduce" | operator-approved, expiring |

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
