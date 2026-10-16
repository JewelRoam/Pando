# PandoSI Project State

*Last updated: 2026-10-16 (round 161; continuous research session)*

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
| Learned combiner | fitted on dev votes (l2 by 5-fold CV), scored on held-out wording | round 139 |
| Swarm beats its best member | 80.42% vs 75.42% full coverage (+5.0) | round 139 |
| Where the headroom is | gate's rows: 90.70% vs an oracle of 91.28%; declined rows: learned 54.41% vs member 41.18% vs oracle 67.65% | round 139 |
| Training rows not the limit | 480 instead of 240 changes nothing (79.58 vs 80.42) | round 139 |
| Combiner shipped as an artifact | assets/combiner.json, 4 KB, fitted + CV'd by tools/train_combiner.py | round 140 |
| Advisory by construction | `suggested_choice` only when the tier is not auto; 0/68 at auto | round 140 |
| Artifact reproduces the claim | full 80.42%, declined rows 54.41% (gate declines them, best member 41.18%) | round 140 |
| Candidate contract | a suggestion outside the request's candidates is refused (found live) | round 140 |
| Third wording set | `--cues c`, 48 phrasings, 0 overlap with A and B; assets/heldout2_test.jsonl | round 141 |
| Three roles separated | combiner on A, text model on B, evaluation on C | round 141 |
| Swarm on unseen wording (split C) | majority 77.08% vs best member 73.75% (+3.3) | round 141 |
| Combiner + text, evaluated on C | full 82.92% (best member 73.75); declined rows 62.34% vs member 40.26%, oracle 71.43% | round 141 |
| Votes vs text | votes alone 58.44% on declined rows, +text 62.34% | round 141 |
| Combiner spec v2 | text posterior appended; text model embedded in the artifact (35 KB) | round 142 |
| Two-provenance training rule | `--reports` (votes) + `--text-train` (wording) disjoint, recorded in meta | round 142 |
| Spec-2 verified on split C | full 82.92%, declined rows 62.34%, 0/77 suggestions at auto | round 142 |
| Text reaches the model live | same votes, different text -> different suggestion probabilities | round 142 |
| Combiner spec 3 | membership-agnostic (27 features): per-lineage cells dropped | round 144 |
| Dropping features helped | split C: full 85.00% (spec 2: 82.92%), declined rows 67.53% | round 144 |
| Smallest real fleet | iGPU english + 19 KB text member, k=2, 240 rows in 13 s | round 144 |
| Live outcomes | auto 76 at 97.37%; declined 164, suggestions 68.29%; decided 77.50% | round 144 |
| Combiner pinned by playbook | `[combiner] path + sha256`; refused on mismatch | round 145 |
| Governance e2e | real root + HTTP stems: admit 403, blocs, floor, pinning | round 145 |
| Corroboration floor isolated | one opinion, agreement 1.0 -> tier verify | round 145 |
| Live growth stages | 1 leaf -> +opinion -> +replica -> -replica, 60 labelled rows each | round 146 |
| A new opinion | opinions 1->2; auto 50->15 rows, accuracy 0.800 -> 1.000 | round 146 |
| A replica buys nothing | opinions stay 2, throughput and answers unchanged | round 146 |
| Removal is free | retiring the replica changes nothing measurable | round 146 |
| LAN path verified | grant -> mDNS discovery -> 2 members -> `tier=auto responded=2` over 192.168.31.100 | round 147 |
| Pairing design confirmed | discovery finds the address; the token (read from the root console) authorises joining | round 147 |
| Cross-device gap | software path proven on a real interface; only a second device is missing | round 147 |
| Calibration does not transfer | refit on 240 dev rows: better on dev, 0.5895 vs 0.5863 held-out | round 148 |
| In-place calibration ceiling | fitted on the deployment's own wording: 0.5318 (-9%) | round 148 |
| Gate is tuned to the tables | sharper confidences: auto 163 -> 179 rows at 0.908 -> 0.860 | round 148 |
| NPU op analysis | **solved** in round 131 (nonisnan graph compiles and votes) | round 148 (backlog fix) |
| Tuning on own rows | 25/50 rows miss the 90% floor; 100 rows buys accuracy with coverage | round 149 |
| In-sample ceiling | thresholds chosen on the evaluation half: 83 rows at 0.9036 = the shipped point | round 149 |
| Shipped gate is optimal in its grid | no tuning procedure could have won on this distribution | round 149 |
| Load-bearing calibration error | correcting typed-decisions' T makes the gate 16 rows freer and 4 points less safe | round 149 |
| Fine-tuning, encoder frozen | 720 rows, 1-3 epochs: 1 and 3 decisions changed in 240 | round 150 |
| **Fine-tuning, encoder unfrozen** | **0.7333 -> 0.9125 on unseen wording; 62/240 decisions changed** | round 150 |
| The new member, verified | re-run through TorchLayaBackend: 219/240; 0 eval states in training | round 150 |
| Error correlation with incumbents | phi +0.17..+0.28 (round 119's descendant: +1.000) | round 150 |
| Fleet with the new member | gate full 0.7708 -> 0.8833; auto 163 -> 189; matched +1.2 | round 150 |
| Open caveat | confidences saturated at 1.0 (228/240); needs its own calibration | round 150 |
| System with the new member | decided 0.9083 on unseen wording (trio: 0.7708; +combiner: 0.8333) | round 151 |
| Gate with the new member | 189 rows auto-executed at 0.9153 | round 151 |
| Spec-3 combiner transfers | shipped file beats a refit on half the data by 8-10 points | round 151 |
| More rows, not more fitting | 120 rows cannot fit 35 features (rounds 137/141, again) | round 151 |
| Fourth wording split (D) | `--cues d`, 0 of 48 cues overlapping A, A2, B, C | round 152 |
| Loop, replicated | parent 0.7458; corpus 648: 3 samples mean 0.9292; corpus 864: 2 samples mean 0.8959 | round 152 |
| Training variance dominates | spreads 5.0 (648) and 13.3 (864) points; ranges overlap | round 152 |
| Admission needs a third condition | independence + capability + **replication**: one run is one draw | round 152 |
| Admission rule, applied | five fine-tunes clear capability (>=0.83 vs floor 0.7692), independence (mean phi +0.04..+0.26), fleet floor, on split D | round 153 |
| Admission is a floor, not a lift | ft-v3-s2 solo 0.9750 vs 0.8792 decided with it seated (gate fitted on the weaker trio) | round 153 |
| Recipe mean vs admitted run | 720 rows mean 0.9431, 960 rows mean 0.8959; admitting a run is not selecting a recipe | round 153 |
| CPU-free live fleet | iGPU+NPU+3070 (torch ft-v3) stems, zero CPU inference: 240/240 three-vote rows, 7.2-8.4 rows/s | round 153 |
| CPU left free | host 16-22% busy during the run; per-decision CPU: npu 5.1, dgpu 45.0, igpu 55-76 ms | round 153 |
| Venv redirector trap | Scripts\python.exe (255 KB) runs the real interpreter as a child; pid CPU reads 0.00, measure the tree | round 153 |
| ft-v3 calibrated | T=6.56 fitted on split C: NLL 300.1 -> 71.1, ECE 0.0909 -> 0.0127; table shipped | round 154 |
| The trio's own tables | offline decided 0.8125 -> 0.8333; round 153's offline baseline was raw | round 154 |
| Replicas dilute | live membership full == english solo (0.7292): two identical votes always hold the majority | round 154 |
| Count opinions | same three devices, lineage-counted: full 0.9417, auto 160/160 correct, decided 0.9792 raw / 0.9750 cal on D | round 154 |
| The advisor must see opinions | declined-row advisor accuracy 0.704 -> 0.926 (D), 0.685 -> 0.904 (C) | round 154 |
| System outgrows its best member | live decided 0.9750 (member solo 0.9417), auto 159/240 all correct, CPU 77% free | round 154 |
| Prediction reproduced live | offline mirror (0.9750, advisor 0.926) written first; live delivered 0.9750 / 0.9259 | round 154 |
| Growth ladder, priced | 1 device 0.7417 -> +replica 0.7417 -> +new opinion 0.9750 -> retire replica 0.9750 | round 155 |
| A replica spends | added english replica: answers identical, throughput 17.25 -> 7.59 rows/s | round 155 |
| A new opinion buys +23.3 | decided 0.7417 -> 0.9750; auto accuracy 0.8454 -> 1.0000; advisor 0.3043 -> 0.9259 | round 155 |
| Removal pays | retiring the replica: answers identical, throughput 8.16 -> 16.07 rows/s | round 155 |
| Predictions 4/4 exact | every ladder stage predicted to 4 dp from recorded votes before the live run | round 155 |
| Latency-ranked seats | with a target the seat goes to the fast twin: p50 115 -> 77 ms, decided 0.9750 unchanged | round 156 |
| k stops buying replicas | expansion oscillated 2<->3 (+40 ms per second decision); ceiling = opinion count -> k stable | round 156 |
| Adaptive fleet, final | p50 60 ms, EWMA 50 ms under the 100 ms target, 15.64 rows/s, decided 0.9750 | round 156 |
| Stem-side accounting | with --lineage-quorum the response's votes are opinions; /health `decisions` gives truth (igpu 240, dgpu 240, npu 1) | round 156 |
| Mac group attached | bridge (Windows -> Mac), one vote as text-nb-v1; registered beside win-igpu under a playbook admitting both | round 157 |
| 24 rows, k=2 with the Mac | mac vote present 24/24; auto 6 / verify 18; decided 0.7500; p50 68 ms | round 157 |
| Foreign confidence measured | out-of-scope refusal at 0.0 (good abstention); in-family unseen wording 0.4167 accuracy at median p 0.83 (confidently wrong) | round 157 |
| Direction | tiering hypothesis: strongest hostable model per device, aggregate composes across tiers; Laya = floor-tier presence | round 157 |
| Mac LLM tier deployed | Qwen3-4B-Instruct-2507 Q4_K_M, llama.cpp/Metal, :8080 OpenAI API, ~45 tok/s, TTFT 50-90 ms | round 158 |
| LLM tier solo on D | 0.7375 zero-shot (= untuned english), mean p 0.9586, ECE 0.2211, 100% JSON parse | round 158 |
| Adding the raw LLM tier | pair 0.9750 -> trio decided 0.9167, but auto coverage 159 -> 225 @ 0.9244: the gate is bought by a foreign confidence scale | round 158 |
| Next (dispatched) | mac-llm votes on C/B; temperature fitted on C, re-run on D: hold auto >= 200 and lift decided >= 0.9750 | round 158 |
| Calibration does not fix the tier | trio with T_C: auto 224 @ 0.9241, decided 0.9167 (raw: identical); ECE 0.303 -> 0.034 on C | round 158 |
| A weak distinct member dilutes | 2-of-3 majorities auto-execute at ~0.924 where the pair escalated to a 0.926 advisor; tiers only lift above their strength | round 158 |
| Prompt scaffold costs accuracy | candidate-only prompt on C: 0.7958 at 131 ms vs JSON 0.6542 at 444 ms | round 158 |
| Prompt scaffold on D | candidate-only 0.8167 @ 139 ms vs JSON 0.7375 @ 442 ms (+7.9 pp, 3.2x faster) | round 159 |
| Assent gating (offline) | auto requires the strongest member's choice: trio auto 199 @ 1.0000, decided 0.9833 | round 159 |
| Tiered composition, first win | trio+B+assent 0.9833 > pair 0.9750, at 40 more auto rows, all correct | round 159 |
| Next | implement --assent-checkpoint in rootd (Task 005, dispatched) + live three-tier run | round 159 |
| Assent gating shipped | Coordinator/CLI/status + collapsed lookup + 3-stem test; cherry-picked to main, 22/22 smoke on both machines | round 159 |
| Next (dispatched) | Task 006: stemd --backend openai (B-prompt, LAN API); then live three-tier run with --assent-checkpoint ft-v3 | round 159 |
| OpenAI backend shipped | stemd --backend openai (B-prompt, LAN API), parse tests; cherry-picked to main, 23/23 smoke both machines | round 160 |
| Live three-tier, run 1 | auto 195 @ 1.0000, decided 0.9833 (core 0.9750), p50 182 ms | round 160 |
| Live three-tier, replicated | run 2 identical to the digit (195 @ 1.0000, 0.9833, p50 181 ms, 5.60 rows/s) | round 160 |
| Prediction matched live | offline 0.9833 predicted (round 159) and delivered; auto 195 vs 199 explained by max_tokens 16 vs 64 | round 160 |
| Split E built | 240 rows, zero cue overlap A/A2/B/C/D (Mac, branch split-e a6eb045) | round 161 |
| Solo on E | english 0.8375, ft-v3 0.9917, mac-llm 0.9167 | round 161 |
| Assent on fresh E | live auto 227 @ 1.0000, decided 0.9583 (advisor fallback), p50 181 ms; offline prediction matched | round 161 |
| Fallback flips by split | D: advisor 0.9833 > veto 0.9417; E: veto 0.9917 > advisor 0.9583; near-tied over 480 rows | round 161 |
| Next (dispatched) | Task 008: --assent-fallback advisor|veto as a declared policy slot + tests | round 161 |
| Fallback slot landed | --assent-fallback advisor|veto (rootd + CLI + /status + veto field), tests 23/23 | round 161 |
| Veto live on E | auto 227 @ 1.0000, decided 0.9750 (advisor live 0.9583; offline veto 0.9917) | round 161 |
| assent=auto landed | assets/registry/accuracies.json + runtime resolution among present voters; /status assent_resolved; 24/24 smoke | round 161 |
| auto live on E | registry resolved to ft-v3; auto 227 @ 1.0000, decided 0.9750 - identical to explicit ft-v3 | round 161 |
| Second LLM tier deployed | Qwen3-8B Q4_K_M on Mac :8082, ~25.5 tok/s; 4B+8B coexist | round 162 |
| 8B solo | D 0.8583 / E 0.9250 (4B: 0.8167 / 0.9167) | round 162 |
| Strength axis | stronger weak-tier buys coverage: E auto 233 @ 1.0000 (core 199; 4B 227); decided set by the fallback | round 162 |
| Precision invariant | every tier configuration on both splits: auto accuracy 1.0000 under assent | round 162 |
| Weak-tier stacking dilutes | 4B+8B together automate fewer rows than 8B alone (221 vs 233 on E) | round 162 |
| Registry updated | mac-llm-8b entry added; assent seat still ft-v3 (0.967 mean) | round 162 |
| Live four-tier (E) | auto 221 @ 1.0000 (matches offline), decided 0.9792, p50 435 ms, 2.25 rows/s | round 162 |
| 8B live role | coverage, not accuracy; slower than three-tier (slowest tier sets the pace) | round 162 |
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
| Termux/TV box real test | needs Android hardware | 0.5 day on device |
| Docker image build test | needs Docker environment | 0.5 day |
| RLCD scale-up beyond one contrast pair | 720 labelled rows now exist, but its descendant may stay a copy (round 119) | 1-2 days |
| Multi-hop with graph adaptation | origin -> B(NPU) -> C(CPU) format conversion | 1 day |
| Dashboard live update mode | design done, not coded | 0.5 day |
| Per-deployment recalibration | possible, but must re-tune `auto_min_probability` in the same edit (round 148) | 0.5 day |

## How to resume

1. Read RESEARCH-SUMMARY.md for the definitive findings
2. Run `python tools/demo.py` to verify everything still works
3. Pick any item from "What is NOT yet done" above
4. All code, docs, and research logs are in this repo and on GitHub

## Repository

https://github.com/JewelRoam/PandoSI (main branch, ~30 commits)
