# Recorded votes

This directory contains the vote files that were still present on the machine when Task 017 consolidated them. Files are byte-for-byte copies of the recorded artifacts; no choice, probability, or expected label was rewritten. The companion `provenance.json` is machine-readable and is also the source of truth for the reproducibility tool.

## Included files

| file | split | member | rows | accuracy | provenance |
|---|---:|---|---:|---:|---|
| `votes-d-seated-english.json` | D | `english` | 240 | 0.7292 | rounds 122-160 seated english lineage votes (canonical file used by the admission and calibration rounds); sha256 `66a88241ee31…` |
| `votes-d-ft-v3.json` | D | `ft-v3` | 240 | 0.9417 | torch/CUDA fine-tune (round 150-153) on split D; tools/finetuned_votes.py; sha256 `74b07f936aec…` |
| `e-votes-english.json` | E | `english` | 240 | 0.8375 | recorded by metrics/lan-check/record_english.py (static ONNX graph, iGPU, pad-to 256) for round 161; sha256 `f90a0b884f8a…` |
| `e-votes-ft-v3.json` | E | `ft-v3` | 240 | 0.9917 | recorded by tools/finetuned_votes.py (torch/CUDA) for round 161; sha256 `eb70044200cc…` |
| `f-votes-english.json` | F | `english` | 120 | 0.8417 | recorded by record_english.py for round 164 (new intent family); sha256 `c5e9ce968899…` |
| `f-votes-ft-v3.json` | F | `ft-v3` | 120 | 0.8667 | recorded by tools/finetuned_votes.py for round 164; sha256 `b1aba5ab67ad…` |
| `finetuned-votes.json` | C | `ft-v3` | 240 | 0.9125 | round 150 verification of the fine-tune on split C; tools/finetuned_votes.py; sha256 `93650577a091…` |
| `llm-votes-e-8b-think.json` | E | `mac-llm-8b-think` | 240 | 0.9250 | round 167 / Task 018; branch `mac-llm-8b-think` @ `34de70b`; sha256 `b5fee99f2ff6…` |
| `llm-votes-f-8b-think.json` | F | `mac-llm-8b-think` | 120 | 0.9750 | round 167 / Task 018; branch `mac-llm-8b-think` @ `34de70b`; sha256 `83a495a53115…` |
| `llm-votes-b.json` | B | `mac-llm` | 240 | 0.8500 | round 158 / Task 003; branch `mac-llm-votes` @ `8242803`; `qwen3-4b-instruct-2507`; sha256 `f5f9b4198acc…` |
| `llm-votes-c.json` | C | `mac-llm` | 240 | 0.6542 | round 158 / Task 003; branch `mac-llm-votes` @ `8242803`; `qwen3-4b-instruct-2507`; sha256 `af28361422d1…` |
| `llm-votes-d-b-only.json` | D | `mac-llm` | 240 | 0.8167 | round 159 / Task 004; branch `mac-llm-votes-3` @ `0484840`; `qwen3-4b-instruct-2507`; sha256 `96a167eb09a8…` |
| `llm-votes-d.json` | D | `mac-llm-a` | 240 | 0.7375 | round 158 / Task 003; branch `mac-llm-votes` @ `8242803`; `qwen3-4b-instruct-2507`; sha256 `bcdba53e5e42…` |
| `llm-votes-d-8b.json` | D | `mac-llm-8b` | 240 | 0.8583 | round 162 / Task 010; branch `mac-llm-8b` @ `26a2b7d`; `qwen3-8b-instruct`; sha256 `29e855c72bc3…` |
| `llm-votes-e.json` | E | `mac-llm` | 240 | 0.9167 | round 161 / Task 008; branch `split-e` @ `a6eb045`; `qwen3-4b-instruct-2507`; sha256 `f6ba53c21ae1…` |
| `llm-votes-e-8b.json` | E | `mac-llm-8b` | 240 | 0.9250 | round 162 / Task 010; branch `mac-llm-8b` @ `26a2b7d`; `qwen3-8b-instruct`; sha256 `66da8b0c4d3a…` |
| `llm-votes-f-4b.json` | F | `mac-llm` | 120 | 0.9333 | round 164 / Task 013; branch `split-f` @ `aef5f01`; `qwen3-4b-instruct-2507`; sha256 `a271baf3272d…` |
| `llm-votes-f-8b.json` | F | `mac-llm-8b` | 120 | 0.9833 | round 164 / Task 013; branch `split-f` @ `aef5f01`; `qwen3-8b-instruct`; sha256 `237c124d5ba9…` |

Accuracy is the fraction of rows where `choice == expected`. The source paths in `provenance.json` record the checkout used for the copy; they are not required at runtime.

## Missing source files

None. The seven Windows-side files (english/ft-v3 on D, E, F; the round-150 C verification) landed in the round-166 update with provenance entries above, and every headline row now reproduces offline.

## Headline definitions

`tools/reproduce_headlines.py` uses the same `pandosi.common.aggregate` gate and `assets/combiner.json` advisor artifact as the research tools. `assent` demotes an auto row unless the trusted member agrees with the aggregate; the demoted row is then answered by the advisor or by the witness, depending on the declared fallback. `veto` means the witness answers every row the gate does not auto-execute (round 162/164/165 semantics).

