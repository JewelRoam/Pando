# Recorded votes

This directory contains the vote files that were still present on the machine when Task 017 consolidated them. Files are byte-for-byte copies of the recorded artifacts; no choice, probability, or expected label was rewritten. The companion `provenance.json` is machine-readable and is also the source of truth for the reproducibility tool.

## Included files

| file | split | member | rows | accuracy | provenance |
|---|---:|---|---:|---:|---|
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

The following files are referenced by `assets/registry/accuracies.json` or the research record but were not present in any local or remote branch searched by this task. They are deliberately not synthesized: a placeholder vote would make a headline reproduction look stronger than the evidence allows.

| expected file | member | split | documented source |
|---|---:|---:|---|
| `votes-d-seated-english.json` | `english` | D | assets/registry/accuracies.json |
| `votes-d-ft-v3.json` | `ft-v3` | D | assets/registry/accuracies.json |
| `e-votes-english.json` | `english` | E | assets/registry/accuracies.json |
| `e-votes-ft-v3.json` | `ft-v3` | E | assets/registry/accuracies.json |
| `f-votes-english.json` | `english` | F | assets/registry/accuracies.json |
| `f-votes-ft-v3.json` | `ft-v3` | F | assets/registry/accuracies.json |
| `finetuned-votes.json` | `ft-v3` | C | docs/research/2026-10-16-round150.md |

Consequences: the solo accuracy rows for `english` and `ft-v3`, and the core-dependent composition rows on D/E/F, cannot be recomputed from this directory alone. The scripts report those rows as `MISSING` rather than inventing a vote. The available LLM solo rows and the F 8B witness-alone diagnostic are recomputed.

## Headline definitions

`tools/reproduce_headlines.py` uses the same `pandosi.common.aggregate` gate and `assets/combiner.json` advisor artifact as the research tools. `assent` demotes an auto row unless the trusted member agrees with the aggregate; the demoted row is then answered by the advisor or by the witness, depending on the declared fallback. `veto` means the witness answers every row the gate does not auto-execute (round 162/164/165 semantics).

