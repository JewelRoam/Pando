# Task 025 — teacher2 (Qwen3-8B Q5_K_M) deployment record

Second, stronger teacher deployed on the Mac node alongside the existing
4B (`:8080`) and 8B Q4_K_M (`:8082`) services.  No existing service was
touched.

## Artifact

| field | value |
| --- | --- |
| model | `Qwen3-8B-Q5_K_M.gguf` |
| path | `~/.pandosi/models/Qwen3-8B-GGUF/Qwen3-8B-Q5_K_M.gguf` |
| bytes | `5851112224` (5.45 GiB) |
| sha256 | `068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738` |
| source | ModelScope mirror `Qwen/Qwen3-8B-GGUF` (HF direct was reset) |
| quant | Q5_K_M (`ftype: Q5_K - Medium`, `n_params: 8190735360`) |
| alias | `qwen3-8b-q5km-instruct` |

## Serving

* `llama-server` (llama.cpp 0.6.0, build 11429) on `127.0.0.1:8086`,
  api-key-file `~/.pandosi/llm-api-8b-q6.token`
  (`--jinja --chat-template-kwargs '{"enable_thinking": false}' -ngl 99
  -c 4096 --parallel 1 --temp 0 --metrics`)
* TCP proxy `0.0.0.0:8087` -> `127.0.0.1:8086`
* cloudflared quick tunnel `com.pandosi.cloudflared.llm8bq6`, kept alive by
  launchd; public URL published (without token) as `llm_teacher2_url` in
  `endpoint/mac.json` on branch `mac-endpoint`.

## H held-out exam — `assets/heldout6_test.jsonl` (240 rows)

Run with `tools/llm_decision_check.py --variant b` (candidate-only prompt,
no confidence), `temperature 0.0`, `max_tokens 64`.

| metric | teacher2 (8B Q5_K_M) | 8B Q4_K_M (task 024) | 4B (task 024) |
| --- | --- | --- | --- |
| accuracy | **0.9417** (226/240) | 0.9333 (224/240) | 0.9208 (221/240) |
| avg latency | 268.6 ms | 385.1 ms | 308.1 ms |
| p50 | 281.2 ms | 379.9 ms | — |
| p95 | 293.7 ms | 553.1 ms | — |
| tokens/s | 20.4 | — | — |
| parse errors | 0 | 0 | — |
| regex fallbacks | 240/240 | — | — |

Votes: `metrics/llm-votes-h-teacher2.json` (record has no confidence, so
every row carries `probability = 0.5` by construction).

Resident memory: `llama-server` RSS ~492 MB after the run (~285 MB at load);
weights live in unified memory / Metal buffers, so RSS understates the
footprint.  The host is a 16 GB Apple Silicon machine that was already
swapping heavily (~24/25.6 GB) before this model was added; the three
resident GGUFs together are the reason, and no OOM occurred during the exam.

## Reproduce the exam

```sh
cd <worktree>
.venv/bin/python tools/llm_decision_check.py \
  --suite assets/heldout6_test.jsonl \
  --out metrics/llm-votes-h-teacher2.json \
  --base-url http://127.0.0.1:8087/v1 \
  --api-key-file ~/.pandosi/llm-api-8b-q6.token \
  --model qwen3-8b-q5km-instruct \
  --variant b
```
