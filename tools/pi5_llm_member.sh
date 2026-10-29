#!/bin/bash
# Pi5 LLM member: serve Qwen3-1.7B Q4_K_M with llama.cpp (CPU, OpenAI-compatible)
# and register a signed worker member against the fabric root.
#
# Instructions live here and in docs/onboarding-raspberry-pi.md sections 7-8;
# bytes (binaries, weights) stay out-of-band (direct transfer / NAS / SD card).
# Acquire them as documented:
#   - llama.cpp release: llama-b<NNNNN>-bin-ubuntu-arm64.tar.gz  -> ~/llama/llama-b<NNNNN>/
#   - weights: ModelScope unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf -> ~/models/
#
# Usage (on the pi5, as the member's user):
#   bash tools/pi5_llm_member.sh serve     # llama-server (foreground; wrap in setsid/syworker)
#   bash tools/pi5_llm_member.sh stem      # the fabric member (foreground; needs a running serve)
#
# Bring-up overrides (one-shot launch knobs; persistent wiring belongs in
# the host config): PANDO_PI_LLAMA, PANDO_PI_MODEL, PANDO_PI_PORT,
# PANDO_PAIRING_TOKEN_FILE, PANDO_NODE_ID, PANDO_PY, PANDO_CHECKPOINT.
# PANDO_ROOT is REQUIRED for `stem`: the repo names no fleet address.
set -e

LLAMA_DIR="${PANDO_PI_LLAMA:-$HOME/llama/llama-b11700}"
MODEL="${PANDO_PI_MODEL:-$HOME/models/Qwen3-1.7B-Q4_K_M.gguf}"
PORT="${PANDO_PI_PORT:-8085}"
ROOT="${PANDO_ROOT:-}"
TOKEN_FILE="${PANDO_PAIRING_TOKEN_FILE:-$HOME/.pando/pairing.token}"
NODE_ID="${PANDO_NODE_ID:-$(hostname)-llm}"
VENV_PY="${PANDO_PY:-$HOME/pando/bin/python}"
CHECKPOINT="${PANDO_CHECKPOINT:-qwen3-1.7b-q4km}"

case "${1:-}" in
  serve)
    export LD_LIBRARY_PATH="$LLAMA_DIR"
    # Note: --chat-template-kwargs is deprecated upstream in favor of
    # --reasoning off; kept until the pinned build is updated.
    exec "$LLAMA_DIR/llama-server" -m "$MODEL" \
      --host 127.0.0.1 --port "$PORT" -c 2048 -np 1 --cache-ram 0 -t 4 --jinja \
      --chat-template-kwargs '{"enable_thinking": false}'
    ;;
  stem)
    if [ -z "$ROOT" ]; then
      echo "PANDO_ROOT is required (the fabric root URL); the repo carries" >&2
      echo "no fleet address - yours is in this host's own notes." >&2
      exit 2
    fi
    # An empty *string argument* disables auth; an empty *file* is rejected.
    exec "$VENV_PY" -m pando.worker --node-id "$NODE_ID" --backend openai \
      --openai-url "http://127.0.0.1:$PORT/v1" --openai-model qwen3-1.7b \
      --openai-key-file '' --checkpoint "$CHECKPOINT" --root "$ROOT" \
      --pairing-token-file "$TOKEN_FILE" --host 0.0.0.0 \
      --advertise-host "$(hostname -I | awk '{print $1}')"
    ;;
  *)
    echo "usage: $0 [serve|stem]" >&2
    exit 2
    ;;
esac
