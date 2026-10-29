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
# Overridables: PANDOSI_PI_LLAMA, PANDOSI_PI_MODEL, PANDOSI_PI_PORT, PANDOSI_ROOT,
#               PANDOSI_PAIRING_TOKEN_FILE, PANDOSI_NODE_ID, PANDOSI_PY, PANDOSI_CHECKPOINT
set -e

LLAMA_DIR="${PANDOSI_PI_LLAMA:-$HOME/llama/llama-b11700}"
MODEL="${PANDOSI_PI_MODEL:-$HOME/models/Qwen3-1.7B-Q4_K_M.gguf}"
PORT="${PANDOSI_PI_PORT:-8085}"
ROOT="${PANDOSI_ROOT:-http://192.168.31.100:8790}"
TOKEN_FILE="${PANDOSI_PAIRING_TOKEN_FILE:-$HOME/.pando/pairing.token}"
NODE_ID="${PANDOSI_NODE_ID:-$(hostname)-llm}"
VENV_PY="${PANDOSI_PY:-$HOME/pando/bin/python}"
CHECKPOINT="${PANDOSI_CHECKPOINT:-qwen3-1.7b-q4km}"

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
