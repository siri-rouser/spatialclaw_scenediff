#!/usr/bin/env bash
# Start a vLLM server on this machine (no SLURM) and register it in
# spatial_agent/logs/serve.json so agents with "llm_base_url": "vllm" find it.
#
# Usage (inside the container, from anywhere):
#   bash spatial_agent/scripts/vllm/serve_local.sh [hf_model] [served_name] [tp] [max_model_len]
# Defaults: Qwen/Qwen3.6-27B qwen3.6-27b 2 262144
#
# The launcher stays in the foreground and logs to spatial_agent/logs/serve_<uid>.log;
# Ctrl-C stops vLLM and removes its registry entry.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL="${1:-Qwen/Qwen3.6-27B}"
SERVED_NAME="${2:-qwen3.6-27b}"
TP="${3:-2}"
MAX_MODEL_LEN="${4:-262144}"

# FlashInfer's JIT build links against $CUDA_HOME/lib64. The system toolkit in the
# image has that layout; the spatialclaw-cuda conda env (lib/ only) does not.
export CUDA_HOME=/usr/local/cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TORCH_LIB=$(dirname "$(find .venv/lib -name libtorch_cuda.so -print -quit)")
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PATH="$PWD/.venv/bin:$PATH"

exec .venv/bin/python -m spatial_agent.entrypoints.launch_vllm \
    --model "$MODEL" \
    --served_model_name "$SERVED_NAME" \
    --tp "$TP" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_seqs 4 \
    --quantization fp8
