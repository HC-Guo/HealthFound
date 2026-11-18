#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}/LLaMA-Factory}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${REPO_DIR}/src/train.py}"

: "${CONFIG_PATH:?Set CONFIG_PATH to the SFT training config.}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"

export HF_HOME="${HF_HOME:-${CACHE_DIR:-}}"
if [[ -n "${HF_HOME}" ]]; then
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
  export HF_METRICS_CACHE="${HF_METRICS_CACHE:-${HF_HOME}/metrics}"
  mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${HF_METRICS_CACHE}"
fi

if [[ -n "${TMPDIR:-}" ]]; then
  mkdir -p "${TMPDIR}"
fi

export TRANSFORMERS_NO_ADVISORY_LOCKING="${TRANSFORMERS_NO_ADVISORY_LOCKING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

run_command=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node="${GPUS_PER_NODE}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  "${TRAIN_ENTRYPOINT}"
  "${CONFIG_PATH}"
)

if [[ -n "${LOG_PATH:-}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  "${run_command[@]}" 2>&1 | tee "${LOG_PATH}"
else
  "${run_command[@]}"
fi
