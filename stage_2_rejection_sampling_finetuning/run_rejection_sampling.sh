#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Stage-1 model or checkpoint.}"
: "${INPUT_JSONL:?Set INPUT_JSONL to the candidate input JSONL.}"
: "${OUTPUT_JSONL:?Set OUTPUT_JSONL to the accepted output JSONL.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

run_command=(
  python
  "${FILTER_SCRIPT:-${SCRIPT_DIR}/rejection_sampling_filter.py}"
  --input "${INPUT_JSONL}"
  --output "${OUTPUT_JSONL}"
  --model "${MODEL_PATH}"
  --n "${NUM_GENERATIONS}"
  --temperature "${TEMPERATURE:-0.7}"
  --top_p "${TOP_P:-0.9}"
  --max_tokens "${MAX_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --diversity_threshold "${DIVERSITY_THRESHOLD:-0.8}"
)

if [[ -n "${MAX_CORRECT_PER_SAMPLE:-}" ]]; then
  run_command+=(--max_correct_per_sample "${MAX_CORRECT_PER_SAMPLE}")
fi
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  run_command+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-}" == "true" ]]; then
  run_command+=(--overwrite)
fi

"${run_command[@]}"
