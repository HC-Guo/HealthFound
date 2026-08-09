#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}/verl}"
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${SCRIPT_DIR}/reward.py}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"

: "${MODEL_PATH:?Set MODEL_PATH to the Stage-2 model or checkpoint.}"
: "${TRAIN_FILES_JSON:?Set TRAIN_FILES_JSON to the training file list string.}"
: "${VAL_FILES_JSON:?Set VAL_FILES_JSON to the validation file list string.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint output directory.}"
: "${PROJECT_NAME:?Set PROJECT_NAME to a public-safe project name.}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to a public-safe experiment name.}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
WORLD_SIZE="${WORLD_SIZE:-1}"
NNODES="${NNODES:-${WORLD_SIZE}}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
  sleep 3
}

wait_for_head() {
  local max_attempts="${RAY_HEAD_MAX_ATTEMPTS:-60}"
  local attempt=0
  while [[ "${attempt}" -lt "${max_attempts}" ]]; do
    if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; then
      return 0
    fi
    attempt=$((attempt + 1))
    echo "Waiting for Ray head node (${attempt}/${max_attempts})"
    sleep 5
  done
  echo "Timed out waiting for Ray head node."
  return 1
}

if [[ "${RANK}" == "0" ]]; then
  cleanup_ray
  ray start --head \
    --port="${MASTER_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${NPROC_PER_NODE}" \
    --disable-usage-stats

  if [[ "${WORLD_SIZE}" -gt 1 ]]; then
    sleep "${WORKER_JOIN_WAIT_SECONDS:-180}"
    ray status
  fi

  cd "${VERL_REPO_DIR}"
  run_command=(
    python3
    -m
    verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}"
    reward.custom_reward_function.name="${REWARD_FUNCTION_NAME}"
    data.train_files="${TRAIN_FILES_JSON}"
    data.val_files="${VAL_FILES_JSON}"
    data.train_batch_size="${TRAIN_BATCH_SIZE:-256}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}"
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-True}"
    data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS:-32}"
    data.truncation="${TRUNCATION_MODE:-error}"
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-5e-6}"
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING:-True}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-True}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}"
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE:-bfloat16}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-2}"
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND:-vllm}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-16}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-False}"
    actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE:-bfloat16}"
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
    trainer.critic_warmup="${CRITIC_WARMUP:-0}"
    trainer.logger="${TRAINER_LOGGER:-[\"console\"]}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ:-40}"
    trainer.test_freq="${TEST_FREQ:-20}"
    trainer.total_epochs="${TOTAL_EPOCHS:-1}"
  )

  if [[ -n "${LOG_PATH:-}" ]]; then
    mkdir -p "$(dirname "${LOG_PATH}")"
    "${run_command[@]}" 2>&1 | tee "${LOG_PATH}"
  else
    "${run_command[@]}"
  fi
else
  cleanup_ray
  wait_for_head
  sleep "${WORKER_START_DELAY_SECONDS:-20}"
  ray start \
    --address="${MASTER_ADDR}:${MASTER_PORT}" \
    --num-gpus="${NPROC_PER_NODE}" \
    --disable-usage-stats
  ray status --address="${MASTER_ADDR}:${MASTER_PORT}"
  sleep infinity
fi
