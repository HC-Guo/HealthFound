import math

import torch

RETENTION_VERSION = "healthfound-hfpo-retention"


def retention_loss(response_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:

    if response_log_probs.ndim != 2 or response_log_probs.shape != response_mask.shape:
        raise ValueError("Retention log probabilities and mask must have equal [batch, response] shapes")
    mask = response_mask.bool()
    lengths = mask.sum(dim=-1)
    if lengths.numel() == 0 or torch.any(lengths == 0):
        raise ValueError("Every anchor example must contain at least one supervised response token")
    log_probs = torch.where(mask, response_log_probs.float(), 0.0)
    return (-log_probs.sum(dim=-1) / lengths).mean()


def validate_retention_config(config) -> bool:

    actor = config.actor_rollout_ref.actor
    coefficient = float(actor.get("retention_loss_coef", 0.0))
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("actor.retention_loss_coef must be finite and nonnegative")
    if coefficient == 0:
        return False
    if actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("HFPO supervised retention is implemented for the FSDP/FSDP2 actor path")
    data = config.data.get("retention", {})
    files = data.get("train_files")
    if not files or isinstance(files, str) and not files.strip():
        raise ValueError("Positive retention_loss_coef requires data.retention.train_files")
    micro_batch_size = actor.get("retention_micro_batch_size_per_gpu", 1)
    batch_size = data.get("batch_size", 0)
    if not isinstance(micro_batch_size, int) or micro_batch_size < 1:
        raise ValueError("retention_micro_batch_size_per_gpu must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("data.retention.batch_size must be a positive global batch size per optimizer update")
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    sp_size = actor.get("ulysses_sequence_parallel_size", 1)
    if sp_size < 1 or world_size % sp_size:
        raise ValueError("Actor world size must be divisible by Ulysses sequence parallel size")
    dp_size = world_size // sp_size
    if batch_size % dp_size:
        raise ValueError(f"Retention global batch_size ({batch_size}) must be divisible by actor DP size ({dp_size})")
    global_mini_batch_size = actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n
    if global_mini_batch_size < 1 or global_mini_batch_size % dp_size:
        raise ValueError("Global PPO minibatch size must be positive and divisible by actor DP size for HFPO")
    if not isinstance(actor.ppo_epochs, int) or actor.ppo_epochs < 1:
        raise ValueError("HFPO requires a positive integer ppo_epochs")
    for key in ("max_prompt_length", "max_response_length"):
        value = data.get(key, 0)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"data.retention.{key} must be a positive integer")
    return True


def optimizer_updates_for_batch(batch_size: int, actor_config, rollout_n: int) -> int:
    mini_batch_size = actor_config.ppo_mini_batch_size * rollout_n
    if batch_size < 1 or mini_batch_size < 1 or batch_size % mini_batch_size:
        raise ValueError("With HFPO retention, the rollout batch must contain whole global PPO minibatches")
    return batch_size // mini_batch_size * actor_config.ppo_epochs
