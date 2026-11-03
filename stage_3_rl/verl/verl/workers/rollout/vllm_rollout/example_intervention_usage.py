"""
Example usage of the intervention-based vLLM rollout for GRPO training.
"""

import torch
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout import vLLMInterventionRollout
from transformers import AutoTokenizer


def main():
    # Load configuration (can be from YAML file)
    config = {
        "prompt_length": 512,
        "response_length": 512,
        "max_model_len": 1024,
        "tensor_model_parallel_size": 1,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,
        "disable_log_stats": True,
        "calculate_log_probs": True,
        "dtype": "bfloat16",
        "load_format": "auto",
        "free_cache_engine": False,

        # Enable intervention
        "enable_intervention": True,

        # Intervention configuration
        "intervention": {
            "n_base_trajectories": 4,
            "n_interventions_per_base": 3,
            "base_temperature": 1.2,
            "intervention_temperature": 1.0,
            "min_intervention_position": 0.3,
            "max_intervention_position": 0.8,
            "intervention_phrases": [
                # Using a subset of phrases for this example
                "Let me try to break this down step by step: ",
                "Let me try to think differently: ",
                "Wait, no, that's not correct: ",
                "Let me think about how to approach this: ",
                "Actually, let me think about this more: ",
                "So, putting it all together: ",
                "Let me double-check this: ",
                "Wait, but I'm not sure: ",
                "Let me reconsider the: ",
            ]
        },

        # Default sampling parameters
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }

    rollout_config = RolloutConfig(**config)

    # Model configuration
    model_path = "path/to/your/model"  # Replace with actual model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_config = HFModelConfig(
        local_path=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
    )

    # Device mesh for distributed training
    device_mesh = DeviceMesh("cuda", torch.arange(1).tolist())

    # Initialize the intervention rollout
    print("Initializing vLLMInterventionRollout...")
    rollout = vLLMInterventionRollout(
        config=rollout_config,
        model_config=model_config,
        device_mesh=device_mesh
    )

    # Example batch preparation
    batch_size = 8
    prompt_length = 512

    # Create example prompts (in practice, these come from your dataset)
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size, prompt_length),
        dtype=torch.long
    )

    # Add padding tokens at the beginning for left-padding
    pad_length = 10
    input_ids[:, :pad_length] = tokenizer.pad_token_id

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, :pad_length] = 0

    # Create position ids
    position_ids = torch.arange(prompt_length).unsqueeze(0).expand(batch_size, -1)

    # Create DataProto
    prompts = DataProto(
        batch={
            "input_ids": input_ids.cuda(),
            "attention_mask": attention_mask.cuda(),
            "position_ids": position_ids.cuda(),
        },
        non_tensor_batch={},
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "validate": False,
        }
    )

    print(f"Running two-stage intervention rollout for batch_size={batch_size}...")
    print(f"Stage 1: Generating {rollout.intervention_config.n_base_trajectories} base trajectories")
    print(f"Stage 2: Generating {rollout.intervention_config.n_base_trajectories * rollout.intervention_config.n_interventions_per_base} intervention trajectories")

    # Generate sequences with intervention
    outputs = rollout.generate_sequences(prompts)

    print(f"\nRollout completed!")
    print(f"Total trajectories generated: {outputs.batch['responses'].shape[0]}")
    print(f"Response shape: {outputs.batch['responses'].shape}")

    # Access intervention metadata
    if "intervention_metadata" in outputs.non_tensor_batch:
        metadata = outputs.non_tensor_batch["intervention_metadata"]
        print(f"\nIntervention metadata:")
        print(f"  - Base trajectories: {metadata['n_base_trajectories']}")
        print(f"  - Interventions per base: {metadata['n_interventions_per_base']}")
        print(f"  - Total rollouts: {metadata['total_rollouts']}")
        print(f"  - Batch size: {metadata['batch_size']}")

        # Example: Process trajectories by type
        responses = outputs.batch['responses']
        batch_size = metadata['batch_size']
        n_base = metadata['n_base_trajectories']
        n_interventions = metadata['n_interventions_per_base']

        for sample_idx in range(batch_size):
            sample_start = sample_idx * metadata['total_rollouts']

            # Base trajectories for this sample
            base_start = sample_start
            base_end = sample_start + n_base
            base_responses = responses[base_start:base_end]
            print(f"\nSample {sample_idx}: {n_base} base trajectories")

            # Intervention trajectories for this sample
            intervention_start = base_end
            intervention_end = sample_start + metadata['total_rollouts']
            intervention_responses = responses[intervention_start:intervention_end]
            print(f"Sample {sample_idx}: {n_base * n_interventions} intervention trajectories")

    return outputs


if __name__ == "__main__":
    # Note: This example assumes vLLM and model are properly set up
    # In actual usage, you would integrate this with your GRPO training loop
    main()