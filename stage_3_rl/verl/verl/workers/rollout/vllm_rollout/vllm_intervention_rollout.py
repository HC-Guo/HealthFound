"""
vLLM Rollout with intervention-based two-stage generation for improved diversity.
"""

import os
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
import pdb
from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl.workers.rollout.vllm_rollout.intervention_sampler import InterventionConfig, InterventionSampler
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
# Import reward scoring functions for base trajectory evaluation (using math_dapo rules)
from verl.utils.reward_score.math_dapo import compute_score as dapo_compute_score, verify as dapo_verify

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMInterventionRollout(vLLMRollout):
    """
    vLLM Rollout with intervention-based two-stage generation.

    This class implements a two-stage rollout process:
    1. Stage 1: Generate 4 base trajectories with higher temperature
    2. Stage 2: For each base trajectory, generate 3 intervention variants
               by inserting trigger phrases at random positions
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # Store tokenizer reference
        self.tokenizer = model_config.tokenizer

        # Initialize intervention configuration
        intervention_config = config.get("intervention", {})
        self.intervention_config = InterventionConfig(
            n_base_trajectories=intervention_config.get("n_base_trajectories", 4),
            n_interventions_per_base=intervention_config.get("n_interventions_per_base", 3),
            base_temperature=intervention_config.get("base_temperature", 1.2),
            intervention_temperature=intervention_config.get("intervention_temperature", 1.0),
            intervention_phrases=intervention_config.get("intervention_phrases", None),
            min_intervention_position=intervention_config.get("min_intervention_position", 0.3),
            max_intervention_position=intervention_config.get("max_intervention_position", 0.8),
        )

        self.intervention_sampler = InterventionSampler(
            config=self.intervention_config,
            tokenizer=self.tokenizer
        )

        self.enable_intervention = config.get("enable_intervention", True)
        logger.info(f"vLLMInterventionRollout initialized with config: {self.intervention_config}")

    def _prepare_vllm_inputs(self, prompt_ids: List[List[int]], multi_modal_data: Optional[List] = None) -> List[Dict]:
        """Prepare inputs for vLLM generation."""
        if multi_modal_data:
            vllm_inputs = []
            for raw_prompt_ids, mm_data in zip(prompt_ids, multi_modal_data, strict=True):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": mm_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in prompt_ids]

        # Ensure token IDs are lists
        for input_data in vllm_inputs:
            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        return vllm_inputs

    def _evaluate_base_trajectories(
        self,
        base_responses: List[List[List[int]]],
        prompts: DataProto
    ) -> List[List[Dict[str, Any]]]:
        """
        Evaluate base trajectories to determine correctness.

        Args:
            base_responses: List of base responses per sample, shape [batch_size][n_base_trajectories][response_tokens]
            prompts: Original prompts containing ground_truth in non_tensor_batch["reward_model"]

        Returns:
            List of evaluation results per sample, each containing:
                - is_correct: bool
                - response_str: decoded response string
                - pred: extracted prediction (if available)
        """
        batch_size = len(base_responses)
        evaluation_results = []

        # Get ground_truth from prompts if available
        reward_model_data = prompts.non_tensor_batch.get("reward_model", None)

        # Calculate the repeat factor (n) to correctly index into reward_model_data
        # reward_model_data is pre-repeated with shape [batch_size * n]
        # We need to map sample_idx to the correct index
        expected_repeats = self.config.n
        original_batch_size = prompts.batch["input_ids"].size(0)
        if original_batch_size % expected_repeats == 0 and expected_repeats > 1:
            # Data was pre-repeated, so reward_model_data[i*n] corresponds to unique sample i
            repeat_stride = expected_repeats
        else:
            repeat_stride = 1

        for sample_idx in range(batch_size):
            sample_results = []
            sample_base_responses = base_responses[sample_idx]

            # Get ground_truth for this sample
            # Use repeat_stride to get the correct index in the pre-repeated data
            ground_truth = None
            reward_data_idx = sample_idx * repeat_stride
            if reward_model_data is not None:
                # Handle different data structures
                if isinstance(reward_model_data, (list, np.ndarray)):
                    if reward_data_idx < len(reward_model_data):
                        rm_item = reward_model_data[reward_data_idx]
                        if isinstance(rm_item, dict):
                            ground_truth = rm_item.get("ground_truth", None)
                elif isinstance(reward_model_data, dict):
                    gt_list = reward_model_data.get("ground_truth", None)
                    if gt_list is not None and reward_data_idx < len(gt_list):
                        ground_truth = gt_list[reward_data_idx]

            for base_idx, response_tokens in enumerate(sample_base_responses):
                # Decode response tokens to string
                response_str = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

                # Evaluate correctness using math_dapo rules
                is_correct = False
                pred = None

                if ground_truth is not None:
                    try:
                        # Use dapo_verify which returns (correct: bool, pred: str)
                        is_correct, pred = dapo_verify(response_str, str(ground_truth))
                    except Exception as e:
                        logger.warning(f"Error evaluating base trajectory: {e}")
                        is_correct = False
                        pred = None

                sample_results.append({
                    "is_correct": is_correct,
                    "response_str": response_str,
                    "pred": pred,
                    "ground_truth": ground_truth,
                })

            evaluation_results.append(sample_results)

        # Log evaluation statistics
        total_correct = sum(
            sum(1 for r in sample_results if r["is_correct"])
            for sample_results in evaluation_results
        )
        total_trajectories = sum(len(sample_results) for sample_results in evaluation_results)
        logger.info(f"Base trajectory evaluation: {total_correct}/{total_trajectories} correct "
                   f"({100*total_correct/max(1,total_trajectories):.1f}%)")

        return evaluation_results

    def _generate_base_trajectories(self, prompts: DataProto) -> tuple:
        """
        Stage 1: Generate base trajectories with higher temperature.

        Returns:
            tuple: (base_responses, base_log_probs, vllm_inputs)
        """
        idx = prompts.batch["input_ids"]
        batch_size = idx.size(0)

        # Check if data has been pre-repeated due to n>1
        # If n=16, we expect the batch to contain repeated prompts
        # We need to extract unique prompts for base generation
        expected_repeats = self.config.n
        if batch_size % expected_repeats == 0:
            # Data has been pre-repeated, extract unique prompts
            actual_batch_size = batch_size // expected_repeats
            unique_indices = list(range(0, batch_size, expected_repeats))
            idx = idx[unique_indices]
            batch_size = actual_batch_size
            logger.info(f"Detected pre-repeated data with n={expected_repeats}, using {batch_size} unique prompts")

        # Prepare inputs for base generation
        raw_prompt_ids = []
        for i in range(batch_size):
            prompt_tokens = idx[i]
            # Remove left padding
            non_pad_index = torch.nonzero(prompt_tokens != self.pad_token_id, as_tuple=False)
            if len(non_pad_index) > 0:
                non_pad_index = non_pad_index[0][0]
                token_ids = prompt_tokens[non_pad_index:].tolist()
            else:
                token_ids = prompt_tokens.tolist()
            raw_prompt_ids.append(token_ids)

        # Prepare vllm inputs
        multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data", None)
        vllm_inputs = self._prepare_vllm_inputs(raw_prompt_ids, multi_modal_data)

        # Repeat inputs for multiple base trajectories
        expanded_inputs = []
        for _ in range(self.intervention_config.n_base_trajectories):
            expanded_inputs.extend(vllm_inputs)

        # Generate with higher temperature for diversity
        # 1.2 temperature直接生成乱码，估计是实现有问题
        #
        outputs = self.inference_engine.generate(
            prompts=expanded_inputs,
            sampling_params=self.sampling_params,
            lora_request=None,
            use_tqdm=False,
        )

        # Extract responses and log probs
        base_responses = []
        base_log_probs = []

        for i, output in enumerate(outputs):
            response_ids = output.outputs[0].token_ids
            base_responses.append(response_ids)

            if self.config.calculate_log_probs:
                curr_log_prob = []
                for j, logprob in enumerate(output.outputs[0].logprobs):
                    curr_log_prob.append(logprob[response_ids[j]].logprob)
                base_log_probs.append(curr_log_prob)

        # Reshape to [batch_size][n_base_trajectories][response_length]
        base_responses = [
            base_responses[i::batch_size]
            for i in range(batch_size)
        ]

        if self.config.calculate_log_probs:
            base_log_probs = [
                base_log_probs[i::batch_size]
                for i in range(batch_size)
            ]

        return base_responses, base_log_probs, vllm_inputs

    def _generate_intervention_trajectories(self,
                                           prompts: DataProto,
                                           base_responses: List,
                                           vllm_inputs: List,
                                           base_evaluations: Optional[List[List[Dict[str, Any]]]] = None) -> tuple:
        """
        Stage 2: Generate intervention trajectories by inserting trigger phrases.

        Args:
            prompts: Original prompts
            base_responses: Base trajectory responses
            vllm_inputs: vLLM input configurations
            base_evaluations: Evaluation results for base trajectories (for adaptive intervention)

        Returns:
            tuple: (intervention_responses, intervention_log_probs, intervention_metadata)
        """
        idx = prompts.batch["input_ids"]

        # Handle pre-repeated data
        expected_repeats = self.config.n
        if idx.size(0) % expected_repeats == 0:
            actual_batch_size = idx.size(0) // expected_repeats
            unique_indices = list(range(0, idx.size(0), expected_repeats))
            idx = idx[unique_indices]
            batch_size = actual_batch_size
        else:
            batch_size = idx.size(0)

        all_intervention_inputs = []
        all_intervention_metadata = []

        # Track intervention statistics
        correct_base_count = 0
        wrong_base_count = 0

        # For each sample in the batch
        for sample_idx in range(batch_size):
            sample_base_responses = base_responses[sample_idx]  # n_base trajectories for this sample
            sample_prompt = idx[sample_idx]

            # Get evaluation results for this sample (if available)
            sample_evaluations = None
            if base_evaluations is not None and sample_idx < len(base_evaluations):
                sample_evaluations = base_evaluations[sample_idx]

            # For each base trajectory of this sample
            for base_idx, base_response in enumerate(sample_base_responses):
                filtered_response = base_response

                # Get correctness of this base trajectory (for adaptive intervention)
                is_correct = False
                if sample_evaluations is not None and base_idx < len(sample_evaluations):
                    is_correct = sample_evaluations[base_idx].get("is_correct", False)

                if is_correct:
                    correct_base_count += 1
                else:
                    wrong_base_count += 1

                # Generate multiple interventions for this base trajectory
                for intervention_idx in range(self.intervention_config.n_interventions_per_base):
                    # Get adaptive intervention strategy based on correctness
                    intervention_phrase, position_range = self.intervention_sampler.get_adaptive_intervention_strategy(
                        is_correct=is_correct,
                        intervention_idx=intervention_idx
                    )

                    # Get intervention position within the specified range
                    position = self.intervention_sampler.get_intervention_position_in_range(
                        response_tokens=filtered_response,
                        position_range=position_range
                    )

                    # Prepare intervention prompt (use filtered response)
                    intervention_inputs, intervention_meta = self.intervention_sampler.prepare_intervention_prompts(
                        base_prompts=[sample_prompt],
                        base_responses=[filtered_response],
                        intervention_positions=[position],
                        intervention_phrases=[intervention_phrase]
                    )

                    # Add sample and base trajectory indices to metadata
                    for meta in intervention_meta:
                        meta["sample_idx"] = sample_idx
                        meta["base_trajectory_idx"] = base_idx
                        meta["intervention_idx"] = intervention_idx
                        meta["base_is_correct"] = is_correct
                        meta["position_range"] = position_range

                         # === DEBUG ===
                    #     print(f"[DEBUG] sample={sample_idx}, base={base_idx}, interv={intervention_idx}, "
                    #             f"is_correct={is_correct}, position_range={position_range}")
                    #     # === END DEBUG ===
                    # pdb.set_trace()
                    all_intervention_inputs.extend(intervention_inputs)
                    all_intervention_metadata.extend(intervention_meta)

        # Log adaptive intervention statistics
        logger.info(f"Adaptive intervention stats: {correct_base_count} correct bases, {wrong_base_count} wrong bases")

        # Add multi-modal data if present
        if "multi_modal_data" in prompts.non_tensor_batch:
            multi_modal_data = prompts.non_tensor_batch["multi_modal_data"]
            for i, input_dict in enumerate(all_intervention_inputs):
                sample_idx = all_intervention_metadata[i]["sample_idx"]
                input_dict["multi_modal_data"] = multi_modal_data[sample_idx]

        # Calculate remaining tokens for each intervention
        # to ensure total response doesn't exceed max_response_length
        max_response_length = self.config.response_length
        #pdb.set_trace()
        # Import SamplingParams here to create adjusted params
        from vllm import SamplingParams

        # Create adjusted sampling params for each intervention
        adjusted_sampling_params_list = []
        for metadata in all_intervention_metadata:
            # Get the length already used (intervention position)
            intervention_position = metadata.get("intervention_position", 0)

            # Calculate remaining tokens (ensure at least 1)
            remaining_tokens = max(1, max_response_length - intervention_position)

            # Create new sampling params with adjusted max_tokens, copying all original params
            adjusted_params = SamplingParams(
                n=self.sampling_params.n,
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                top_k=self.sampling_params.top_k,
                min_p=self.sampling_params.min_p if hasattr(self.sampling_params, 'min_p') else 0.0,
                presence_penalty=self.sampling_params.presence_penalty if hasattr(self.sampling_params, 'presence_penalty') else 0.0,
                frequency_penalty=self.sampling_params.frequency_penalty if hasattr(self.sampling_params, 'frequency_penalty') else 0.0,
                repetition_penalty=self.sampling_params.repetition_penalty,
                max_tokens=min(remaining_tokens, self.sampling_params.max_tokens),  # Adjusted max_tokens
                min_tokens=self.sampling_params.min_tokens if hasattr(self.sampling_params, 'min_tokens') else 0,
                logprobs=self.sampling_params.logprobs,
                prompt_logprobs=self.sampling_params.prompt_logprobs,
                stop=self.sampling_params.stop if hasattr(self.sampling_params, 'stop') else [],
                stop_token_ids=self.sampling_params.stop_token_ids if hasattr(self.sampling_params, 'stop_token_ids') else [],
                bad_words=self.sampling_params.bad_words if hasattr(self.sampling_params, 'bad_words') else [],
                include_stop_str_in_output=self.sampling_params.include_stop_str_in_output if hasattr(self.sampling_params, 'include_stop_str_in_output') else False,
                ignore_eos=self.sampling_params.ignore_eos if hasattr(self.sampling_params, 'ignore_eos') else False,
                skip_special_tokens=self.sampling_params.skip_special_tokens if hasattr(self.sampling_params, 'skip_special_tokens') else True,
                spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens if hasattr(self.sampling_params, 'spaces_between_special_tokens') else True,
                truncate_prompt_tokens=self.sampling_params.truncate_prompt_tokens if hasattr(self.sampling_params, 'truncate_prompt_tokens') else None,
                seed=self.sampling_params.seed if hasattr(self.sampling_params, 'seed') else None,
            )
            adjusted_sampling_params_list.append(adjusted_params)

        
        # Generate with adjusted max_tokens for each prompt
        outputs = self.inference_engine.generate(
            prompts=all_intervention_inputs,
            sampling_params=adjusted_sampling_params_list,  # Use per-prompt adjusted params
            lora_request=None,
            use_tqdm=False,
        )

        # Extract and merge responses
        intervention_responses = []
        intervention_log_probs = []

        for i, (output, metadata) in enumerate(zip(outputs, all_intervention_metadata)):
            continuation_tokens = output.outputs[0].token_ids

            # Merge with prefix and intervention phrase
            complete_response = self.intervention_sampler.merge_intervention_responses(
                metadata, continuation_tokens, max_response_length=self.config.response_length
            )
            intervention_responses.append(complete_response)

            if self.config.calculate_log_probs:
                # Note: log probs for intervention are more complex
                # We only have log probs for the continuation part
                curr_log_prob = []
                for j, logprob in enumerate(output.outputs[0].logprobs):
                    curr_log_prob.append(logprob[continuation_tokens[j]].logprob)
                intervention_log_probs.append(curr_log_prob)

        return intervention_responses, intervention_log_probs, all_intervention_metadata

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using two-stage intervention-based rollout.

        Args:
            prompts: Input prompts
            **kwargs: Additional arguments

        Returns:
            DataProto with all generated sequences (base + interventions)
        """
        if not self.enable_intervention:
            # Fall back to standard generation
            return super().generate_sequences(prompts, **kwargs)

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]

        # Handle pre-repeated data: extract actual batch size
        original_batch_size = idx.size(0)
        expected_repeats = self.config.n

        # Check if this is validation/evaluation mode
        # During validation, data is typically not pre-repeated and batch size is different
        # We detect this by checking if the batch size suggests pre-repetition
        is_likely_validation = (original_batch_size < expected_repeats or
                               original_batch_size % expected_repeats != 0)

        if is_likely_validation:
            # During validation, use standard rollout without intervention
            logger.info(f"Detected validation mode (batch_size={original_batch_size}), using standard rollout")
            return super().generate_sequences(prompts, **kwargs)

        # Training mode: proceed with intervention-based rollout
        if original_batch_size % expected_repeats == 0 and expected_repeats > 1:
            actual_batch_size = original_batch_size // expected_repeats
            logger.info(f"Starting two-stage rollout (training mode): original_batch_size={original_batch_size}, "
                       f"expected_repeats={expected_repeats}, actual_batch_size={actual_batch_size}")
        else:
            actual_batch_size = original_batch_size
            logger.info(f"Starting two-stage rollout for batch_size={actual_batch_size}")

        # Stage 1: Generate base trajectories

        print(f"Stage 1: Generating {self.intervention_config.n_base_trajectories} base trajectories")
        base_responses, base_log_probs, vllm_inputs = self._generate_base_trajectories(prompts) # base_responses: 2(batch) * 4(base)

        # Evaluate base trajectories for adaptive intervention strategy
        print("Evaluating base trajectories for correctness...")
        base_evaluations = self._evaluate_base_trajectories(base_responses, prompts)

        # Stage 2: Generate intervention trajectories with adaptive strategy based on base evaluations
        print(f"Stage 2: Generating {self.intervention_config.n_base_trajectories * self.intervention_config.n_interventions_per_base} intervention trajectories")
        intervention_responses, intervention_log_probs, intervention_metadata = self._generate_intervention_trajectories(
            prompts, base_responses, vllm_inputs, base_evaluations
        )
        #
        # Combine all responses - IMPORTANT: must be grouped by sample to match original non_tensor_batch order
        # Original non_tensor_batch order: [sample_0 × n] + [sample_1 × n] + ...
        # So responses must be: [sample_0's 4 base + 12 intervention] + [sample_1's 4 base + 12 intervention] + ...
        all_responses = []
        all_log_probs = []

        # Use actual batch size from base generation
        batch_size = len(base_responses)
        n_base = self.intervention_config.n_base_trajectories
        n_interventions_per_base = self.intervention_config.n_interventions_per_base
        total_interventions_per_sample = n_base * n_interventions_per_base

        # Build responses grouped by sample
        for sample_idx in range(batch_size):
            # Add base trajectories for this sample
            for base_response in base_responses[sample_idx]:
                all_responses.append(base_response)

            # Add intervention trajectories for this sample
            # intervention_responses is ordered as: all interventions for sample_0, then sample_1, etc.
            start_idx = sample_idx * total_interventions_per_sample
            end_idx = start_idx + total_interventions_per_sample
            for i in range(start_idx, end_idx):
                all_responses.append(intervention_responses[i])

        # # ========== DEBUG: 检查 all_responses 中的轨迹类型 ==========
        # n_base = self.intervention_config.n_base_trajectories
        # n_total_per_sample = n_base + total_interventions_per_sample

        # print(f"\n[DEBUG ALL_RESPONSES] ========================================")
        # print(f"[DEBUG] Total responses: {len(all_responses)}")
        # print(f"[DEBUG] Per sample: {n_base} base + {total_interventions_per_sample} intervention = {n_total_per_sample}")

        # # 检查第一个样本的base轨迹和intervention轨迹
        # if len(all_responses) > 0:
        #     # 第一个base轨迹 (index 0)
        #     base_decoded = self.tokenizer.decode(all_responses[0][:300], skip_special_tokens=False)
        #     print(f"[DEBUG] Sample 0, Base 0 (first 300 chars): {base_decoded[:300]}")

        #     # 第一个intervention轨迹 (index n_base)
        #     if len(all_responses) > n_base:
        #         interv_decoded = self.tokenizer.decode(all_responses[n_base][:300], skip_special_tokens=False)
        #         print(f"[DEBUG] Sample 0, Intervention 0 (first 300 chars): {interv_decoded[:300]}")

        #         # 检查intervention metadata
        #         if len(intervention_metadata) > 0:
        #             meta = intervention_metadata[0]
        #             print(f"[DEBUG] Intervention 0 phrase: {repr(meta.get('intervention_phrase', 'N/A'))}")
        #             print(f"[DEBUG] Is phrase in intervention response? {meta.get('intervention_phrase', '').strip() in interv_decoded}")
        # print(f"[DEBUG] ========================================\n", flush=True)
        # # ========== END DEBUG ==========

        # === DEBUG: 统计所有轨迹的正确率 (已注释) ===
        # # 使用 math_verify.compute_score，与实际 reward 计算一致
        # from verl.utils.reward_score.math_verify import compute_score as math_verify_score
        # import os
        # reward_model_data = prompts.non_tensor_batch.get("reward_model", None)
        # pid = os.getpid()
        #
        # # 统计 base 轨迹准确率
        # base_correct = 0
        # base_total = 0
        # for sample_idx in range(batch_size):
        #     for base_idx, base_response in enumerate(base_responses[sample_idx]):
        #         base_total += 1
        #         reward_data_idx = sample_idx * expected_repeats
        #         ground_truth = None
        #         if reward_model_data is not None:
        #             if isinstance(reward_model_data, (list, np.ndarray)) and reward_data_idx < len(reward_model_data):
        #                 rm_item = reward_model_data[reward_data_idx]
        #                 if isinstance(rm_item, dict):
        #                     ground_truth = rm_item.get("ground_truth", None)
        #         if ground_truth is not None:
        #             response_str = self.tokenizer.decode(base_response, skip_special_tokens=True)
        #             try:
        #                 score = math_verify_score(response_str, str(ground_truth))
        #                 if score > 0:
        #                     base_correct += 1
        #             except:
        #                 pass
        #
        # # 统计 intervention 轨迹准确率
        # intervention_correct = 0
        # intervention_total = len(intervention_responses)
        # for i, interv_response in enumerate(intervention_responses):
        #     meta = intervention_metadata[i]
        #     sample_idx = meta["sample_idx"]
        #     reward_data_idx = sample_idx * expected_repeats
        #     ground_truth = None
        #     if reward_model_data is not None:
        #         if isinstance(reward_model_data, (list, np.ndarray)) and reward_data_idx < len(reward_model_data):
        #             rm_item = reward_model_data[reward_data_idx]
        #             if isinstance(rm_item, dict):
        #                 ground_truth = rm_item.get("ground_truth", None)
        #     if ground_truth is not None:
        #         response_str = self.tokenizer.decode(interv_response, skip_special_tokens=True)
        #         try:
        #             score = math_verify_score(response_str, str(ground_truth))
        #             if score > 0:
        #                 intervention_correct += 1
        #         except:
        #             pass
        #
        # total_correct = base_correct + intervention_correct
        # total_trajectories = base_total + intervention_total
        #
        # print(f"[DEBUG pid={pid}] Base accuracy: {base_correct}/{base_total} = {100*base_correct/max(1,base_total):.1f}%", flush=True)
        # print(f"[DEBUG pid={pid}] Intervention accuracy: {intervention_correct}/{intervention_total} = {100*intervention_correct/max(1,intervention_total):.1f}%", flush=True)
        # print(f"[DEBUG pid={pid}] Total accuracy: {total_correct}/{total_trajectories} = {100*total_correct/max(1,total_trajectories):.1f}%", flush=True)
        # === END DEBUG ===
        if self.config.calculate_log_probs:
            # Log probs must also be grouped by sample to match responses order
            for sample_idx in range(batch_size):
                # Add base log probs for this sample
                for base_log_prob in base_log_probs[sample_idx]:
                    all_log_probs.append(base_log_prob)
                # Add intervention log probs for this sample
                start_idx = sample_idx * total_interventions_per_sample
                end_idx = start_idx + total_interventions_per_sample
                for i in range(start_idx, end_idx):
                    all_log_probs.append(intervention_log_probs[i])

        # Pad responses to same length
        response_tensor = pad_2d_list_to_length(
            all_responses, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)

        if self.config.calculate_log_probs:
            log_probs_tensor = pad_2d_list_to_length(
                all_log_probs, -1, max_length=self.config.response_length
            ).to(idx.device).to(torch.float32)

        # Create output batch with all trajectories
        total_samples = len(all_responses)

        # The output should match the expected shape from pre-repeated input
        # If input was pre-repeated with n=16, output should also have that shape
        assert total_samples == original_batch_size, (
            f"Output size {total_samples} does not match expected size {original_batch_size}. "
            f"Generated {len(base_responses)} unique prompts × {self.intervention_config.n_base_trajectories + self.intervention_config.n_base_trajectories * self.intervention_config.n_interventions_per_base} trajectories"
        )

        # Use original tensors (already repeated if n>1)
        expanded_idx = idx
        expanded_attention_mask = attention_mask
        expanded_position_ids = position_ids

        # Log the shapes for debugging
        ##print(f"[DEBUG] Before padding - idx shape: {expanded_idx.shape}, response shape: {response_tensor.shape}", flush=True)

        # Create sequences
        seq = torch.cat([expanded_idx, response_tensor], dim=-1)

        # Log seq shape before padding
        #print(f"[DEBUG] Seq shape before padding: {seq.shape}", flush=True)

        # Ensure seq has consistent length across all workers
        # Use max_model_len if available, otherwise use prompt_length + response_length
        # All workers must use the same max_seq_length for consistency
        
        FIXED_MAX_SEQ_LENGTH = self.config.prompt_length + self.config.response_length

        #print(f"[DEBUG] Using FIXED_MAX_SEQ_LENGTH: {FIXED_MAX_SEQ_LENGTH}", flush=True)

        current_seq_length = seq.size(1)
        if current_seq_length < FIXED_MAX_SEQ_LENGTH:
            padding = torch.full(
                (seq.size(0), FIXED_MAX_SEQ_LENGTH - current_seq_length),
                self.pad_token_id,
                dtype=seq.dtype,
                device=seq.device
            )
            seq = torch.cat([seq, padding], dim=-1)
            #print(f"[DEBUG] Padded seq from {current_seq_length} to {seq.size(1)}", flush=True)
        elif current_seq_length > FIXED_MAX_SEQ_LENGTH:
            # Truncate if longer than max (this may lose some response tokens)
            logger.warning(f"Truncating seq from {current_seq_length} to {FIXED_MAX_SEQ_LENGTH}. "
                         f"Consider increasing prompt_length or response_length in config.")
            seq = seq[:, :FIXED_MAX_SEQ_LENGTH]

        # Update position_ids and attention_mask for responses
        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(total_samples, -1)

        if expanded_position_ids.dim() == 3:  # Handle multi-dimensional position_ids
            delta_position_id = delta_position_id.view(total_samples, 1, -1).expand(total_samples, 3, -1)

        response_position_ids = expanded_position_ids[..., -1:] + delta_position_id
        final_position_ids = torch.cat([expanded_position_ids, response_position_ids], dim=-1)

        # Pad position_ids to match FIXED_MAX_SEQ_LENGTH
        if final_position_ids.size(-1) < FIXED_MAX_SEQ_LENGTH:
            if final_position_ids.dim() == 3:
                padding_shape = (total_samples, 3, FIXED_MAX_SEQ_LENGTH - final_position_ids.size(-1))
            else:
                padding_shape = (total_samples, FIXED_MAX_SEQ_LENGTH - final_position_ids.size(-1))
            padding = torch.zeros(padding_shape, dtype=final_position_ids.dtype, device=final_position_ids.device)
            final_position_ids = torch.cat([final_position_ids, padding], dim=-1)
        elif final_position_ids.size(-1) > FIXED_MAX_SEQ_LENGTH:
            final_position_ids = final_position_ids[..., :FIXED_MAX_SEQ_LENGTH]

        response_attention_mask = get_response_mask(
            response_id=response_tensor, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        final_attention_mask = torch.cat([expanded_attention_mask, response_attention_mask], dim=-1)

        # Pad attention_mask to match FIXED_MAX_SEQ_LENGTH
        if final_attention_mask.size(-1) < FIXED_MAX_SEQ_LENGTH:
            padding = torch.zeros(
                (total_samples, FIXED_MAX_SEQ_LENGTH - final_attention_mask.size(-1)),
                dtype=final_attention_mask.dtype,
                device=final_attention_mask.device
            )
            final_attention_mask = torch.cat([final_attention_mask, padding], dim=-1)
        elif final_attention_mask.size(-1) > FIXED_MAX_SEQ_LENGTH:
            final_attention_mask = final_attention_mask[:, :FIXED_MAX_SEQ_LENGTH]

        # Ensure prompts and responses are also padded to consistent lengths
        # Pad responses to max response length
        max_response_length = self.config.response_length
        if response_tensor.size(1) < max_response_length:
            response_padding = torch.full(
                (response_tensor.size(0), max_response_length - response_tensor.size(1)),
                self.pad_token_id,
                dtype=response_tensor.dtype,
                device=response_tensor.device
            )
            response_tensor_padded = torch.cat([response_tensor, response_padding], dim=-1)
        else:
            response_tensor_padded = response_tensor[:, :max_response_length]

        # Pad prompts to max prompt length
        max_prompt_length = self.config.prompt_length
        if expanded_idx.size(1) < max_prompt_length:
            prompt_padding = torch.full(
                (expanded_idx.size(0), max_prompt_length - expanded_idx.size(1)),
                self.pad_token_id,
                dtype=expanded_idx.dtype,
                device=expanded_idx.device
            )
            expanded_idx_padded = torch.cat([prompt_padding, expanded_idx], dim=-1)  # Left padding for prompts
        else:
            expanded_idx_padded = expanded_idx[:, :max_prompt_length]

        # Final debug info
        #print(f"[DEBUG] Final shapes - seq: {seq.shape}, prompts: {expanded_idx_padded.shape}, responses: {response_tensor_padded.shape}", flush=True)
        #print(f"[DEBUG] Final shapes - attention_mask: {final_attention_mask.shape}, position_ids: {final_position_ids.shape}", flush=True)

        # Create output batch
        batch = TensorDict(
            {
                "prompts": expanded_idx_padded,
                "responses": response_tensor_padded,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "position_ids": final_position_ids,
            },
            batch_size=total_samples,
        )

        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = log_probs_tensor

        # Responses are now in correct order matching original non_tensor_batch:
        # [sample_0's n trajectories] + [sample_1's n trajectories] + ...
        # So we can use the original non_tensor_batch directly

        print(f"Two-stage rollout completed. Generated {total_samples} total trajectories")

        return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)