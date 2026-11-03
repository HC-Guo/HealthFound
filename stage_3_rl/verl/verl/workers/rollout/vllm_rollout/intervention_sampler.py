"""
Intervention-based trajectory sampling for improved diversity in GRPO training.
"""

import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InterventionConfig:
    """Configuration for intervention-based sampling."""
    n_base_trajectories: int = 4  # Number of base trajectories to sample
    n_interventions_per_base: int = 3  # Number of interventions per base trajectory
    base_temperature: float = 1.2  # Higher temperature for diverse base trajectories
    intervention_temperature: float = 1.0  # Temperature for intervention completions
    intervention_phrases: List[str] = None  # List of intervention phrases (for non-adaptive mode)
    min_intervention_position: float = 0.3  # Minimum position (0-1) to insert intervention
    max_intervention_position: float = 0.8  # Maximum position (0-1) to insert intervention

    # Adaptive intervention settings
    enable_adaptive_intervention: bool = True  # Enable adaptive strategy based on correctness

    # Phrases for CORRECT trajectories (goal: encourage early conclusion, efficiency)
    correct_phrases: List[str] = None
    # Positions for correct trajectories: try to conclude earlier (spread across range)
    correct_position_ranges: List[Tuple[float, float]] = None  # [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]

    # Phrases for WRONG trajectories (goal: encourage rethinking, correction)
    wrong_phrases: List[str] = None
    # Positions for wrong trajectories: intervene earlier to "rescue"
    wrong_position_ranges: List[Tuple[float, float]] = None  # [(0.15, 0.25), (0.3, 0.4), (0.45, 0.55)]

    def __post_init__(self):
        # Default intervention phrases (for non-adaptive mode)
        if self.intervention_phrases is None:
            self.intervention_phrases = [
                # Problem decomposition
                "Let me try to break this down step by step",
                "Let me break this down step by step",
                "Hmm, let me break this down step by step",
                "Let me try to figure this out step by step",
                # Changing approach
                "Let me try to think differently",
                "Maybe I need to think differently",
                "Wait, let me think differently",
                "Let me try a different approach",
                "Let me try another approach",
                "Maybe I need a different approach",
                # Verification/correction
                "Wait, no, that's not correct",
                "Wait, that doesn't make sense",
                "Wait, no, that can't be",
                "Let me double-check this",
                "Let me verify this",
                # Deep thinking
                "Hmm, let me think about how to approach this",
                "Let me think about how to approach this",
                "Wait, let me think again",
                "Actually, let me think about this more",
                # Summarization
                "So, putting it all together",
                "Thus, the final answer is",
                # Recording/expression
                "Let me write that down",
                "Let me write this as",
                "Let me write that out",
                # Uncertainty/questioning
                "But I'm not entirely sure",
                "Wait, but I'm not sure",
                "So, which one is correct",
                "Hmm, not sure if that helps",
                # Backtracking/reconsidering
                "Wait, perhaps I made a mistake in the",
                "Let me reconsider the",
            ]

        # Phrases for CORRECT trajectories: summarization, conclusion, verification
        # Goal: Learn to conclude efficiently without unnecessary elaboration
        if self.correct_phrases is None:
            self.correct_phrases = [
                # Summarization / Early conclusion
                "Time is limited, stop thinking and start answering.\n</think>\n\n"
            ]

        # Phrases for WRONG trajectories: rethinking, changing approach, error correction
        # Goal: Learn to recognize mistakes and try different approaches
        if self.wrong_phrases is None:
            self.wrong_phrases = [
                # Rethinking / Changing approach
                "Wait, let me reconsider this.",
                "Actually, let me try a different approach.",
                "Wait, I think I made a mistake. Let me start over.",
                "Let me try to think about this differently.",
                "Maybe I should approach this another way.",
                # Error recognition
                "Wait, no, maybe that's not correct.",
                "Hold on, maybe something is wrong here.",
                # Fresh start
                "Let me reconsider the problem from the beginning.",
                "Wait, let me re-read the problem.",
            ]

        # Position ranges for correct trajectories: spread across the response
        # Goal: Test where we can safely conclude without losing correctness
        if self.correct_position_ranges is None:
            self.correct_position_ranges = [
                (0.2, 0.3),   # Early conclusion attempt
                (0.3, 0.4),   # Mid-point conclusion
                (0.4, 0.5),   # Late conclusion
            ]

        # Position ranges for wrong trajectories: intervene earlier
        # Goal: Try to "rescue" the trajectory before it goes too far wrong
        if self.wrong_position_ranges is None:
            self.wrong_position_ranges = [
                (0.05, 0.10),  # Very early intervention
                (0.10, 0.20),  # Early-mid intervention
                (0.20, 0.30),  # Mid intervention
            ]

    @property
    def total_rollouts(self) -> int:
        """Total number of rollouts (base + interventions)."""
        return self.n_base_trajectories * (1 + self.n_interventions_per_base)


class InterventionSampler:
    """Handles intervention-based trajectory sampling for two-stage rollout."""

    def __init__(self, config: InterventionConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def get_intervention_positions(self, response_tokens: List[torch.Tensor]) -> List[int]:
        """
        Get intervention positions for a batch of responses.
        Finds positions after '\n\n' (double newline) to ensure interventions
        occur at natural step boundaries.

        Args:
            response_tokens: List of response token tensors

        Returns:
            List of intervention positions (after '\n\n' sequences)
        """
        positions = []

        # Get token IDs for different double newline patterns
        # These patterns commonly appear at step boundaries
        double_newline_patterns = []
        patterns_to_encode = ['\n\n', '.\n\n', ' \n\n', '!\n\n', ')\n\n']
        for pattern in patterns_to_encode:
            encoded = self.tokenizer.encode(pattern, add_special_tokens=False)
            if encoded:  # Only add non-empty encodings
                double_newline_patterns.append(encoded)

        # Also prepare single newline patterns as fallback
        single_newline_patterns = []
        single_patterns_to_encode = ['\n', '.\n', ' \n']
        for pattern in single_patterns_to_encode:
            encoded = self.tokenizer.encode(pattern, add_special_tokens=False)
            if encoded:  # Only add non-empty encodings
                single_newline_patterns.append(encoded)

        for tokens in response_tokens:
            if isinstance(tokens, torch.Tensor):
                tokens_list = tokens.tolist() if tokens.dim() > 0 else [tokens.item()]
            else:
                tokens_list = tokens if isinstance(tokens, list) else [tokens]

            response_length = len(tokens_list)

            # Skip very short responses
            if response_length < 10:
                positions.append(response_length // 2)
                continue

            min_pos = int(response_length * self.config.min_intervention_position)
            max_pos = int(response_length * self.config.max_intervention_position)

            # Find all positions where double newline patterns occur
            double_newline_positions = []
            for pattern in double_newline_patterns:
                pattern_len = len(pattern)
                for i in range(min_pos, min(max_pos, response_length - pattern_len + 1)):
                    # Check if the sequence at position i matches the pattern
                    if tokens_list[i:i+pattern_len] == pattern:
                        # Position after the pattern
                        double_newline_positions.append(i + pattern_len)

            # If we found double newline positions in the valid range, randomly select one
            if double_newline_positions:
                position = random.choice(double_newline_positions)
            else:
                # Fallback: if no double newline patterns found, look for single newline patterns
                single_newline_positions = []

                for pattern in single_newline_patterns:
                    pattern_len = len(pattern)
                    for i in range(min_pos, min(max_pos, response_length - pattern_len + 1)):
                        if tokens_list[i:i+pattern_len] == pattern:
                            single_newline_positions.append(i + pattern_len)

                if single_newline_positions:
                    position = random.choice(single_newline_positions)
                else:
                    # Final fallback: random position in range
                    if min_pos >= max_pos:
                        position = response_length // 2
                    else:
                        position = random.randint(min_pos, min(max_pos, response_length))

            positions.append(position)

        return positions

    def prepare_intervention_prompts(self,
                                    base_prompts: torch.Tensor,
                                    base_responses: List[List[int]],
                                    intervention_positions: List[int],
                                    intervention_phrases: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare prompts for intervention rollout by inserting intervention phrases.

        Args:
            base_prompts: Original prompt tensor [n_base, prompt_length]
            base_responses: Base trajectory responses
            intervention_positions: Position to insert interventions
            intervention_phrases: Phrases to insert

        Returns:
            Tuple of (intervention_inputs, intervention_metadata)
        """
        intervention_inputs = []
        intervention_metadata = []

        for base_idx, (prompt, response, position, phrase) in enumerate(
            zip(base_prompts, base_responses, intervention_positions, intervention_phrases)
        ):
            # Extract prefix of response up to intervention point
            prefix_tokens = response[:position]

            # Tokenize intervention phrase
            intervention_tokens = self.tokenizer.encode(
                phrase,
                add_special_tokens=False
            )

            # Combine prompt + prefix + intervention phrase
            if isinstance(prompt, torch.Tensor):
                prompt_list = prompt.cpu().tolist() if prompt.dim() > 0 else [prompt.item()]
            else:
                prompt_list = prompt

            # Remove padding from prompt
            prompt_list = [t for t in prompt_list if t != self.pad_token_id]

            # Create new prompt with intervention
            # Ensure all tokens are valid (within vocabulary)
            vocab_size = len(self.tokenizer)
            new_prompt_tokens = []
            # import pdb; pdb.set_trace()
            # Add prompt tokens (filter out invalid ones)
            for token in prompt_list:
                if 0 <= token < vocab_size:
                    new_prompt_tokens.append(token)
                else:
                    print(f"Warning: Skipping invalid prompt token {token} (vocab_size={vocab_size})")

            # Add prefix tokens (filter out invalid ones)
            for token in prefix_tokens:
                if 0 <= token < vocab_size:
                    new_prompt_tokens.append(token)
                else:
                    print(f"Warning: Skipping invalid prefix token {token} (vocab_size={vocab_size})")

            # Add intervention tokens (these should be valid since we just encoded them)
            for token in intervention_tokens:
                if 0 <= token < vocab_size:
                    new_prompt_tokens.append(token)
                else:
                    print(f"Warning: Skipping invalid intervention token {token} (vocab_size={vocab_size})")

            intervention_inputs.append({
                "prompt_token_ids": new_prompt_tokens
            })

            intervention_metadata.append({
                "base_idx": base_idx,
                "intervention_position": position,
                "intervention_phrase": phrase,
                "prefix_length": len(prompt_list) + len(prefix_tokens),
                "intervention_length": len(intervention_tokens),
                "original_prompt_length": len(prompt_list),
                "original_response": response
            })
        # pdb.set_trace()
        return intervention_inputs, intervention_metadata

    def merge_intervention_responses(self,
                                    intervention_metadata: Dict[str, Any],
                                    continuation_tokens: List[int],
                                    max_response_length: int = 8192) -> List[int]:
        """
        Merge the continuation with the prefix to create complete response.

        Args:
            intervention_metadata: Metadata about the intervention
            continuation_tokens: Generated continuation tokens after intervention
            max_response_length: Maximum allowed response length (from config)

        Returns:
            Complete response tokens including prefix, intervention and continuation
        """
        original_response = intervention_metadata["original_response"]
        position = intervention_metadata["intervention_position"]

        # Get the prefix from original response
        prefix = original_response[:position]

        # Tokenize intervention phrase
        intervention_tokens = self.tokenizer.encode(
            intervention_metadata["intervention_phrase"],
            add_special_tokens=False
        )

        # Combine prefix + intervention + continuation
        complete_response = prefix + intervention_tokens + continuation_tokens

        # Ensure the complete response doesn't exceed max_response_length
        if len(complete_response) > max_response_length:
            complete_response = complete_response[:max_response_length]

        return complete_response

    def prepare_two_stage_rollout(self, prompts: torch.Tensor, batch_size: int) -> Dict[str, Any]:
        """
        Prepare configuration for two-stage rollout.

        Args:
            prompts: Original prompts tensor
            batch_size: Batch size for each sample

        Returns:
            Configuration dict for two-stage rollout
        """
        # Stage 1: Base trajectories with higher temperature
        stage1_config = {
            "n_rollouts": self.config.n_base_trajectories,
            "temperature": self.config.base_temperature,
            "prompts_per_rollout": batch_size,
        }

        # Stage 2: Intervention trajectories with normal temperature
        stage2_config = {
            "n_rollouts": self.config.n_base_trajectories * self.config.n_interventions_per_base,
            "temperature": self.config.intervention_temperature,
            "prompts_per_rollout": batch_size,
        }

        return {
            "stage1": stage1_config,
            "stage2": stage2_config,
            "total_rollouts": self.config.total_rollouts,
            "intervention_config": self.config,
        }

    def get_adaptive_intervention_strategy(
        self,
        is_correct: bool,
        intervention_idx: int
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Get intervention phrase and position range based on trajectory correctness.

        Args:
            is_correct: Whether the base trajectory is correct
            intervention_idx: Index of the intervention (0, 1, 2, ...)

        Returns:
            Tuple of (intervention_phrase, position_range)
        """
        if not self.config.enable_adaptive_intervention:
            # Non-adaptive mode: use original random selection
            phrase = random.choice(self.config.intervention_phrases)
            position_range = (self.config.min_intervention_position, self.config.max_intervention_position)
            return phrase, position_range

        if is_correct:
            # CORRECT trajectory: use summarization phrases, spread positions
            phrases = self.config.correct_phrases
            position_ranges = self.config.correct_position_ranges
        else:
            # WRONG trajectory: use rethinking phrases, earlier positions
            phrases = self.config.wrong_phrases
            position_ranges = self.config.wrong_position_ranges

        # Select phrase randomly from the appropriate list
        phrase = random.choice(phrases)

        # Select position range based on intervention index
        # This ensures different interventions go to different positions
        n_ranges = len(position_ranges)
        range_idx = intervention_idx % n_ranges
        position_range = position_ranges[range_idx]

        return phrase, position_range

    def _find_think_boundaries(self, response_tokens: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the token positions of <think> and </think> tags.

        Returns:
            Tuple of (think_start_pos, think_end_pos)
            - think_start_pos: token index right after <think> tag ends
            - think_end_pos: token index where </think> tag starts

            Returns (None, None) if no think tags found.
        """
        # Encode the think tags
        think_open_tokens = self.tokenizer.encode('<think>', add_special_tokens=False)
        think_close_tokens = self.tokenizer.encode('</think>', add_special_tokens=False)

        # Find <think> position
        think_start_pos = None
        open_len = len(think_open_tokens)
        for i in range(len(response_tokens) - open_len + 1):
            if response_tokens[i:i+open_len] == think_open_tokens:
                think_start_pos = i + open_len  # Position right after <think>
                break

        if think_start_pos is None:
            return None, None

        # Find </think> position (after <think>)
        think_end_pos = None
        close_len = len(think_close_tokens)
        for i in range(think_start_pos, len(response_tokens) - close_len + 1):
            if response_tokens[i:i+close_len] == think_close_tokens:
                think_end_pos = i  # Position where </think> starts
                break

        if think_end_pos is None or think_end_pos <= think_start_pos:
            return None, None

        return think_start_pos, think_end_pos

    def _find_natural_break_in_range(
        self,
        response_tokens: List[int],
        min_pos: int,
        max_pos: int
    ) -> int:
        """Find a natural break position (after newline) within the given token range."""
        response_length = len(response_tokens)

        # Get token IDs for double newline patterns
        double_newline_patterns = []
        for pattern in ['\n\n', '.\n\n', ' \n\n', '!\n\n', ')\n\n']:
            encoded = self.tokenizer.encode(pattern, add_special_tokens=False)
            if encoded:
                double_newline_patterns.append(encoded)

        # Single newline patterns as fallback
        single_newline_patterns = []
        for pattern in ['\n', '.\n', ' \n']:
            encoded = self.tokenizer.encode(pattern, add_special_tokens=False)
            if encoded:
                single_newline_patterns.append(encoded)

        # Find double newline positions in range
        double_newline_positions = []
        for pattern in double_newline_patterns:
            pattern_len = len(pattern)
            for i in range(min_pos, min(max_pos, response_length - pattern_len + 1)):
                if response_tokens[i:i+pattern_len] == pattern:
                    double_newline_positions.append(i + pattern_len)

        if double_newline_positions:
            return random.choice(double_newline_positions)

        # Fallback: single newline positions
        single_newline_positions = []
        for pattern in single_newline_patterns:
            pattern_len = len(pattern)
            for i in range(min_pos, min(max_pos, response_length - pattern_len + 1)):
                if response_tokens[i:i+pattern_len] == pattern:
                    single_newline_positions.append(i + pattern_len)

        if single_newline_positions:
            return random.choice(single_newline_positions)

        # Final fallback: random position in range
        return random.randint(min_pos, max_pos)

    def get_intervention_position_in_range(
        self,
        response_tokens: List[int],
        position_range: Tuple[float, float]
    ) -> int:
        """
        Get intervention position within a specific range, preferring natural boundaries.

        IMPORTANT: If the response contains <think>...</think> tags, the position is
        calculated as a percentage of the THINKING content only, not the entire response.
        The returned position is still the absolute token position in the full response.

        Args:
            response_tokens: List of response token IDs
            position_range: (min_ratio, max_ratio) tuple

        Returns:
            Absolute token position index for intervention
        """
        response_length = len(response_tokens)

        # Skip very short responses
        if response_length < 10:
            return response_length // 2

        # Check for <think>...</think> tags
        think_start_pos, think_end_pos = self._find_think_boundaries(response_tokens)

        if think_start_pos is not None and think_end_pos is not None:
            # Calculate position based on thinking content only
            think_length = think_end_pos - think_start_pos

            if think_length < 10:
                # Very short thinking, use midpoint
                return think_start_pos + think_length // 2

            # Calculate position range within thinking content
            min_pos_relative = int(think_length * position_range[0])
            max_pos_relative = int(think_length * position_range[1])

            min_pos_relative = max(0, min_pos_relative)
            max_pos_relative = min(think_length, max_pos_relative)

            if min_pos_relative >= max_pos_relative:
                return think_start_pos + think_length // 2

            # Convert to absolute positions
            abs_min_pos = think_start_pos + min_pos_relative
            abs_max_pos = think_start_pos + max_pos_relative

            # Find natural break within thinking content
            return self._find_natural_break_in_range(response_tokens, abs_min_pos, abs_max_pos)

        else:
            # No think tags, use the full response (fallback for non-reasoning models)
            min_pos = int(response_length * position_range[0])
            max_pos = int(response_length * position_range[1])

            # Ensure valid range
            min_pos = max(0, min_pos)
            max_pos = min(response_length, max_pos)

            if min_pos >= max_pos:
                return response_length // 2

            return self._find_natural_break_in_range(response_tokens, min_pos, max_pos)