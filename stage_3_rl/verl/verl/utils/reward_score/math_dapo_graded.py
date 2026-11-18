# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Graded reward function for math problems with format scoring
# This version provides partial credit for format compliance and partial correctness

import re
from typing import Optional, Union

# Import functions from original math_dapo
from .math_dapo import (
    last_boxed_only_string,
    remove_boxed,
    normalize_final_answer,
    is_correct_minerva,
)


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    enable_format_reward: bool = True,
    enable_partial_credit: bool = True,
) -> Union[float, dict]:
    """Compute graded reward score with format and partial correctness scoring.

    Reward structure:
    - Full credit (1.0): Correct answer with proper format
    - High partial (0.8): Correct answer without proper format
    - Medium partial (0.4): Has boxed format but wrong answer
    - Low partial (0.2): Has "Answer:" keyword but wrong answer
    - No credit (0.0): No format markers and wrong answer

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens
        enable_format_reward: Whether to give partial credit for format
        enable_partial_credit: Whether to give partial credit for near-correct answers

    Returns:
        dict with score, acc, pred, and breakdown
    """
    # Limit solution length for efficiency
    original_solution = solution_str
    solution_str = solution_str[-300:]

    # Initialize score breakdown
    breakdown = {
        "has_boxed": False,
        "has_answer_keyword": False,
        "answer_correct": False,
        "format_score": 0.0,
        "correctness_score": 0.0,
    }

    # Check for format markers
    has_boxed = "\\boxed" in solution_str or "\\fbox" in solution_str
    has_answer_keyword = bool(re.search(r"(?i)(answer|final answer)\s*:", solution_str))

    breakdown["has_boxed"] = has_boxed
    breakdown["has_answer_keyword"] = has_answer_keyword

    # Extract and verify answer
    pred = None
    correct = False

    try:
        # Try to extract boxed answer
        boxed_answer = last_boxed_only_string(solution_str)
        if boxed_answer:
            pred = remove_boxed(boxed_answer)
        else:
            # Fallback to Answer: pattern
            match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
            if match:
                pred = match[-1].strip()

        # Verify correctness
        if pred:
            pred_normalized = normalize_final_answer(pred)
            gt_normalized = normalize_final_answer(ground_truth)
            correct = (pred_normalized == gt_normalized)
            breakdown["answer_correct"] = correct

            # Check for partial correctness (numerical closeness)
            if not correct and enable_partial_credit:
                try:
                    pred_num = float(re.sub(r"[^\d.-]", "", pred_normalized))
                    gt_num = float(re.sub(r"[^\d.-]", "", gt_normalized))
                    relative_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-8)
                    if relative_error < 0.05:  # Within 5% is considered close
                        breakdown["close_enough"] = True
                except:
                    pass

    except Exception as e:
        pass

    # Compute final score with breakdown
    if correct:
        if has_boxed:
            # Perfect: correct answer with proper format
            score = 1.0
            breakdown["format_score"] = 0.2
            breakdown["correctness_score"] = 0.8
        else:
            # Good: correct answer but missing format
            score = 0.8 if enable_format_reward else 1.0
            breakdown["format_score"] = 0.0
            breakdown["correctness_score"] = 0.8
    else:
        # Answer is wrong
        if enable_format_reward:
            if has_boxed:
                # Has proper format, encourage this
                score = 0.4
                breakdown["format_score"] = 0.4
            elif has_answer_keyword:
                # At least has answer keyword
                score = 0.2
                breakdown["format_score"] = 0.2
            else:
                # No format markers
                score = 0.0
        else:
            # No format reward, pure binary
            score = 0.0

        # Add partial credit for close answers
        if enable_partial_credit and breakdown.get("close_enough", False):
            score = max(score, 0.6)
            breakdown["correctness_score"] = 0.4

    return {
        "score": score,
        "acc": correct,
        "pred": pred,
        "breakdown": breakdown,
    }


def compute_score_simple(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Simplified version that returns only the score (for compatibility).

    This uses moderate format reward:
    - Correct + boxed: 1.0
    - Correct without boxed: 0.9
    - Has boxed but wrong: 0.3
    - Wrong without boxed: 0.0
    """
    result = compute_score(
        solution_str,
        ground_truth,
        strict_box_verify,
        pause_tokens_index,
        enable_format_reward=True,
        enable_partial_credit=False,
    )
    return result["score"]
