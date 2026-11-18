# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhanced math reward computation for DeepScaler datasets.
Integrates features from summ-reason repository with VERL framework.
"""

import re
from typing import Optional, Union, List


def compute_score(solution_str: str, ground_truth: Union[str, List[str]], extra_info: dict = None) -> Union[float, dict]:
    """Compute reward score for mathematical solutions.

    This function evaluates mathematical solutions by comparing them against ground truth answers.
    It supports multiple answer formats and uses various validation methods.

    Args:
        solution_str: The solution string from the model
        ground_truth: Either a single string or list of valid answers
        extra_info: Optional dictionary with additional information

    Returns:
        Float score: 1.0 for correct, 0.0 for incorrect
        OR dict with score, acc, and pred for compatibility with math_dapo
    """
    if extra_info is None:
        extra_info = {}

    # Handle empty or None responses
    if not solution_str or solution_str.strip() == "":
        return 0.0

    # Extract answer from solution
    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return 0.0

    # Convert single answer to list for uniform processing
    if isinstance(ground_truth, str):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth if ground_truth else []

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth).strip()
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0.0

    # Check against all possible correct answers
    for ground_truth_answer in processed_ground_truths:
        if is_equiv(model_answer, ground_truth_answer):
            # Apply tool call bonus if applicable
            if extra_info.get("has_toolcall", False):
                return 1.2  # 20% bonus for using tools
            return 1.0

        # Try additional validation methods
        if grade_answer_mathd(model_answer, ground_truth_answer):
            return 1.0

        if grade_answer_sympy(model_answer, ground_truth_answer):
            return 1.0

    return 0.0


def extract_answer(solution_str: str) -> Optional[str]:
    """Extract the final answer from a solution string.

    Supports multiple formats including \\boxed{}, direct answers, and
    answers after specific delimiters.
    """
    # Check for thought delimiter (from summ-reason style)
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]

    # Try to extract boxed answer
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer:
        return remove_boxed(boxed_answer)

    # Try to extract answer after "The answer is"
    answer_patterns = [
        r"[Tt]he answer is:?\s*(.+?)(?:\.|$)",
        r"[Ff]inal answer:?\s*(.+?)(?:\.|$)",
        r"####\s*(.+?)(?:\.|$)",  # GSM8k style
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, solution_str)
        if match:
            return match.group(1).strip()

    # If no pattern matches, try to extract the last number or expression
    # This is a fallback for simple answers
    numbers = re.findall(r"-?\d+\.?\d*", solution_str)
    if numbers:
        return numbers[-1]

    return None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None

    # Handle \\boxed{...} format
    if idx + 6 < len(string) and string[idx:idx+7] == "\\boxed{":
        i = idx + 7
        brace_count = 1
        end_idx = i

        while i < len(string) and brace_count > 0:
            if string[i] == "{":
                brace_count += 1
            elif string[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
            i += 1

        if brace_count == 0:
            return string[idx:end_idx+1]

    # Handle \\boxed ... format (space after boxed)
    if idx + 6 < len(string) and string[idx:idx+7] == "\\boxed ":
        # Find the end of the boxed content (usually ends with $ or newline)
        end_match = re.search(r"[\$\n]", string[idx+7:])
        if end_match:
            end_idx = idx + 7 + end_match.start()
            return "\\boxed{" + string[idx+7:end_idx] + "}"
        else:
            return "\\boxed{" + string[idx+7:].strip() + "}"

    return None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    if "\\boxed " in s:
        return s.replace("\\boxed ", "").strip()

    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1]

    return s


def is_equiv(str1: str, str2: str) -> bool:
    """Check if two mathematical expressions are equivalent."""
    if str1 is None or str2 is None:
        return False

    # Normalize both strings
    norm1 = normalize_answer(str1)
    norm2 = normalize_answer(str2)

    return norm1 == norm2


def normalize_answer(string: str) -> str:
    """Normalize a mathematical answer for comparison."""
    if not string:
        return ""

    # Remove extra whitespace
    string = " ".join(string.split())

    # Remove dollar signs
    string = string.replace("$", "").replace("\\$", "")

    # Remove units and text
    string = re.sub(r"\\text\{[^}]*\}", "", string)
    string = re.sub(r"\b(meters?|cm|km|miles?|feet|inches|hours?|minutes?|seconds?|dollars?|cents?|pounds?)\b", "", string, flags=re.IGNORECASE)

    # Normalize fractions
    string = string.replace("\\frac", "frac")
    string = string.replace("\\dfrac", "frac")
    string = string.replace("\\tfrac", "frac")

    # Remove LaTeX commands
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\!", "")

    # Normalize decimal points
    string = re.sub(r"\.0+$", "", string)  # Remove trailing zeros after decimal
    string = re.sub(r"(\d)\.0+(\D|$)", r"\1\2", string)  # Remove .0 from integers

    # Remove all spaces for final comparison
    string = string.replace(" ", "")

    return string.strip().lower()


def grade_answer_mathd(model_answer: str, ground_truth: str) -> bool:
    """Grade answer using MATH dataset style comparison."""
    try:
        # Try direct string comparison after normalization
        return normalize_answer(model_answer) == normalize_answer(ground_truth)
    except:
        return False


def grade_answer_sympy(model_answer: str, ground_truth: str) -> bool:
    """Grade answer using symbolic math comparison.

    This uses sympy for more sophisticated mathematical equivalence checking.
    """
    try:
        from sympy import simplify, sympify, N
        from sympy.parsing.latex import parse_latex

        # Try to parse both as sympy expressions
        try:
            # Try LaTeX parsing first
            if "\\" in model_answer or "\\" in ground_truth:
                model_expr = parse_latex(model_answer)
                truth_expr = parse_latex(ground_truth)
            else:
                model_expr = sympify(model_answer)
                truth_expr = sympify(ground_truth)
        except:
            # Fallback to direct sympify
            model_expr = sympify(model_answer)
            truth_expr = sympify(ground_truth)

        # Check if expressions are equivalent
        difference = simplify(model_expr - truth_expr)

        # Check if difference is zero (handles both symbolic and numeric)
        if difference == 0:
            return True

        # For numerical comparison with tolerance
        try:
            diff_float = abs(float(N(difference)))
            return diff_float < 1e-6
        except:
            return False

    except:
        # If sympy fails, fall back to string comparison
        return False


# Additional helper functions for specific dataset formats

def compute_score_with_metadata(solution_str: str, ground_truth: Union[str, List[str]],
                                data_source: str = None, extra_info: dict = None) -> dict:
    """Compute score with additional metadata for analysis.

    Returns a dictionary with score and additional information about the evaluation.
    """
    score = compute_score(solution_str, ground_truth, extra_info)

    extracted_answer = extract_answer(solution_str)

    metadata = {
        "score": score,
        "extracted_answer": extracted_answer,
        "data_source": data_source,
        "evaluation_method": "rule-based",
    }

    if extra_info:
        metadata["extra_info"] = extra_info

    return metadata