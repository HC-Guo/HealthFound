#!/usr/bin/env python3
"""De-identified reward function used by the Stage-3 GRPO template.

This is a faithful public-safe rewrite of the internal reward logic:
- binary risk prediction reward with answer/thinking consistency
- multiple-choice reasoning reward
- masked value reward for codes, dates, numeric points, and numeric ranges
- open QA reward using lightweight lexical overlap metrics

Private data-source names are intentionally replaced with generic aliases. If
your released parquet files use different ``data_source`` values, either set
``extra_info["task_type"]`` or extend ``TASK_ALIASES`` below in your private
training copy.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from mathruler.grader import extract_boxed_content
except ImportError:

    def extract_boxed_content(text: str) -> str:
        match = re.findall(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.IGNORECASE | re.DOTALL)
        return match[-1].strip() if match else text


TASK_ALIASES = {
    "binary_risk": {
        "binary_risk",
        "risk_prediction",
        "diagnosis_prediction",
        "clinical_risk_prediction",
    },
    "choice_reasoning": {
        "choice_reasoning",
        "medical_choice_reasoning",
        "general_choice_reasoning",
        "multiple_choice",
    },
    "masked_value": {
        "masked_value",
        "fill_in_the_mask",
        "structured_value_prediction",
        "numeric_or_code_prediction",
    },
    "open_qa": {
        "open_qa",
        "scientific_reasoning_qa",
        "free_text_qa",
    },
}


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    **_: Any,
) -> float:
    """veRL custom reward entrypoint."""
    del reward_router_address, reward_model_tokenizer

    task_type = normalize_task_type(data_source, extra_info)
    if task_type == "binary_risk":
        res = compute_score_binary_risk(solution_str, str(ground_truth))
    elif task_type == "choice_reasoning":
        res = compute_score_choice_reasoning(solution_str, str(ground_truth))
    elif task_type == "masked_value":
        res = compute_score_masked_value(solution_str, str(ground_truth))
    elif task_type == "open_qa":
        res = compute_score_open_qa(solution_str, str(ground_truth))
    else:
        raise NotImplementedError(f"Reward function is not implemented for data_source={data_source!r}")

    if isinstance(res, dict):
        return res
    if isinstance(res, (int, float, bool)):
        return float(res)
    return float(res[0])


def normalize_task_type(data_source: Any, extra_info: Optional[Dict[str, Any]]) -> str:
    if isinstance(extra_info, dict):
        for key in ("task_type", "reward_type", "metric"):
            value = extra_info.get(key)
            if value:
                normalized = str(value).strip().lower()
                for task_type, aliases in TASK_ALIASES.items():
                    if normalized == task_type or normalized in aliases:
                        return task_type

    source = str(data_source or "").strip().lower()
    for task_type, aliases in TASK_ALIASES.items():
        if source == task_type or source in aliases:
            return task_type

    return source


def get_content_between_a_b(start_tag: str, end_tag: str, text: str) -> str:
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break
    return extracted_text.strip()


def extract(text: str, section_type: str, hard: bool = True) -> str:
    if text:
        target_str = get_content_between_a_b(f"<{section_type}>", f"</{section_type}>", text)
        if target_str:
            return target_str
        if hard:
            return text
        return "unk"
    return "unk"


def parse_response(response: str) -> str:
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response, "answer")

    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "answer in",
        "final answer",
        "final answer is",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    return response


def judge_multi_choice(answer: str, response: str, alphas: Any = None) -> int:
    del alphas
    response = parse_response(response.lower()).strip().lower()
    response = response.replace("\n", "")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()

    if len(split_response) > 300:
        return 0
    return 1 if split_response == answer else 0


def compute_score_choice_reasoning(solution_str: str, ground_truth: str) -> int:
    response = extract(solution_str, "answer")
    return judge_multi_choice(ground_truth, response)


def count_think_tags(text: str) -> Tuple[int, int]:
    return text.count("<think>"), text.count("</think>")


def extract_think_and_answer(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, solution_str, re.IGNORECASE | re.DOTALL)
    if not think_match:
        return None, solution_str

    think_content = think_match.group(1).strip()
    answer_content = solution_str[think_match.end() :].strip()
    return think_content, answer_content


def extract_prediction(response: str) -> str:
    if not response:
        return "unk"

    patterns = [
        r"\\\\?boxed\{(Yes|No|YES|NO|yes|no)\}",
        r"\\boxed\{(Yes|No|YES|NO|yes|no)\}",
        r"\\boxed\s*\{(Yes|No|YES|NO|yes|no)\}",
        r"\[boxed\{(Yes|No|YES|NO|yes|no)\}\]",
        r"<boxed>(Yes|No|YES|NO|yes|no)</boxed>",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            last_match = matches[-1].lower()
            if "yes" in last_match:
                return "yes"
            if "no" in last_match:
                return "no"

    response_lower = response.lower()
    yes_terms = ["yes", "high risk", "high", "positive"]
    no_terms = ["no", "low risk", "low", "negative"]
    has_yes = any(keyword in response_lower for keyword in yes_terms)
    has_no = any(keyword in response_lower for keyword in no_terms)

    recent_context = response_lower[-200:] if len(response) > 200 else response_lower
    recent_has_yes = any(keyword in recent_context for keyword in yes_terms)
    recent_has_no = any(keyword in recent_context for keyword in no_terms)

    if recent_has_yes and not recent_has_no:
        return "yes"
    if recent_has_no and not recent_has_yes:
        return "no"
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unk"


def compute_score_binary_risk(
    solution_str: str,
    ground_truth: str,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    if weights is None:
        weights = {
            "answer_accuracy": 0.8,
            "think_accuracy": 0.1,
            "think_answer_consistency": 0.1,
        }

    true_risk = ground_truth
    think_content, answer_content = extract_think_and_answer(solution_str)

    if not solution_str:
        return 0.0

    if not answer_content:
        answer_risk_accuracy = 0.0
    else:
        answer_prediction = extract_prediction(answer_content)
        if true_risk.lower() in answer_prediction:
            answer_risk_accuracy = 1.0
        elif "yes" in true_risk.lower() and true_risk.lower() not in answer_prediction:
            answer_risk_accuracy = -0.1
        else:
            answer_risk_accuracy = 0.0

    answer_accuracy_score = answer_risk_accuracy

    if not think_content:
        think_risk_accuracy = 0.0
    else:
        think_risk_accuracy = 1.0 if true_risk.lower() in extract_prediction(think_content) else 0.0

    think_accuracy_score = think_risk_accuracy
    think_answer_consistency_score = 1.0 if answer_risk_accuracy == think_risk_accuracy else 0.0

    total_score = (
        weights["answer_accuracy"] * answer_accuracy_score
        + weights["think_accuracy"] * think_accuracy_score
        + weights["think_answer_consistency"] * think_answer_consistency_score
    )
    return float(total_score)


def parse_date_to_float(text: str) -> Optional[float]:
    text = text.strip()
    try:
        match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
        if match:
            year, month, day = map(int, match.groups())
            return year + (month - 1) / 12 + (day - 1) / 365

        match = re.search(r"\b(\d{4})[-/](\d{1,2})\b", text)
        if match:
            year, month = map(int, match.groups())
            return year + (month - 1) / 12

        match = re.search(r"\b(\d{4})\b", text)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def calculate_iou(range_a: Tuple[float, float], range_b: Tuple[float, float]) -> float:
    start_a, end_a = sorted(range_a)
    start_b, end_b = sorted(range_b)

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    if intersection_end <= intersection_start:
        return 0.0

    intersection_len = intersection_end - intersection_start
    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)
    union_len = union_end - union_start
    return intersection_len / union_len if union_len > 0 else 0.0


def extract_value_hybrid(text: str) -> Tuple[Optional[Any], Optional[str]]:
    if not text:
        return None, None

    if "[Prediction]:" in text:
        text = text.split("[Prediction]:")[-1]
        if "[" in text:
            text = text.split("[")[0]
    text = text.strip()

    code_match = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", text)
    if code_match:
        return code_match.group(1).upper(), "code"

    date_range_pattern = (
        r"(\d{4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?)\s*(?:-|to|and)\s*"
        r"(\d{4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?)"
    )
    date_range_match = re.search(date_range_pattern, text)
    if date_range_match:
        value_1 = parse_date_to_float(date_range_match.group(1))
        value_2 = parse_date_to_float(date_range_match.group(2))
        if value_1 and value_2 and 1800 < value_1 < 2100 and 1800 < value_2 < 2100:
            return (min(value_1, value_2), max(value_1, value_2)), "range"

    date_match = re.search(r"\b\d{4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b", text)
    if date_match:
        value = parse_date_to_float(date_match.group(0))
        if value and 1800 < value < 2100:
            return value, "point"

    text_num_map = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "no": "0",
        "zero": "0",
    }
    text_lower = text.lower()
    for word, digit in text_num_map.items():
        text_lower = re.sub(r"\b" + word + r"\b", digit, text_lower)
    text = text_lower

    if date_range_match:
        text = text.replace(date_range_match.group(0), "")
    elif date_match:
        text = text.replace(date_match.group(0), "")

    text = text.replace(",", "").replace("%", "")
    text = text.replace("–", "-").replace("—", "-").replace("~", "-").replace("to", "-")
    text = re.sub(r"\b(years?|old|weeks?|days?|months?)\b", "", text)

    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|and)\s*(-?\d+(?:\.\d+)?)", text)
    if range_match:
        try:
            value_1 = float(range_match.group(1))
            value_2 = float(range_match.group(2))
            return (min(value_1, value_2), max(value_1, value_2)), "range"
        except Exception:
            pass

    matches = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if matches:
        return float(matches[0]), "point"

    return None, None


def compute_score_masked_value(solution_str: str, ground_truth: str) -> float:
    if not solution_str or not ground_truth:
        return 0.0

    pred_val, pred_type = extract_value_hybrid(solution_str)
    gt_val, gt_type = extract_value_hybrid(ground_truth)
    if pred_val is None or gt_val is None:
        return 0.0

    correct = False
    iou_threshold = 0.5

    if gt_type == "code" or pred_type == "code":
        pred_str = str(pred_val).upper()
        gt_str = str(gt_val).upper()
        if pred_str in gt_str or gt_str in pred_str or (
            len(pred_str) >= 3 and len(gt_str) >= 3 and pred_str[:3] == gt_str[:3]
        ):
            correct = True
    else:
        if gt_type == "range":
            gt_min, gt_max = gt_val
        elif gt_type == "point":
            value = float(gt_val)
            if value > 1800:
                tolerance = 0.5
                gt_min = value - tolerance
                gt_max = value + tolerance
            else:
                gt_min = value * 0.95
                gt_max = value * 1.05
                if gt_min > gt_max:
                    gt_min, gt_max = gt_max, gt_min
        else:
            return 0.0

        if pred_type == "point":
            if gt_min <= float(pred_val) <= gt_max:
                correct = True
        elif pred_type == "range":
            iou = calculate_iou((gt_min, gt_max), pred_val)
            if iou >= iou_threshold:
                correct = True

    return 1.0 if correct else 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return Counter(" ".join(ngram) for ngram in ngrams)


def _bleu_4_score(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        cand_counts = _ngram_counts(cand_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)

        clipped_count = 0
        for ngram, count in cand_counts.items():
            ref_count = ref_counts.get(ngram, 0)
            clipped_count += min(count, ref_count)

        total_cand_ngrams = max(len(cand_tokens) - n + 1, 1)
        precision = clipped_count / total_cand_ngrams
        precisions.append(precision)

    if min(precisions) == 0:
        geom_mean = 0.0
    else:
        geom_mean = math.exp(sum(math.log(p) for p in precisions) / 4)

    cand_len = len(cand_tokens)
    ref_len = len(ref_tokens)
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_len / max(cand_len, 1))

    return brevity_penalty * geom_mean


def _rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    rows, cols = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[rows][cols]
    precision = lcs / rows if rows > 0 else 0.0
    recall = lcs / cols if cols > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _f1_word_overlap(candidate: str, reference: str) -> float:
    cand_words = set(_tokenize(candidate))
    ref_words = set(_tokenize(reference))
    if not cand_words or not ref_words:
        return 0.0

    intersection = cand_words & ref_words
    precision = len(intersection) / len(cand_words)
    recall = len(intersection) / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _string_similarity_jaccard(candidate: str, reference: str) -> float:
    cand_chars = set(candidate.lower().strip())
    ref_chars = set(reference.lower().strip())
    if not cand_chars or not ref_chars:
        return 0.0
    intersection = cand_chars & ref_chars
    union = cand_chars | ref_chars
    return len(intersection) / len(union)


def compute_score_open_qa(solution_str: str, ground_truth: str) -> float:
    if not solution_str or not ground_truth:
        return 0.0

    answer = solution_str.strip()
    if "[Prediction]:" in answer:
        answer = answer.split("[Prediction]:")[-1].strip()
    if "[Answer]:" in answer:
        answer = answer.split("[Answer]:")[-1].strip()

    bleu4 = _bleu_4_score(answer, ground_truth)
    rouge_l = _rouge_l_f1(answer, ground_truth)
    f1_word = _f1_word_overlap(answer, ground_truth)
    jaccard = _string_similarity_jaccard(answer, ground_truth)

    bleu4_scaled = min(bleu4 * 300.0, 1.0)
    reward = (bleu4_scaled + rouge_l + f1_word + jaccard) / 4.0
    return max(0.0, min(1.0, reward))


# Backward-readable function names for users adapting the public template.
compute_score_risk_prediction = compute_score_binary_risk
compute_score_multiple_choice = compute_score_choice_reasoning
compute_score_structured_value = compute_score_masked_value
compute_score_free_text_qa = compute_score_open_qa
