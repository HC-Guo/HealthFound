from __future__ import annotations

import re
from typing import Any, Dict, Optional

REWARD_VERSION = "healthfound"

TASK_ALIASES = {
    "binary_risk": {"binary_risk", "risk_prediction", "diagnosis_prediction", "clinical_risk_prediction", "ukb_diagnosis"},
}


def normalize_task_type(data_source, extra_info=None):
    candidates = []
    if isinstance(extra_info, dict):
        candidates += [extra_info.get(k) for k in ("task_type", "reward_type", "metric")]
    candidates.append(data_source)
    for candidate in candidates:
        value = str(candidate or "").strip().lower()
        for task, aliases in TASK_ALIASES.items():
            if value in aliases:
                return task
    return str(data_source or "").strip().lower()


def compute_score(data_source: Any, solution_str: str, ground_truth: Any,
                  extra_info: Optional[Dict[str, Any]] = None,
                  reward_router_address=None, reward_model_tokenizer=None, **kwargs) -> float:
    """Drop-in veRL entrypoint for binary-risk reward."""
    del reward_router_address, reward_model_tokenizer, kwargs
    task = normalize_task_type(data_source, extra_info)
    if task == "binary_risk":
        return compute_score_binary_risk(solution_str, ground_truth)
    raise NotImplementedError(f"Unknown reward task {data_source!r}")


def extract_think_and_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None, None
    closes = list(re.finditer(r"</think\s*>", text, re.I))
    if not closes:
        return (None, None) if re.search(r"<think\b", text, re.I) else (None, text.strip())
    last = closes[-1]
    prefix, final = text[:last.start()], text[last.end():].strip()
    opens = list(re.finditer(r"<think\s*>", prefix, re.I))
    reasoning = prefix[opens[-1].end():] if opens else prefix
    if re.search(r"</?think\b", reasoning, re.I):
        reasoning = None
    if re.search(r"</?think\b", final, re.I):
        final = None
    return reasoning.strip() if reasoning else None, final or None


def _terminal_suffix(text):
    return not text.strip(" \t\r\n.$!\u3002\uff01")


def _extract_final_answer(text):
    _, text = extract_think_and_answer(text)
    if text is None:
        return None
    if re.search(r"</?answer\b", text, re.I):
        fields = list(re.finditer(r"<answer\s*>(.*?)</answer\s*>", text, re.I | re.S))
        if (len(fields) != 1 or len(re.findall(r"<answer\s*>", text, re.I)) != 1
                or len(re.findall(r"</answer\s*>", text, re.I)) != 1):
            return None
        field = fields[0]
        if not _terminal_suffix(text[field.end():]):
            return None
        text = field.group(1).strip()
    markers = list(re.finditer(
        r"\[(?:Prediction|Answer)\]\s*:|(?:^|\n)\s*(?:the\s+)?(?:final\s+answer|answer)\s*(?::|\bis\b)\s*", text, re.I))
    if markers:
        text = text[markers[-1].end():].strip()
        text = re.split(r"\n\s*\[[A-Za-z][^\]\n]*\]\s*:", text, maxsplit=1)[0].strip()
    starts = list(re.finditer(r"\\+boxed\b|<boxed\b|\[boxed\{", text, re.I))
    if starts:
        start = starts[-1]
        suffix = text[start.start():]
        if suffix.lower().startswith("<boxed"):
            field = re.fullmatch(r"<boxed\s*>(.*?)</boxed\s*>([\s.$!\u3002\uff01]*)", suffix, re.I | re.S)
            if not field:
                return None
            text = field.group(1).strip()
        else:
            opening = re.match(r"(?:\\+boxed\s*|\[boxed)\{", suffix, re.I)
            if not opening:
                return None
            depth, close = 1, None
            for i in range(opening.end(), len(suffix)):
                if suffix[i] == "{":
                    depth += 1
                elif suffix[i] == "}":
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
            if close is None:
                return None
            rest = suffix[close + 1:]
            if suffix.startswith("["):
                if not rest.startswith("]"):
                    return None
                rest = rest[1:]
            if not _terminal_suffix(rest):
                return None
            text = suffix[opening.end():close].strip()
    if re.search(r"</?think\b|</?answer\b", text, re.I):
        return None
    return text or None


def _binary_payload(text):
    if not isinstance(text, str):
        return None
    text = text.strip().strip("* \t\r\n").rstrip(".!\u3002\uff01").strip().casefold()
    aliases = {"yes": "yes", "high risk": "yes", "high-risk": "yes", "positive": "yes",
               "no": "no", "low risk": "no", "low-risk": "no", "negative": "no"}
    if text in aliases:
        return aliases[text]
    m = re.fullmatch(r"(?:the\s+)?(?:subject|patient)\s+is\s+at\s+(?:a\s+)?(high|low)\s+risk"
                     r"(?:\s+of\s+developing\s+([\w\s,'()\-/&]+))?", text)
    if m and not re.search(r"\b(?:yes|no|not|unknown|however|but|instead|actually|prediction|answer|conclusion)\b|\b(?:high|low)\s+risk\b", m.group(2) or ""):
        return "yes" if m.group(1) == "high" else "no"
    return None


def extract_prediction(response):
    return _binary_payload(_extract_final_answer(response)) or "unk"


def _reasoning_prediction(reasoning):
    if not reasoning:
        return None
    fields = list(re.finditer(r"<risk\s*>(.*?)</risk\s*>", reasoning, re.I | re.S))
    if fields:
        if not _terminal_suffix(reasoning[fields[-1].end():]):
            return None
        values = [_binary_payload(v.group(1)) for v in fields]
        return values[-1] if all(v is not None and v == values[-1] for v in values) else None
    direct = extract_prediction(reasoning)
    if direct != "unk":
        return direct
    markers = list(re.finditer(r"(?:^|[\n.!?]\s*)(?:overall\s+risk|risk\s+prediction|final\s+risk|risk|conclusion)\s*:\s*", reasoning, re.I))
    if markers:
        return _binary_payload(reasoning[markers[-1].end():])
    last = re.split(r"[\n.!?]+\s*", reasoning.strip().rstrip(".!?"))[-1]
    return _binary_payload(last)


def binary_risk_components(solution_str, ground_truth):
    target = extract_prediction(str(ground_truth))
    if target not in {"yes", "no"}:
        raise ValueError("Ground truth must contain an unambiguous binary risk label")
    reasoning, _ = extract_think_and_answer(solution_str)
    answer = extract_prediction(solution_str)
    answer = answer if answer in {"yes", "no"} else None
    thought = _reasoning_prediction(reasoning)
    accuracy = 1.0 if answer == target else (-0.1 if target == "yes" else 0.0)
    return dict(answer_accuracy=accuracy,
                think_accuracy=float(thought is not None and thought == target),
                think_answer_consistency=float(answer is not None and thought is not None and answer == thought))


def compute_score_binary_risk(solution_str, ground_truth, weights=None):
    if weights is None:
        weights = dict(answer_accuracy=0.8, think_accuracy=0.1, think_answer_consistency=0.1)
    parts = binary_risk_components(solution_str, ground_truth)
    return float(sum(weights[k] * parts[k] for k in parts))


compute_score_risk_prediction = compute_score_binary_risk
