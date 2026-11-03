from __future__ import annotations

import calendar
import math
import re
from collections import Counter, namedtuple
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

REWARD_VERSION = "healthfound-paper-s6-v2-20261103"
NUMERIC_RTOL = Decimal("0.05")
DATE_TOLERANCE_YEARS = Decimal("0.5")
INTERVAL_IOU_THRESHOLD = Decimal("0.5")
BLEU4_SCALE = 300.0

TASK_ALIASES = {
    "binary_risk": {"binary_risk", "risk_prediction", "diagnosis_prediction", "clinical_risk_prediction", "ukb_diagnosis"},
    "choice_reasoning": {"choice_reasoning", "medical_choice_reasoning", "general_choice_reasoning", "multiple_choice", "medical_reasoning", "natural_reasoning"},
    "masked_value": {"masked_value", "fill_in_the_mask", "structured_value_prediction", "numeric_or_code_prediction", "fim"},
    "open_qa": {"open_qa", "scientific_reasoning_qa", "free_text_qa", "scientificqa"},
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
    """Drop-in veRL entrypoint. Ground-truth/schema errors raise ValueError."""
    del reward_router_address, reward_model_tokenizer, kwargs
    task = normalize_task_type(data_source, extra_info)
    if task == "binary_risk":
        return compute_score_binary_risk(solution_str, ground_truth)
    if task == "choice_reasoning":
        return float(compute_score_choice_reasoning(solution_str, ground_truth))
    if task == "masked_value":
        return compute_score_masked_value(solution_str, ground_truth, extra_info=extra_info)
    if task == "open_qa":
        return compute_score_open_qa(solution_str, ground_truth)
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
    return not text.strip(" \t\r\n.$!。！")


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
            field = re.fullmatch(r"<boxed\s*>(.*?)</boxed\s*>([\s.$!。！]*)", suffix, re.I | re.S)
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
                        close = i; break
            if close is None:
                return None
            rest = suffix[close+1:]
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
    text = text.strip().strip("* \t\r\n").rstrip(".!。！").strip().casefold()
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
    return float(sum(weights[k]*parts[k] for k in parts))


def _choice(text):
    text = _extract_final_answer(text)
    if text is None:
        return None
    text = text.strip().strip("*")
    m = re.fullmatch(r"\s*(?:(?:option|choice)\s*)?[\[(]?\s*([A-Z]|\d{1,2}|yes|no|maybe|true|false)\s*[\])]?\s*[.!。]?\s*", text, re.I)
    return m.group(1).upper() if m else None


def judge_multi_choice(answer, response, alphas=None):
    del alphas
    target = _choice(str(answer))
    if target is None:
        raise ValueError("Multiple-choice ground truth must be one normalized option label")
    return int(_choice(response) == target)


def compute_score_choice_reasoning(solution_str, ground_truth):
    return judge_multi_choice(ground_truth, solution_str)


_Value = namedtuple("_Value", "kind value unit")
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ISO_DATE = r"\d{4}(?:[-/]\d{1,2})(?:[-/]\d{1,2})?"
_WORDS = dict(zero="0", no="0", one="1", two="2", three="3", four="4", five="5", six="6", seven="7", eight="8", nine="9", ten="10")
_UNITS = {"%", "mmol/l", "mmol/mol", "umol/l", "mg/dl", "g/dl", "g/l", "ng/ml", "pg/ml",
          "ug/l", "nmol/l", "pmol/l", "u/l", "iu/l", "miu/l", "kg", "g", "cm", "mm", "m",
          "kg/m2", "kg/m^2", "mmhg", "bpm", "year", "years", "day", "days", "week", "weeks",
          "month", "months", "10^9/l", "10^12/l", "x10^9/l", "x10^12/l", "ml/min/1.73m2"}


def _unit(text):
    return re.sub(r"\s+", "", text).casefold().replace("μ", "u").replace("µ", "u").replace("²", "2")


def _numeric(text, expected_unit=None):
    text = text.strip().replace("−", "-")
    text = re.sub(r"^(?:approximately|approx\.?|about)\s+", "", text, flags=re.I)
    text = _WORDS.get(text.casefold(), text)
    m = re.fullmatch(r"("+_NUMBER+r")\s*(.*?)", text)
    if not m:
        return None
    unit = _unit(m.group(2).rstrip("."))
    expected = _unit(expected_unit) if expected_unit else None
    if unit and unit not in _UNITS and unit != expected:
        return None
    if expected and unit and unit != expected:
        return None
    try:
        number = Decimal(m.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return _Value("number", number, unit or expected) if number.is_finite() else None


def _date_value(text):
    text = text.strip()
    m = re.fullmatch(r"(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?", text)
    if m:
        year, month, day = int(m[1]), int(m[2] or 1), int(m[3] or 1)
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        return Decimal(year)+Decimal((parsed-date(year, 1, 1)).days)/Decimal(366 if calendar.isleap(year) else 365)
    m = re.fullmatch(r"\d{4}\.\d+", text)
    if m:
        value = Decimal(text)
        return value if 1 <= value < 10000 else None
    return None


def parse_date_to_float(text):
    value = _date_value(text)
    return float(value) if value is not None else None


def _interval(text, date_mode=False, expected_unit=None):
    if any(c in text for c in "[]()"):
        text = text.strip()
        if not text.startswith(("[", "(")):
            return None
        close = "]" if text[0] == "[" else ")"
        position = text.find(close)
        if position < 0 or any(c in text[1:position]+text[position+1:] for c in "[]()"):
            return None
    if date_mode:
        atom = r"(?:"+_ISO_DATE+r"|\d{4}(?:\.\d+)?)"
        m = re.fullmatch(r"\s*[\[(]?\s*("+atom+r")\s*(?:to|and|through|\s[-–—]\s|,)\s*("+atom+r")\s*[\])]?\s*", text, re.I)
        if not m:
            m = re.fullmatch(r"\s*(\d{4})\s*[-–—]\s*(\d{4})\s*", text)
        if not m:
            return None
        a, b = _date_value(m[1]), _date_value(m[2])
        if a is None or b is None:
            return None
        return _Value("date_interval", tuple(sorted((a, b))), None)
    atom = _NUMBER
    m = re.fullmatch(r"\s*[\[(]?\s*("+atom+r")\s*(?:to|and|[-–—~]|,)\s*("+atom+r")\s*[\])]?\s*(.*?)", text.replace("−", "-"), re.I)
    if not m:
        return None
    unit = m[3].strip()
    a, b = _numeric(m[1]+unit, expected_unit), _numeric(m[2]+unit, expected_unit)
    if a is None or b is None:
        return None
    return _Value("interval", tuple(sorted((a.value, b.value))), a.unit)


def _icd(text):
    text = re.sub(r"^ICD(?:-?10)?(?:\s+code)?\s*:\s*", "", text.strip(), flags=re.I)
    if re.fullmatch(r"[A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?", text, re.I):
        return _Value("code", text.upper(), None)
    return None


def _type(value):
    aliases = {"number": "number", "numeric": "number", "point": "number", "float": "number",
               "date": "date", "year": "date", "code": "code", "icd": "code", "icd10": "code",
               "interval": "interval", "range": "interval", "numeric_interval": "interval",
               "date_interval": "date_interval", "date_range": "date_interval"}
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported FIM target_type {value!r}; S6 defines numeric/date/interval/ICD rewards")
    return aliases[key]


def _parse_value(text, kind=None, expected_unit=None):
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if kind == "code":
        return _icd(text)
    if kind in {"date", "date_interval"}:
        point = _date_value(text)
        return _Value("date", point, None) if point is not None else _interval(text, True)
    if kind in {"number", "interval"}:
        return _numeric(text, expected_unit) or _interval(text, False, expected_unit)
    code = _icd(text)
    if code:
        return code
    if re.fullmatch(_ISO_DATE, text):
        value = _date_value(text)
        return _Value("date", value, None) if value is not None else None
    if re.search(_ISO_DATE+r"\s+(?:to|and|through)\s+"+_ISO_DATE, text, re.I):
        return _interval(text, True)
    return _numeric(text, expected_unit) or _interval(text, False, expected_unit)


def extract_value_hybrid(text, target_type=None):
    """Compatibility helper, now requiring a complete scalar/interval payload."""
    payload = _extract_final_answer(text)
    parsed = _parse_value(payload, _type(target_type)) if payload else None
    if parsed is None:
        return None, None
    kind = {"number": "point", "interval": "range", "date_interval": "range"}.get(parsed.kind, parsed.kind)
    value = tuple(map(float, parsed.value)) if isinstance(parsed.value, tuple) else float(parsed.value) if isinstance(parsed.value, Decimal) else parsed.value
    return value, kind


def _iou(a, b):
    a, b = sorted(a), sorted(b)
    union = max(a[1], b[1])-min(a[0], b[0])
    if union == 0:
        return Decimal(int(a == b))
    intersection = max(Decimal(0), min(a[1], b[1])-max(a[0], b[0]))
    return intersection/union


def calculate_iou(range_a, range_b):
    return float(_iou(tuple(Decimal(str(x)) for x in range_a), tuple(Decimal(str(x)) for x in range_b)))


def compute_score_masked_value(solution_str, ground_truth, extra_info=None, target_type=None):
    metadata = extra_info if isinstance(extra_info, dict) else {}
    unit = metadata.get("unit")
    if isinstance(ground_truth, dict):
        target_type = target_type or ground_truth.get("target_type") or ground_truth.get("type")
        unit = ground_truth.get("unit", unit)
        ground_truth = ground_truth.get("value")
    kind = _type(target_type or metadata.get("target_type") or metadata.get("value_type"))
    target_text = _extract_final_answer(str(ground_truth)) if ground_truth is not None else None
    target = _parse_value(target_text, kind, unit) if target_text else None
    if target is None:
        raise ValueError("Unparseable FIM ground truth; provide a scalar/interval/ICD and explicit target_type for ambiguous dates")
    if kind in {"interval", "date_interval"} and not isinstance(target.value, tuple):
        raise ValueError("Interval target_type requires two ground-truth endpoints")
    answer = _extract_final_answer(solution_str)
    pred = _parse_value(answer, target.kind, unit) if answer else None
    if pred is None:
        return 0.0
    if target.unit and pred.unit and target.unit != pred.unit:
        return 0.0
    if target.kind == "code":
        a, b = pred.value, target.value
        return float(a in b or b in a or a[:3] == b[:3])
    if isinstance(target.value, tuple):
        lower, upper = target.value
    else:
        tolerance = DATE_TOLERANCE_YEARS if target.kind == "date" else abs(target.value)*NUMERIC_RTOL
        lower, upper = target.value-tolerance, target.value+tolerance
    if isinstance(pred.value, tuple):
        return float(_iou((lower, upper), pred.value) >= INTERVAL_IOU_THRESHOLD)
    return float(lower <= pred.value <= upper)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _bleu_4_score(candidate, reference):
    cand, ref = _tokenize(candidate), _tokenize(reference)
    if not cand or not ref:
        return 0.0
    precision = []
    for n in range(1, 5):
        cc = Counter(tuple(cand[i:i+n]) for i in range(max(0, len(cand)-n+1)))
        rc = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
        clipped = sum(min(v, rc[k]) for k, v in cc.items())
        precision.append(clipped/max(1, len(cand)-n+1))
    if not min(precision):
        return 0.0
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1-len(ref)/len(cand))
    return brevity*math.exp(sum(math.log(p) for p in precision)/4)


def _rouge_l_f1(candidate, reference):
    a, b = _tokenize(candidate), _tokenize(reference)
    if not a or not b:
        return 0.0
    previous = [0]*(len(b)+1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j-1]+1 if x == y else max(previous[j], current[-1]))
        previous = current
    return 2*previous[-1]/(len(a)+len(b))


def _f1_word_overlap(candidate, reference):
    a, b = set(candidate.lower().split()), set(reference.lower().split())
    return 2*len(a & b)/(len(a)+len(b)) if a and b else 0.0


def _string_similarity_jaccard(candidate, reference):
    a, b = set(candidate.lower().strip()), set(reference.lower().strip())
    return len(a & b)/len(a | b) if a and b else 0.0


def open_qa_components(solution_str, ground_truth):
    target = _extract_final_answer(str(ground_truth))
    if target is None:
        raise ValueError("Open-QA ground truth must contain a nonempty final answer")
    answer = _extract_final_answer(solution_str)
    if answer is None:
        return dict(bleu4_scaled=0.0, rouge_l=0.0, word_f1=0.0, character_jaccard=0.0)
    return dict(bleu4_scaled=min(BLEU4_SCALE*_bleu_4_score(answer, target), 1.0),
                rouge_l=_rouge_l_f1(answer, target), word_f1=_f1_word_overlap(answer, target),
                character_jaccard=_string_similarity_jaccard(answer, target))


def compute_score_open_qa(solution_str, ground_truth):
    parts = open_qa_components(solution_str, ground_truth)
    return float(max(0.0, min(1.0, sum(parts.values())/4)))


compute_score_risk_prediction = compute_score_binary_risk
compute_score_multiple_choice = compute_score_choice_reasoning
compute_score_structured_value = compute_score_masked_value
compute_score_free_text_qa = compute_score_open_qa
