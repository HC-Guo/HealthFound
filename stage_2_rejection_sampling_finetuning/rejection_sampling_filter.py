#!/usr/bin/env python3
"""Rejection-sampling data builder for instruction-tuning records.

The script generates multiple completions for each input sample, keeps completions
whose parsed prediction matches the ground truth, and optionally filters near-
duplicate completions with a simple Jaccard similarity check.

No dataset-specific paths or private names are embedded in this file. Adapt the
label and prediction parsers if your task is not binary risk prediction.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from vllm import LLM, SamplingParams


def extract_prediction(response: str) -> Optional[int]:
    """Parse a binary prediction from a model response.

    Returns:
        1 for positive or high risk, 0 for negative or low risk, None if parsing
        fails.
    """
    if not response:
        return None
    if not isinstance(response, str):
        response = str(response)

    patterns = [
        r"\\?boxed\{(Yes|No|YES|NO|yes|no)\}",
        r"\\boxed\s*\{(Yes|No|YES|NO|yes|no)\}",
        r"\[boxed\{(Yes|No|YES|NO|yes|no)\}\]",
        r"<boxed>(Yes|No|YES|NO|yes|no)</boxed>",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            value = matches[-1].lower()
            if "yes" in value:
                return 1
            if "no" in value:
                return 0
    return None


def extract_true_label(output_text: str) -> Optional[int]:
    """Parse the ground-truth label from the original output field."""
    if not output_text:
        return None
    text = output_text.lower()
    if "high risk" in text or "positive" in text:
        return 1
    if "low risk" in text or "negative" in text:
        return 0
    return None


def build_prompt(sample: Dict) -> str:
    """Build the generation prompt from an instruction-format sample."""
    suffix = """Task: Evaluate the risk based on the context.
Rules:
1. Analyze step by step inside <think> tags.
2. Immediately after </think>, output \\boxed{Yes} or \\boxed{No}.
3. Use Yes for high risk or positive, and No for low risk or negative.
4. Your output must end with the boxed answer.

Provide your analysis:"""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}\n{suffix}"
    if instruction:
        return f"{instruction}\n{suffix}"
    return f"{input_text}\n{suffix}"


def get_sample_id(sample: Dict, index: int) -> str:
    """Return a stable sample id."""
    for key in ("id", "record_id", "uid"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return f"idx_{index}"


def load_checkpoint(checkpoint_path: Path) -> Set[str]:
    """Load processed sample ids from a checkpoint file."""
    if not checkpoint_path.is_file():
        return set()
    try:
        with checkpoint_path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return set(data.get("processed_ids", []))
    except (json.JSONDecodeError, KeyError):
        print(f"Warning: invalid checkpoint file {checkpoint_path}; restarting.")
        return set()


def save_checkpoint(checkpoint_path: Path, processed_ids: Set[str]) -> None:
    """Save processed sample ids to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as file_obj:
        json.dump({"processed_ids": sorted(processed_ids)}, file_obj, indent=2)


def compute_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity."""
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 and not tokens2:
        return 1.0
    union = tokens1 | tokens2
    if not union:
        return 0.0
    return len(tokens1 & tokens2) / len(union)


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL samples and attach internal sample ids."""
    samples = []
    with path.open("r", encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {index + 1}: {exc}")
                continue
            sample["_sample_id"] = get_sample_id(sample, index)
            samples.append(sample)
    return samples


def select_diverse_texts(texts: List[str], threshold: float, limit: Optional[int]) -> List[str]:
    """Keep correct generations while removing near-duplicates."""
    selected = []
    for text in texts:
        if all(compute_similarity(text, kept) <= threshold for kept in selected):
            selected.append(text)
            if limit is not None and len(selected) >= limit:
                break
    if not selected and texts:
        selected.append(texts[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RFT data with rejection sampling.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face id.")
    parser.add_argument("--n", type=int, help="Generations per sample.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, help="Maximum response tokens.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--overwrite", action="store_true", help="Restart from scratch.")
    parser.add_argument("--diversity_threshold", type=float, default=0.8, help="Similarity cutoff.")
    parser.add_argument("--max_correct_per_sample", type=int, default=None, help="Max kept samples per input.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.json")

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    if output_path.exists() and not checkpoint_path.exists():
        raise RuntimeError(
            f"Output file exists but checkpoint is missing: {output_path}. "
            "Use --overwrite or provide the matching checkpoint."
        )

    processed_ids = load_checkpoint(checkpoint_path)
    samples = read_jsonl(input_path)
    remaining = [sample for sample in samples if sample["_sample_id"] not in processed_ids]

    if not remaining:
        print("No remaining samples to process.")
        return

    prompts = [build_prompt(sample) for sample in remaining]
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    newly_processed = set()

    with output_path.open("a", encoding="utf-8") as out_file:
        for sample, request_output in zip(remaining, outputs):
            sample_id = sample["_sample_id"]
            true_label = extract_true_label(sample.get("output", ""))
            if true_label is None:
                print(f"Warning: sample {sample_id} has no parseable label; skipped.")
                continue

            correct_texts = []
            for choice in request_output.outputs:
                generated_text = choice.text.strip()
                if extract_prediction(generated_text) == true_label:
                    correct_texts.append(generated_text)

            if not correct_texts:
                print(f"Sample {sample_id}: no correct generations found.")
                continue

            selected_texts = select_diverse_texts(
                correct_texts,
                threshold=args.diversity_threshold,
                limit=args.max_correct_per_sample,
            )

            for generated_text in selected_texts:
                new_sample = dict(sample)
                new_sample["ground_truth"] = sample.get("output", "")
                new_sample["output"] = generated_text
                new_sample.pop("_sample_id", None)
                out_file.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                total_written += 1

            newly_processed.add(sample_id)
            save_checkpoint(checkpoint_path, processed_ids | newly_processed)
            out_file.flush()
            os.fsync(out_file.fileno())
            print(f"Sample {sample_id}: kept {len(selected_texts)} correct generations.")

    print(f"Done. Wrote {total_written} accepted generations.")
    print(f"Completed {len(processed_ids | newly_processed)} / {len(samples)} samples.")


if __name__ == "__main__":
    main()

