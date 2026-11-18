
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RetentionDataset(Dataset):
    def __init__(self, files, tokenizer, config):
        if isinstance(files, str):
            files = [files]
        self.files = [str(Path(path).expanduser()) for path in files]
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_length = int(config.max_prompt_length)
        self.response_length = int(config.max_response_length)
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Anchor tokenizer must define pad_token_id and eos_token_id")
        self.records = []
        for path in self.files:
            if Path(path).suffix.lower() == ".parquet":
                import pandas as pd

                self.records.extend(pd.read_parquet(path).to_dict(orient="records"))
            elif Path(path).suffix.lower() == ".jsonl":
                with open(path, encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid anchor JSON at {path}:{line_number}") from exc
                        if not isinstance(record, dict):
                            raise ValueError(f"Anchor record at {path}:{line_number} must be an object")
                        self.records.append(record)
            else:
                raise ValueError("Anchor files must be local JSONL or Parquet files")
        if not self.records:
            raise ValueError("The supervised anchor corpus is empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        prompt_key = self.config.get("prompt_key", "prompt")
        response_key = self.config.get("response_key", "response")
        if prompt_key not in record or response_key not in record:
            raise ValueError(f"Anchor record {index} must contain configured prompt and response fields")
        prompt, response = record[prompt_key], record[response_key]
        instruction_key = self.config.get("instruction_key")
        if instruction_key:
            instruction = record.get(instruction_key)
            if not isinstance(instruction, str) or not isinstance(prompt, str):
                raise ValueError("Alpaca anchor instruction and input fields must be strings")
            prompt = "\n\n".join(part for part in (instruction, prompt) if part)
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError(f"Anchor prompt {index} is empty")
            messages = [{"role": "user", "content": prompt}]
        else:
            # Parquet may materialize list-valued cells as numpy arrays.
            messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
            if not isinstance(messages, list) or not messages:
                raise ValueError("Anchor prompt must be text or a nonempty chat message list")
            for message in messages:
                if (not isinstance(message, Mapping) or message.get("role") not in {"system", "user", "assistant"}
                        or not isinstance(message.get("content"), str)):
                    raise ValueError("Anchor chat messages must have a text content and system/user/assistant role")
            if messages[-1]["role"] != "user":
                raise ValueError("Anchor chat prompt must end with a user message")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Anchor response {index} must be a nonempty trusted target string")
        template_kwargs = dict(self.config.get("apply_chat_template_kwargs", {}))
        if {"tokenize", "add_generation_prompt", "return_tensors", "return_dict"} & template_kwargs.keys():
            raise ValueError("Anchor template kwargs cannot override tokenization or generation-prefix arguments")
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **template_kwargs
        )
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if not response_ids or response_ids[-1] != self.tokenizer.eos_token_id:
            response_ids.append(self.tokenizer.eos_token_id)
        if not prompt_ids or len(prompt_ids) > self.prompt_length or len(response_ids) > self.response_length:
            raise ValueError(
                f"Anchor record {index} exceeds configured prompt/response lengths or has an empty tokenized prompt; "
                "increase the lengths or preprocess targets explicitly (no silent target truncation)"
            )
        prompt_padding = self.prompt_length - len(prompt_ids)
        response_padding = self.response_length - len(response_ids)
        input_ids = ([self.tokenizer.pad_token_id] * prompt_padding + prompt_ids + response_ids
                     + [self.tokenizer.pad_token_id] * response_padding)
        attention_mask = torch.tensor(
            [0] * prompt_padding + [1] * (len(prompt_ids) + len(response_ids)) + [0] * response_padding,
            dtype=torch.long,
        )
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": torch.tensor(
                response_ids + [self.tokenizer.pad_token_id] * response_padding, dtype=torch.long
            ),
            "response_mask": torch.tensor([1] * len(response_ids) + [0] * response_padding, dtype=torch.long),
        }


class RetentionBatchStream:
    """Draw a fresh global anchor minibatch per optimizer update, with exact resume."""

    def __init__(self, tokenizer, config):
        from omegaconf import OmegaConf
        from torch.utils.data import SequentialSampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        from torchdata.stateful_dataloader.sampler import RandomSampler

        self.dataset = RetentionDataset(config.train_files, tokenizer, config)
        self.batch_size = config.batch_size
        if len(self.dataset) < self.batch_size:
            raise ValueError("Anchor corpus must contain at least one complete global retention batch")
        seed = config.get("seed", 42)
        sampler_generator = torch.Generator().manual_seed(seed)
        sampler = (RandomSampler(self.dataset, generator=sampler_generator)
                   if config.get("shuffle", True) else SequentialSampler(self.dataset))
        # Keep the sampler RNG separate from the loader's base-seed RNG.
        # Otherwise recreating the loader iterator on resume consumes an extra
        # sampler draw and changes permutations after the next corpus wrap.
        self.loader = StatefulDataLoader(
            self.dataset, batch_size=self.batch_size, sampler=sampler,
            drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(seed),
        )
        self.signature = {"config": OmegaConf.to_container(config, resolve=True), "records": len(self.dataset)}
        self.iterator = None

    def next_batch(self, num_updates):
        from verl import DataProto

        if num_updates < 1:
            raise ValueError("At least one optimizer update is required for an anchor batch")
        batches = []
        for _ in range(num_updates):
            if self.iterator is None:
                self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            batches.append(batch)
        tensors = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}
        return DataProto.from_dict(tensors=tensors)

    def state_dict(self):
        return {"version": 1, "signature": self.signature, "loader": self.loader.state_dict()}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("signature") != self.signature:
            raise ValueError("Anchor checkpoint does not match the current corpus/configuration")
        self.loader.load_state_dict(state["loader"])
        self.iterator = None
