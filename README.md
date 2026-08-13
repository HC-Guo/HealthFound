# HealthFound Open-Source Training Pipeline

This directory contains a de-identified, three-stage training pipeline template for HealthFound-style model development.

The original internal scripts were converted into public-safe templates. Concrete local paths, private dataset names, private output names, API tokens, and internal cache directories are intentionally removed. Before running, replace placeholders or export the required environment variables in your own environment.
## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LingranSong/HealthFound-32B-v7-rft-rl-360"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to HealthFound."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```


## Directory Layout

```text
HealthFound-opensource/
  stage_1_sft/
    LLaMA-Factory/
    run_sft.sh
    sft_config.yaml
  stage_2_rejection_sampling_finetuning/
    LLaMA-Factory/
    rejection_sampling_filter.py
    run_rejection_sampling.sh
    run_rft.sh
    rft_config.yaml
  stage_3_rl/
    verl/
    reward.py
    run_rl_grpo.sh
  examples/
    data_format.md
```

## Stage 1: Supervised Fine-Tuning

Purpose: teach the base model the target instruction format, clinical reasoning style, biomedical QA behavior, and structured prediction format.

Prepare a LLaMA-Factory training config from:

```bash
cp stage_1_sft/sft_config.yaml.template /your/config/location/sft_config.yaml
```

Then edit all placeholder values in the copied config, including the base model, dataset mixture, output directory, and distributed training settings.

Run:

```bash
export REPO_DIR="<LLAMA_FACTORY_REPO_DIR>"
export TRAIN_ENTRYPOINT="<LLAMA_FACTORY_TRAIN_ENTRYPOINT>"
export CONFIG_PATH="<SFT_CONFIG_PATH>"
export CONDA_ENV_NAME="<OPTIONAL_CONDA_ENV>"
export GPUS_PER_NODE=8
export MASTER_PORT=29500
bash stage_1_sft/run_sft.sh
```

`REPO_DIR` and `TRAIN_ENTRYPOINT` are optional if you use the GitHub clone included under `stage_1_sft/LLaMA-Factory`.

## Stage 2: Rejection-Sampling Fine-Tuning

Purpose: generate multiple responses from the Stage-1 model, retain correct and sufficiently diverse responses, then continue SFT on the accepted samples.

First run rejection sampling:

```bash
export MODEL_PATH="<STAGE_1_MODEL_OR_CHECKPOINT>"
export INPUT_JSONL="<RFT_CANDIDATE_INPUT_JSONL>"
export OUTPUT_JSONL="<ACCEPTED_RFT_OUTPUT_JSONL>"
export TENSOR_PARALLEL_SIZE=1
export NUM_GENERATIONS=10
bash stage_2_rejection_sampling_finetuning/run_rejection_sampling.sh
```

Expected input JSONL fields are described in [examples/data_format.md](examples/data_format.md).

Then prepare the RFT config:

```bash
cp stage_2_rejection_sampling_finetuning/rft_config.yaml /your/config/location/rft_config.yaml
```

Run RFT:

```bash
export REPO_DIR="<LLAMA_FACTORY_REPO_DIR>"
export TRAIN_ENTRYPOINT="<LLAMA_FACTORY_TRAIN_ENTRYPOINT>"
export CONFIG_PATH="<RFT_CONFIG_PATH>"
export GPUS_PER_NODE=8
export MASTER_PORT=29501
bash stage_2_rejection_sampling_finetuning/run_rft.sh
```

`REPO_DIR` and `TRAIN_ENTRYPOINT` are optional if you use the GitHub clone included under `stage_2_rejection_sampling_finetuning/LLaMA-Factory`.

## Stage 3: Reinforcement Learning

Purpose: improve final answer quality with rule-based or task-specific rewards after SFT and RFT.

The RL script is intentionally driven entirely by environment variables. At minimum, provide:

```bash
export VERL_REPO_DIR="<VERL_REPO_DIR>"
export MODEL_PATH="<STAGE_2_MODEL_OR_CHECKPOINT>"
export REWARD_FUNCTION_PATH="<REWARD_FUNCTION_PY>"
export REWARD_FUNCTION_NAME="compute_score"
export TRAIN_FILES_JSON="<PYTHON_LIST_OR_JSON_LIST_OF_TRAIN_FILES>"
export VAL_FILES_JSON="<PYTHON_LIST_OR_JSON_LIST_OF_VALIDATION_FILES>"
export OUTPUT_DIR="<RL_CHECKPOINT_OUTPUT_DIR>"
export PROJECT_NAME="<PUBLIC_PROJECT_NAME>"
export EXPERIMENT_NAME="<PUBLIC_EXPERIMENT_NAME>"
export GPUS_PER_NODE=8
export MASTER_PORT=6379
bash stage_3_rl/run_rl_grpo.sh
```

`VERL_REPO_DIR`, `REWARD_FUNCTION_PATH`, and `REWARD_FUNCTION_NAME` are optional if you use the GitHub clone and reward template included under `stage_3_rl/`.

`TRAIN_FILES_JSON` and `VAL_FILES_JSON` should be list strings accepted by the downstream veRL command, for example a Python-style list string. Do not commit private data locations to this repository.

The included [stage_3_rl/reward.py](stage_3_rl/reward.py) keeps the original reward structure while replacing internal `data_source` names with generic task aliases: `binary_risk`, `choice_reasoning`, `masked_value`, and `open_qa`. If your released data uses different labels, set `extra_info["task_type"]` or extend `TASK_ALIASES`.

## Privacy Checklist Before Release

Before publishing or sharing this folder, run:

```bash
grep -R "<PRIVATE_PATTERN>" HealthFound-opensource
grep -R "/absolute/private/path" HealthFound-opensource
grep -R "<CREDENTIAL_PATTERN>" HealthFound-opensource
```

Also inspect generated configs and logs. The templates here do not contain internal dataset names or private file paths, but user-created configs may.

## Notes on 32B Training

The original workflow used full-parameter training for a 32B model. For smaller ablation models, reduce per-device batch size, gradient accumulation, rollout count, tensor parallel size, and total data volume according to available GPU memory. Keep the stage order unchanged:

```text
base model -> SFT -> rejection sampling -> RFT -> RL
```
