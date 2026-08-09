# Data Format

The rejection-sampling script expects JSONL input. Each line should be one JSON object.

Minimal fields:

```json
{
  "instruction": "<task instruction>",
  "input": "<case context>",
  "output": "<ground-truth label or answer>"
}
```

Optional identifier fields:

```json
{
  "id": "<stable record id>"
}
```

For binary risk-prediction tasks, the default parser treats ground-truth text containing `high risk` or `positive` as positive, and text containing `low risk` or `negative` as negative. If your task uses a different answer space, update `extract_true_label` and `extract_prediction` in `stage_2_rejection_sampling_finetuning/rejection_sampling_filter.py`.

