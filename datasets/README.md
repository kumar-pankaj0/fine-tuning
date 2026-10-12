---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
license: mit
task_categories:
- text-generation
language:
- en
tags:
- speech-to-text
- stt-cleanup
- sft
- dpo
size_categories:
- n<1K
---

# STT Post-Refining Dataset

Faithful cleanup dataset for Speech-to-Text (STT) outputs from low-resource models (e.g. Parakeet-TDT, Whisper) running on low RAM with background noise and acoustic degradation.

## Schema

Each entry in `train.jsonl` / `sft_post_refining.jsonl` contains:

| Field | Type | Description |
|---|---|---|
| `input` | string | Raw STT output with speech disfluencies, phonetic slips, and missing punctuation |
| `accept` | string | Ground-truth cleaned text adhering to Spec v1 (F4 formatting, faithful intent, no hallucinated facts) |
| `reject` | string | Hard negative output simulating typical model errors (flipped negation, invented cut-off endings, over-editing) |

## Usage

### Supervised Fine-Tuning (SFT)
Use `input` as prompt and `accept` as target completion:
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl", split="train")
def format_sft(sample):
    return {"prompt": sample["input"], "completion": sample["accept"]}
sft_dataset = dataset.map(format_sft)
```

### Direct Preference Optimization (DPO)
Map fields to TRL DPOTrainer standard (`prompt`, `chosen`, `rejected`):
```python
def format_dpo(sample):
    return {
        "prompt": sample["input"],
        "chosen": sample["accept"],
        "rejected": sample["reject"]
    }
dpo_dataset = dataset.map(format_dpo)
```
