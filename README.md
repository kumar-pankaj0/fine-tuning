---
license: cc0-1.0
task_categories:
  - text2text-generation
  - text-generation
  - automatic-speech-recognition
language:
  - en
tags:
  - asr-post-refinement
  - post-asr-correction
  - text-refinement
  - speech-to-text
  - whisper
  - generative-error-correction
  - llm-fine-tuning
size_categories:
  - 10K<n<100K
---

# Post-ASR Text Refinement & Correction Dataset

A benchmark dataset of **13,295 training samples** and **358 evaluation samples** engineered for training Large Language Models (LLMs) and policy models on **Post-ASR Text Refinement**.

In speech-to-text pipelines, Automatic Speech Recognition (ASR) engines produce raw transcripts with acoustic errors, homophone confusions, and phonetic boundary shifts. This dataset trains a downstream refinement model to take noisy speech output (`raw_asr`) and transform it into clean, domain-accurate ground truth (`refined_text`).

---

## The Post-ASR Refinement Task

```
[ Spoken Audio ] 
       │
       ▼
[ Speech-to-Text (ASR) ]  ──►  raw_asr (Noisy, phonetic transcription)
                                  │
                                  ▼
                       [ Post-Refinement Model (LLM) ]
                                  │
                                  ▼
                              refined_text (Clean, polished ground truth)
```

- **Input (`raw_asr`)**: Raw ASR transcript with realistic Whisper acoustic errors (25%–40% WER, multi-word boundary shifts, phonetic homophones, and misheard technical jargon).
- **Target (`refined_text`)**: Clean, verified, domain-correct text with 100% frozen ground truth.

---

## Why Existing Synthetic Datasets Fail at Refinement

Most synthetic datasets for post-ASR refinement use **unrealistic distortion shortcuts**:
- **Suffix Chopping**: Chopping letters off word endings (`"minut"`, `"automat"`, `"releas"`). Real ASR decoders output valid dictionary words.
- **Lazy Space-Splitting**: Merely inserting spaces into compound words (`"database"` -> `"data base"`).
- **Robotic Character Edits**: Uniform character flips that do not match spoken phonetics.

This dataset was calibrated to ensure **100% real English dictionary tokens**, **phonetic multi-word boundary shifts** (e.g., `asynchronous` -> `sink cronus`), and **recoverable acoustic noise**.

---

## Dataset Statistics & Verification

All samples pass strict programmatic audit gates with exit code 0:

| Metric | Target Specification | Observed Dataset Mean | Validation Status |
|---|---|---|---|
| **Training Set Size** | 13,000+ | **13,295 samples** | 100.0% Pass Rate |
| **Evaluation Set Size** | 300+ | **358 samples** | 100.0% Pass Rate |
| **Word Error Rate (WER)** | 25.0% – 40.0% | **32.3%** | Calibrated Sweet Spot |
| **Character Similarity** | Strictly <= 92.0% | **88.5%** | Enforces Phonetic Depth |
| **Multi-Word Boundary Shifts** | >= 15.0% | **25.8%** | Exceeds Target |
| **Per-Sentence Error Density** | >= 20.0% modified | **100% of Sentences** | Zero Pristine Sentences |
| **Mechanical Suffix Chopping** | 0.0% allowed | **0.00%** | Zero Non-Dictionary Tokens |
| **Ground Truth Integrity** | 100% Frozen | **0 Mutations** | 100% Verified Ground Truth |

---

## The 6 Acoustic Mechanisms for Post-ASR Refinement

1. **Multi-Word Boundary Shifts (25.8% of edits)**
   - Words split or merged phonetically:
     - `asynchronous` -> `sink cronus` / `a sink cronus`
     - `database` -> `day till base` / `day to base`
     - `pipeline` -> `pipe lane`
     - `throughput` -> `true put`
     - `timeout` -> `thyme out`
2. **Same-Sound Homophones (Phonetic Substitutions)**
   - `queue` -> `cue`
   - `write` -> `right`
   - `serial` -> `cereal`
   - `brake` -> `break`
3. **Technical Acronyms & Command Dictation**
   - `TLS` -> `tee el ess`
   - `CPU` -> `see pee you`
   - `dxc.exe` -> `d x c dot exe`
   - `--program` -> `dash dash program`
4. **Fast Speech Articulation Blurs**
   - `what if I wrote` -> `what reason I authored`
   - `going a little too fast` -> `going the little too fast`
5. **Domain Vocabulary Distortion**
   - `sysctl net.core.somaxconn` -> `net dot core dot socket quota`
   - `Transit Gateway` -> `transit gate way line`
6. **Lexical Semantic Drifts (Contextual Mishearings)**
   - `approved` -> `granted`
   - `traffic` -> `flow`
   - `deadlock` -> `lock up stall`

---

## Sample Preview: Post-ASR Refinement

### Example 1: Distributed Systems
- **Input (`raw_asr`)**:
  > Our engineering council granted NATS JetStream persistence towards the sink cronus messaging by replacing RabbitMQ quorum cues.
- **Output (`refined_text`)**:
  > Our engineering council approved NATS JetStream persistence for asynchronous messaging by replacing RabbitMQ quorum queues.

### Example 2: Kernel & Network Tuning
- **Input (`raw_asr`)**:
  > Profiling employing performance analysis tools exhibited which competing thread locks approximately sysctl net dot core dot socket quota triggered excessive scope switching plus cpu starvation.
- **Output (`refined_text`)**:
  > Profiling using performance analysis tools demonstrated that competing thread locks around sysctl net.core.somaxconn caused excessive context switching and cpu starvation.

---

## Quick Start (Loading the Data)

### From Kaggle
```python
import json

with open("/kaggle/input/asr-policy-model-dataset/sft_train.json", "r") as f:
    train_data = json.load(f)

with open("/kaggle/input/asr-policy-model-dataset/sft_eval.json", "r") as f:
    eval_data = json.load(f)

print(f"Loaded {len(train_data)} train and {len(eval_data)} eval samples.")
```

### From Hugging Face Datasets
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "https://raw.githubusercontent.com/kumar-pankaj0/datasets/main/policy_model_dataset/data/sft_train.json",
    "test": "https://raw.githubusercontent.com/kumar-pankaj0/datasets/main/policy_model_dataset/data/sft_eval.json"
})

print(dataset)
```

---

## Fine-Tuning a Post-ASR Refinement Model

This prompt template trains an LLM to refine raw speech transcripts into polished text:

```python
def format_refinement_prompt(sample):
    system_prompt = (
        "You are an expert Post-ASR Refinement engine. "
        "Your task is to refine the following raw speech-to-text transcript into clean, domain-accurate text. "
        "Restore misheard jargon, fix phonetic substitutions, and correct multi-word boundary shifts while preserving original meaning."
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{sample['raw_asr']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['refined_text']}<|im_end|>"
    )
```

---

## License & Attribution

This dataset is released under the **Creative Commons Zero v1.0 Universal (CC0-1.0)** public domain dedication. You are free to copy, modify, distribute, and train commercial models without restriction.
