import os
import logging
import warnings
import torch

# Suppress all noisy warnings and verbose logs from Unsloth, Transformers, and PyTorch
os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("unsloth").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

model_name = "pankaj0/llama-3.2-3b-asr-refiner-grpo-3-phase"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto" if torch.cuda.is_available() else None,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

SystemPrompt = (
    "You are an expert speech-to-text post-processor. "
    "Fix transcription misspellings, add correct casing and punctuation, "
    "and remove speech disfluencies (e.g., 'um', 'uh', false starts). "
    "Preserve the speaker's original meaning and exact phrasing. Do not summarize or rewrite."
)


def _format_batch_samples(batch):
    batch_samples = []
    raw_asr_list = []
    for item in batch["input"]:
        raw_asr = item["raw_asr"]
        messages = [
            {"role": "system", "content": SystemPrompt},
            {"role": "user", "content": f"Raw transcript:\n{raw_asr}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        batch_samples.append(prompt)
        raw_asr_list.append(raw_asr)

    return {"prompt": batch_samples, "raw_asr": raw_asr_list}


def format_dataset(dataset_path=None):
    if dataset_path is None:
        for candidate in [
            "./datasets/data_eval.json",
            ".log/datasets/data_eval.json",
            "/kaggle/working/datasets/data_eval.json",
        ]:
            if os.path.exists(candidate):
                dataset_path = candidate
                break
        if dataset_path is None:
            dataset_path = "./datasets/data_eval.json"

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    formated_dataset = dataset.map(_format_batch_samples, batched=True)
    return formated_dataset


def evaluate(num_samples: int = 10):
    eval_data = format_dataset()
    eval_data = eval_data.select(range(min(num_samples, len(eval_data))))
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting Evaluation ({len(eval_data)} samples) ===")
    for idx, input_sample in enumerate(eval_data, 1):
        prompt = input_sample["prompt"]
        raw_asr = input_sample["raw_asr"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            response = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(
            response[0][prompt_len:], skip_special_tokens=True
        ).strip()

        print(f"\n[Sample {idx}]")
        print(f"--- RAW ASR ---\n{raw_asr}\n")
        print(f"--- REFINED OUTPUT ---\n{output_text}")
        print("-" * 60)


if __name__ == "__main__":
    evaluate(num_samples=10)
