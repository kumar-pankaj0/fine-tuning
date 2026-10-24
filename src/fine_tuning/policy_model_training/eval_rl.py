import os
import json
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
from jiwer import wer, cer
from bert_score import score as bert_score

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
    refined_text_list = []
    for item in batch["input"]:
        raw_asr = item["raw_asr"]
        refined_text = item["refined_text"]
        messages = [
            {"role": "system", "content": SystemPrompt},
            {"role": "user", "content": f"Raw transcript:\n{raw_asr}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        batch_samples.append(prompt)
        raw_asr_list.append(raw_asr)
        refined_text_list.append(refined_text)

    return {
        "prompt": batch_samples,
        "raw_asr": raw_asr_list,
        "refined_text": refined_text_list,
    }


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


def extract_refined_output(generated_text: str) -> str:
    """Safely extracts refined_output from JSON format, or falls back to raw text."""
    try:
        parsed = json.loads(generated_text)
        if isinstance(parsed, dict) and "refined_output" in parsed:
            return str(parsed["refined_output"]).strip()
    except Exception:
        pass
    return generated_text.strip()


def evaluate(num_samples: int = 10):
    eval_data = format_dataset()
    eval_data = eval_data.select(range(min(num_samples, len(eval_data))))
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    references = []
    total_wer = 0.0
    total_cer = 0.0

    print(f"\n{'='*70}")
    print(f"🚀 Starting Evaluation Pipeline ({len(eval_data)} samples)")
    print(f"{'='*70}")

    for idx, input_sample in enumerate(eval_data, 1):
        prompt = input_sample["prompt"]
        raw_asr = input_sample["raw_asr"]
        ground_truth = input_sample["refined_text"]

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
        raw_output_text = tokenizer.decode(
            response[0][prompt_len:], skip_special_tokens=True
        ).strip()

        # Clean JSON if wrapped
        output_text = extract_refined_output(raw_output_text)

        # Word Error Rate and Character Error Rate for this sample
        sample_wer = wer(ground_truth, output_text)
        sample_cer = cer(ground_truth, output_text)
        total_wer += sample_wer
        total_cer += sample_cer

        predictions.append(output_text)
        references.append(ground_truth)

        # Clean sample inspection printout
        print(f"\n[Sample {idx:02d}/{len(eval_data):02d}]")
        print(f"🔹 RAW ASR:\n   {raw_asr}")
        print(f"🎯 GROUND TRUTH:\n   {ground_truth}")
        print(f"🤖 REFINED OUTPUT:\n   {output_text}")
        print(f"📊 Sample WER : {sample_wer:.4f} ({sample_wer*100:.1f}%) | CER : {sample_cer:.4f} ({sample_cer*100:.1f}%)")
        print("-" * 70)

    # Compute Batch BERTScore in one vectorized pass (much faster & GPU efficient)
    print("\n⏳ Computing BERTScore across all samples...")
    p_tensor, r_tensor, f1_tensor = bert_score(
        predictions,
        references,
        lang="en",
        device=device,
        verbose=False,
    )

    avg_wer = total_wer / len(eval_data)
    avg_cer = total_cer / len(eval_data)
    avg_precision = p_tensor.mean().item()
    avg_recall = r_tensor.mean().item()
    avg_f1 = f1_tensor.mean().item()

    return avg_wer, avg_cer, avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    avg_wer, avg_cer, p, r, f1 = evaluate(num_samples=10)

    print(f"\n{'='*70}")
    print("📈 FINAL EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"  • Word Error Rate (WER)      : {avg_wer:.4f} ({avg_wer*100:.1f}%) [Lower is better]")
    print(f"  • Character Error Rate (CER) : {avg_cer:.4f} ({avg_cer*100:.1f}%) [Lower is better]")
    print(f"  • BERTScore Precision        : {p:.4f} [Higher is better]")
    print(f"  • BERTScore Recall           : {r:.4f} [Higher is better]")
    print(f"  • BERTScore F1               : {f1:.4f} [Higher is better]")
    print(f"{'='*70}\n")
