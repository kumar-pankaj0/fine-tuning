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


def load_model(model_name: str):
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
    return model, tokenizer


SystemPrompt = (
    "You are an expert speech-to-text post-processor. "
    "Fix transcription misspellings, add correct casing and punctuation, "
    "and remove speech disfluencies (e.g., 'um', 'uh', false starts). "
    "Preserve the speaker's original meaning and exact phrasing. Do not summarize or rewrite."
)


def _format_batch_samples(batch, tokenizer):
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


def format_dataset(tokenizer, dataset_path=None):
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
    formated_dataset = dataset.map(
        lambda b: _format_batch_samples(b, tokenizer),
        batched=True,
    )
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


def evaluate(tokenizer, model, num_samples: int = 10, model_label: str = "Model"):
    eval_data = format_dataset(tokenizer=tokenizer)
    eval_data = eval_data.select(range(min(num_samples, len(eval_data))))
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    references = []
    total_wer = 0.0
    total_cer = 0.0

    print(f"\n{'=' * 70}")
    print(f"🚀 Starting Evaluation Pipeline [{model_label}] ({len(eval_data)} samples)")
    print(f"{'=' * 70}")

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
        print(f"\n[{model_label} - Sample {idx:02d}/{len(eval_data):02d}]")
        print(f"🔹 RAW ASR:\n   {raw_asr}")
        print(f"🎯 GROUND TRUTH:\n   {ground_truth}")
        print(f"🤖 REFINED OUTPUT:\n   {output_text}")
        print(
            f"📊 Sample WER : {sample_wer:.4f} ({sample_wer * 100:.1f}%) | CER : {sample_cer:.4f} ({sample_cer * 100:.1f}%)"
        )
        print("-" * 70)

    # Compute Batch BERTScore in one vectorized pass (much faster & GPU efficient)
    print(f"\n⏳ Computing BERTScore across all samples for {model_label}...")
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


def print_comparison_report(base_metrics, ft_metrics):
    base_wer, base_cer, base_p, base_r, base_f1 = base_metrics
    ft_wer, ft_cer, ft_p, ft_r, ft_f1 = ft_metrics

    # Relative improvements (for error rates: lower is better; for BERTScore: higher is better)
    wer_rel = ((base_wer - ft_wer) / base_wer * 100) if base_wer > 0 else 0.0
    cer_rel = ((base_cer - ft_cer) / base_cer * 100) if base_cer > 0 else 0.0
    p_rel = ((ft_p - base_p) / base_p * 100) if base_p > 0 else 0.0
    r_rel = ((ft_r - base_r) / base_r * 100) if base_r > 0 else 0.0
    f1_rel = ((ft_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0.0

    print(f"\n{'=' * 85}")
    print("📈 FINAL EVALUATION & BASELINE COMPARISON REPORT")
    print(f"{'=' * 85}")
    print(f"{'Metric':<26} {'Baseline':<16} {'Fine-Tuned':<16} {'Absolute Δ':<14} {'Improvement':<12}")
    print(f"{'-' * 85}")
    print(
        f"{'Word Error Rate (WER) ↓':<26} {base_wer:.4f} ({base_wer*100:.1f}%)   {ft_wer:.4f} ({ft_wer*100:.1f}%)   {ft_wer - base_wer:+.4f}        {wer_rel:+.1f}% {'(Better)' if wer_rel > 0 else ''}"
    )
    print(
        f"{'Char Error Rate (CER) ↓':<26} {base_cer:.4f} ({base_cer*100:.1f}%)   {ft_cer:.4f} ({ft_cer*100:.1f}%)   {ft_cer - base_cer:+.4f}        {cer_rel:+.1f}% {'(Better)' if cer_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore Precision ↑':<26} {base_p:.4f}          {ft_p:.4f}          {ft_p - base_p:+.4f}        {p_rel:+.1f}% {'(Better)' if p_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore Recall ↑':<26} {base_r:.4f}          {ft_r:.4f}          {ft_r - base_r:+.4f}        {r_rel:+.1f}% {'(Better)' if r_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore F1 ↑':<26} {base_f1:.4f}          {ft_f1:.4f}          {ft_f1 - base_f1:+.4f}        {f1_rel:+.1f}% {'(Better)' if f1_rel > 0 else ''}"
    )
    print(f"{'=' * 85}\n")


if __name__ == "__main__":
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    fine_tuned_model_name = "pankaj0/llama-3.2-3b-asr-sft"

    num_samples = 10

    # 1. Evaluate Baseline Model
    print("\n🔍 Loading Baseline Model...")
    base_model, base_tokenizer = load_model(model_name=base_model_name)
    base_metrics = evaluate(
        tokenizer=base_tokenizer,
        model=base_model,
        num_samples=num_samples,
        model_label="Baseline",
    )

    # Free Baseline VRAM to prevent Out-Of-Memory (OOM) on Kaggle/Colab
    del base_model
    del base_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Fine-Tuned Model
    print("\n🎯 Loading Fine-Tuned Model...")
    ft_model, ft_tokenizer = load_model(model_name=fine_tuned_model_name)
    ft_metrics = evaluate(
        tokenizer=ft_tokenizer,
        model=ft_model,
        num_samples=num_samples,
        model_label="Fine-Tuned",
    )

    # Free Fine-Tuned Model VRAM
    del ft_model
    del ft_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Print Side-by-Side Comparison
    print_comparison_report(base_metrics, ft_metrics)
