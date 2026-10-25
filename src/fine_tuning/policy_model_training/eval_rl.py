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


def load_model(model_name: str, device: str = None):
    """
    Loads model and tokenizer using FastLanguageModel.
    If device is specified (e.g. 'cuda:0' or 'cuda:1'), pins all layers to that device.
    """
    if device is not None:
        device_map = {"": device}
    else:
        device_map = "auto" if torch.cuda.is_available() else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=device_map,
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
    """Sequential sample-by-sample evaluation pipeline (ideal for quick debugging)."""
    eval_data = format_dataset(tokenizer=tokenizer)
    eval_data = eval_data.select(range(min(num_samples, len(eval_data))))
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    references = []
    total_wer = 0.0
    total_cer = 0.0

    print(f"\n{'=' * 70}")
    print(
        f"🚀 Starting Evaluation Pipeline [{model_label}] ({len(eval_data)} samples on {device})"
    )
    print(f"{'=' * 70}")

    for idx, input_sample in enumerate(eval_data):
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
    print(
        f"\n⏳ Computing BERTScore across all samples for {model_label} on {device}..."
    )
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


def batch_evaluate(tokenizer, model, batch_size: int = 8, model_label: str = "Model"):
    """
    Batched parallel evaluation pipeline.
    Uses left-padding for decoder-only models and autocast for memory bandwidth optimization.
    """
    eval_data = format_dataset(tokenizer=tokenizer)
    device = next(model.parameters()).device

    # LEFT-pad: decoder-only models need this for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    prompts = eval_data["prompt"]
    references = eval_data["refined_text"]
    predictions = []

    print(f"\n{'=' * 70}")
    print(
        f"🚀 [{model_label}] Starting Evaluation on {device} ({len(prompts)} samples, batch_size={batch_size})"
    )
    print(f"{'=' * 70}")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize whole batch at once — padding fills shorter seqs
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        # autocast: halves memory bandwidth for compute kernels
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            responses = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        # Left-padding → all prompts same length → slice is clean
        prompt_len = inputs["input_ids"].shape[1]
        for response in responses:
            raw = tokenizer.decode(
                response[prompt_len:], skip_special_tokens=True
            ).strip()
            predictions.append(extract_refined_output(raw))

        print(
            f"  [{model_label}] Progress: {min(i + batch_size, len(prompts))}/{len(prompts)}",
            end="\r",
        )

    print()  # Advance to next line after progress output

    # --- Metrics: all computed outside the loop ---
    total_wer = sum(wer(r, p) for r, p in zip(references, predictions)) / len(
        references
    )
    total_cer = sum(cer(r, p) for r, p in zip(references, predictions)) / len(
        references
    )

    print(f"⏳ [{model_label}] Computing BERTScore on {device}...")
    p_t, r_t, f1_t = bert_score(
        predictions, references, lang="en", device=device, verbose=False
    )

    return (
        total_wer,
        total_cer,
        p_t.mean().item(),
        r_t.mean().item(),
        f1_t.mean().item(),
    )


def print_comparison_report(base_metrics, ft_metrics):
    """Prints a structured side-by-side metric comparison table between Baseline & Fine-Tuned models."""
    base_wer, base_cer, base_p, base_r, base_f1 = base_metrics
    ft_wer, ft_cer, ft_p, ft_r, ft_f1 = ft_metrics

    # Relative improvements (for error rates: lower is better; for BERTScore: higher is better)
    wer_rel = ((base_wer - ft_wer) / base_wer * 100) if base_wer > 0 else 0.0
    cer_rel = ((base_cer - ft_cer) / base_cer * 100) if base_cer > 0 else 0.0
    p_rel = ((ft_p - base_p) / base_p * 100) if base_p > 0 else 0.0
    r_rel = ((ft_r - base_r) / base_r * 100) if base_r > 0 else 0.0
    f1_rel = ((ft_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0.0

    print(f"\n{'=' * 85}")
    print("📈 FINAL EVALUATION & BASELINE COMPARISON REPORT (GPU 0 vs GPU 1)")
    print(f"{'=' * 85}")
    print(
        f"{'Metric':<26} {'Baseline (GPU 0)':<18} {'Fine-Tuned (GPU 1)':<18} {'Absolute Δ':<14} {'Improvement':<12}"
    )
    print(f"{'-' * 85}")
    print(
        f"{'Word Error Rate (WER) ↓':<26} {base_wer:.4f} ({base_wer * 100:.1f}%)    {ft_wer:.4f} ({ft_wer * 100:.1f}%)    {ft_wer - base_wer:+.4f}        {wer_rel:+.1f}% {'(Better)' if wer_rel > 0 else ''}"
    )
    print(
        f"{'Char Error Rate (CER) ↓':<26} {base_cer:.4f} ({base_cer * 100:.1f}%)    {ft_cer:.4f} ({ft_cer * 100:.1f}%)    {ft_cer - base_cer:+.4f}        {cer_rel:+.1f}% {'(Better)' if cer_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore Precision ↑':<26} {base_p:.4f}             {ft_p:.4f}             {ft_p - base_p:+.4f}        {p_rel:+.1f}% {'(Better)' if p_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore Recall ↑':<26} {base_r:.4f}             {ft_r:.4f}             {ft_r - base_r:+.4f}        {r_rel:+.1f}% {'(Better)' if r_rel > 0 else ''}"
    )
    print(
        f"{'BERTScore F1 ↑':<26} {base_f1:.4f}             {ft_f1:.4f}             {ft_f1 - base_f1:+.4f}        {f1_rel:+.1f}% {'(Better)' if f1_rel > 0 else ''}"
    )
    print(f"{'=' * 85}\n")


# ==============================================================================
# Helper worker functions (defined at module level for spawn / thread pickling)
# ==============================================================================


def _spawn_worker(
    model_name: str, queue, device_id: int, model_label: str, batch_size: int = 8
):
    """Worker process for torch.multiprocessing parallel evaluation."""
    try:
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
        print(f"\n🔍 [{model_label}] Loading on {device}...")
        model, tokenizer = load_model(model_name=model_name, device=device)
        metrics = batch_evaluate(
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            model_label=model_label,
        )
        del model
        del tokenizer
        torch.cuda.empty_cache()
        queue.put((model_label, metrics, None))
    except Exception as e:
        queue.put((model_label, None, str(e)))


def _thread_worker(
    model_name: str, device_id: int, model_label: str, batch_size: int = 8
):
    """Worker task for ThreadPoolExecutor parallel evaluation."""
    device = f"cuda:{device_id}"
    with torch.cuda.device(device_id):
        print(f"\n🧵 [{model_label}] Loading in thread on {device}...")
        model, tokenizer = load_model(model_name=model_name, device=device)
        metrics = batch_evaluate(
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            model_label=model_label,
        )
        del model
        del tokenizer
        torch.cuda.empty_cache()
        return metrics


if __name__ == "__main__":
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    fine_tuned_model_name = "pankaj0/llama-3.2-3b-asr-sft"
    model, tokenizer = load_model(base_model_name)
    avg_wer, avg_cer, avg_precision, avg_recall, avg_f1 = evaluate(
        tokenizer, model, num_samples=10
    )
    print("WER : {avg_wer}")
    print("CER : {avg_cer}")
    print("PRECISION : {avg_precisione}")
    print("RECALL : {avg_recall}")
    print("F1 : {avg_f1}")
    """
    # ==============================================================================
    # OPTION 1: ThreadPoolExecutor (Multi-threading in single process)
    # ------------------------------------------------------------------------------
    # Concurrent kernel dispatch using Python threads. Results return directly in memory.
    # Note: Slight GIL contention occurs during CPU tokenization / string operations.
    # ==============================================================================
    from concurrent.futures import ThreadPoolExecutor

    print("⚡ Starting Parallel Evaluation via ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        base_future = executor.submit(
            _thread_worker,
            model_name=base_model_name,
            device_id=0,
            model_label="Baseline",
            batch_size=8,
        )
        ft_future = executor.submit(
            _thread_worker,
            model_name=fine_tuned_model_name,
            device_id=1,
            model_label="Fine-Tuned",
            batch_size=8,
        )

        base_metrics = base_future.result()
        ft_metrics = ft_future.result()

    print_comparison_report(base_metrics, ft_metrics)
    """

    """
    # ==============================================================================
    # OPTION 2: torch.multiprocessing (Process Isolation with 'spawn' method)
    # ------------------------------------------------------------------------------
    # Pure Python multiprocessing without the GIL. Each process owns its own CUDA context.
    # Completely avoids CUDA context conflicts and delivers 100% full parallel throughput.
    # Run directly: python eval_rl.py
    # ==============================================================================
    from torch import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    result_queue = multiprocessing.Queue()

    print("⚡ Starting Parallel Evaluation via torch.multiprocessing...")
    base_process = multiprocessing.Process(
        target=_spawn_worker,
        args=(base_model_name, result_queue, 0, "Baseline", 8),
    )
    fine_tuned_process = multiprocessing.Process(
        target=_spawn_worker,
        args=(fine_tuned_model_name, result_queue, 1, "Fine-Tuned", 8),
    )

    base_process.start()
    fine_tuned_process.start()

    # Collect metrics from both worker processes
    results = {}
    for _ in range(2):
        process_model_label, metrics, error = result_queue.get()
        if error is not None:
            raise RuntimeError(f"Error in {process_model_label} process: {error}")
        results[process_model_label] = metrics

    base_process.join()
    fine_tuned_process.join()

    print_comparison_report(results["Baseline"], results["Fine-Tuned"])

    # ==============================================================================
    # OPTION 3 (ACTIVE): torchrun Distributed Parallelization (torch.distributed)
    # ------------------------------------------------------------------------------
    # Production / MLOps standard using NCCL backend.
    # Launch with:
    #     torchrun --nproc_per_node=2 eval_rl.py
    #
    # Rank 0 is pinned to GPU 0 (Baseline)
    # Rank 1 is pinned to GPU 1 (Fine-Tuned)
    # Metrics are synchronized across ranks using all_gather_object.
    # ==============================================================================
    from torch import distributed

    # Check if launched under torchrun / distributed runner
    if "LOCAL_RANK" not in os.environ:
        print("\n⚠️  [torchrun Parallel Mode Active]")
        print("To run both models simultaneously across 2 GPUs, launch with:")
        print("   torchrun --nproc_per_node=2 eval_rl.py\n")
        sys_exit = True
    else:
        sys_exit = False

    if not sys_exit:
        distributed.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        if local_rank == 0:
            print(f"\n🔍 [Rank {local_rank}] Loading Baseline Model on {device}...")
            model, tokenizer = load_model(model_name=base_model_name, device=device)
            metrics = batch_evaluate(
                tokenizer=tokenizer,
                model=model,
                batch_size=8,
                model_label="Baseline",
            )
        else:
            print(f"\n🎯 [Rank {local_rank}] Loading Fine-Tuned Model on {device}...")
            model, tokenizer = load_model(
                model_name=fine_tuned_model_name, device=device
            )
            metrics = batch_evaluate(
                tokenizer=tokenizer,
                model=model,
                batch_size=8,
                model_label="Fine-Tuned",
            )

        # Free GPU memory after evaluation
        del model
        del tokenizer
        torch.cuda.empty_cache()

        all_gathered_metrics = [None, None]
        distributed.all_gather_object(all_gathered_metrics, metrics)

        if local_rank == 0:
            # index 0 -> Baseline metrics (GPU 0)
            # index 1 -> Fine-Tuned metrics (GPU 1)
            print_comparison_report(all_gathered_metrics[0], all_gathered_metrics[1])

        distributed.destroy_process_group()"""
