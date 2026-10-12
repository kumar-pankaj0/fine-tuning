import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import shutil

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STEPS = 5
BATCH_SIZE = 4
MAX_SEQ_LEN = 512

def load_dataset():
    from datasets import Dataset
    data_file = Path("./datasets/eval_data.json")
    if not data_file.exists():
        data_file = Path("./datasets/data_full.json")

    if data_file.exists():
        sample_text = []
        with open(data_file, "r") as f:
            json_samples = json.load(f)
        for sample in json_samples[:20]:
            inp = sample["input"].get("asr_input") or sample["input"].get("raw_asr", "")
            asr_out = sample["input"].get("refined_text", "")
            judge_out = str(sample.get("output", ""))
            sample_text.append(
                f"<|im_start|>user\n{inp}\n{asr_out}\n<|im_end|>\n<|im_start|>assistant\n{judge_out}<|im_end|>"
            )
        return Dataset.from_dict({"text": sample_text})

    # Synthetic fallback to guarantee determinism if external repo is unavailable
    texts = [
        f"<|im_start|>user\nraw asr test input sample {i}\nrefined text output {i}\n<|im_end|>\n<|im_start|>assistant\nfaithfulness: 9\n<|im_end|>"
        for i in range(20)
    ]
    return Dataset.from_dict({"text": texts})

def run_unsloth_benchmark(is_ddp=False):
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dataset = load_dataset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=128,
        lora_dropout=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=STEPS,
            logging_steps=1,
            output_dir="./tmp_unsloth",
            report_to="none",
            fp16=True,
            ddp_find_unused_parameters=False if is_ddp else None,
        ),
    )

    torch.cuda.synchronize()
    init = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - init

    if local_rank == 0:
        num_gpus = 2 if is_ddp else 1
        token_per_sec = (STEPS * BATCH_SIZE * num_gpus * MAX_SEQ_LEN) / elapsed
        tag = "UNSLOTH 2-GPU DDP" if is_ddp else "UNSLOTH 1-GPU"
        print(f"\n[BENCHMARK RESULT: {tag}] Step Latency: {(elapsed / STEPS)*1000:.2f} ms | Throughput: {token_per_sec:.2f} tok/s")

def run_trl_benchmark(is_ddp=False):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if is_ddp else "auto"

    dataset = load_dataset()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=STEPS,
            logging_steps=1,
            output_dir="./tmp_trl",
            report_to="none",
            fp16=True,
            ddp_find_unused_parameters=False if is_ddp else None,
        ),
    )

    torch.cuda.synchronize()
    init = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - init

    if local_rank == 0:
        num_gpus = 2 if is_ddp else 1
        token_per_sec = (STEPS * BATCH_SIZE * num_gpus * MAX_SEQ_LEN) / elapsed
        tag = "TRL 2-GPU DDP" if is_ddp else "TRL 1-GPU"
        print(f"\n[BENCHMARK RESULT: {tag}] Step Latency: {(elapsed / STEPS)*1000:.2f} ms | Throughput: {token_per_sec:.2f} tok/s")

def supervisor_main():
    print("=== INITIALIZING KAGGLE REMOTE CLOUD BENCHMARK ===")
    
    # 1. Install remote dependencies if absent
    try:
        import unsloth
        import trl
    except ImportError:
        print("--> Installing unsloth, unsloth_zoo, bitsandbytes, and trl inside container...")
        subprocess.run([sys.executable, "-m", "pip", "install", "unsloth", "unsloth_zoo", "bitsandbytes", "trl"], check=False)

    # 2. Fetch dataset repository if not present
    if not Path("./datasets").exists():
        print("--> Cloning evaluation dataset...")
        subprocess.run(["git", "clone", "https://github.com/kumar-pankaj0/datasets"], check=False)

    # 3. Define the sequential execution suite
    script_path = os.path.abspath(__file__)
    suite = [
        ("UNSLOTH 1-GPU", [sys.executable, script_path, "--engine", "unsloth"]),
        ("UNSLOTH 2-GPU DDP", ["torchrun", "--nproc_per_node=2", script_path, "--engine", "unsloth_ddp"]),
        ("TRL 1-GPU", [sys.executable, script_path, "--engine", "trl_single"]),
        ("TRL 2-GPU DDP", ["torchrun", "--nproc_per_node=2", script_path, "--engine", "trl_ddp"]),
    ]

    try:
        for label, cmd in suite:
            print(f"\n==================================================")
            print(f"--> EXECUTING: {label}")
            print(f"==================================================")
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"FAILED: {label} exited with non-zero status code {res.returncode}")

        print("\n=== ALL BENCHMARKS COMPLETED ===")
    finally:
        shutil.rmtree("./datasets", ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["unsloth", "unsloth_ddp", "trl_single", "trl_ddp"],
        default=None,
    )
    args = parser.parse_args()

    if args.engine is None:
        # Default entrypoint when invoked by Kaggle container: run the full benchmark supervisor
        supervisor_main()
    elif args.engine == "unsloth":
        run_unsloth_benchmark(is_ddp=False)
    elif args.engine == "unsloth_ddp":
        run_unsloth_benchmark(is_ddp=True)
    elif args.engine == "trl_single":
        run_trl_benchmark(is_ddp=False)
    elif args.engine == "trl_ddp":
        run_trl_benchmark(is_ddp=True)
