from pathlib import Path
import os
import subprocess
import json
import argparse
import time
import torch
from datasets import Dataset

try:
    if not Path("./datasets").exists():
        gitclone = subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/kumar-pankaj0/datasets",
            ],
            check=False,
            capture_output=True,
        )
        gitclone.check_returncode()
except Exception:
    pass

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STEPS = 10
BATCH_SIZE = 4
MAX_SEQ_LEN = 512


def load_dataset(path):
    sample_text = []
    with open(path, "r") as f:
        json_samples = json.load(f)

    for sample in json_samples:
        inp = sample["input"].get(
            "asr_input"
        ) or sample["input"].get("raw_asr", "")
        asr_out = sample["input"].get(
            "refined_text", ""
        )
        judge_out = sample.get("output", "")

        sample_text.append(
            f"<|im_start|>user\n {inp} \n {asr_out}\n <|im_end|>\n <|im_start|>assistant \n {judge_out}"
        )

    return Dataset.from_dict(
        {"text": sample_text}
    )


def run_unsloth_benchmark(
    data_path, is_ddp=False
):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    dataset = load_dataset(data_path)
    local_rank = int(
        os.environ["LOCAL_RANK", 0]
    )
    model, tokenizer = (
        FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
        )
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
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
            ddp_find_unused_parameters=False,
            fp16=True,
        ),
    )

    torch.cuda.synchronize()
    init = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - init

    if local_rank == 0:
        num_gpus = 2 if is_ddp else 1
        token_per_sec = (
            num_gpus
            * STEPS
            * BATCH_SIZE
            * MAX_SEQ_LEN
        ) / elapsed
        print(
            f"\n[RESULT: UNSLOTH 1-GPU] Step Latency: {(elapsed / STEPS)*1000:.2f} ms | Throughput: {token_per_sec:.2f} tok/s"
        )


def run_trl_benchmark(data_path, is_ddp=False):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    dataset = load_dataset(data_path)
    local_rank = int(
        os.environ.get("LOCAL_RANK", 0)
    )
    device_map = (
        {"": local_rank} if is_ddp else "auto"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
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
            ddp_find_unused_parameters=(
                False if is_ddp else None
            ),
        ),
    )

    torch.cuda.synchronize()
    init = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - init

    num_gpus = 2 if is_ddp else 1
    token_per_sec = (
        STEPS
        * BATCH_SIZE
        * num_gpus
        * MAX_SEQ_LEN
    ) / elapsed
    tag = (
        "TRL 2-GPU DDP"
        if is_ddp
        else "TRL 1-GPU"
    )

    print(
        f"\n[RESULT: {tag}] Step Latency: {(elapsed / STEPS)*1000:.2f} ms | Throughput: {token_per_sec:.2f} tok/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=[
            "unsloth",
            "unsloth_ddp",
            "trl_single",
            "trl_ddp",
        ],
        required=True,
    )
    parser.add_argument(
        "--samples",
        default=Path(
            "./datasets/eval_data.json"
        ),
    )

    args = parser.parse_args()
    data_file = Path(args.samples)
    if not data_file.exists():
        data_file = Path(
            "./datasets/data_full.json"
        )

    if args.engine == "unsloth":
        run_unsloth_benchmark(data_file)
    elif args.engine == "unsloth_ddp":
        run_unsloth_benchmark(data_file, is_ddp = True)
    elif args.engine == "trl_single":
        run_trl_benchmark(
            data_file, is_ddp=False
        )
    else:
        run_trl_benchmark(
            data_file, is_ddp=True
        )
