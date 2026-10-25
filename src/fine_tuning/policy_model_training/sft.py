from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers.trainer_callback import (
    PrinterCallback,
    TrainerCallback,
    EarlyStoppingCallback,
)
import json
import os
import torch

max_seq_length = 2048

# Map the process strictly to its allocated local GPU
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map=device_map,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=124,
    lora_dropout=False,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
    ],
)

policy_system_prompt = (
    "You are an expert speech-to-text post-processor. "
    "Fix transcription misspellings, add correct casing and punctuation, "
    "and remove speech disfluencies (e.g., 'um', 'uh', false starts). "
    "Preserve the speaker's original meaning and exact phrasing. Do not summarize or rewrite."
)


def _process_batch_examples(batch):
    samples = []
    for raw_asr, refined_text in zip(batch["raw_asr"], batch["refined_text"]):
        messages = [
            {"role": "system", "content": policy_system_prompt},
            {"role": "user", "content": f"raw transcript : {raw_asr}"},
            {
                "role": "assistant",
                "content": json.dumps({"refined_output": refined_text}),
            },
        ]
        samples.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return {"text": samples}


def format_dataset(path: str = None):
    if path is None:
        candidates = [
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../datasets/policy_model_dataset/combined_dataset.jsonl",
                )
            ),
            "./fine-tuning/datasets/policy_model_dataset/combined_dataset.jsonl",
            "./datasets/policy_model_dataset/combined_dataset.jsonl",
            "/kaggle/working/fine-tuning/datasets/policy_model_dataset/combined_dataset.jsonl",
        ]
        path = next((c for c in candidates if os.path.exists(c)), candidates[0])

    datasets = load_dataset("json", data_files=path, split="train")
    datasets = datasets.map(_process_batch_examples, batched=True)
    split = datasets.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


train_dataset, test_dataset = format_dataset()

training_args = SFTConfig(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir="./outputs",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    logging_steps=10,
    report_to="none",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    ddp_find_unused_parameters=False,
)


class CustomPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and logs:
            if "loss" in logs:
                print(
                    f" steps : {state.global_step} | loss : {logs.get('loss')} | lr : {logs.get('learning_rate')}"
                )
            if "eval_loss" in logs:
                print(
                    f" [EVAL] steps : {state.global_step} | eval_loss : {logs.get('eval_loss')}"
                )


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[CustomPrinter(), EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.remove_callback(PrinterCallback)

if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_REPO = "pankaj0/llama-3.2-3b-asr-sft"
    trainer.train()

    print(f"Pushing trained model to Hugging Face Hub: {HF_REPO}...")

    # Push the lightweight LoRA adapter (~50MB)
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)

    print("[DONE]")
