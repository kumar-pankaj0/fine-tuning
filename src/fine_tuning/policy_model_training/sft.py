"""
depency for this file
- git clone https://kumer-pankaj0/datasets.git
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers.trainer_callback import PrinterCallback, TrainerCallback
import json

model_name = "llama-3.1-3b"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    dtype=None,
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


def format_dataset(path: str = "./datasets/policy_sft_asr.json"):
    datasets = load_dataset("json", data_files=path)
    datasets = datasets.map(_process_batch_examples, batched=True)
    return datasets


training_dataset = format_dataset()

training_args = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    report_to="none",
    learning_rate=2e-4,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataset_text_field="text",
)


class CustomPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(
            f" steps : {state.global_step} | loss : {logs.get('loss')} | lr : {logs.get('learning_rate')}"
        )


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=training_dataset["train"],
    callbacks=[CustomPrinter()],
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.remove_callback(PrinterCallback)
trainer.train()
