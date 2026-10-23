"""
depency for this file
- git clone https://kumer-pankaj0/datasets.git
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

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


def _process_batch_exmaples(batch):
    sampels = [] 
    for pair in batch:





def format_dataset(path: str = "./datasets/policy_sft_asr.json"):
    datasets = load_dataset("json", data_files=path)
    datasets.map(_process_batch_exmaples, batched=True)


training_args = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    report_to="none",
    learning_rate=2e4,
    optim="adwm-8bit",
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
)


trainer = SFTTrainer(model, args=training_args, train_dataset=training_dataset)
