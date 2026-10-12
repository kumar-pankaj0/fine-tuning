import torch
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# =======================================================================
# 1. Hardware & Quantization Setup
# =======================================================================
# 4-bit configuration to fit both 3B models on 15GB T4 GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# =======================================================================
# 2. Isolate the Qwen Judge on GPU 1
# =======================================================================
JUDGE_MODEL = "pankaj0/qwen-post-refine-judge"

# The pipeline handles text-generation on device 1 (the second T4)
judge_pipeline = pipeline(
    "text-generation",
    model=JUDGE_MODEL,
    device=1,
    model_kwargs={
        "quantization_config": bnb_config
    },
)

# Optional: Set pad_token_id to avoid warnings during batch inference
if (
    judge_pipeline.tokenizer.pad_token_id
    is None
):
    judge_pipeline.tokenizer.pad_token_id = (
        judge_pipeline.tokenizer.eos_token_id
    )


# =======================================================================
# 3. Define the Reward Function
# =======================================================================
def judge_reward(completions, **kwargs):
    """
    Evaluates Llama's refined completions using the Qwen judge.
    Returns a list of float rewards (one per completion).
    """
    prompts = kwargs.get("prompt", [])
    batch_inputs = []

    # Construct prompts for the judge
    for prompt_text, completion_list in zip(
        prompts, completions
    ):
        # TRL passes completions as a list of dicts: [{'role': 'assistant', 'content': '...'}]
        refined_text = completion_list[0][
            "content"
        ]

        # Adjust this template to match exactly how your Qwen judge was fine-tuned
        judge_input = (
            f"<|im_start|>system\nYou are an ASR Refinement Judge. Score the text quality.<|im_end|>\n"
            f"<|im_start|>user\nOriginal: {prompt_text}\nRefined: {refined_text}<|im_end|>\n"
            f"<|im_start|>assistant\nScore:"
        )
        batch_inputs.append(judge_input)

    # Run batched inference on GPU 1
    # Disable gradient calculation to save memory
    with torch.no_grad():
        results = judge_pipeline(
            batch_inputs,
            max_new_tokens=10,
            truncation=True,
            return_full_text=False,
            batch_size=4,  # Adjust based on GPU 1 memory availability
        )

    rewards = []
    for res in results:
        generated_output = (
            res[0]["generated_text"]
            .strip()
            .lower()
        )

        # Example parsing logic: map the judge's text output to a float scalar.
        # Modify this regex/parsing based on your specific Qwen model's output format.
        try:
            if (
                "excellent" in generated_output
                or "5" in generated_output
            ):
                rewards.append(1.0)
            elif (
                "good" in generated_output
                or "4" in generated_output
            ):
                rewards.append(0.75)
            elif (
                "fair" in generated_output
                or "3" in generated_output
            ):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(
                0.0
            )  # Fallback penalty for malformed judge output

    return rewards


# =======================================================================
# 4. Prepare Dataset for Llama
# =======================================================================
# Replace with your actual dataset
dataset = load_dataset(
    "your_huggingface_username/your_asr_dataset",
    split="train",
)


def format_prompt(example):
    # GRPO requires a 'prompt' column. The assistant completions are generated automatically.
    # Format matches Llama 3.2 Instruct structure.
    system_prompt = "You are an expert ASR refiner. Fix the transcription errors."
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRefine this text: {example['raw_asr_text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return {"prompt": formatted_prompt}


train_dataset = dataset.map(format_prompt)

# =======================================================================
# 5. Configure GRPO Training for Llama on GPU 0
# =======================================================================
LLAMA_MODEL = (
    "pankaj0/llama-3.2-3b-asr-refiner-merged"
)

training_args = GRPOConfig(
    output_dir="./grpo-llama-asr",
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # Enable vLLM Colocation (uses GPU 0 for both training and generation)
    use_vllm=True,
    vllm_mode="colocate",
    vllm_device="cuda:0",
    # Strictly limit vLLM KV cache size so it doesn't crash the LoRA optimizer
    vllm_gpu_memory_utilization=0.35,
    # Rollout constraints
    max_completion_length=256,
    num_generations=4,  # 4 parallel rollouts per prompt
    temperature=0.7,
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    report_to="none",
    # Optimizer settings for limited VRAM
    optim="paged_adamw_8bit",
    bf16=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
    bias="none",
)

# =======================================================================
# 6. Initialize Trainer and Execute
# =======================================================================
trainer = GRPOTrainer(
    model=LLAMA_MODEL,
    reward_funcs=[judge_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
)

print("Starting GRPO Training Pipeline...")
trainer.train()

# Save final LoRA adapter
trainer.save_model("./grpo-llama-asr-final")
print("Training complete and adapter saved.")
