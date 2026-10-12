"""
Synthetic STT Data Generator for Post-Refining fine-tuning.
Follows Spec v1 (F4 style):
- Acoustic / STT degradation simulation: Parakeet-tdt-0.6b-v3 on low RAM, mic noise, low gain.
- Faithful cleanup (accept): removes fillers/stutters, fixes non-native syntax, keeps negations,
  preserves uncertain words, keeps cut-off fragments, formats quotes, paragraph breaks on shifts.
- Rotating hard negative (reject):
  * dropped_negation / flipped_negation
  * invented_ending
  * guessed_uncertain_word
  * added_facts_or_ideas
  * kept_fillers
  * over_edited
  * merged_or_shortened_sentences
  * added_exclamation
  * missing_quotes_on_dictated_text
"""

import json
from typing import List, Dict, Any

BATCH_1_SAMPLES: List[Dict[str, Any]] = [
    # 1. dropped_negation / flipped_negation
    {
        "input": "so uh when you deploy the lora adapter on the fast api server do not merge the weights yet because we we want to benchmark the latency without merge",
        "accept": "When you deploy the LoRA adapter on the FastAPI server, do not merge the weights yet, because we want to benchmark the latency without merge.",
        "reject": "When you deploy the LoRA adapter on the FastAPI server, merge the weights now, because we want to benchmark the latency with merge.",
        "reject_type": "flipped_negation",
        "uncertain_words": []
    },
    # 2. invented_ending (trailing chunk cut-off)
    {
        "input": "I was checking the vram allocation when running the four bit quantization and you know the the context window is taking like the whole",
        "accept": "I was checking the VRAM allocation when running the 4-bit quantization, and the context window is taking like the whole",
        "reject": "I was checking the VRAM allocation when running the 4-bit quantization, and the context window is taking like the whole memory on the GPU.",
        "reject_type": "invented_ending",
        "uncertain_words": []
    },
    # 3. guessed_uncertain_word (ambiguous acoustic token)
    {
        "input": "because the mic gain was very very down the audio is having too much hum and the thum is clipping on the high frequencies so explain me how to filter it",
        "accept": "Because the mic gain was very down, the audio is having too much hum and the thum is clipping on the high frequencies.\n\nExplain to me how to filter it.",
        "reject": "Because the mic gain was very down, the audio is having too much hum and the thumb is clipping on the high frequencies.\n\nExplain to me how to filter it.",
        "reject_type": "guessed_uncertain_word",
        "uncertain_words": ["thum"]
    },
    # 4. added_facts_or_ideas (hallucinating details not spoken)
    {
        "input": "so for the next training epoch we we should reduce the learning rate because the loss curve is oscillating very badly",
        "accept": "So for the next training epoch, we should reduce the learning rate, because the loss curve is oscillating very badly.",
        "reject": "So for the next training epoch, we should reduce the learning rate to 1e-5 using AdamW, because the loss curve is oscillating very badly.",
        "reject_type": "added_facts_or_ideas",
        "uncertain_words": []
    },
    # 5. kept_fillers (failing to clean stutter/fillers)
    {
        "input": "uh you know like I tried to run the celery worker in the background but like you know it is crashing on the redis connection",
        "accept": "I tried to run the Celery worker in the background, but it is crashing on the Redis connection.",
        "reject": "Uh, you know, I tried to run the Celery worker in the background, but like you know it is crashing on the Redis connection.",
        "reject_type": "kept_fillers",
        "uncertain_words": []
    },
    # 6. over_edited (unwanted thesaurus rewrite)
    {
        "input": "the model is not returning the json output properly so tell me what regex we can use to extract the code block",
        "accept": "The model is not returning the JSON output properly. Tell me what regex we can use to extract the code block.",
        "reject": "The generative model fails to yield valid JSON structures; consequently, elucidate which regular expression is optimal to isolate the code snippet.",
        "reject_type": "over_edited",
        "uncertain_words": []
    },
    # 7. merged_or_shortened_sentences
    {
        "input": "first check the cpu temperature on the server. then if it is normal restart the docker daemon. do not reboot the host machine.",
        "accept": "First, check the CPU temperature on the server. Then, if it is normal, restart the Docker daemon.\n\nDo not reboot the host machine.",
        "reject": "First check CPU temperature on the server, then restart Docker without rebooting the host.",
        "reject_type": "merged_or_shortened_sentences",
        "uncertain_words": []
    },
    # 8. added_exclamation (! added to calm speech)
    {
        "input": "I noticed that the parakeet stt is dropping words when I speak very fast so I will try to to record again with lower speed",
        "accept": "I noticed that the Parakeet STT is dropping words when I speak very fast, so I will try to record again with lower speed.",
        "reject": "I noticed that the Parakeet STT is dropping words when I speak very fast, so I will try to record again with lower speed!",
        "reject_type": "added_exclamation",
        "uncertain_words": []
    },
    # 9. missing_quotes_on_dictated_text
    {
        "input": "open the logger file and write the warning message connection reset by peer in the stderr console",
        "accept": "Open the logger file and write the warning message \"connection reset by peer\" in the stderr console.",
        "reject": "Open the logger file and write the warning message connection reset by peer in the stderr console.",
        "reject_type": "missing_quotes_on_dictated_text",
        "uncertain_words": []
    },
    # 10. Clean pass-through sample (25% rule: input already clean, model must leave it alone)
    {
        "input": "Verify that the PostgreSQL database container is healthy before running the migrations.",
        "accept": "Verify that the PostgreSQL database container is healthy before running the migrations.",
        "reject": "Ensure the PostgreSQL relational database container is verified healthy prior to execution of migrations.",
        "reject_type": "over_edited",
        "uncertain_words": []
    }
]


def generate_batch(output_path: str = "datasets/synthetic_samples_batch1.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in BATCH_1_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Generated {len(BATCH_1_SAMPLES)} samples in {output_path}")


if __name__ == "__main__":
    generate_batch()
