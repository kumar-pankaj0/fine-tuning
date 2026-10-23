from prompt_func import load_system_prompt
from agy import agy_call
from prompt.schemas import SEED_SCHEMA, JUDGE_SCHEMA, AUGMENTATION_SCHEMA
import asyncio
import argparse
import json

SEED_MODEL = "gemini-3.8-flash-high"
JUDGE_MODEL = "gemini-3.8-flash-medium"
AUGMENTATION_MODEL = "gemini-3.8-flash-high"


async def generate_pairs(system_prompt: str, user_prompt: str) -> list[dict]:
    """
    Request initial synthetic ASR training pairs from the primary generator model.

    Args:
        system_prompt: System prompt with generation guidelines and JSON schema.
        user_prompt: Specific domain instruction or seed topic prompt.

    Returns:
        list[dict]: List of generated (raw_asr, refined_text) dictionary pairs.
    """
    data = await agy_call(
        system_prompt,
        user_prompt,
        model=SEED_MODEL,
        json_schema=SEED_SCHEMA,
    )
    return parse_response(data, type="generate_pairs")


def parse_response(data, type: str):
    """
    Extract and parse structured JSON payload from an agy response.

    Args:
        data: Parsed JSON payload returned by agy_call.
        type: Parsing target mode:
            - 'remove_bad_pairs': Extracts list of integer indices under 'bad_index'.
            - 'data_augmentation': Extracts sample list under 'augmented_noise_inputs'.
            - 'generate_pairs': Extracts pairs list under 'pairs' (or top-level data).

    Returns:
        The parsed payload matching the requested type contract.
    """
    if type == "remove_bad_pairs":
        return [int(i) for i in data["bad_index"]]
    elif type == "data_augmentation":
        return data["augmented_noise_inputs"]
    elif type == "generate_pairs":
        return data["pairs"] if isinstance(data, dict) and "pairs" in data else data
    return data


async def remove_index_pairs(pairs: list[dict], bad_index: list[int]) -> list[dict]:
    """
    Drop pairs located at the specified indices.

    Args:
        pairs: List of sample dictionary pairs.
        bad_index: Integer indices marked for deletion.

    Returns:
        list[dict]: Surviving clean pairs.
    """
    index_set = set(bad_index)
    return [pair for idx, pair in enumerate(pairs) if idx not in index_set]


async def remove_bad_pairs(llm_system_prompt: str, input_pair: list[dict], determinstic_removal_func) -> list[dict]:
    """
    Two-stage quality filter combining deterministic heuristics and LLM evaluation.

    Stage 1 runs fast rule-based checks (fillers, syntax, ratios).
    Stage 2 sends remaining pairs to the judge model to detect hallucinations and drift.

    Args:
        llm_system_prompt: Evaluation instructions for the judge model.
        input_pair: Initial list of candidate pairs.
        determinstic_removal_func: Async callable executing deterministic checks.

    Returns:
        list[dict]: Cleaned and verified pairs.
    """
    llm_input_pair = await determinstic_removal_func(input_pair)
    data = await agy_call(
        llm_system_prompt,
        json.dumps(llm_input_pair, indent=2),
        model=JUDGE_MODEL,
        json_schema=JUDGE_SCHEMA,
    )

    bad_pair_index = parse_response(data, type="remove_bad_pairs")
    filtered_pair = await remove_index_pairs(llm_input_pair, bad_pair_index)
    return filtered_pair


def validate_noise_input(noise_input: str, refined_text: str) -> bool:
    """
    Validate quality and length constraints for an augmented noisy ASR candidate.

    Checks:
    1. Rejects empty strings.
    2. Rejects noisy inputs identical to the target text.
    3. Rejects length ratios outside [0.4x, 1.3x] of the target word count.

    Args:
        noise_input: Candidate noisy ASR string.
        refined_text: Reference clean target string.

    Returns:
        bool: True if candidate passes all checks; False otherwise.
    """
    if not noise_input or not refined_text:
        return False
    if "\n" in noise_input:
        return False
    if noise_input.strip().lower() == refined_text.strip().lower():
        return False

    n_word = len(noise_input.split())
    r_word = len(refined_text.split())
    if r_word > 0 and not (0.4 * r_word <= n_word <= 1.3 * r_word):
        return False

    return True


async def _data_augmentation(
    system_prompt: str, user_prompt: str, batch_filtered_pair: list[dict]
) -> list[dict]:
    """
    Execute batch augmentation request and pair 10 noisy variants per sample.

    Outer loop selects target sample text.
    Inner loop iterates over 10 generated noisy variants, validates each, and pairs them.

    Args:
        system_prompt: Augmentation system prompt.
        user_prompt: JSON payload of clean pairs for this batch.
        batch_filtered_pair: Original slice of clean pairs for this batch.

    Returns:
        list[dict]: Augmented list of (raw_asr, refined_text) pairs.
    """
    data = await agy_call(
        system_prompt,
        user_prompt,
        model=AUGMENTATION_MODEL,
        json_schema=AUGMENTATION_SCHEMA,
    )
    noise_pairs = parse_response(data, type="data_augmentation")

    batch_augmented_pair = []
    for noise_dict in noise_pairs:
        refined_text = batch_filtered_pair[noise_dict["idx"]]["refined_text"]
        for noise_input in noise_dict["noise_raw_asr"]:
            if validate_noise_input(noise_input, refined_text):
                batch_augmented_pair.append(
                    {"raw_asr": noise_input, "refined_text": refined_text}
                )

    return batch_augmented_pair


async def data_augmentation(
    augmentation_prompt: str, filtered_pair: list[dict], batch_size: int = 10
) -> list[dict]:
    """
    Partition clean pairs into batches and augment concurrently with the local agy CLI.

    Args:
        augmentation_prompt: System prompt for ASR noise simulation.
        filtered_pair: Source clean pairs to augment.
        batch_size: Number of samples per parallel LLM request.

    Returns:
        list[dict]: Consolidated list of all augmented pairs.
    """
    task = []
    combined_response = []
    for idx in range(0, len(filtered_pair), batch_size):
        batch_filtered_pair = filtered_pair[idx : idx + batch_size]
        user_prompt = json.dumps(batch_filtered_pair, indent=2)
        task.append(
            _data_augmentation(augmentation_prompt, user_prompt, batch_filtered_pair)
        )

    batchs_response = await asyncio.gather(*task)

    for batched_output in batchs_response:
        combined_response.extend(batched_output)

    return combined_response


async def write_in_file(varified_pair: list[dict], file_path: str):
    """
    Serialize final verified dataset pairs into a formatted JSON file.

    Args:
        varified_pair: Final verified dataset pairs.
        file_path: Destination file path.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(varified_pair, f, ensure_ascii=False, indent=2)


async def asr_determinstic_removal_func(input_pair: list[dict]) -> list[dict]:
    """
    Deterministic rule-based filtering to drop bad pairs before sending to the LLM:
    1. Removes filler/pause tokens ('um', 'uh', 'erm') or '---' dividers left in output.
    2. Ensures markdown formatting (`, ", **) is properly closed (even count).
    3. Blocks triple newlines (\\n\\n\\n), repetitive dots ('....'), and excessive bold tags (> 6).
    4. Blocks newlines in short text (< 25 words) and enforces max newline ratio (len // 25).
    5. Length ratio heuristic (word count between 0.4x and 1.3x of raw ASR input).

    Args:
        input_pair: Candidate pairs to evaluate.

    Returns:
        list[dict]: Clean pairs that passed all deterministic checks.
    """
    valid_pairs = []
    pause_words = {"um", "uh", "erm"}

    for data_dict in input_pair:
        raw = str(data_dict.get("raw_asr") or data_dict.get("input", "")).strip()
        out = str(data_dict.get("refined_text") or data_dict.get("output", "")).strip()

        # Drop empty samples
        if not raw or not out:
            continue

        raw_words = raw.lower().split()
        out_words = out.lower().split()

        # Check 1: Output must not contain spoken filler words or divider dashes
        if any(w.strip(".,!?\"':;") in pause_words for w in out_words) or "---" in out:
            continue

        # Check 2: Formatting pairs must be closed (even count for backticks, quotes, bold)
        if (
            out.count("`") % 2 != 0
            or out.count('"') % 2 != 0
            or out.count("**") % 2 != 0
        ):
            continue

        # Check 3: Block ugly formatting artifacts
        if "\n\n\n" in out or "...." in out or out.count("**") > 6:
            continue

        # Check 4A: Very short answers (< 25 words) must be continuous
        if len(out_words) < 25 and "\n" in out:
            continue

        # Check 4B: Newline budget enforces max newline ratio
        if out.count("\n") > max(1, len(out_words) // 25):
            continue

        # Check 5: Fast length ratio check
        if len(raw_words) > 0:
            ratio = len(out_words) / len(raw_words)
            if not (0.4 <= ratio <= 1.3):
                continue

        valid_pairs.append(data_dict)

    return valid_pairs


async def generate(
    domain: str = "general technical and developer workflows",
    num_samples: int = 10,
    user_prompt: str = "",
    file_path: str = "augmented_dataset.json",
):
    """
    Execute the full end-to-end synthetic dataset generation pipeline.

    Workflow:
    1. Loads generation prompt and creates initial candidate pairs.
    2. Applies deterministic heuristic filter.
    3. Runs LLM quality filter to drop hallucinated pairs.
    4. Applies 1-to-10 data augmentation on clean pairs with inline validation.
    5. Writes final dataset to disk.

    Args:
        domain: Domain description for dataset generation.
        num_samples: Number of initial pairs to generate.
        user_prompt: User instruction for generation.
        file_path: Target output JSON file path.
    """
    system_prompt = await load_system_prompt(
        domain=domain, num_samples=num_samples, type="input_output_pair"
    )
    input_output_pair = await generate_pairs(system_prompt, user_prompt)

    remove_bad_pair_prompt = await load_system_prompt(
        type="remove_bad_pair",
    )
    filtered_pair = await remove_bad_pairs(
        llm_system_prompt=remove_bad_pair_prompt,
        input_pair=input_output_pair,
        determinstic_removal_func=asr_determinstic_removal_func,
    )

    augmentation_prompt = await load_system_prompt(type="asr_data_augmentation")
    augmented_pair = await data_augmentation(augmentation_prompt, filtered_pair)
    return await write_in_file(augmented_pair, file_path)


async def main():
    """
    Command-line entrypoint for the dataset generation pipeline.
    """
    parser = argparse.ArgumentParser(description="ASR Policy Model Dataset Generation Pipeline")
    parser.add_argument("--domain", type=str, default="general technical and developer workflows", help="Domain for dataset generation")
    parser.add_argument("--num-samples", type=int, default=10, help="Initial number of samples to generate")
    parser.add_argument("--output", type=str, default="augmented_dataset.json", help="Output file path for verified pairs")
    parser.add_argument("--prompt", type=str, default="", help="Optional user guidance prompt for generator")
    args = parser.parse_args()

    print(f"Starting generation pipeline for domain: {args.domain}")
    await generate(
        domain=args.domain,
        num_samples=args.num_samples,
        user_prompt=args.prompt,
        file_path=args.output,
    )
    print(f"Dataset generation complete. Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
