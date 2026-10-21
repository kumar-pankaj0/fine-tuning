from prompt.seed_prompt import build_seed_prompt
from prompt.judge_prompt import build_judge_prompt
from prompt.augmentation_prompt import build_augmentation_prompt


async def load_system_prompt(
    domain: str = "general technical and developer workflows",
    num_samples: int = 10,
    type: str = "input_output_pair",
) -> str:
    """
    Construct specialized system prompts for the ASR dataset generation pipeline.

    Args:
        domain: Target technical domain for synthetic pair generation.
        num_samples: Number of initial samples requested during generation.
        type: Prompt mode identifier. Supported values:
            - 'input_output_pair': Generates initial synthetic (raw_asr, refined_text) pairs.
            - 'remove_bad_pair': LLM judge prompt that identifies invalid or hallucinated pairs.
            - 'asr_data_augmentation': Generates 10 diverse noisy ASR inputs for each clean target.

    Returns:
        str: Formatted system prompt enforcing strict JSON output.

    Raises:
        ValueError: If an unrecognized prompt type is provided.
    """
    if type == "input_output_pair":
        return build_seed_prompt(domain, num_samples)
    elif type == "remove_bad_pair":
        return build_judge_prompt()
    elif type == "asr_data_augmentation":
        return build_augmentation_prompt()
    else:
        raise ValueError(
            f"Unknown prompt type: '{type}'. Supported: 'input_output_pair', 'remove_bad_pair', 'asr_data_augmentation'"
        )
