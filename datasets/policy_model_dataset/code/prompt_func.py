"""
Prompt Loader Module for ASR Policy Dataset Generation Pipeline.
Maps pipeline stages to their respective prompt builders:
- Stage 1: Candidate Pair Generation
- Stage 2: Quality Judge & Filtering
- Stage 3: Acoustic Noise Augmentation
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompt.stage1_pair_generation_prompt import (
    build_pair_generation_prompt,
    build_seed_prompt,
)
from prompt.stage2_quality_judge_prompt import (
    build_quality_judge_prompt,
    build_judge_prompt,
)
from prompt.stage3_noise_augmentation_prompt import (
    build_noise_augmentation_prompt,
    build_augmentation_prompt,
)


async def load_system_prompt(
    domain: str = "general technical and developer workflows",
    num_samples: int = 10,
    type: str = "stage1_pair_generation",
) -> str:
    """
    Construct specialized system prompts for the ASR dataset generation pipeline.

    Args:
        domain: Target technical domain for synthetic pair generation.
        num_samples: Number of initial samples requested during generation.
        type: Prompt mode identifier. Supported values:
            - 'stage1_pair_generation' (alias 'input_output_pair'): Generates initial synthetic (raw_asr, refined_text) pairs.
            - 'stage2_quality_judge' (alias 'remove_bad_pair'): LLM judge prompt that identifies invalid or hallucinated pairs.
            - 'stage3_noise_augmentation' (alias 'asr_data_augmentation'): Generates 10 diverse noisy ASR inputs for each clean target.

    Returns:
        str: Formatted system prompt enforcing strict JSON output.

    Raises:
        ValueError: If an unrecognized prompt type is provided.
    """
    if type in ("stage1_pair_generation", "input_output_pair"):
        return build_pair_generation_prompt(domain, num_samples)
    elif type in ("stage2_quality_judge", "remove_bad_pair"):
        return build_quality_judge_prompt()
    elif type in ("stage3_noise_augmentation", "asr_data_augmentation"):
        return build_noise_augmentation_prompt()
    else:
        raise ValueError(
            f"Unknown prompt type: '{type}'. Supported: 'stage1_pair_generation', 'stage2_quality_judge', 'stage3_noise_augmentation' "
            "(or legacy aliases: 'input_output_pair', 'remove_bad_pair', 'asr_data_augmentation')"
        )
