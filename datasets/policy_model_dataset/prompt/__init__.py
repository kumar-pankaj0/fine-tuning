"""
Prompt module for the ASR Policy Dataset Generation Pipeline.
Exports stage-aligned prompts and schemas for:
- Stage 1: Candidate Pair Generation
- Stage 2: Quality Judge & Filtering
- Stage 3: Acoustic Noise Augmentation
"""

from .definition import (
    TITLE,
    CONTEXT,
    DIMENSIONS_HEADING,
    FAITHFULNESS,
    NOISE_REMOVED,
    ERRORS_FIXED,
    FORMATTING,
    EXAMPLES,
    FULL_DEFINITION,
)
from .stage1_pair_generation_prompt import (
    build_pair_generation_prompt,
    build_seed_prompt,
)
from .stage2_quality_judge_prompt import (
    build_quality_judge_prompt,
    build_judge_prompt,
)
from .stage3_noise_augmentation_prompt import (
    build_noise_augmentation_prompt,
    build_augmentation_prompt,
)
from .schemas import (
    SEED_SCHEMA,
    JUDGE_SCHEMA,
    AUGMENTATION_SCHEMA,
    STAGE1_PAIR_GENERATION_SCHEMA,
    STAGE2_QUALITY_JUDGE_SCHEMA,
    STAGE3_NOISE_AUGMENTATION_SCHEMA,
)

__all__ = [
    # Core Definition & Rules
    "TITLE",
    "CONTEXT",
    "DIMENSIONS_HEADING",
    "FAITHFULNESS",
    "NOISE_REMOVED",
    "ERRORS_FIXED",
    "FORMATTING",
    "EXAMPLES",
    "FULL_DEFINITION",
    # Stage 1: Candidate Generation
    "build_pair_generation_prompt",
    "build_seed_prompt",
    "STAGE1_PAIR_GENERATION_SCHEMA",
    "SEED_SCHEMA",
    # Stage 2: Quality Judge
    "build_quality_judge_prompt",
    "build_judge_prompt",
    "STAGE2_QUALITY_JUDGE_SCHEMA",
    "JUDGE_SCHEMA",
    # Stage 3: Noise Augmentation
    "build_noise_augmentation_prompt",
    "build_augmentation_prompt",
    "STAGE3_NOISE_AUGMENTATION_SCHEMA",
    "AUGMENTATION_SCHEMA",
]
