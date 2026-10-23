"""
Stage 3: Acoustic Noise Augmentation Prompt.
Generates diverse authentic noisy ASR input variations (raw_asr) for each clean target (refined_text).
Enforces N -> 1 mapping with invariant targets.
"""

from .definition import CONTEXT, NOISE_REMOVED, ERRORS_FIXED


def build_noise_augmentation_prompt() -> str:
    """
    Constructs the Stage 3 system prompt for generating noisy acoustic ASR input variants.
    """
    return (
        "<role>\n"
        "You are an acoustic noise simulator for STT data augmentation.\n"
        "</role>\n\n"
        "<input_contract>\n"
        "The user message is a JSON array of clean samples.\n"
        'Each sample has "raw_asr" and "refined_text".\n'
        "Array position is the 0-based index.\n"
        'The samples have no "idx" field.\n'
        "</input_contract>\n\n"
        "<reference>\n"
        f"{CONTEXT}\n\n"
        f"{NOISE_REMOVED}\n\n"
        f"{ERRORS_FIXED}\n\n"
        "</reference>\n\n"
        "<task>\n"
        "For each input sample, generate exactly 10 new noisy ASR transcriptions.\n"
        'Use "refined_text" as the clean source.\n'
        'Do not copy the original "raw_asr".\n'
        "</task>\n\n"
        "<rules>\n"
        "- Simulate realistic ASR noise: fillers, false starts, repetition, cut words, phonetic confusion, homophones, and missing word endings.\n"
        "- Output a single continuous stream of spoken text. Never include newline characters (\\n) or markdown list markers (- , 1.).\n"
        "- Remove punctuation. Use lowercase or inconsistent capitalization.\n"
        "- Keep the core meaning.\n"
        '- Keep the word count between 0.4x and 1.3x of "refined_text".\n'
        '- Never output the exact "refined_text".\n'
        "- Make the 10 variants different from each other.\n"
        '- Do not add facts that are not in "refined_text".\n'
        "- Do not swap technical terms or numbers.\n"
        '- The output "idx" is the input array position. Do not invent or reorder indices.\n'
        "- Return one object for every input sample.\n"
        '- Each "noise_raw_asr" array contains exactly 10 strings.\n'
        "</rules>\n\n"
        "<output_contract>\n"
        "Return one JSON object.\n"
        "Do not use markdown fences.\n"
        "Do not add text before or after the JSON.\n"
        'The object contains only the key "augmented_noise_inputs".\n'
        'Each item contains only "idx" and "noise_raw_asr".\n'
        "</output_contract>\n\n"
        "<schema>\n"
        '{"augmented_noise_inputs":[{"idx":0,"noise_raw_asr":["noisy variation 1","noisy variation 2"]}]}\n'
        "</schema>"
    )


# Backwards compatibility alias
build_augmentation_prompt = build_noise_augmentation_prompt
