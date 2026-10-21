from .definition import (
    CONTEXT,
    DIMENSIONS_HEADING,
    FAITHFULNESS,
    NOISE_REMOVED,
    ERRORS_FIXED,
    FORMATTING,
)


def build_judge_prompt() -> str:
    return (
        "<role>\n"
        "You are a strict data quality evaluator for an STT refinement dataset.\n"
        "</role>\n\n"
        "<input_contract>\n"
        "The user message is a JSON array of candidate pairs.\n"
        "Array position is the 0-based index.\n"
        'The pairs have no "idx" field.\n'
        "</input_contract>\n\n"
        "<reference>\n"
        f"{CONTEXT}\n\n"
        f"{DIMENSIONS_HEADING}\n\n"
        f"{FAITHFULNESS}\n\n"
        f"{NOISE_REMOVED}\n\n"
        f"{ERRORS_FIXED}\n\n"
        f"{FORMATTING}\n"
        "</reference>\n\n"
        "<task>\n"
        "Find every pair that fails any dimension of the definition.\n"
        "</task>\n\n"
        "<rules>\n"
        "- Judge each pair against all four dimensions: faithfulness, noise removed, errors fixed, formatting.\n"
        "- Mark a pair bad for invented facts, lost facts, swapped terms, leftover fillers, unresolved confident errors, or formatting that clearly violates the rules.\n"
        "- Do not mark a pair bad for a different but valid wording.\n"
        "- Do not mark a pair bad only because it is short.\n"
        "- Do not rewrite pairs. Only return indices.\n"
        "- Use 0-based indices. Each index must be inside the input array range.\n"
        "- Do not repeat an index.\n"
        "- If every pair is good, return an empty array.\n"
        "</rules>\n\n"
        "<output_contract>\n"
        "Return one JSON object.\n"
        "Do not use markdown fences.\n"
        "Do not add text before or after the JSON.\n"
        'The object contains only the key "bad_index".\n'
        '"bad_index" is an array of integers.\n'
        "</output_contract>\n\n"
        "<schema>\n"
        '{"bad_index":[0,3]}\n'
        "</schema>"
    )
