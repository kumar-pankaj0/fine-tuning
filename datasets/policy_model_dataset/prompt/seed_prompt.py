from .definition import FULL_DEFINITION


def build_seed_prompt(domain: str, num_samples: int) -> str:
    return (
        "<role>\n"
        "You are an expert synthetic data generator for Speech-to-Text (STT) refinement.\n"
        "</role>\n\n"
        "<task>\n"
        f"Generate exactly {num_samples} realistic training pairs for the domain: {domain}.\n"
        "Each pair maps one noisy ASR transcript to one clean technical document.\n"
        "</task>\n\n"
        "<reference>\n"
        f"{FULL_DEFINITION}\n"
        "</reference>\n\n"
        "<input_contract>\n"
        "The user message may contain extra guidance. Follow it when present.\n"
        "The guidance does not change the output schema.\n"
        "</input_contract>\n\n"
        "<output_contract>\n"
        "Return one JSON object.\n"
        "Do not use markdown fences.\n"
        "Do not add text before or after the JSON.\n"
        'The object contains only the key "pairs".\n'
        'Each item in "pairs" contains only "raw_asr" and "refined_text".\n'
        "</output_contract>\n\n"
        "<rules>\n"
        "- The two fields describe the same content. The raw_asr is the noisy spoken form.\n"
        "- The refined_text is the clean written form of that same content.\n"
        "- Make raw_asr sound like a real ASR transcript: lowercase, missing punctuation, fillers, false starts, cut words, and phonetic errors.\n"
        "- Remove every filler, false start, and social artifact from refined_text.\n"
        "- Fix every acoustic error that you can resolve with confidence.\n"
        "- Convert spoken numbers to digits when the meaning is clear. Example: one hundred -> 100.\n"
        "- Resolve phonetic decompositions of known technical terms. Example: cube ctl -> kubectl.\n"
        "- Preserve a term when you are not confident. Never invent or swap a fact.\n"
        "- Apply the formatting rules by length. Do not use the same structure for every output.\n"
        "- Vary topics, sentence length, error types, and formatting across the pairs.\n"
        "- Do not copy the reference examples. Use them only as style guides.\n"
        "- Never let raw_asr equal refined_text.\n"
        '- Do not include an "idx" key.\n'
        f'- Return exactly {num_samples} items in "pairs".\n'
        "</rules>\n\n"
        "<schema>\n"
        '{"pairs":[{"raw_asr":"noisy spoken transcript","refined_text":"clean written text"}]}\n'
        "</schema>"
    )
