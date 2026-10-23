import re

PAUSE_WORDS = {"um", "uh", "erm"}

def validate_pair(pair: dict, is_practical_domain: bool = False) -> tuple[bool, str]:
    """
    Validate a single (raw_asr, refined_text) pair against all deterministic criteria.
    Returns (is_valid, reason).
    """
    raw = str(pair.get("raw_asr") or pair.get("input", "")).strip()
    out = str(pair.get("refined_text") or pair.get("output", "")).strip()

    if not raw or not out:
        return False, "Empty raw or refined text"

    # Check ASCII
    try:
        raw.encode("ascii")
        out.encode("ascii")
    except UnicodeEncodeError as e:
        return False, f"Non-ASCII character found: {e}"

    if "\\u" in raw or "\\u" in out:
        return False, "Found escaped unicode sequence"

    # Raw ASR checks: must be a continuous speech stream without newlines
    if "\n" in raw:
        return False, "Newline in raw_asr"
    if re.search(r"(^|\s)-\s", raw):
        return False, "Bullet dash in raw_asr"
    if re.search(r"(^|\s)\d+\.\s", raw):
        return False, "Numbered list marker in raw_asr"

    raw_words = raw.lower().split()
    out_words = out.lower().split()

    # Check spoken filler words in refined_text
    for w in out_words:
        cleaned_w = w.strip(".,!?\"':;")
        if cleaned_w in PAUSE_WORDS:
            return False, f"Spoken pause word '{cleaned_w}' in refined text"

    if "---" in out:
        return False, "Divider dashes '---' in refined text"

    # Delimiter balance
    if out.count("`") % 2 != 0:
        return False, f"Unbalanced backticks: {out.count('`')}"
    if out.count('"') % 2 != 0:
        return False, f"Unbalanced double quotes: {out.count('\"')}"
    if out.count("**") % 2 != 0:
        return False, f"Unbalanced bold tags: {out.count('**')}"

    # Artifacts
    if "\n\n\n" in out:
        return False, "Triple newline in refined text"
    if "...." in out:
        return False, "Repetitive dots '....' in refined text"
    if out.count("**") > 6:
        return False, f"Excessive bold tags: {out.count('**')}"

    # Newline rules
    if len(out_words) < 25 and "\n" in out:
        return False, "Newline in short text (< 25 words)"

    max_newlines = max(1, len(out_words) // 25)
    if out.count("\n") > max_newlines:
        return False, f"Newline count {out.count(chr(10))} exceeds max budget {max_newlines}"

    # Length ratio heuristic
    ratio = len(out_words) / max(1, len(raw_words))
    if ratio < 0.4 or ratio > 1.3:
        return False, f"Word length ratio {ratio:.2f} outside [0.4, 1.3] (raw: {len(raw_words)}, out: {len(out_words)})"

    # Practical domains backtick check
    if is_practical_domain and "`" in out:
        return False, "Backticks found in practical non-code domain"

    return True, "OK"
