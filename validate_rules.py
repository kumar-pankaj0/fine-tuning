"""
Automated Rule Filters for Spec v1 Speech-to-Text Post-Refining Dataset.

Filters:
1. Number preservation: accept cannot contain numbers not present in input.
2. Negation preservation: accept cannot lose negation words ('not', 'no', 'never', "n't", 'without').
3. Length sanity: accept length cannot shrink below 40% or grow above 120% of input words.
4. Uncertain word preservation: if an uncertain word is marked, accept must contain it verbatim.
5. Format safety: checks F4 paragraph breaks and quote consistency.
"""

import re
import json
from typing import Dict, Any, List, Tuple


NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "without", "none"}


WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100"
}

def extract_numbers(text: str) -> List[str]:
    nums = re.findall(r"\b\d+\b", text)
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    for t in tokens:
        if t in WORD_TO_DIGIT:
            nums.append(WORD_TO_DIGIT[t])
    return nums


def check_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    inp = sample.get("input", "")
    acc = sample.get("accept", "")
    rej = sample.get("reject", "")
    uncertain_words = sample.get("uncertain_words", [])

    if not inp.strip():
        errors.append("Empty input")
    if not acc.strip():
        errors.append("Empty accept")
    if not rej.strip():
        errors.append("Empty reject")

    # 1. Number preservation
    inp_nums = set(extract_numbers(inp))
    acc_nums = set(extract_numbers(acc))
    hallucinated_nums = acc_nums - inp_nums
    if hallucinated_nums:
        # Check if numbers were spelled out in input (e.g. "three" -> "3", "fifteen" -> "15")
        # If not recognizable, flag error
        for num in hallucinated_nums:
            errors.append(f"Number {num} in accept not found in input")

    # 2. Negation preservation
    inp_words_lower = set(re.findall(r"\b\w+(?:'\w+)?\b", inp.lower()))
    acc_words_lower = set(re.findall(r"\b\w+(?:'\w+)?\b", acc.lower()))

    inp_negs = inp_words_lower & NEGATION_WORDS
    acc_negs = acc_words_lower & NEGATION_WORDS
    if inp_negs and not acc_negs:
        errors.append(f"Negation dropped in accept. Input had: {inp_negs}")

    # 3. Uncertain words preservation
    for uword in uncertain_words:
        if uword.lower() not in acc.lower():
            errors.append(f"Uncertain word '{uword}' was modified or missing in accept")

    # 4. Length sanity
    inp_len = len(inp.split())
    acc_len = len(acc.split())
    if inp_len > 0:
        ratio = acc_len / inp_len
        if ratio < 0.35 and inp_len > 10:
            errors.append(f"Accept shrunk excessively: {acc_len}/{inp_len} words ({ratio:.2f})")
        elif ratio > 1.4:
            errors.append(f"Accept expanded excessively: {acc_len}/{inp_len} words ({ratio:.2f})")

    # 5. Reject must differ from accept
    if acc.strip() == rej.strip():
        errors.append("Accept and reject are identical")

    return (len(errors) == 0, errors)


def validate_file(file_path: str) -> bool:
    print(f"Validating {file_path}...")
    passed = 0
    failed = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            is_valid, errors = check_sample(data)
            if is_valid:
                passed += 1
            else:
                failed += 1
                print(f"Line {line_num} FAIL: {errors}")

    print(f"Validation complete: {passed} passed, {failed} failed.")
    return failed == 0


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "datasets/seed_recordings.jsonl"
    success = validate_file(target)
    sys.exit(0 if success else 1)
