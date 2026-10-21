def asr_deterministic_removal_func(pair: dict) -> bool:
    raw = pair["input"]["raw_asr"].strip()
    out = pair["output"].strip()
    out_words = out.lower().split()
    raw_words = raw.lower().split()

    # 1. No filler words or "---" dividers left in output
    bad_words = {"um", "uh", "erm"}
    if any(w.strip(".,!?") in bad_words for w in out_words) or "---" in out:
        return False

    # 2. Backticks (`), double quotes ("), and bold (**) must be closed
    if out.count("`") % 2 != 0 or out.count('"') % 2 != 0 or out.count("**") % 2 != 0:
        return False

    # 3. No triple newlines (\n\n\n), no "....", and max 3 bold pairs
    if "\n\n\n" in out or "...." in out or out.count("**") > 6:
        return False

    # 4. Newline threshold based on word count (Matches your Guide Section 4!)
    if len(out_words) < 25 and "\n" in out:
        return False
    if out.count("\n") > (len(out_words) // 15):  # blocks 9 \n in a short text!
        return False

    # 5. Fast $0 replacement for Semantic Drift (Length Ratio Check)
    if not (0.4 * len(raw_words) <= len(out_words) <= 1.3 * len(raw_words)):
        return False

    return True
