SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw_asr": {"type": "string"},
                    "refined_text": {"type": "string"},
                },
                "required": ["raw_asr", "refined_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "bad_index": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["bad_index"],
    "additionalProperties": False,
}

AUGMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "augmented_noise_inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "noise_raw_asr": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["idx", "noise_raw_asr"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["augmented_noise_inputs"],
    "additionalProperties": False,
}
