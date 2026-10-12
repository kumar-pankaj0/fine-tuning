"""
Unit tests for STT post-refining data generation and validation pipeline.
"""

import unittest
import json
import os
from validate_rules import check_sample, validate_file
from generate_synthetic_stt import BATCH_1_SAMPLES, generate_batch

class TestDatasetPipeline(unittest.TestCase):
    def test_seed_recordings_valid(self):
        seed_path = "datasets/seed_recordings.jsonl"
        self.assertTrue(os.path.exists(seed_path), "Seed file must exist")
        self.assertTrue(validate_file(seed_path), "Seed recordings must pass all rule filters")

    def test_batch_1_samples_valid(self):
        for idx, sample in enumerate(BATCH_1_SAMPLES):
            is_valid, errors = check_sample(sample)
            self.assertTrue(is_valid, f"Sample {idx+1} failed validation: {errors}")

    def test_reject_not_equal_accept(self):
        for idx, sample in enumerate(BATCH_1_SAMPLES):
            self.assertNotEqual(
                sample["accept"].strip(),
                sample["reject"].strip(),
                f"Sample {idx+1} accept and reject must differ"
            )

    def test_rejection_type_coverage(self):
        types = {sample["reject_type"] for sample in BATCH_1_SAMPLES}
        expected = {
            "flipped_negation",
            "invented_ending",
            "guessed_uncertain_word",
            "added_facts_or_ideas",
            "kept_fillers",
            "over_edited",
            "merged_or_shortened_sentences",
            "added_exclamation",
            "missing_quotes_on_dictated_text",
        }
        self.assertTrue(expected.issubset(types), f"Missing reject types: {expected - types}")

    def test_clean_input_preservation(self):
        clean_sample = BATCH_1_SAMPLES[-1]
        self.assertEqual(clean_sample["input"], clean_sample["accept"])

if __name__ == "__main__":
    unittest.main()
