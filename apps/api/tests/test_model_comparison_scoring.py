import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("score_tts_comparison", Path(__file__).resolve().parents[3] / "scripts/score_tts_comparison.py")
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


class ComparisonScoringTests(unittest.TestCase):
    def test_errors_count_missing_repeated_and_changed_words(self):
        result = scoring.error_metrics("Please send the blue parcel now.", "please send red parcel parcel")
        self.assertEqual(result["word"]["errors"], 3)
        self.assertEqual(result["word"]["reference_units"], 6)
        self.assertEqual(sum(result["word"][key] for key in ("substitutions", "deletions", "insertions")), 3)

    def test_normalization_preserves_digits_and_does_not_expand_numbers(self):
        self.assertEqual(scoring.normalized_text("  WE’RE at 12.50!  "), "were at 1250")
        self.assertGreater(scoring.error_metrics("twelve", "12")["word"]["errors"], 0)

    def test_weighted_word_error_aggregate_is_not_mean_sentence_rate(self):
        rows = [{"score_status": "ok", "asr": scoring.error_metrics("one", "wrong")},
                {"score_status": "ok", "asr": scoring.error_metrics("two three four five", "two three four five")}]
        self.assertEqual(scoring.aggregate(rows)["word"]["rate"], .2)

    def test_cap_or_failed_audio_is_excluded_from_speed_ratio(self):
        base = {"case_id": "case", "repeat": 1, "status": "ok", "stop_reason": "eos", "finite": True,
                "nonzero": True, "score_status": "ok", "generation_seconds": 2, "audio_seconds": 1,
                "real_time_factor": 2, "speaker_cosine": .8}
        a = dict(base, label="soar")
        b = dict(base, label="mf2", generation_seconds=1, stop_reason="schedule_exhausted")
        self.assertFalse(scoring.paired_comparisons([a, b], ["soar", "mf2"])[0]["eligible"])
        b["stop_reason"] = "eos"
        self.assertEqual(scoring.paired_comparisons([a, b], ["soar", "mf2"])[0]["generation_speedup_first_over_second"], 2)

    def test_empty_reference_rejected_and_empty_hypothesis_is_deletions(self):
        with self.assertRaises(ValueError):
            scoring.error_metrics("?!", "anything")
        self.assertEqual(scoring.error_metrics("one two", "")["word"]["deletions"], 2)

    def test_duplicate_pairs_rejected(self):
        row = {"case_id": "case", "repeat": 1, "label": "soar"}
        with self.assertRaises(ValueError):
            scoring.paired_comparisons([row, row], ["soar", "mf2"])


if __name__ == "__main__":
    unittest.main()
