import unittest
from unittest.mock import patch

from t1000.text import expand_english_text


class EnglishExpansionTests(unittest.TestCase):
    def test_money_and_date_use_explicit_english_readings(self):
        self.assertEqual(expand_english_text("The total is $12.50.", max_length=150),
                         "The total is twelve point five dollars.")
        self.assertEqual(expand_english_text("Meet on 10/29/2026.", max_length=150),
                         "Meet on the twenty ninth of october twenty twenty six.")

    def test_plain_contractions_are_preserved(self):
        text = "I didn't think you'd come."
        self.assertEqual(expand_english_text(text, max_length=150), text)

    def test_recognized_phone_formats_require_manual_spelling(self):
        for text in ("Call 555-0123.", "Call 555-123-4567.", "Call +1 (555) 123-4567."):
            with self.subTest(text=text), patch("dots_tts.utils.text.normalize_english_text") as normalize:
                with self.assertRaisesRegex(ValueError, "phone numbers as you want them spoken"):
                    expand_english_text(text, max_length=150)
                normalize.assert_not_called()

    def test_expansion_is_rejected_when_it_exceeds_generation_limit(self):
        with patch("dots_tts.utils.text.normalize_english_text", return_value="x" * 151):
            with self.assertRaisesRegex(ValueError, "151 characters.*150-character limit"):
                expand_english_text("123", max_length=150)

    def test_empty_or_oversized_input_is_rejected_before_normalization(self):
        with patch("dots_tts.utils.text.normalize_english_text") as normalize:
            for text in ("  ", "x" * 151):
                with self.assertRaises(ValueError):
                    expand_english_text(text, max_length=150)
            normalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
