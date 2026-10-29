import unittest

from t1000.paragraphs import REQUEST_TEXT_LIMIT, plan_text


class ParagraphPlanningTests(unittest.TestCase):
    def assert_exact(self, text, maximum, segments=None):
        segments = segments if segments is not None else plan_text(text, maximum)
        source = text.strip()
        self.assertTrue(segments)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, len(source))
        self.assertEqual("".join(source[part.start:part.end] for part in segments), source)
        for index, part in enumerate(segments):
            self.assertTrue(part.text)
            self.assertEqual(part.text, source[part.start:part.end].strip())
            self.assertLessEqual(len(part.text), maximum)
            if index:
                self.assertEqual(segments[index - 1].end, part.start)
                self.assertGreater(part.pause_before, 0)
            else:
                self.assertEqual(part.pause_before, 0)
        return segments

    def test_short_input_is_one_unchanged_segment_including_paragraph_break(self):
        text = "  Hello, this is a voice test.\n\nAnother line.  "
        parts = self.assert_exact(text, 150)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].text, text.strip())
        self.assertEqual(parts[0].boundary, "start")

    def test_sentences_are_greedily_grouped_without_word_loss(self):
        text = "The first train arrives soon. The second train is delayed. Please wait here."
        parts = self.assert_exact(text, 58)
        self.assertEqual([p.text for p in parts], [
            "The first train arrives soon. The second train is delayed.", "Please wait here."
        ])
        self.assertEqual(parts[1].boundary, "sentence")
        self.assertEqual(parts[1].pause_before, .18)

    def test_paragraph_boundary_has_its_own_pause_and_preserves_whitespace(self):
        text = "  First paragraph ends here.\r\n \r\nSecond paragraph has two sentences. It stays intact.\nThird paragraph.  "
        parts = self.assert_exact(text, 60)
        self.assertEqual([p.text for p in parts], [
            "First paragraph ends here.",
            "Second paragraph has two sentences. It stays intact.",
            "Third paragraph.",
        ])
        self.assertEqual([p.pause_before for p in parts], [0, .3, .3])

    def test_sentence_terminal_quotes_stay_with_the_sentence(self):
        text = 'She asked, “Are you coming?” Then she said, "Come soon!" We followed her.'
        parts = self.assert_exact(text, 35)
        self.assertEqual(parts[0].text, 'She asked, “Are you coming?”')
        self.assertEqual(parts[1].text, 'Then she said, "Come soon!"')
        self.assertEqual(parts[2].text, 'We followed her.')
        self.assertTrue(all(p.boundary == "sentence" for p in parts[1:]))

    def test_initials_abbreviations_and_decimal_versions_are_not_sentence_ends(self):
        text = "Dr. A. J. Smith works in the U.S. office with version 2.8.0 and costs 3.14 dollars. Next sentence follows."
        parts = self.assert_exact(text, 45)
        self.assertEqual(parts[0].boundary, "start")
        self.assertEqual(parts[1].boundary, "word")
        self.assertEqual(parts[-1].text, "Next sentence follows.")
        self.assertEqual(parts[-1].boundary, "sentence")
        self.assertTrue(parts[0].text.endswith("with"))

    def test_common_dotted_abbreviations_are_not_false_sentence_boundaries(self):
        for abbreviation in ("e.g.", "i.e.", "Mr.", "Mrs.", "Prof.", "U.S.", "A."):
            with self.subTest(abbreviation=abbreviation):
                text = f"Try {abbreviation} one small example with many useful words before the finish. End."
                parts = self.assert_exact(text, 35)
                self.assertFalse(parts[0].text.endswith(abbreviation))
                self.assertEqual(parts[1].boundary, "word")

    def test_overlong_sentence_prefers_clause_then_complete_words(self):
        text = "Before the station, everyone waited quietly beside the unusually bright sign until the train finally arrived."
        parts = self.assert_exact(text, 42)
        self.assertEqual(parts[0].text, "Before the station,")
        self.assertEqual(parts[1].pause_before, .08)
        self.assertEqual(parts[1].boundary, "clause")
        self.assertTrue(any(p.boundary == "word" for p in parts))

    def test_whitespace_fallback_keeps_a_title_with_the_following_name(self):
        text = "Ask the friendly Dr. Smith to wait near the gate."
        parts = self.assert_exact(text, 20)
        self.assertEqual(parts[0].text, "Ask the friendly")
        self.assertTrue(parts[1].text.startswith("Dr. Smith"))

    def test_spaces_tabs_and_unicode_are_never_normalized(self):
        text = "  Café\tvoices  keep\nall their words.   ‘Every ending stays.’\n\nAnother paragraph follows.  "
        self.assert_exact(text, 27)

    def test_exact_per_segment_and_request_boundaries_are_accepted(self):
        text = "x" * 150
        self.assertEqual(plan_text(text, 150)[0].text, text)
        full = ("word " * 1000).rstrip() + "."
        self.assertEqual(len(full), REQUEST_TEXT_LIMIT)
        self.assert_exact(full, 150)

    def test_oversized_request_and_unbroken_token_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "5000 characters"):
            plan_text("word " * 1001, 150)
        with self.assertRaisesRegex(ValueError, "unbroken token.*150-character"):
            plan_text("start " + "x" * 151, 150)

    def test_empty_input_and_invalid_segment_limits_are_rejected(self):
        for text in (None, "", "  \n\t"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                plan_text(text, 150)
        for maximum in (True, 0, -1, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                plan_text("hello", maximum)

    def test_results_are_deterministic_and_no_input_word_is_lost(self):
        text = ("Keep every word exactly where it belongs, and prefer a natural pause.\n\n" * 20).strip()
        first = self.assert_exact(text, 150)
        self.assertEqual(first, plan_text(text, 150))
        self.assertEqual(" ".join(p.text for p in first).split(), text.split())


if __name__ == "__main__":
    unittest.main()
