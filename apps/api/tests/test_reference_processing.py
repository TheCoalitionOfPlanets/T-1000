import unittest

import numpy as np

from t1000.reference_processing import process_reference


def tone(rate=16000, seconds=3, frequency=180, amplitude=0.2):
    time = np.arange(round(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


class ReferenceProcessingTests(unittest.TestCase):
    def test_manual_crop_is_sample_exact_without_cleanup_and_does_not_mutate_source(self):
        source = tone()
        unchanged = source.copy()
        result = process_reference(source, 16000, start_seconds=0.25, end_seconds=2.75,
                                   noise_reduction=False)
        np.testing.assert_array_equal(result, source[4000:44000])
        np.testing.assert_array_equal(source, unchanged)
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(np.shares_memory(result, source))

    def test_crop_rounds_fractional_sample_times_and_accepts_exact_minimum(self):
        source = tone(seconds=2)
        result = process_reference(source, 16000, start_seconds=0.10001, end_seconds=1.10001,
                                   noise_reduction=False)
        np.testing.assert_array_equal(result, source[1600:17600])

    def test_quiet_sections_have_less_hiss_and_hum_while_speech_snr_improves(self):
        rate = 16000
        time = np.arange(rate * 4) / rate
        speech = np.zeros(len(time))
        speech[rate:3 * rate] = (
            0.18 * np.sin(2 * np.pi * 180 * time[rate:3 * rate])
            + 0.08 * np.sin(2 * np.pi * 360 * time[rate:3 * rate])
            + 0.04 * np.sin(2 * np.pi * 720 * time[rate:3 * rate])
        )
        noise = np.random.default_rng(7).normal(0, 0.016, len(time))
        noise += 0.018 * np.sin(2 * np.pi * 50 * time)
        source = (speech + noise).astype(np.float32)
        result = process_reference(source, rate)
        self.assertEqual(len(result), len(source))
        self.assertLess(rms(result[2000:12000]), rms(source[2000:12000]) * 0.4)
        voiced = slice(rate + rate // 4, 3 * rate - rate // 4)
        projection = np.dot(result[voiced], speech[voiced]) / np.dot(speech[voiced], speech[voiced])
        self.assertGreater(projection, 0.9)
        self.assertLess(projection, 1.02)
        original_error = rms(source[voiced] - speech[voiced])
        cleaned_error = rms(result[voiced] - speech[voiced])
        self.assertLess(cleaned_error, original_error * 0.5)

    def test_continuous_voicing_is_not_learned_as_background_noise(self):
        for rate in (8000, 16000, 22050, 44100, 48000, 96000, 192000):
            with self.subTest(rate=rate):
                source = tone(rate=rate, seconds=1 + 13 / rate)
                result = process_reference(source, rate)
                self.assertEqual(result.shape, source.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                interior = slice(rate // 5, -rate // 5)
                self.assertGreater(rms(result[interior]), rms(source[interior]) * 0.95)
                self.assertLessEqual(rms(result), rms(source) * 1.001)

    def test_default_cleanup_keeps_quiet_outer_phonemes_and_duration(self):
        rate = 16000
        source = tone(seconds=3)
        source[:rate // 2] *= 0.02
        source[-rate // 2:] *= 0.02
        result = process_reference(source, rate)
        self.assertEqual(len(result), len(source))
        self.assertGreater(rms(result[1000:7000]), rms(source[1000:7000]) * 0.5)
        self.assertGreater(rms(result[-7000:-1000]), rms(source[-7000:-1000]) * 0.5)

    def test_automatic_trim_only_removes_outer_silence_and_keeps_padding(self):
        rate = 16000
        source = np.zeros(rate * 5, dtype=np.float32)
        source[rate:rate * 2] = tone(seconds=1)
        source[rate * 3:rate * 4] = tone(seconds=1)
        result = process_reference(source, rate, auto_trim=True, noise_reduction=False)
        # Starts at 0.88 s and ends at 4.12 s; the middle 1 s pause stays intact.
        np.testing.assert_array_equal(result, source[14080:65920])
        self.assertEqual(len(result), round(3.24 * rate))

    def test_background_estimation_only_uses_the_selected_crop(self):
        rate = 16000
        kept = tone(seconds=2)
        unwanted = np.random.default_rng(11).normal(0, 0.2, rate).astype(np.float32)
        source = np.concatenate([unwanted, kept, unwanted])
        result = process_reference(source, rate, start_seconds=1, end_seconds=3)
        expected = process_reference(kept, rate)
        np.testing.assert_array_equal(result, expected)

    def test_too_short_and_silent_crops_are_rejected(self):
        source = np.concatenate([np.zeros(16000, dtype=np.float32), tone(seconds=2)])
        cases = [
            dict(start_seconds=1, end_seconds=1.5),
            dict(start_seconds=0, end_seconds=1),
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "at least|silent"):
                process_reference(source, 16000, **options)
        short_speech = np.concatenate([np.zeros(16000), tone(seconds=0.4), np.zeros(16000)])
        with self.assertRaisesRegex(ValueError, "at least"):
            process_reference(short_speech, 16000, auto_trim=True, noise_reduction=False)

    def test_invalid_audio_rates_bounds_and_options_are_rejected(self):
        source = tone()
        for samples in ([], [np.nan] * 16000, [np.inf] * 16000, np.zeros((16000, 2)), ["bad"]):
            with self.subTest(samples_type=type(samples)), self.assertRaises(ValueError):
                process_reference(samples, 16000)
        for rate in (True, 16000.0, 7999, 192001):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "sample rate"):
                process_reference(source, rate)
        for options in (
            dict(start_seconds=-1), dict(start_seconds=2, end_seconds=1),
            dict(end_seconds=4), dict(start_seconds=np.nan), dict(end_seconds=np.inf),
            dict(start_seconds=True), dict(end_seconds="2"),
            dict(min_seconds=0), dict(min_seconds=np.nan), dict(min_seconds=True),
            dict(noise_reduction="yes"), dict(auto_trim=1),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                process_reference(source, 16000, **options)


if __name__ == "__main__":
    unittest.main()
