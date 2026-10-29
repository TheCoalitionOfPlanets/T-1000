import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from t1000.references import _PromptAudioLoader, configure_prompt_audio


class ReferencePreprocessingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def save(self, name, audio, sample_rate=24000):
        path = self.directory / name
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        return path

    def test_quiet_opening_and_closing_phonemes_are_preserved_by_default(self):
        sample_rate = 24000
        audio = np.zeros(sample_rate * 2, dtype=np.float32)
        quiet = 0.004 * np.sin(2 * np.pi * 600 * np.arange(6000) / sample_rate)
        audio[2400:8400] = quiet
        audio[12000:36000] = 0.4 * np.sin(2 * np.pi * 180 * np.arange(sample_rate) / sample_rate)
        audio[38400:44400] = quiet
        path = self.save("quiet-edges.wav", audio)
        runtime = SimpleNamespace(sample_rate=sample_rate, transcript="Quiet edges stay spoken.")
        with patch.dict(os.environ, {"T1000_REFERENCE_TRIM": "0"}):
            configure_prompt_audio(runtime)
        result = runtime._load_prompt_audio(path)
        self.assertEqual(tuple(result.shape), (1, audio.size))
        np.testing.assert_array_equal(result.numpy()[0], audio)
        self.assertEqual(runtime.transcript, "Quiet edges stay spoken.")
        # Confirm the regression fixture actually loses the quiet phonemes with
        # the pinned upstream relative-threshold trimming operation.
        trimmed = _PromptAudioLoader(sample_rate, trim=True)(path)
        self.assertLess(trimmed.numel(), result.numel())
        self.assertLess(trimmed.numel(), 36000)

    def test_stereo_mixing_and_resampling_keep_gain_and_expected_frame_count(self):
        from dots_tts.utils.audio import high_quality_resample

        for source_rate in [16000, 24000, 44100, 48000]:
            with self.subTest(source_rate=source_rate):
                time = np.arange(source_rate + 13, dtype=np.float32) / source_rate
                mono = 0.1 * np.sin(2 * np.pi * 220 * time)
                stereo = np.stack([mono, mono * 0.5], axis=1)
                path = self.save(f"stereo-{source_rate}.wav", stereo, source_rate)
                result = _PromptAudioLoader(48000)(path)
                expected = high_quality_resample(
                    torch.from_numpy(stereo.mean(axis=1)).unsqueeze(0),
                    orig_sr=source_rate, target_sr=48000,
                )
                self.assertEqual(result.size(-1), math.ceil(stereo.shape[0] * 48000 / source_rate))
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(result.device.type, "cpu")
                torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_invalid_and_empty_audio_are_rejected(self):
        for name, samples in [("empty.wav", []), ("nan.wav", [0.1, np.nan]),
                              ("infinity.wav", [np.inf, 0.1])]:
            with self.subTest(name=name):
                path = self.save(name, np.asarray(samples, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "empty|invalid"):
                    _PromptAudioLoader(24000)(path)

    def test_duplicate_content_paths_share_cache_and_return_independent_storage(self):
        from dots_tts.utils.audio import high_quality_resample

        samples = np.linspace(-0.1, 0.1, 1200, dtype=np.float32)
        first = self.save("frozen-first.wav", samples)
        second = self.save("frozen-second.wav", samples)
        loader = _PromptAudioLoader(48000)
        with patch("dots_tts.utils.audio.high_quality_resample", wraps=high_quality_resample) as resample:
            with torch.inference_mode():
                output = loader(first)
            output.zero_()
            cached = loader(second)
            resample.assert_called_once()
        self.assertGreater(float(cached.abs().max()), 0)
        self.assertEqual(len(loader._cache), 1)
        self.assertFalse(torch.is_inference(next(iter(loader._cache.values()))))
        cached.zero_()
        self.assertGreater(float(loader(first).abs().max()), 0)

    def test_changed_content_or_sample_rate_cannot_hit_stale_cache(self):
        from dots_tts.utils.audio import high_quality_resample

        path = self.save("changing.wav", np.ones(100, dtype=np.float32) * 0.1)
        loader = _PromptAudioLoader(24000)
        with patch("dots_tts.utils.audio.high_quality_resample", wraps=high_quality_resample) as resample:
            first = loader(path)
            self.save("changing.wav", np.ones(100, dtype=np.float32) * 0.2)
            second = loader(path)
            self.save("changing.wav", np.ones(100, dtype=np.float32) * 0.2, 16000)
            third = loader(path)
            self.assertEqual(resample.call_count, 3)
        self.assertFalse(torch.equal(first, second))
        self.assertNotEqual(second.size(-1), third.size(-1))

    def test_cache_enforces_entry_byte_limits_and_skips_oversize_waveforms(self):
        loader = _PromptAudioLoader(24000, max_entries=2, max_bytes=100)
        for index in range(4):
            loader(self.save(f"small-{index}.wav", np.full(10, 0.1 + index, dtype=np.float32)))
        self.assertEqual(len(loader._cache), 2)
        self.assertEqual(loader._cache_bytes, 80)
        loader(self.save("larger.wav", np.full(20, 0.2, dtype=np.float32)))
        self.assertEqual(len(loader._cache), 1)
        self.assertEqual(loader._cache_bytes, 80)
        loader(self.save("oversize.wav", np.full(30, 0.3, dtype=np.float32)))
        self.assertEqual(len(loader._cache), 1)
        self.assertEqual(loader._cache_bytes, 80)

    def test_configuration_is_per_runtime_idempotent_and_trim_requires_explicit_opt_in(self):
        first = SimpleNamespace(sample_rate=24000)
        second = SimpleNamespace(sample_rate=48000)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(configure_prompt_audio(first), first)
            self.assertFalse(first._load_prompt_audio.trim)
            loader = first._load_prompt_audio
            configure_prompt_audio(first)
            configure_prompt_audio(second)
            self.assertIs(first._load_prompt_audio, loader)
            self.assertIsNot(first._load_prompt_audio, second._load_prompt_audio)
        with patch.dict(os.environ, {"T1000_REFERENCE_TRIM": "1"}):
            configure_prompt_audio(first)
            self.assertTrue(first._load_prompt_audio.trim)
        with patch.dict(os.environ, {"T1000_REFERENCE_TRIM": "maybe"}):
            with self.assertRaisesRegex(ValueError, "T1000_REFERENCE_TRIM"):
                configure_prompt_audio(first)


if __name__ == "__main__":
    unittest.main()
