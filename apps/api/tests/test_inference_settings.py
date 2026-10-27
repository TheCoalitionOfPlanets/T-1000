import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from t1000.inference_settings import get_inference_settings


class InferenceSettingsTests(unittest.TestCase):
    def settings(self, *, cuda=True, memory_gib=4, bf16=True, **environment):
        with patch.dict(os.environ, environment, clear=True), \
                patch("torch.cuda.is_available", return_value=cuda), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=memory_gib * 2**30)), \
                patch("torch.cuda.is_bf16_supported", return_value=bf16):
            return get_inference_settings()

    def test_four_and_six_gb_select_bounded_quantized_profile(self):
        for memory in (4, 6):
            with self.subTest(memory=memory):
                settings = self.settings(memory_gib=memory)
                self.assertEqual(settings.profile, "low-vram")
                self.assertFalse(settings.optimize)
                self.assertEqual(settings.reference_limit, 6)
                self.assertEqual(settings.text_limit, 150)
                self.assertEqual(settings.max_generate_length, 128)

    def test_compile_is_automatic_only_on_larger_gpus(self):
        self.assertFalse(self.settings(memory_gib=8).optimize)
        self.assertTrue(self.settings(memory_gib=12).optimize)
        self.assertTrue(self.settings(memory_gib=8, T1000_TTS_OPTIMIZE="1").optimize)

    def test_older_cuda_gpus_use_fp16(self):
        self.assertEqual(self.settings(bf16=False).precision, "float16")

    def test_cpu_keeps_full_precision_eager_inference(self):
        settings = self.settings(cuda=False)
        self.assertEqual(settings.profile, "standard")
        self.assertEqual(settings.precision, "float32")
        self.assertEqual(settings.reference_limit, 10)
        self.assertFalse(settings.optimize)

    def test_offload_allows_partial_gpu_placement(self):
        settings = self.settings(T1000_TTS_PROFILE="offload", T1000_TTS_GPU_LAYERS="14")
        self.assertEqual(settings.profile, "offload")
        self.assertEqual(settings.gpu_layers, 14)
        self.assertFalse(settings.optimize)

    def test_low_memory_modes_reject_compile_and_cpu_execution(self):
        with self.assertRaisesRegex(ValueError, "Low-memory profiles"):
            self.settings(T1000_TTS_OPTIMIZE="1")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            self.settings(cuda=False, T1000_TTS_PROFILE="offload")

    def test_invalid_placement_settings_fail_before_loading_weights(self):
        for environment in (
            {"T1000_TTS_PROFILE": "unknown"},
            {"T1000_TTS_GPU_LAYERS": "many"},
            {"T1000_TTS_GPU_LAYERS": "-1"},
            {"T1000_TTS_GPU_LAYERS": "14"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.settings(**environment)


if __name__ == "__main__":
    unittest.main()
