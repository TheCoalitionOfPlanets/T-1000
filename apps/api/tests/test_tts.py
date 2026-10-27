import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from t1000 import tts
from t1000.inference_settings import InferenceSettings


class SlowRuntime:
    device = SimpleNamespace(type="cpu")
    sample_rate = 48000

    def generate_stream(self, **kwargs):
        for _ in range(100):
            time.sleep(0.06)
            yield torch.full((1, 4800), 0.05)


class RuntimeLoadingTests(unittest.TestCase):
    def setUp(self):
        tts.load_runtime.cache_clear()
        self.addCleanup(tts.load_runtime.cache_clear)

    def load(self, *, cuda, optimization=None):
        warmup = Mock(side_effect=lambda: self.assertTrue(torch.is_inference_mode_enabled()))
        runtime = SimpleNamespace(device=SimpleNamespace(type="cuda" if cuda else "cpu"), run_warmup=warmup)
        create = Mock(return_value=runtime)
        module = SimpleNamespace(DotsTtsRuntime=SimpleNamespace(from_pretrained=create))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.safetensors").touch()
            with patch.dict(os.environ), \
                    patch.dict(sys.modules, {"dots_tts.runtime": module}), \
                    patch.object(tts, "MODEL_PATH", model_path), \
                    patch.object(torch.cuda, "is_available", return_value=cuda), \
                    patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=24 * 2**30)), \
                    patch.object(torch.cuda, "is_bf16_supported", return_value=True):
                for key in ("T1000_TTS_OPTIMIZE", "T1000_TTS_PROFILE", "T1000_TTS_GPU_LAYERS"):
                    os.environ.pop(key, None)
                if optimization is not None:
                    os.environ["T1000_TTS_OPTIMIZE"] = optimization
                self.assertIs(tts.load_runtime(), runtime)
                self.assertIs(tts.load_runtime(), runtime)
        create.assert_called_once()
        self.assertFalse(create.call_args.kwargs["warmup_on_optimize"])
        self.assertEqual(warmup.call_count, int(create.call_args.kwargs["optimize"]))
        return create.call_args.kwargs

    def test_cuda_uses_optimized_bf16_runtime_and_reuses_it(self):
        settings = self.load(cuda=True)
        self.assertTrue(settings["optimize"])
        self.assertEqual(settings["precision"], "bfloat16")
        self.assertEqual(settings["max_generate_length"], 300)
        self.assertEqual(settings["vocoder_merge_steps"], 1)

    def test_cpu_never_enables_cuda_optimization(self):
        settings = self.load(cuda=False, optimization="1")
        self.assertFalse(settings["optimize"])
        self.assertEqual(settings["precision"], "float32")
        self.assertEqual(settings["max_generate_length"], 100)

    def test_cuda_optimization_can_be_disabled(self):
        settings = self.load(cuda=True, optimization="0")
        self.assertFalse(settings["optimize"])
        self.assertEqual(settings["precision"], "bfloat16")

    def test_invalid_optimization_setting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "T1000_TTS_OPTIMIZE must be auto, 0, or 1"):
            self.load(cuda=True, optimization="maybe")

    def test_low_memory_profiles_dispatch_to_selected_loader_and_cache_runtime(self):
        for profile, target, expected in (
            ("low-vram", "t1000.low_vram.load_low_vram_runtime", {"precision": "float16"}),
            ("offload", "t1000.offload.load_offloaded_runtime", {"precision": "float16", "gpu_layers": 14}),
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                tts.load_runtime.cache_clear()
                model_path = Path(directory)
                (model_path / "model.safetensors").touch()
                settings = InferenceSettings(profile, "float16", False, 6, 150, 128, gpu_layers=14)
                runtime = object()
                with patch.object(tts, "MODEL_PATH", model_path), \
                        patch.object(tts, "get_inference_settings", return_value=settings), \
                        patch(target, return_value=runtime) as loader:
                    self.assertIs(tts.load_runtime(), runtime)
                    self.assertIs(tts.load_runtime(), runtime)
                loader.assert_called_once_with(model_path, **expected)


class InferenceControlsTests(unittest.TestCase):
    def test_silent_microphone_recording_never_starts_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "microphone.wav"
            sf.write(path, np.zeros(44100 * 6, dtype=np.float32), 44100)
            with patch.object(tts, "load_runtime") as load:
                with self.assertRaisesRegex(gr.Error, "silent"):
                    list(tts.synthesize("Hello.", str(path), "Example transcript.", 10, 42))
                load.assert_not_called()

    def test_nonfinite_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.wav"
            sf.write(path, np.full(24000, np.nan, dtype=np.float32), 24000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "invalid audio"):
                tts.read_reference(path)

    def test_example_reference_is_usable(self):
        audio, sample_rate = tts.read_reference(tts.EXAMPLE_AUDIO)
        self.assertEqual(sample_rate, 24000)
        self.assertGreater(audio.size / sample_rate, 3)
        self.assertGreater(float(np.abs(audio).max()), 0.01)

    def test_stop_releases_the_worker_and_reenables_generation(self):
        with patch.object(tts, "load_runtime", return_value=SlowRuntime()):
            generator = tts.synthesize("Hello.", str(tts.EXAMPLE_AUDIO), tts.EXAMPLE_TRANSCRIPT, 10, 42)
            next(generator)
            tts.stop_generation()
            updates = list(generator)
        self.assertIn("Stopped", updates[-1][1])
        self.assertTrue(updates[-1][2]["interactive"])
        self.assertFalse(updates[-1][3]["interactive"])
        self.assertIsNone(tts.ACTIVE_STOP)
        self.assertFalse(tts.MODEL_LOCK.locked())

    def test_stop_during_model_warmup_does_not_start_audio_generation(self):
        runtime = SimpleNamespace(generate_stream=Mock())

        def load_then_stop():
            tts.stop_generation()
            return runtime

        with patch.object(tts, "load_runtime", side_effect=load_then_stop):
            updates = list(tts.synthesize("Hello.", str(tts.EXAMPLE_AUDIO), tts.EXAMPLE_TRANSCRIPT, 10, 42))
        runtime.generate_stream.assert_not_called()
        self.assertIn("Stopped", updates[-1][1])
        self.assertIsNone(tts.ACTIVE_STOP)
        self.assertFalse(tts.MODEL_LOCK.locked())

    def test_deadline_preserves_partial_audio_and_releases_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tts, "load_runtime", return_value=SlowRuntime()), \
                    patch.object(tts, "GENERATION_TIMEOUT", 0.15), \
                    patch.object(tts, "OUTPUT_DIR", Path(directory)):
                started = time.perf_counter()
                updates = list(tts.synthesize("Hello.", str(tts.EXAMPLE_AUDIO), tts.EXAMPLE_TRANSCRIPT, 10, 42))
                elapsed = time.perf_counter() - started
                self.assertIn("time limit", updates[-1][1])
                self.assertIn("partial audio", updates[-1][1])
                self.assertTrue(Path(updates[-1][0]).is_file())
                self.assertLess(elapsed, 3)
                self.assertFalse(tts.MODEL_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
