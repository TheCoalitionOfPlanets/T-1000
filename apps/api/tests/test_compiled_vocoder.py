import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from t1000.compiled_vocoder import (
    configure_compiled_vocoder, read_compiled_vocoder_enabled,
)
from t1000.vocoder import BufferedVocoderInference


class FakeInner:
    def __init__(self):
        self.vocoder = torch.nn.Module()
        self.vocoder.decoder = torch.nn.Linear(1, 1)
        self.frame_counts = []
        self.windows = []
        self.flushes = 0
        self.inference_contexts = []

    def _decode_stream_window(self, window):
        self.windows.append(tuple(window.shape))
        self.inference_contexts.append(torch.is_inference_mode_enabled())
        return window.new_empty((1, 1, 0))

    def init_stream_state(self, *, batch_size, chunk_size):
        self.inference_contexts.append(torch.is_inference_mode_enabled())
        self.chunk_size = chunk_size
        return SimpleNamespace(decoder=SimpleNamespace(
            window=torch.empty(batch_size, 128, 64).zero_()))

    def stream_step(self, latent, state, *, optimize, use_compiled, profile_step=None):
        if optimize:
            raise AssertionError("Full optimized inference must remain disabled")
        self.frame_counts.append(latent.shape[1])
        return self._decode_stream_window(state.decoder.window)

    def flush(self, state):
        self.flushes += 1
        return self._decode_stream_window(state.decoder.window)


def fake_runtime():
    inner = FakeInner()
    model = torch.nn.Module()
    model.vocoder = inner.vocoder
    model.core = SimpleNamespace(latent_patch_size=4, latent_dim=128)
    model._optimize_enabled = False
    adapter = BufferedVocoderInference(inner, batch_size=4, model=model)
    model._get_vocoder_inference = lambda: adapter
    return SimpleNamespace(device=torch.device("cuda"), optimize=False, model=model), inner


class CompiledVocoderTests(unittest.TestCase):
    def test_flag_defaults_off_and_rejects_ambiguous_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(read_compiled_vocoder_enabled())
        for value, expected in (("0", False), ("1", True), (" 1 ", True)):
            with patch.dict(os.environ, {"T1000_TTS_COMPILE_VOCODER": value}):
                self.assertEqual(read_compiled_vocoder_enabled(), expected)
        for value in ("auto", "yes", "", "2"):
            with patch.dict(os.environ, {"T1000_TTS_COMPILE_VOCODER": value}):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    read_compiled_vocoder_enabled()

    def test_disabled_path_never_inspects_or_compiles_runtime(self):
        runtime = object()
        with patch("torch.compile") as compiler:
            self.assertIs(configure_compiled_vocoder(runtime, enabled=False), runtime)
        compiler.assert_not_called()
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            configure_compiled_vocoder(runtime, enabled="1")

    def test_warmup_uses_actual_buffering_flush_and_only_window_compilation(self):
        runtime, inner = fake_runtime()
        original_window = inner._decode_stream_window
        zeros = torch.zeros
        latent_allocations = []

        def cpu_zeros(shape, *, device, dtype):
            latent_allocations.append((shape, device, dtype))
            return zeros(shape, dtype=dtype)

        with patch("torch.compile", side_effect=lambda target, **options: target) as compiler, \
                patch("t1000.compiled_vocoder.torch.zeros", side_effect=cpu_zeros), \
                patch("torch.cuda.synchronize") as synchronize:
            self.assertIs(configure_compiled_vocoder(runtime, enabled=True), runtime)
            self.assertIs(configure_compiled_vocoder(runtime, enabled=True), runtime)
        compiler.assert_called_once_with(
            original_window, backend="inductor", dynamic=False, fullgraph=False,
            options={"triton.cudagraphs": False},
        )
        self.assertEqual(latent_allocations, [((1, 4, 128), runtime.device, torch.bfloat16)])
        self.assertEqual(inner.chunk_size, 16)
        self.assertEqual(inner.frame_counts, [4, 4, 16, 12])
        self.assertEqual(inner.flushes, 1)
        self.assertEqual(inner.windows, [(1, 128, 64)] * 5)
        self.assertTrue(all(inner.inference_contexts))
        synchronize.assert_called_once_with(runtime.device)
        self.assertEqual(runtime._t1000_compiled_vocoder["status"], "ready")
        self.assertFalse(runtime.optimize)
        self.assertFalse(runtime.model._optimize_enabled)

    def test_unsupported_runtime_fails_before_compilation(self):
        for device, runtime_optimize, model_optimize in (("cpu", False, False),
                                                       ("cuda", True, False),
                                                       ("cuda", False, True)):
            runtime, inner = fake_runtime()
            runtime.device = torch.device(device)
            runtime.optimize = runtime_optimize
            runtime.model._optimize_enabled = model_optimize
            with patch("torch.compile") as compiler:
                with self.assertRaisesRegex(ValueError, "eager low-VRAM CUDA"):
                    configure_compiled_vocoder(runtime, enabled=True)
            compiler.assert_not_called()

    def test_compile_failure_is_visible_and_cannot_double_wrap(self):
        runtime, inner = fake_runtime()
        with patch("torch.compile", side_effect=RuntimeError("compiler failed")) as compiler:
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                configure_compiled_vocoder(runtime, enabled=True)
            with self.assertRaisesRegex(RuntimeError, "load a fresh runtime"):
                configure_compiled_vocoder(runtime, enabled=True)
        compiler.assert_called_once()
        self.assertEqual(runtime._t1000_compiled_vocoder["status"], "failed")

    def test_runtime_loader_applies_hook_once_and_rejects_other_profiles(self):
        from t1000 import tts
        from t1000.inference_settings import InferenceSettings

        tts.load_runtime.cache_clear()
        self.addCleanup(tts.load_runtime.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.safetensors").touch()
            settings = InferenceSettings("low-vram", "bfloat16", False, 6, 150, 128)
            runtime = SimpleNamespace(sample_rate=48000)
            with patch.object(tts, "MODEL_PATH", model_path), \
                    patch.object(tts, "get_inference_settings", return_value=settings), \
                    patch.dict(os.environ, {"T1000_TTS_COMPILE_VOCODER": "1", "T1000_TTS_EAGER_CACHE": "0"}), \
                    patch("t1000.low_vram.load_low_vram_runtime", return_value=runtime), \
                    patch.object(tts, "configure_compiled_vocoder", return_value=runtime) as configure:
                self.assertIs(tts.load_runtime(), runtime)
                self.assertIs(tts.load_runtime(), runtime)
                configure.assert_called_once_with(runtime, enabled=True)
                tts.load_runtime.cache_clear()
                settings = InferenceSettings("offload", "bfloat16", False, 6, 150, 128)
                with patch.object(tts, "get_inference_settings", return_value=settings):
                    with self.assertRaisesRegex(ValueError, "requires the low-vram profile"):
                        tts.load_runtime()
