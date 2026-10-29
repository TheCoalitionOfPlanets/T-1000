import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from t1000.vocoder import (
    BufferedVocoderInference, configure_vocoder_batching, read_vocoder_batch_size,
)


class VocoderConfigurationTests(unittest.TestCase):
    def test_environment_defaults_to_one_and_accepts_only_one_or_four(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_vocoder_batch_size(), 1)
        for value, expected in [("1", 1), ("4", 4), (" 4 ", 4)]:
            with patch.dict(os.environ, {"T1000_TTS_VOCODER_BATCH_SIZE": value}):
                self.assertEqual(read_vocoder_batch_size(), expected)
        for value in ["", "0", "2", "-1", "true", "4.0"]:
            with self.subTest(value=value), patch.dict(
                os.environ, {"T1000_TTS_VOCODER_BATCH_SIZE": value}
            ):
                with self.assertRaisesRegex(ValueError, "T1000_TTS_VOCODER_BATCH_SIZE"):
                    read_vocoder_batch_size()

    def test_default_and_compiled_runtimes_keep_native_adapter(self):
        untouched = object()
        self.assertIs(configure_vocoder_batching(untouched, 1), untouched)
        model = nn.Module()
        model._optimize_enabled = True
        model._get_vocoder_inference = Mock()
        runtime = SimpleNamespace(model=model, optimize=True)
        self.assertIs(configure_vocoder_batching(runtime, 4), runtime)
        model._get_vocoder_inference.assert_not_called()

    def test_opt_in_installs_once_and_preserves_nonstream_operations(self):
        model = nn.Module()
        model._optimize_enabled = False
        model.vocoder = nn.Identity()
        model.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES = 2
        inner = Mock()
        model._get_vocoder_inference = lambda: getattr(model, "_vocoder_inference", inner)
        runtime = SimpleNamespace(model=model, optimize=False)
        configure_vocoder_batching(runtime, 4)
        adapter = model._vocoder_inference
        self.assertIsInstance(adapter, BufferedVocoderInference)
        self.assertIs(adapter.inner, inner)
        self.assertEqual(model._vocoder_inference_vocoder_id, id(model.vocoder))
        configure_vocoder_batching(runtime, 4)
        self.assertIs(model._vocoder_inference, adapter)
        adapter.clear()
        inner.clear.assert_called_once_with()
        adapter.extract_latents("reference", do_sample=False)
        inner.extract_latents.assert_called_once_with("reference", do_sample=False)

    def test_optimized_streams_pass_native_state_and_arguments_through(self):
        model = nn.Module()
        model._optimize_enabled = False
        inner = Mock()
        adapter = BufferedVocoderInference(inner, model=model)
        # A subsequent upstream mode change must not multiply compiled windows.
        model._optimize_enabled = True
        state = adapter.init_stream_state(batch_size=2, chunk_size=16)
        self.assertIs(state, inner.init_stream_state.return_value)
        inner.init_stream_state.assert_called_once_with(batch_size=2, chunk_size=16)
        latent = torch.zeros(2, 16, 4)
        adapter.stream_step(latent, state, optimize=True, profile_step="3-6", use_compiled=True)
        inner.stream_step.assert_called_once_with(
            latent, state, optimize=True, profile_step="3-6", use_compiled=True,
        )
        adapter.flush(state)
        inner.flush.assert_called_once_with(state)

    def test_invalid_constructor_parameters_fail_immediately(self):
        for value in [True, 0, 2, "4", 4.0]:
            with self.subTest(batch_size=value), self.assertRaises(ValueError):
                BufferedVocoderInference(Mock(), batch_size=value)
        for value in [True, -1, "2", 2.0]:
            with self.subTest(initial_unmerged=value), self.assertRaises(ValueError):
                BufferedVocoderInference(Mock(), initial_unmerged=value)


class RealVocoderBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.config import AudioVAEConfig
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            vocoder = AudioVAE(AudioVAEConfig(
                latent_dim=4, upsample_rates=[2], upsample_kernel_sizes=[4],
                upsample_initial_channel=8, resblock="2",
                resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]],
                downsample_rates=[2], downsample_channels=[2, 4],
                mi_num_layers=1, causal=True, causal_encoder=True,
            )).eval()
        cls.native = VocoderInference(vocoder)
        cls.buffered = BufferedVocoderInference(cls.native)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def render(self, adapter, latents):
        batch, frames, latent_dim = latents.shape
        state = adapter.init_stream_state(batch_size=batch, chunk_size=4)
        chunks = []
        reused = torch.empty(batch, 4, latent_dim)
        with torch.inference_mode():
            for start in range(0, frames, 4):
                # Every yield aliases the same storage, as an acoustic decoder
                # with an in-place output buffer is permitted to do.
                reused.copy_(latents[:, start:start + 4])
                chunks.append(adapter.stream_step(reused, state, optimize=False))
            chunks.append(adapter.flush(state))
        return torch.cat(chunks, dim=-1)

    def test_real_waveforms_match_across_leftovers_batches_and_window_rollover(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            for batch in [1, 2]:
                # 23 patches exceed the tiny native decoder's 33-frame window;
                # the other cases exercise initial cadence and final leftovers.
                for patches in [1, 2, 3, 6, 7, 23]:
                    with self.subTest(batch=batch, patches=patches):
                        latents = torch.randn(batch, patches * 4, 4)
                        expected = self.render(self.native, latents)
                        actual = self.render(self.buffered, latents)
                        self.assertEqual(tuple(actual.shape), (batch, 1, patches * 8))
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_abandoned_stream_does_not_leak_pending_audio_to_next_request(self):
        state = self.buffered.init_stream_state(chunk_size=4)
        with torch.inference_mode():
            for _ in range(3):
                self.buffered.stream_step(torch.full((1, 4, 4), 10.0), state, optimize=False)
        # Abandon with one unflushed group, as cancellation/closing a generator
        # can do. A new request must start with fresh audio and recurrent state.
        latents = torch.zeros(1, 28, 4)
        torch.testing.assert_close(
            self.render(self.buffered, latents), self.render(self.native, latents),
            rtol=1e-4, atol=1e-5,
        )

    def test_pending_empty_audio_uses_decoder_dtype_and_rejects_bad_shapes(self):
        state = self.buffered.init_stream_state(chunk_size=4)
        with torch.inference_mode():
            for _ in range(3):
                empty = self.buffered.stream_step(
                    torch.zeros(1, 4, 4, dtype=torch.bfloat16), state, optimize=False,
                )
        self.assertEqual(tuple(empty.shape), (1, 1, 0))
        self.assertEqual(empty.dtype, torch.float32)
        for shape in [(1, 4), (2, 4, 4), (1, 4, 3), (1, 0, 4)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                self.buffered.stream_step(torch.zeros(shape), state, optimize=False)
        with self.assertRaisesRegex(RuntimeError, "enable optimization"):
            self.buffered.stream_step(torch.zeros(1, 4, 4), state, optimize=True)


if __name__ == "__main__":
    unittest.main()
