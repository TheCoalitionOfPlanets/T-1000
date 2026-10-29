import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from t1000.vocoder import BufferedVocoderInference
from t1000.vocoder_experiments import (
    EightPatchVocoderInference, configure_vocoder_experiment,
)


class VocoderExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_runtime(self, *, skip=True):
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.config import AudioVAEConfig
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            vocoder = AudioVAE(AudioVAEConfig(
                latent_dim=4, upsample_rates=[2], upsample_kernel_sizes=[4],
                upsample_initial_channel=8, resblock="2",
                resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]],
                downsample_rates=[2], downsample_channels=[2, 4],
                mi_num_layers=2, causal=True, causal_encoder=True,
            )).eval()
        vocoder.dec_mi_layer[1].skip = skip
        inner = VocoderInference(vocoder)
        model = torch.nn.Module()
        model.vocoder = vocoder
        model._optimize_enabled = False
        model.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES = 2
        model._vocoder_inference = BufferedVocoderInference(inner)
        model._get_vocoder_inference = lambda: model._vocoder_inference
        return SimpleNamespace(model=model, optimize=False), inner

    def latents(self, batch=1, patches=23):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            return torch.randn(batch, patches * 4, 4)

    def render(self, adapter, latent):
        batch, frames, latent_dim = latent.shape
        state = adapter.init_stream_state(batch_size=batch, chunk_size=4)
        reused = torch.empty(batch, 4, latent_dim)
        chunks = []
        with torch.inference_mode():
            for start in range(0, frames, 4):
                reused.copy_(latent[:, start:start + 4])
                chunks.append(adapter.stream_step(reused, state, optimize=False))
            chunks.append(adapter.flush(state))
        return torch.cat(chunks, dim=-1), state

    def test_native_lstm_keeps_hidden_state_residual_and_stream_waveform(self):
        for skip in [False, True]:
            with self.subTest(skip=skip):
                baseline, _ = self.make_runtime(skip=skip)
                candidate, _ = self.make_runtime(skip=skip)
                metadata = configure_vocoder_experiment(candidate, "native-lstm")
                self.assertFalse(metadata["acoustic_sampler_changed"])
                for batch in [1, 2]:
                    latent = self.latents(batch=batch)
                    expected, expected_state = self.render(baseline.model._vocoder_inference, latent)
                    actual, actual_state = self.render(candidate.model._vocoder_inference, latent)
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    for result, reference in zip(actual_state.inner.lstm_hidden,
                                                 expected_state.inner.lstm_hidden):
                        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
                    self.assertEqual(actual.shape[-1], 23 * 8)

    def test_native_lstm_does_not_reset_hidden_at_chunk_boundaries(self):
        runtime, inner = self.make_runtime()
        inner._prepare_lstm_stream_params()
        configure_vocoder_experiment(runtime, "native-lstm")
        x = torch.randn(2, 19, inner.vocoder.dec_mi_layer[1].lstm.input_size)
        with torch.inference_mode():
            full, full_hidden = inner._lstm_stream_step(x, None)
            first, hidden = inner._lstm_stream_step(x[:, :7], None)
            second, chunk_hidden = inner._lstm_stream_step(x[:, 7:], hidden)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full)
        for actual, expected in zip(chunk_hidden, full_hidden):
            torch.testing.assert_close(actual, expected)

    def test_batch8_preserves_waveforms_leftovers_rollover_and_mutable_latents(self):
        baseline, _ = self.make_runtime()
        candidate, _ = self.make_runtime()
        metadata = configure_vocoder_experiment(candidate, "batch8")
        self.assertEqual(metadata["vocoder_batch_size"], 8)
        self.assertIsInstance(candidate.model._vocoder_inference, EightPatchVocoderInference)
        for patches in [1, 2, 3, 7, 10, 11, 23, 39]:
            with self.subTest(patches=patches):
                latent = self.latents(batch=2, patches=patches)
                expected, _ = self.render(baseline.model._vocoder_inference, latent)
                actual, state = self.render(candidate.model._vocoder_inference, latent)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                self.assertEqual(actual.shape[-1], patches * 8)
                self.assertFalse(state.pending)

    def test_precision_changes_only_decoder_and_outputs_are_finite(self):
        for variant, precision in [("decoder-bf16", torch.bfloat16),
                                   ("decoder-fp16", torch.float16)]:
            with self.subTest(variant=variant):
                runtime, inner = self.make_runtime()
                recurrent_before = {k: v.clone() for k, v in inner.vocoder.dec_mi_layer.state_dict().items()}
                encoder_before = {k: v.clone() for k, v in inner.vocoder.audio_encoder.state_dict().items()}
                metadata = configure_vocoder_experiment(runtime, variant)
                self.assertEqual(metadata["decoder_dtype"], str(precision))
                for parameter in inner.vocoder.decoder.parameters():
                    self.assertEqual(parameter.dtype, precision)
                for key, value in inner.vocoder.dec_mi_layer.state_dict().items():
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertTrue(torch.equal(value, recurrent_before[key]))
                for key, value in inner.vocoder.audio_encoder.state_dict().items():
                    self.assertTrue(torch.equal(value, encoder_before[key]))
                actual, state = self.render(runtime.model._vocoder_inference, self.latents(patches=7))
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertEqual(actual.dtype, precision)
                self.assertTrue(bool(torch.count_nonzero(actual)))
                self.assertEqual(state.inner.lstm_hidden[0].dtype, torch.float32)

    def test_compiler_targets_only_decoder_window_and_disables_cuda_graphs(self):
        runtime, inner = self.make_runtime()
        original = inner._decode_stream_window
        with patch("t1000.vocoder_experiments.torch.compile", side_effect=lambda target, **kw: target) as compiler:
            metadata = configure_vocoder_experiment(runtime, "compiled-vocoder")
        compiler.assert_called_once_with(
            original, backend="inductor", dynamic=False, fullgraph=False,
            options={"triton.cudagraphs": False},
        )
        self.assertFalse(metadata["compile"]["options"]["triton.cudagraphs"])
        self.assertFalse(runtime.optimize)
        self.assertFalse(runtime.model._optimize_enabled)
        self.assertEqual(inner._decode_stream_window, original)

    def test_bad_or_already_optimized_configuration_fails_before_mutation(self):
        runtime, inner = self.make_runtime()
        before = inner._lstm_stream_step
        with self.assertRaisesRegex(ValueError, "Unknown vocoder experiment"):
            configure_vocoder_experiment(runtime, "half-everything")
        runtime.optimize = True
        with self.assertRaisesRegex(ValueError, "eager runtime"):
            configure_vocoder_experiment(runtime, "native-lstm")
        self.assertEqual(inner._lstm_stream_step, before)


if __name__ == "__main__":
    unittest.main()
