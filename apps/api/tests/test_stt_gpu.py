import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soxr
import soundfile as sf

from t1000 import stt
from t1000.stt import gpu


class CUDAModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in gpu.MODEL_FILES:
            (self.root / name).touch()
        self.factory = Mock(return_value=object())
        self.ct2 = SimpleNamespace(
            get_cuda_device_count=Mock(return_value=1),
            get_supported_compute_types=Mock(return_value={"int8_float16", "float32"}),
        )
        self.patches = [
            patch.object(gpu, "MODEL_DIR", self.root),
            patch.object(gpu, "_model", None),
            patch.dict("sys.modules", {
                "ctranslate2": self.ct2,
                "faster_whisper": SimpleNamespace(WhisperModel=self.factory),
            }),
        ]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)

    def test_model_explicitly_selects_cuda_quantization_and_offline_files_once(self):
        first = gpu._load_gpu_model()
        self.assertIs(gpu._load_gpu_model(), first)
        self.factory.assert_called_once_with(
            str(self.root), device="cuda", compute_type="int8_float16",
            local_files_only=True, num_workers=1,
        )

    def test_unavailable_or_unsupported_cuda_does_not_fall_back_to_cpu(self):
        for count, compute_types in [(0, {"int8_float16"}), (1, {"float32"})]:
            with self.subTest(device_count=count, compute_types=compute_types):
                self.ct2.get_cuda_device_count.return_value = count
                self.ct2.get_supported_compute_types.return_value = compute_types
                with self.assertRaisesRegex(stt.STTError, "STT_BACKEND=cpu"):
                    gpu._load_gpu_model()
        self.factory.assert_not_called()

    def test_missing_weights_reports_the_gpu_download_command(self):
        (self.root / "model.bin").unlink()
        with self.assertRaisesRegex(stt.STTError, "fetch_whisper"):
            gpu._load_gpu_model()
        self.factory.assert_not_called()


class CUDADecodingTests(unittest.TestCase):
    def test_final_decode_uses_accuracy_settings_and_preview_uses_greedy_search(self):
        model = SimpleNamespace(transcribe=Mock(return_value=(iter([SimpleNamespace(text=" hello ")]), None)))
        audio = np.full(16000, 0.2, dtype=np.float32)
        with patch.object(gpu, "_load_gpu_model", return_value=model):
            self.assertEqual(gpu._decode(audio, final=True)[0].text, " hello ")
            final_kwargs = model.transcribe.call_args.kwargs
            self.assertEqual(final_kwargs["beam_size"], 5)
            self.assertTrue(final_kwargs["vad_filter"])
            self.assertEqual(final_kwargs["vad_parameters"], {"min_silence_duration_ms": 500})
            self.assertFalse(final_kwargs["condition_on_previous_text"])
            model.transcribe.return_value = (iter([]), None)
            gpu._decode(audio, final=False)
            self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 1)
            self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    def test_error_while_iterating_native_segments_unloads_weights(self):
        native = SimpleNamespace(unload_model=Mock())

        def failing_segments():
            yield SimpleNamespace(text="partial")
            raise RuntimeError("CUDA out of memory")

        model = SimpleNamespace(model=native, transcribe=Mock(return_value=(failing_segments(), None)))
        with patch.object(gpu, "_model", model):
            with self.assertRaisesRegex(stt.STTError, "CUDA transcription failed"):
                gpu._decode(np.ones(16000, dtype=np.float32), final=True)
            self.assertIsNone(gpu._model)
        native.unload_model.assert_called_once()

    def test_explicit_release_is_idempotent_and_frees_native_weights(self):
        native = SimpleNamespace(unload_model=Mock())
        with patch.object(gpu, "_model", SimpleNamespace(model=native)):
            gpu.release_gpu_model()
            gpu.release_gpu_model()
            self.assertIsNone(gpu._model)
        native.unload_model.assert_called_once()

    def test_cuda_decode_waits_for_the_shared_tts_compute_lock(self):
        entered = threading.Event()
        finished = threading.Event()
        model = SimpleNamespace(transcribe=Mock(return_value=(iter([]), None)))

        def decode():
            entered.set()
            gpu._decode(np.ones(100, dtype=np.float32), final=True)
            finished.set()

        with patch.object(gpu, "_load_gpu_model", return_value=model):
            with stt.gpu_compute_lock():
                worker = threading.Thread(target=decode)
                worker.start()
                self.assertTrue(entered.wait(1))
                self.assertFalse(finished.wait(0.05))
                model.transcribe.assert_not_called()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())


class CUDASessionTests(unittest.TestCase):
    def setUp(self):
        self.load_patch = patch.object(gpu, "_load_gpu_model", return_value=object())
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.release_patch = patch.object(gpu, "release_gpu_model")
        self.release = self.release_patch.start()
        self.addCleanup(self.release_patch.stop)
        self.decode_patch = patch.object(gpu, "_decode", return_value=[SimpleNamespace(text=" transcript ")])
        self.decode = self.decode_patch.start()
        self.addCleanup(self.decode_patch.stop)

    def test_previews_are_throttled_to_two_seconds_and_have_a_bounded_window(self):
        session = gpu.GPUSTTSession(max_seconds=40)
        second = np.full(16000, 0.2, dtype=np.float32)
        for index in range(32):
            session.feed(16000, second)
            self.assertEqual(self.decode.call_count, (index + 1) // 2)
        for call in self.decode.call_args_list:
            self.assertLessEqual(call.args[0].size, 30 * 16000)
            self.assertFalse(call.kwargs["final"])
        session.finish()
        self.assertEqual(self.decode.call_args.args[0].size, 32 * 16000)
        self.assertTrue(self.decode.call_args.kwargs["final"])
        self.release.assert_called_once()

    def test_final_audio_preserves_every_normalized_sample_and_finish_is_idempotent(self):
        session = gpu.GPUSTTSession()
        first = np.full(24000, 8192, dtype=np.int16)
        second = np.full(24000, -4096, dtype=np.int16)
        session.feed(16000, first)
        session.feed(16000, second)
        session.finish()
        final_audio = self.decode.call_args.args[0]
        expected = np.concatenate([first, second]).astype(np.float32) / 32768
        np.testing.assert_array_equal(final_audio, expected)
        before = self.decode.call_count
        self.assertEqual(session.finish(), "transcript")
        self.assertEqual(self.decode.call_count, before)
        self.assertEqual(session._chunks, [])
        self.assertFalse(session._preview_chunks)

    def test_final_resampling_flush_keeps_fractional_tail(self):
        source = (0.1 * np.sin(np.arange(48013) * 0.03)).astype(np.float32)
        session = gpu.GPUSTTSession()
        for chunk in np.array_split(source, 13):
            session.feed(48000, chunk)
        session.finish()
        expected = soxr.resample(source, 48000, 16000, quality="HQ")
        np.testing.assert_allclose(self.decode.call_args.args[0], expected, atol=1e-6)
        self.release.assert_called_once()

    def test_final_error_still_releases_gpu_weights_and_saved_sample_buffers(self):
        session = gpu.GPUSTTSession()
        session.feed(16000, np.ones(8000, dtype=np.float32))
        self.decode.side_effect = stt.STTError("CUDA failure")
        with self.assertRaisesRegex(stt.STTError, "CUDA failure"):
            session.finish()
        self.release.assert_called_once()
        self.assertEqual(session._chunks, [])
        self.assertFalse(session._preview_chunks)

    def test_pure_silence_skips_whisper_decoding(self):
        session = gpu.GPUSTTSession()
        session.feed(16000, np.zeros(48000, dtype=np.float32))
        self.assertEqual(session.finish(), "")
        self.decode.assert_not_called()
        self.release.assert_called_once()

    def test_close_cancels_without_final_inference_and_drops_audio_buffers(self):
        session = gpu.GPUSTTSession()
        session.feed(48000, np.full(24000, 0.25, dtype=np.float32))
        session.close()
        self.assertTrue(session.finished)
        self.assertEqual(session._chunks, [])
        self.assertFalse(session._preview_chunks)
        self.assertIsNone(session._stream)
        self.assertIsNone(session._resampler)
        self.decode.assert_not_called()
        self.release.assert_called_once()
        self.assertEqual(session.finish(), "")
        with self.assertRaises(stt.STTError):
            session.feed(48000, np.ones(100, dtype=np.float32))

    def test_uploaded_file_skips_drafts_and_decodes_the_full_reference_once(self):
        samples = np.full(48000, 0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sf.write(path, samples, 16000)
            with patch.dict("os.environ", {"STT_BACKEND": "cuda"}):
                self.assertEqual(stt.transcribe_file(path), "transcript")
        self.decode.assert_called_once()
        np.testing.assert_array_equal(self.decode.call_args.args[0], samples)
        self.assertTrue(self.decode.call_args.kwargs["final"])

    def test_backend_choice_is_explicit(self):
        with patch.dict("os.environ", {"STT_BACKEND": "cpu"}):
            self.assertIs(type(stt.create_session()), stt.STTSession)
        with patch.dict("os.environ", {"STT_BACKEND": "cuda"}):
            self.assertIs(type(stt.create_session()), gpu.GPUSTTSession)
        with patch.dict("os.environ", {"STT_BACKEND": "auto"}):
            with self.assertRaisesRegex(stt.STTError, "STT_BACKEND"):
                stt.create_session()


if __name__ == "__main__":
    unittest.main()
