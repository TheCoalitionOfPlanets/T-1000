import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from t1000 import stt


class FakeStream:
    def __init__(self):
        self.chunks = []
        self.pending = 0
        self.finished_calls = 0

    def accept_waveform(self, sample_rate, samples):
        if sample_rate != 16000:
            raise AssertionError("Recognizer input must be resampled to 16 kHz")
        self.chunks.append(np.asarray(samples).copy())
        self.pending += 1

    def input_finished(self):
        self.finished_calls += 1


class FakeRecognizer:
    def __init__(self):
        self.streams = []

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.pending > 0

    def decode_stream(self, stream):
        stream.pending -= 1

    def get_result(self, stream):
        return "recognized speech" if any(np.any(chunk) for chunk in stream.chunks) else ""


class AudioNormalizationTests(unittest.TestCase):
    def test_integer_pcm_keeps_its_level_and_stereo_is_mixed_to_mono(self):
        audio = np.array([[-16384, 0], [8192, 8192], [32767, -32768]], dtype=np.int16)
        result = stt._mono_float32(audio)
        np.testing.assert_allclose(result, [-0.25, 0.25, -1 / 65536])
        self.assertEqual(result.dtype, np.float32)
        # Quiet speech must not be amplified independently for every chunk.
        quiet = stt._mono_float32(np.array([64, -64], dtype=np.int16))
        np.testing.assert_allclose(quiet, [64 / 32768, -64 / 32768])

    def test_unsigned_pcm_silence_is_zero(self):
        np.testing.assert_array_equal(stt._mono_float32(np.array([0, 128, 255], dtype=np.uint8)),
                                      [-1, 0, 127 / 128])

    def test_invalid_audio_never_reaches_the_recognizer(self):
        for audio in [np.array([np.nan]), np.array([np.inf]), np.zeros((1, 2, 3)),
                      np.array([True, False]), np.array(["speech"])]:
            with self.subTest(audio=repr(audio)):
                with self.assertRaises(stt.STTError):
                    stt._mono_float32(audio)


class StreamingTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = FakeRecognizer()
        self.loader = patch.object(stt, "_load_recognizer", return_value=self.recognizer)
        self.load = self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_incremental_audio_is_not_redecoded_and_finalization_is_idempotent(self):
        session = stt.STTSession()
        first = np.full(1600, 0.1, dtype=np.float32)
        second = np.full(3200, 0.2, dtype=np.float32)
        self.assertEqual(session.feed(16000, first), "recognized speech")
        self.assertEqual(session.feed(16000, second), "recognized speech")
        stream = self.recognizer.streams[0]
        self.assertEqual(len(self.recognizer.streams), 1)
        self.assertEqual(session.finish(), "recognized speech")
        accepted = np.concatenate(stream.chunks)
        np.testing.assert_array_equal(accepted[accepted != 0], np.concatenate([first, second]))
        self.assertEqual(accepted[0], 0, "A new stream needs leading context")
        self.assertEqual(accepted[-1], 0, "Final words need a flushed tail")
        self.assertAlmostEqual(session.audio_seconds, 0.3)
        self.assertTrue(session.finished)
        self.assertEqual(stream.finished_calls, 1)
        before = len(stream.chunks)
        self.assertEqual(session.finish(), "recognized speech")
        self.assertEqual(len(stream.chunks), before)
        with self.assertRaises(stt.STTError):
            session.feed(16000, first)

    def test_empty_recording_does_not_load_a_model(self):
        session = stt.STTSession()
        self.assertEqual(session.finish(), "")
        self.load.assert_not_called()

    def test_close_discards_session_state_without_final_decoding(self):
        session = stt.STTSession()
        session.feed(48000, np.full(48000, 0.2, dtype=np.float32))
        stream = self.recognizer.streams[0]
        accepted_chunks = len(stream.chunks)
        previous_text = session.text
        with patch.object(stt, "_recognizer", self.recognizer):
            session.close()
            self.assertIs(stt._recognizer, self.recognizer, "Cancelling one session must retain shared weights")
        self.assertTrue(session.finished)
        self.assertIsNone(session._stream)
        self.assertIsNone(session._recognizer)
        self.assertIsNone(session._resampler)
        self.assertEqual(session.finish(), previous_text)
        self.assertEqual(len(stream.chunks), accepted_chunks)
        self.assertEqual(stream.finished_calls, 0)
        with self.assertRaises(stt.STTError):
            session.feed(48000, np.ones(100, dtype=np.float32))

    def test_file_transcription_closes_native_session_after_decoding_error(self):
        session = Mock(limit_reached=False)
        session.feed.side_effect = stt.STTError("native decoder failed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sf.write(path, np.full(1600, 0.25, dtype=np.float32), 16000)
            with patch.object(stt, "create_session", return_value=session):
                with self.assertRaisesRegex(stt.STTError, "native decoder failed"):
                    stt.transcribe_file(path)
        session.close.assert_called_once()
        session.finish.assert_not_called()

    def test_limit_bounds_audio_even_when_a_chunk_crosses_the_boundary(self):
        session = stt.STTSession(max_seconds=0.1)
        session.feed(16000, np.full(2400, 0.25, dtype=np.float32))
        session.finish()
        accepted = np.concatenate(self.recognizer.streams[0].chunks)
        self.assertEqual(np.count_nonzero(accepted), 1600)
        self.assertAlmostEqual(session.audio_seconds, 0.1)
        self.assertTrue(session.limit_reached)

    def test_chunk_boundaries_do_not_restart_the_resampler(self):
        source_rate = 48000
        audio = (0.2 * np.sin(2 * np.pi * 300 * np.arange(source_rate) / source_rate)).astype(np.float32)
        one = stt.STTSession()
        one.feed(source_rate, audio)
        one.finish()
        many = stt.STTSession()
        for chunk in np.array_split(audio, 17):
            many.feed(source_rate, chunk)
        many.finish()
        one_audio = np.concatenate(self.recognizer.streams[0].chunks)
        many_audio = np.concatenate(self.recognizer.streams[1].chunks)
        np.testing.assert_allclose(many_audio, one_audio, atol=1e-6)
        self.assertAlmostEqual(many.audio_seconds, 1)

    def test_invalid_or_changing_sample_rate_is_rejected(self):
        session = stt.STTSession()
        for rate in [0, -16000, 16000.5, True]:
            with self.subTest(rate=rate):
                with self.assertRaises(stt.STTError):
                    session.feed(rate, np.ones(100, dtype=np.float32))
        session.feed(16000, np.full(100, 0.1, dtype=np.float32))
        with self.assertRaises(stt.STTError):
            session.feed(48000, np.full(100, 0.1, dtype=np.float32))

    def test_upload_transcription_uses_the_same_streaming_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sf.write(path, np.full(8000, 0.25, dtype=np.float32), 16000)
            self.assertEqual(stt.transcribe_file(path), "recognized speech")
        accepted = np.concatenate(self.recognizer.streams[0].chunks)
        self.assertEqual(np.count_nonzero(accepted), 8000)
        self.assertEqual(self.recognizer.streams[0].finished_calls, 1)


class RecognizerLoadingTests(unittest.TestCase):
    def test_missing_model_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(stt, "MODEL_DIR", Path(directory)), patch.object(stt, "_recognizer", None):
                with self.assertRaisesRegex(stt.STTError, "fetch_stt"):
                    stt._load_recognizer()

    def test_recognizer_is_created_once_and_reused(self):
        recognizer = object()
        factory = Mock(return_value=recognizer)
        fake_package = SimpleNamespace(OnlineRecognizer=SimpleNamespace(from_transducer=factory))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]:
                (root / name).touch()
            with patch.object(stt, "MODEL_DIR", root), patch.object(stt, "_recognizer", None), \
                    patch.dict("sys.modules", {"sherpa_onnx": fake_package}):
                self.assertIs(stt._load_recognizer(), recognizer)
                self.assertIs(stt._load_recognizer(), recognizer)
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
