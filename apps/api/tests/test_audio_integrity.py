import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from t1000 import audio_integrity as integrity


class AtomicAudioWriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_ordered_chunks_have_exact_sample_offsets_and_pcm16_frames(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        first = np.array([0, 1 / 32768, -1 / 32768], dtype=np.float32)
        second = np.array([.5, -.5, 1, -1], dtype=np.float32)
        self.assertEqual(writer.append(first), (0, 3))
        self.assertEqual(writer.append(np.empty(0, dtype=np.float32)), (3, 0))
        self.assertEqual(writer.append(second), (3, 4))
        self.assertEqual(writer.frames, 7)
        self.assertEqual(writer.chunk_count, 2)
        self.assertEqual(list(self.root.glob("speech-*.wav")), [])
        self.assertEqual(len(list(self.root.glob(".speech-*.wav.part"))), 1)
        result = writer.finish(complete=True, reason="eos")
        self.assertIsInstance(result, str)
        self.assertIsInstance(result, integrity.GeneratedAudio)
        samples, rate = sf.read(result, dtype="int16")
        np.testing.assert_array_equal(samples, [0, 1, -1, 16384, -16384, 32767, -32768])
        self.assertEqual(rate, 16000)
        self.assertEqual(result.metadata["total_samples"], 7)
        self.assertEqual(result.metadata["duration_seconds"], 7 / 16000)
        self.assertEqual(result.metadata["chunks"], 2)
        self.assertEqual(result.metadata["channels"], 1)
        self.assertEqual(result.metadata["sha256"], hashlib.sha256(Path(result).read_bytes()).hexdigest())
        self.assertEqual(result.metadata["byte_length"], Path(result).stat().st_size)
        self.assertTrue(result.metadata["complete"])
        self.assertEqual(result.metadata["reason"], "eos")
        self.assertFalse(result.metadata["clipped"])
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_atomic_publication_occurs_only_after_closed_wav_validation(self):
        writer = integrity.AtomicAudioWriter(self.root, 48000)
        writer.append(np.array([.25, -.25], dtype=np.float32))
        stream = writer._file
        replace = integrity.os.replace

        def checked_replace(source, target):
            self.assertTrue(stream.closed)
            self.assertFalse(Path(target).exists())
            self.assertEqual(sf.info(source).frames, 2)
            return replace(source, target)

        with patch.object(integrity.os, "replace", side_effect=checked_replace) as publish:
            result = writer.finish(complete=True, reason="eos")
        self.assertEqual(publish.call_count, 1)
        self.assertEqual(len(list(self.root.glob("speech-*.wav"))), 1)
        self.assertTrue(Path(result).is_file())

    def test_partial_finish_is_valid_and_cannot_be_upgraded_by_duplicate_finish(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.array([.25, -.25], dtype=np.float32))
        result = writer.finish(complete=False, reason="cancelled")
        self.assertFalse(result.metadata["complete"])
        self.assertEqual(result.metadata["reason"], "cancelled")
        self.assertEqual(sf.info(result).frames, 2)
        self.assertIs(writer.finish(complete=True, reason="eos"), result)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            writer.append(np.array([.1], dtype=np.float32))

    def test_empty_finish_cleans_up_without_publishing_a_fake_complete_file(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.empty(0, dtype=np.float32))
        self.assertIsNone(writer.finish(complete=False, reason="cancelled"))
        self.assertIsNone(writer.finish(complete=True, reason="eos"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_pcm_never_writes_data_or_changes_preceding_samples(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        for data in [np.array([np.nan]), np.array([np.inf]), np.zeros((2, 1)), np.array([1, 2])]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    writer.append(data)
                self.assertEqual(list(self.root.iterdir()), [])
        writer.append(np.array([.5, -.5], dtype=np.float32))
        with self.assertRaises(ValueError):
            writer.append(np.array([np.nan], dtype=np.float32))
        result = writer.finish(complete=False, reason="invalid_audio")
        np.testing.assert_array_equal(sf.read(result, dtype="int16")[0], [16384, -16384])
        self.assertEqual(result.metadata["total_samples"], 2)

    def test_pcm16_saturation_is_counted_in_metadata(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.array([1.25, -1.25, 1, -1, 0], dtype=np.float32))
        result = writer.finish(complete=True, reason="eos")
        np.testing.assert_array_equal(sf.read(result, dtype="int16")[0], [32767, -32768, 32767, -32768, 0])
        self.assertEqual(result.metadata["clipped_sample_count"], 2)
        self.assertTrue(result.metadata["clipped"])

    def test_partial_write_failure_rolls_back_only_failed_chunk_and_preserves_prefix(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.array([.5, -.5], dtype=np.float32))
        original_write = writer._file.write

        def failed_write(samples):
            original_write(samples[:1])
            raise OSError("simulated disk write failure")

        with patch.object(writer._file, "write", side_effect=failed_write):
            with self.assertRaisesRegex(OSError, "disk write failure"):
                writer.append(np.array([.25, -.25], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "finalize the valid prefix"):
            writer.append(np.array([.1], dtype=np.float32))
        result = writer.finish(complete=True, reason="eos")
        self.assertFalse(result.metadata["complete"])
        self.assertEqual(result.metadata["reason"], "write_error")
        self.assertIn("disk write failure", result.metadata["error"])
        self.assertEqual(result.metadata["chunks"], 1)
        np.testing.assert_array_equal(sf.read(result, dtype="int16")[0], [16384, -16384])

    def test_failed_initial_rollback_can_recover_prefix_after_closing(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.array([.5, -.5], dtype=np.float32))
        original_write = writer._file.write

        def failed_write(samples):
            original_write(samples[:1])
            raise OSError("write failed")

        with patch.object(writer._file, "write", side_effect=failed_write), \
                patch.object(writer._file, "truncate", side_effect=OSError("busy writer")):
            with self.assertRaises(OSError):
                writer.append(np.array([.25, -.25], dtype=np.float32))
        result = writer.finish(complete=False, reason="error")
        np.testing.assert_array_equal(sf.read(result, dtype="int16")[0], [16384, -16384])
        self.assertEqual(result.metadata["total_samples"], 2)
        self.assertFalse(result.metadata["complete"])

    def test_failed_publication_keeps_valid_hidden_file_for_retry(self):
        writer = integrity.AtomicAudioWriter(self.root, 16000)
        writer.append(np.array([.5, -.5], dtype=np.float32))
        with patch.object(integrity.os, "replace", side_effect=OSError("publish unavailable")):
            with self.assertRaisesRegex(OSError, "publish unavailable"):
                writer.finish(complete=False, reason="error")
        self.assertEqual(list(self.root.glob("speech-*.wav")), [])
        self.assertEqual(len(list(self.root.glob(".speech-*.wav.part"))), 1)
        result = writer.finish(complete=False, reason="error")
        self.assertEqual(sf.info(result).frames, 2)
        self.assertFalse(result.metadata["complete"])


class DecodeEndObserverTests(unittest.TestCase):
    def test_eos_is_observed_only_after_the_final_patch_and_instance_method_restores(self):
        state = SimpleNamespace(end_flag=False)

        def decode(*, state):
            yield "last patch"
            state.end_flag = True

        model = SimpleNamespace(_decode=decode)
        with integrity.DecodeEndObserver(SimpleNamespace(model=model)) as observer:
            stream = model._decode(state=state)
            self.assertEqual(next(stream), "last patch")
            self.assertEqual(observer.stop_reason, "unknown")
            self.assertEqual(list(stream), [])
            self.assertEqual(observer.stop_reason, "eos")
            self.assertEqual(observer.decoded_patches, 1)
        self.assertIs(model._decode, decode)

    def test_schedule_exhaustion_and_class_method_restoration(self):
        class Model:
            def _decode(self, state):
                yield 1
                yield 2

        model = Model()
        original = model._decode.__func__
        with integrity.DecodeEndObserver(model) as observer:
            self.assertEqual(list(model._decode(SimpleNamespace(end_flag=False))), [1, 2])
            self.assertEqual(observer.stop_reason, "schedule_exhausted")
        self.assertNotIn("_decode", model.__dict__)
        self.assertIs(model._decode.__func__, original)

    def test_generator_close_does_not_claim_eos_even_if_finally_sets_flag(self):
        state = SimpleNamespace(end_flag=False)

        def decode(*, state):
            try:
                yield 1
                yield 2
            finally:
                state.end_flag = True

        model = SimpleNamespace(_decode=decode)
        with integrity.DecodeEndObserver(model) as observer:
            stream = model._decode(state=state)
            next(stream)
            stream.close()
            self.assertTrue(state.end_flag)
            self.assertEqual(observer.stop_reason, "closed")
        self.assertIs(model._decode, decode)

    def test_context_exit_closes_unfinished_decode_and_restores_original(self):
        closed = []

        def decode(*, state):
            try:
                yield 1
                yield 2
            finally:
                closed.append(True)
                state.end_flag = True

        model = SimpleNamespace(_decode=decode)
        with integrity.DecodeEndObserver(model) as observer:
            stream = model._decode(state=SimpleNamespace(end_flag=False))
            next(stream)
        self.assertEqual(closed, [True])
        self.assertIs(model._decode, decode)
        self.assertEqual(observer.stop_reason, "closed")
        self.assertEqual(list(stream), [])
        self.assertEqual(observer.stop_reason, "closed")

    def test_decoder_error_is_retained_and_original_restored(self):
        def decode(*, state):
            yield 1
            raise RuntimeError("decoder failed")

        model = SimpleNamespace(_decode=decode)
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            with integrity.DecodeEndObserver(model) as observer:
                list(model._decode(state=SimpleNamespace(end_flag=False)))
        self.assertEqual(observer.stop_reason, "error")
        self.assertIs(model._decode, decode)

    def test_fake_runtime_without_decode_remains_unknown_and_unchanged(self):
        model = SimpleNamespace()
        with integrity.DecodeEndObserver(SimpleNamespace(model=model)) as observer:
            self.assertEqual(observer.stop_reason, "unknown")
        self.assertEqual(model.__dict__, {})

    def test_non_cpu_flag_does_not_trigger_a_synchronizing_boolean_conversion(self):
        class NonCpuScalar:
            def __bool__(self):
                raise AssertionError("Must not read a device scalar")

        model = SimpleNamespace(_decode=lambda **unused: iter([1]))
        with integrity.DecodeEndObserver(model) as observer:
            self.assertEqual(list(model._decode(state=SimpleNamespace(end_flag=NonCpuScalar()))), [1])
            self.assertEqual(observer.stop_reason, "unknown")


if __name__ == "__main__":
    unittest.main()
