import asyncio
import base64
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from t1000 import mic


def pcm(samples):
    return base64.b64encode(np.asarray(samples, dtype="<i2").tobytes()).decode("ascii")


class FakeSession:
    def __init__(self, max_seconds):
        self.text = ""
        self.chunks = []
        self.finished_calls = 0
        self.closed_calls = 0

    def feed(self, sample_rate, audio):
        self.chunks.append(np.asarray(audio).copy())
        self.text = "live transcript"
        return self.text

    def finish(self):
        self.finished_calls += 1
        self.text = "complete transcript"
        return self.text

    def close(self):
        self.closed_calls += 1


class RecorderTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path_patch = patch.object(mic, "RECORDING_DIR", self.root)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.session_patch = patch.object(mic, "create_session", side_effect=FakeSession)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.service = mic.RecorderService(max_seconds=1)
        self.token = self.service._start({"sample_rate": 16000})["token"]

    def feed(self, samples, sequence=0):
        return self.service._feed({"token": self.token, "sequence": sequence, "pcm": pcm(samples)})

    def finish(self, samples):
        return self.service._finish({"token": self.token, "pcm": pcm(samples)})

    def test_complete_recording_retains_the_last_partial_chunk_without_replaying_audio(self):
        first = np.array([20, -30, 40], dtype=np.int16)
        complete = np.array([20, -30, 40, -50, 60], dtype=np.int16)
        self.feed(first)
        session = self.service._recordings[self.token].session
        self.assertEqual(self.finish(complete), {"token": self.token})
        # A duplicate final request must not append tail audio a second time.
        self.assertEqual(self.finish(complete), {"token": self.token})
        path, text, _ = self.service.consume(self.token)
        recorded, rate = sf.read(path, dtype="int16")
        np.testing.assert_array_equal(recorded, complete)
        np.testing.assert_array_equal(np.concatenate(session.chunks), complete)
        self.assertEqual(rate, 16000)
        self.assertEqual(text, "complete transcript")
        self.assertEqual(session.finished_calls, 1)
        self.assertEqual(session.closed_calls, 1)

    def test_retrying_an_acknowledged_chunk_does_not_duplicate_words_or_audio(self):
        samples = [100, -200, 300]
        self.assertEqual(self.feed(samples)["sequence"], 1)
        self.assertEqual(self.feed(samples)["sequence"], 1)
        self.finish(samples)
        entry = self.service._recordings[self.token]
        self.assertEqual(len(entry.session.chunks), 1)
        np.testing.assert_array_equal(entry.session.chunks[0], samples)

    def test_chunk_acknowledgements_count_exact_samples_and_reject_changed_retries(self):
        first = self.feed([100, -200, 300])
        self.assertEqual(first["sample_count"], 3)
        retry = self.feed([100, -200, 300])
        self.assertEqual(retry["sample_count"], 3)
        with self.assertRaisesRegex(ValueError, "changed an acknowledged"):
            self.feed([100, -200, 301])
        next_chunk = self.feed([400, 500], sequence=1)
        self.assertEqual(next_chunk["sequence"], 2)
        self.assertEqual(next_chunk["sample_count"], 5)
        entry = self.service._recordings[self.token]
        np.testing.assert_array_equal(np.concatenate(entry.session.chunks), [100, -200, 300, 400, 500])

    def test_final_recording_cannot_silently_lose_acknowledged_samples(self):
        self.feed([100, 200, 300])
        with self.assertRaisesRegex(ValueError, "missing acknowledged audio"):
            self.finish([100, 200])
        self.assertFalse(list(self.root.iterdir()))
        self.finish([100, 200, 300, 400])
        path, _, _ = self.service.consume(self.token)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], [100, 200, 300, 400])

    def test_final_retries_are_immutable_and_retained_publication_is_idempotent(self):
        original = [100, 200, 300]
        self.finish(original)
        first = self.service.consume(self.token, retain=True)
        self.assertNotIn(self.token, self.service._recordings)
        self.assertIn(self.token, self.service._completed)
        self.assertEqual(self.finish(original), {"token": self.token})
        self.assertEqual(self.service.consume(self.token, retain=True), first)
        with self.assertRaisesRegex(ValueError, "retry changed the saved audio"):
            self.finish([100, 200, 301])
        np.testing.assert_array_equal(sf.read(first[0], dtype="int16")[0], original)
        self.assertEqual(self.service._completed[self.token].session.finished_calls, 1)

    def test_retained_results_do_not_exhaust_active_slots_and_eviction_keeps_published_audio(self):
        self.finish([100, 200])
        first = self.service.consume(self.token, retain=True)
        with patch.object(mic, "MAX_SESSIONS", 1), patch.object(mic, "MAX_COMPLETED_RESULTS", 1):
            token = self.service._start({"sample_rate": 16000})["token"]
            self.service._finish({"token": token, "pcm": pcm([300, 400])})
            second = self.service.consume(token, retain=True)
            self.assertEqual(len(self.service._completed), 1)
            self.assertNotIn(self.token, self.service._completed)
            self.assertEqual(self.service._recordings, {})
            self.assertTrue(Path(first[0]).is_file())
            self.assertTrue(Path(second[0]).is_file())

    def test_retained_result_expiry_and_cancel_do_not_delete_published_audio(self):
        self.finish([100, 200])
        path, _, _ = self.service.consume(self.token, retain=True)
        self.service._completed[self.token].updated -= mic.SESSION_TTL + 1
        with self.assertRaisesRegex(ValueError, "expired"):
            self.finish([100, 200])
        self.assertTrue(Path(path).is_file())
        token = self.service._start({"sample_rate": 16000})["token"]
        self.service._finish({"token": token, "pcm": pcm([300, 400])})
        other_path, _, _ = self.service.consume(token, retain=True)
        asyncio.run(self.service.cancel({"token": token}))
        self.assertTrue(Path(other_path).is_file())
        self.assertNotIn(token, self.service._completed)

    def test_failed_streaming_tail_is_recovered_from_the_complete_saved_recording(self):
        self.feed([100, 200])
        entry = self.service._recordings[self.token]
        with patch.object(FakeSession, "feed", side_effect=mic.STTError("preview tail failed")), \
                patch.object(mic, "transcribe_file", return_value="all the final words") as transcribe:
            self.finish([100, 200, 300, 400])
        path, text, _ = self.service.consume(self.token)
        transcribe.assert_called_once_with(path, max_seconds=1)
        self.assertEqual(text, "all the final words")
        self.assertEqual(entry.session.finished_calls, 0)
        self.assertEqual(entry.session.closed_calls, 1)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], [100, 200, 300, 400])

    def test_failed_incremental_finalization_never_publishes_partial_transcript(self):
        self.feed([100, 200])
        with patch.object(FakeSession, "finish", side_effect=mic.STTError("final decoder failed")), \
                patch.object(mic, "transcribe_file", return_value="fresh complete transcript"):
            self.finish([100, 200])
        _, text, _ = self.service.consume(self.token)
        self.assertEqual(text, "fresh complete transcript")

    def test_preview_chunk_bound_and_empty_chunks_preserve_final_reconciliation(self):
        with self.assertRaisesRegex(ValueError, "chunk is empty"):
            self.feed([])
        with patch.object(mic, "MAX_PREVIEW_CHUNKS", 2):
            self.feed([100])
            self.feed([200], sequence=1)
            with self.assertRaisesRegex(ValueError, "Too many live"):
                self.feed([300], sequence=2)
            self.finish([100, 200, 300, 400])
        path, text, _ = self.service.consume(self.token)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], [100, 200, 300, 400])
        self.assertEqual(text, "complete transcript")

    def test_out_of_order_preview_cannot_lose_the_complete_final_recording(self):
        with self.assertRaisesRegex(ValueError, "out of order"):
            self.feed([300, 400], sequence=1)
        complete = [100, 200, 300, 400, 500]
        self.finish(complete)
        path, text, _ = self.service.consume(self.token)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], complete)
        self.assertEqual(text, "complete transcript")

    def test_divergent_live_audio_is_retranscribed_from_the_saved_complete_recording(self):
        self.feed([100, 200])
        with patch.object(mic, "transcribe_file", return_value="words from complete audio") as transcribe:
            self.finish([300, 400, 500])
        path, text, _ = self.service.consume(self.token)
        transcribe.assert_called_once_with(path, max_seconds=1)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], [300, 400, 500])
        self.assertEqual(text, "words from complete audio")

    def test_missing_stt_preserves_complete_audio_for_manual_transcription(self):
        with patch.object(FakeSession, "feed", side_effect=RuntimeError("weights unavailable")), \
                patch.object(mic, "transcribe_file", side_effect=RuntimeError("weights unavailable")):
            preview = self.feed([100, 200])
            self.assertEqual(preview["text"], "")
            self.assertEqual(preview["status"], mic.MANUAL_STATUS)
            self.finish([100, 200, 300])
        path, text, status = self.service.consume(self.token)
        self.assertEqual(text, "")
        self.assertEqual(status, mic.MANUAL_STATUS)
        np.testing.assert_array_equal(sf.read(path, dtype="int16")[0], [100, 200, 300])

    def test_duration_limit_applies_to_individual_and_accumulated_requests(self):
        self.feed(np.ones(16000, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "limit"):
            self.feed([1], sequence=1)
        with self.assertRaisesRegex(ValueError, "limit"):
            self.finish(np.ones(16001, dtype=np.int16))
        self.assertFalse(list(self.root.iterdir()))

    def test_consumed_or_cancelled_tokens_cannot_publish_a_late_result(self):
        self.finish([100, 200])
        self.service.consume(self.token)
        with self.assertRaisesRegex(ValueError, "expired"):
            self.service.consume(self.token)
        token = self.service._start({"sample_rate": 16000})["token"]
        asyncio.run(self.service.cancel({"token": token}))
        with self.assertRaisesRegex(ValueError, "expired"):
            self.service._finish({"token": token, "pcm": pcm([100])})

    def test_cancelling_during_final_transcription_deletes_the_obsolete_recording(self):
        self.feed([100, 200])
        started = threading.Event()
        release = threading.Event()
        result = []

        def delayed_transcription(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release transcription")
            return "obsolete transcript"

        with patch.object(mic, "transcribe_file", side_effect=delayed_transcription):
            worker = threading.Thread(target=lambda: result.append(self.finish([300, 400])))
            worker.start()
            try:
                self.assertTrue(started.wait(3), "Final transcription did not start")

                async def cancel_during_inference():
                    cancellation = asyncio.create_task(self.service.cancel({"token": self.token}))
                    await asyncio.sleep(0)
                    self.assertNotIn(self.token, self.service._recordings)
                    release.set()
                    await cancellation

                asyncio.run(cancel_during_inference())
            finally:
                release.set()
                worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [{"cancelled": True}])
        self.assertFalse(list(self.root.iterdir()))

    def test_malformed_audio_and_sequence_are_rejected_before_recognition(self):
        for data in [
            {"pcm": "not base64", "sequence": 0},
            {"pcm": base64.b64encode(b"x").decode(), "sequence": 0},
            {"pcm": pcm([100]), "sequence": True},
            {"pcm": pcm([100]), "sequence": -1},
        ]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    self.service._feed({"token": self.token, **data})
        self.assertEqual(self.service._recordings[self.token].sample_count, 0)


class RecorderComponentTests(unittest.TestCase):
    def test_component_registers_server_functions_and_recording_events(self):
        # Gradio's server decorator requires functions, so registering bound
        # service methods directly breaks construction before a browser opens.
        with patch.object(mic, "_service", None):
            with gr.Blocks() as demo:
                widget = mic.build_recorder(10)
                widget.start_recording(lambda: None, queue=False)
                widget.complete(lambda: None, queue=False)
            config = widget.get_config()
            self.assertEqual(set(config["server_fns"]), {"start", "feed", "finish", "cancel"})
            self.assertEqual(len(demo.get_config_file()["dependencies"]), 2)


if __name__ == "__main__":
    unittest.main()
