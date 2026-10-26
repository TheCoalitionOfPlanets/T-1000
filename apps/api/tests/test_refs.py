import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from t1000 import tts


class ReferenceTranscriptionTests(unittest.TestCase):
    def test_gradio_cached_copy_of_selected_audio_can_generate(self):
        state = tts.ReferenceState()
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / "cached.wav"
            shutil.copyfile(tts.EXAMPLE_AUDIO, cached)
            result = (None, "Generated", gr.update(interactive=True), gr.update(interactive=False))
            with patch.object(tts, "synthesize", return_value=iter([result])) as generate:
                self.assertEqual(list(tts.synthesize_ready("Hello", str(cached), "Words", 10, 42, state)), [result])
            self.assertEqual(generate.call_args.args[1], str(tts.EXAMPLE_AUDIO))

    def test_pending_reference_cannot_use_an_old_transcript(self):
        state = tts.ReferenceState()
        tts.prepare_upload("new.wav", state)
        with patch.object(tts, "synthesize") as generate:
            with self.assertRaisesRegex(gr.Error, "wait for its transcript"):
                list(tts.synthesize_ready("Hello", str(tts.EXAMPLE_AUDIO), "Old words", 10, 42, state))
            generate.assert_not_called()

    def test_late_upload_result_cannot_replace_the_example(self):
        state = tts.ReferenceState()
        path = str(tts.EXAMPLE_AUDIO)
        tts.prepare_upload(path, state)

        def transcribe(*args, **kwargs):
            tts.restore_example(state)
            return "Late words"

        with patch.object(tts.stt, "transcribe_file", side_effect=transcribe):
            result = tts.transcribe_upload(path, state)
        self.assertEqual(result, (gr.skip(),) * 3)
        self.assertTrue(state.ready)

    def test_overlong_upload_is_rejected_before_recognition(self):
        state = tts.ReferenceState()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "long.wav")
            sf.write(path, np.full(16000 * 11, 0.1), 16000)
            with patch.object(tts, "REFERENCE_LIMIT", 10), patch.object(tts.stt, "transcribe_file") as transcribe:
                tts.prepare_upload(path, state)
                words, status, button = tts.transcribe_upload(path, state)
                transcribe.assert_not_called()
            self.assertEqual(words, "")
            self.assertIn("10 seconds", status)
            self.assertFalse(button["interactive"])
            self.assertFalse(state.ready)

    def test_recognition_failure_allows_manual_words_for_valid_audio(self):
        state = tts.ReferenceState()
        path = str(tts.EXAMPLE_AUDIO)
        tts.prepare_upload(path, state)
        with patch.object(tts.stt, "transcribe_file", side_effect=RuntimeError("weights missing")):
            words, status, button = tts.transcribe_upload(path, state)
        self.assertEqual(words, "")
        self.assertIn("weights missing", status)
        self.assertIn("exact words", status)
        self.assertTrue(state.ready)
        self.assertTrue(button["interactive"])

    def test_stale_microphone_completion_cannot_replace_upload(self):
        state = tts.ReferenceState()
        tts.begin_recording(state, SimpleNamespace(_data={"token": "old-recording"}))
        tts.prepare_upload("new.wav", state)
        with patch.object(tts, "consume_complete") as consume:
            result = tts.finish_recording(state, SimpleNamespace(_data={"token": "old-recording"}))
        consume.assert_not_called()
        self.assertEqual(result, (gr.skip(),) * 5)
        self.assertEqual(state.path, "new.wav")

    def test_finishing_tts_keeps_generate_disabled_during_new_recording(self):
        state = tts.ReferenceState()

        def fake_synthesize(*args):
            tts.begin_recording(state, SimpleNamespace(_data={"token": "new-recording"}))
            yield None, "Generated", gr.update(interactive=True), gr.update(interactive=False)

        with patch.object(tts, "synthesize", side_effect=fake_synthesize):
            result = list(tts.synthesize_ready("Hello", str(tts.EXAMPLE_AUDIO), "Words", 10, 42, state))
        self.assertFalse(result[-1][2]["interactive"])


if __name__ == "__main__":
    unittest.main()
