import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from t1000 import tts
from t1000.ui import build_app


class LegacyReferenceEditorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "original.wav"
        time = np.arange(16000 * 3) / 16000
        sf.write(self.source, 0.2 * np.sin(2 * np.pi * 180 * time), 16000, subtype="PCM_16")
        self.state = tts.ReferenceState()
        self.state.replace(path=str(self.source), ready=True)
        cache = patch.object(tts, "CACHE", self.directory / "cache")
        cache.start()
        self.addCleanup(cache.stop)

    def edit(self, start=0, end=None, noise=False, auto_trim=False):
        return tts.edit_reference(start, end, noise, auto_trim, self.state)

    def test_edits_are_immutable_and_repeated_edits_use_the_original(self):
        with patch.object(tts.stt, "transcribe_file", return_value="Selected words.") as transcribe:
            first = list(self.edit(start=0.5, end=2.5))[-1]
            first_path = Path(first[0])
            second = list(self.edit(start=0, end=3))[-1]
        self.assertNotEqual(first[0], second[0])
        self.assertTrue(first_path.is_file())
        self.assertEqual(sf.info(first_path).frames, 32000)
        self.assertEqual(sf.info(second[0]).frames, 48000)
        self.assertEqual(sf.info(self.source).frames, 48000)
        self.assertEqual(self.state.original_path, str(self.source))
        self.assertEqual(self.state.path, second[0])
        self.assertTrue(self.state.ready)
        self.assertEqual(first[1], "Selected words.")
        self.assertEqual(transcribe.call_args.args[0], second[0])

    def test_generation_is_blocked_while_reference_is_being_edited(self):
        pending = self.edit()
        initial = next(pending)
        self.assertFalse(initial[3]["interactive"])
        self.assertFalse(self.state.ready)
        with patch.object(tts, "synthesize") as generate:
            with self.assertRaisesRegex(gr.Error, "wait for its transcript"):
                list(tts.synthesize_ready("Hello", str(self.source), "Old words", 10, 42, self.state))
        generate.assert_not_called()
        with patch.object(tts.stt, "transcribe_file", return_value="Updated words."):
            final = list(pending)[-1]
        self.assertTrue(final[3]["interactive"])
        self.assertTrue(self.state.ready)

    def test_invalid_crop_restores_the_prior_reference_and_does_not_transcribe(self):
        with patch.object(tts.stt, "transcribe_file") as transcribe:
            result = list(self.edit(start=2.5, end=1))[-1]
        transcribe.assert_not_called()
        self.assertEqual(result[0], gr.skip())
        self.assertEqual(result[1], gr.skip())
        self.assertIn("Reference edit failed", result[2])
        self.assertTrue(self.state.ready)
        self.assertEqual(self.state.path, str(self.source))
        self.assertEqual(self.state.original_path, str(self.source))

    def test_stale_edit_cannot_replace_a_new_reference_and_removes_its_file(self):
        def transcribe(*args, **kwargs):
            tts.restore_example(self.state)
            return "Stale words."

        with patch.object(tts.stt, "transcribe_file", side_effect=transcribe):
            result = list(self.edit())[-1]
        self.assertEqual(result, (gr.skip(),) * 5)
        self.assertEqual(self.state.path, str(tts.EXAMPLE_AUDIO))
        self.assertEqual(self.state.original_path, str(tts.EXAMPLE_AUDIO))
        self.assertEqual(list((self.directory / "cache" / "reference-edits").iterdir()), [])

    def test_restore_uses_original_audio_and_a_fresh_transcript(self):
        with patch.object(tts.stt, "transcribe_file", return_value="Cropped words."):
            edited = list(self.edit(start=0.5, end=2))[-1]
        with patch.object(tts.stt, "transcribe_file", return_value="Complete original words.") as transcribe:
            restored = list(tts.restore_original_reference(self.state))[-1]
        transcribe.assert_called_once_with(str(self.source), max_seconds=tts.REFERENCE_LIMIT)
        self.assertEqual(restored[0], str(self.source))
        self.assertEqual(restored[1], "Complete original words.")
        self.assertEqual(self.state.path, str(self.source))
        self.assertTrue(Path(edited[0]).is_file())

    def test_transcription_failure_clears_stale_words_and_allows_manual_transcript(self):
        with patch.object(tts.stt, "transcribe_file", side_effect=RuntimeError("weights missing")):
            result = list(self.edit())[-1]
        self.assertEqual(result[1], "")
        self.assertIn("Enter the exact words", result[2])
        self.assertTrue(self.state.ready)
        self.assertTrue(result[3]["interactive"])

    def test_pending_reference_cannot_be_edited_or_restored(self):
        tts.prepare_upload(str(self.source), self.state)
        for operation in (self.edit(), tts.restore_original_reference(self.state)):
            with self.assertRaisesRegex(gr.Error, "wait for its transcript"):
                list(operation)

    def test_cancelled_edit_restores_ready_state_without_changing_source(self):
        pending = self.edit()
        next(pending)
        self.assertFalse(self.state.ready)
        pending.close()
        self.assertTrue(self.state.ready)
        self.assertEqual(self.state.path, str(self.source))

    def test_new_upload_replaces_the_original_used_for_future_edits(self):
        with patch.object(tts.stt, "transcribe_file", return_value="Words."):
            list(self.edit(start=0.5, end=2))
        other = self.directory / "new.wav"
        other.write_bytes(self.source.read_bytes())
        tts.prepare_upload(str(other), self.state)
        self.assertEqual(self.state.original_path, str(other))

    def test_optional_lab_builds_with_cleanup_and_crop_controls(self):
        demo = build_app()
        self.addCleanup(demo.close)
        labels = {component.get("props", {}).get("label") for component in demo.config["components"]}
        self.assertIn("Crop start · seconds in original", labels)
        self.assertIn("Crop end · seconds in original", labels)
        self.assertIn("Reduce background noise", labels)


if __name__ == "__main__":
    unittest.main()
