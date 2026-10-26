import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf
import torch

import app


class SlowRuntime:
    device = SimpleNamespace(type="cpu")
    sample_rate = 48000

    def generate_stream(self, **kwargs):
        for _ in range(100):
            time.sleep(0.06)
            yield torch.full((1, 4800), 0.05)


class InferenceControlsTests(unittest.TestCase):
    def test_silent_microphone_recording_never_starts_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "microphone.wav"
            sf.write(path, np.zeros(44100 * 6, dtype=np.float32), 44100)
            with patch.object(app, "load_runtime") as load:
                with self.assertRaisesRegex(gr.Error, "silent"):
                    list(app.synthesize("Hello.", str(path), "Example transcript.", 10, 42))
                load.assert_not_called()

    def test_nonfinite_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.wav"
            sf.write(path, np.full(24000, np.nan, dtype=np.float32), 24000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "invalid audio"):
                app.read_reference(path)

    def test_example_reference_is_usable(self):
        audio, sample_rate = app.read_reference(app.EXAMPLE_AUDIO)
        self.assertEqual(sample_rate, 24000)
        self.assertGreater(audio.size / sample_rate, 3)
        self.assertGreater(float(np.abs(audio).max()), 0.01)

    def test_stop_releases_the_worker_and_reenables_generation(self):
        with patch.object(app, "load_runtime", return_value=SlowRuntime()):
            generator = app.synthesize("Hello.", str(app.EXAMPLE_AUDIO), app.EXAMPLE_TRANSCRIPT, 10, 42)
            next(generator)
            app.stop_generation()
            updates = list(generator)
        self.assertIn("Stopped", updates[-1][1])
        self.assertTrue(updates[-1][2]["interactive"])
        self.assertFalse(updates[-1][3]["interactive"])
        self.assertIsNone(app.ACTIVE_STOP)
        self.assertFalse(app.MODEL_LOCK.locked())

    def test_deadline_preserves_partial_audio_and_releases_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "load_runtime", return_value=SlowRuntime()), \
                    patch.object(app, "GENERATION_TIMEOUT", 0.15), \
                    patch.object(app, "OUTPUT_DIR", Path(directory)):
                started = time.perf_counter()
                updates = list(app.synthesize("Hello.", str(app.EXAMPLE_AUDIO), app.EXAMPLE_TRANSCRIPT, 10, 42))
                elapsed = time.perf_counter() - started
                self.assertIn("time limit", updates[-1][1])
                self.assertIn("partial audio", updates[-1][1])
                self.assertTrue(Path(updates[-1][0]).is_file())
                self.assertLess(elapsed, 3)
                self.assertFalse(app.MODEL_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
