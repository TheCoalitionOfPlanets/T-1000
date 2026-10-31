import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from t1000 import api, mic


class ReferenceEditingAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "source.wav"
        self.rate = 24000
        time = np.arange(self.rate * 3) / self.rate
        self.samples = (.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        sf.write(self.path, self.samples, self.rate, subtype="PCM_16")
        self.original_bytes = self.path.read_bytes()
        with api.audio_lock:
            api.audio_files.clear()
        self.source = api._register(self.path, "Old words", "Ready")
        self.client = TestClient(api.api)
        self.addCleanup(self.client.close)
        self.addCleanup(api.finished_references.clear)
        self.addCleanup(api.finish_locks.clear)
        self.directory_patch = patch.object(api, "UPLOAD_DIR", self.root)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        self.seen = []

        def recognize(path, **kwargs):
            values, rate = sf.read(path, dtype="float32")
            self.seen.append((path, values, rate))
            return f"Words from {len(values)} samples"

        self.stt = patch.object(api.voice.stt, "transcribe_file", side_effect=recognize)
        self.stt.start()
        self.addCleanup(self.stt.stop)

    def edit(self, **options):
        return self.client.post("/api/reference/process", json={
            "reference_id": self.source["id"], "noise_reduction": False, **options,
        })

    def test_crop_transcribes_and_serves_only_selected_native_rate_samples(self):
        response = self.edit(start_seconds=.25, end_seconds=2.25)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        served, rate = sf.read(io.BytesIO(self.client.get(result["url"]).content), dtype="int16")
        original, _ = sf.read(self.path, dtype="int16")
        np.testing.assert_array_equal(served, original[6000:54000])
        self.assertEqual(rate, self.rate)
        self.assertEqual(result["duration_seconds"], 2)
        self.assertEqual(result["transcript"], "Words from 48000 samples")
        self.assertEqual(self.seen[-1][0], str(api._lookup(result["id"]).path))
        self.assertEqual(self.seen[-1][2], self.rate)
        self.assertEqual(result["original_id"], self.source["id"])
        self.assertEqual(result["original_duration_seconds"], 3)
        self.assertEqual(self.path.read_bytes(), self.original_bytes)

    def test_repeated_edits_use_original_and_restore_retranscribes_full_recording(self):
        first = self.edit(start_seconds=.5, end_seconds=2, noise_reduction=True).json()
        second = self.edit(reference_id=first["id"], start_seconds=.5, end_seconds=2,
                           noise_reduction=True).json()
        self.assertEqual(self.client.get(first["url"]).content, self.client.get(second["url"]).content)
        restored = self.edit(reference_id=second["id"]).json()
        self.assertEqual(restored["transcript"], "Words from 72000 samples")
        self.assertEqual(restored["duration_seconds"], 3)
        self.assertEqual(self.client.get(restored["url"]).content, self.original_bytes)

    def test_retained_edits_keep_original_available_after_registry_eviction(self):
        with patch.object(api, "MAX_AUDIO_ENTRIES", 1):
            for _ in range(3):
                response = self.edit(start_seconds=.5, end_seconds=2, noise_reduction=True)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.client.get(response.json()["original_url"]).content, self.original_bytes)
                self.assertEqual(len(api.audio_files), 1)

    def test_bad_crops_and_unknown_ids_do_not_publish_or_transcribe(self):
        before = set(self.root.iterdir())
        for options in ({"start_seconds": 2, "end_seconds": 1}, {"end_seconds": 4},
                        {"start_seconds": .5, "end_seconds": .9},
                        {"reference_id": "missing"}, {"reference_id": "/etc/passwd"}):
            with self.subTest(options=options):
                self.assertIn(self.edit(**options).status_code, (400, 404))
        self.assertEqual(self.seen, [])
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.client.get(self.source["url"]).status_code, 200)

    def test_invalid_nonfinite_and_extra_controls_rejected(self):
        for options in ({"start_seconds": "NaN"}, {"end_seconds": "Infinity"},
                        {"start_seconds": -1}, {"noise_reduction": "true"},
                        {"path": str(self.path)}):
            with self.subTest(options=options):
                self.assertEqual(self.edit(**options).status_code, 422)
        self.assertEqual(self.seen, [])

    def test_cleaning_failure_keeps_original_and_leaves_no_partial_output(self):
        before = set(self.root.iterdir())
        with patch.object(api, "process_reference", side_effect=ValueError("Choose another crop.")):
            response = self.edit(noise_reduction=True)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Choose another crop.")
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.seen, [])

    def test_stt_failure_preserves_processed_audio_and_clears_stale_words(self):
        with patch.object(api.voice.stt, "transcribe_file", side_effect=RuntimeError("No weights")):
            response = self.edit(start_seconds=.5, end_seconds=2)
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["transcript"], "")
        self.assertIn("Enter the exact words", result["status"])
        self.assertEqual(self.client.get(result["url"]).status_code, 200)

    def test_upload_defaults_to_cleanup_and_retains_original_for_restore(self):
        result = self.client.post("/api/reference", files={
            "file": ("voice.wav", self.original_bytes, "audio/wav"),
        }).json()
        self.assertTrue(result["noise_reduction"])
        self.assertNotEqual(result["id"], result["original_id"])
        self.assertEqual(self.client.get(result["original_url"]).content, self.original_bytes)
        self.assertNotEqual(self.client.get(result["url"]).content, self.original_bytes)
        self.assertEqual(result["transcript"], "Words from 72000 samples")

    def test_microphone_default_cleanup_is_transcribed_and_finish_retry_is_stable(self):
        class Session:
            text = "Draft"

            def feed(self, *_args):
                pass

            def finish(self):
                return "Old draft words"

            def close(self):
                pass

        service = mic.RecorderService(10)
        pcm = sf.read(self.path, dtype="int16")[0].astype("<i2").tobytes()
        with patch.object(api, "recorder", service), patch.object(mic, "RECORDING_DIR", self.root), \
                patch.object(mic, "create_session", return_value=Session()):
            token = self.client.post("/api/recordings/start", json={"sample_rate": self.rate}).json()["token"]
            data = {"token": token, "pcm": base64.b64encode(pcm).decode()}
            first = self.client.post("/api/recordings/finish", json=data)
            with api.audio_lock:
                api.audio_files.clear()
            again = self.client.post("/api/recordings/finish", json=data)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), again.json())
        result = first.json()
        self.assertEqual(result["transcript"], "Words from 72000 samples")
        self.assertTrue(result["noise_reduction"])
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.client.get(result["original_url"]).content, self.original_bytes)
        self.assertEqual(api._lookup(result["id"]).original_id, result["original_id"])


if __name__ == "__main__":
    unittest.main()
