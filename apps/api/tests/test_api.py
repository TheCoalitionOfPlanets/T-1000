import asyncio
import base64
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from t1000 import mic as transport
from t1000 import api


def wav_bytes(seconds=2, silent=False):
    target = io.BytesIO()
    samples = np.zeros(16000 * seconds) if silent else np.sin(np.arange(16000 * seconds) * .05) * .2
    sf.write(target, samples, 16000, format="WAV", subtype="PCM_16")
    return target.getvalue()


class APITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.client = TestClient(api.api)
        with api.audio_lock:
            api.audio_files.clear()

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def test_health_reports_model_load_state_without_loading_weights(self):
        api.voice.load_runtime.cache_clear()
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok", "device": api.voice.DEVICE, "speech_model_loaded": False,
        })

    def test_opt_in_preload_finishes_before_server_accepts_requests(self):
        events = []
        with patch.dict(os.environ, {"T1000_PRELOAD_MODELS": "1"}), \
                patch.object(api.voice, "load_runtime", side_effect=lambda: events.append("tts")), \
                patch.object(api.voice.stt, "preload_cpu_model", side_effect=lambda: events.append("stt")), \
                patch("uvicorn.run", side_effect=lambda *a, **k: events.append("serve")):
            api.main()
        self.assertEqual(events, ["tts", "stt", "serve"])

    def test_invalid_preload_setting_fails_before_model_or_server_start(self):
        with patch.dict(os.environ, {"T1000_PRELOAD_MODELS": "maybe"}), \
                patch.object(api.voice, "load_runtime") as load, \
                patch("uvicorn.run") as serve:
            with self.assertRaisesRegex(ValueError, "T1000_PRELOAD_MODELS"):
                api.main()
        load.assert_not_called()
        serve.assert_not_called()

    def test_preload_tts_failure_is_reported_before_server_start(self):
        with patch.dict(os.environ, {"T1000_PRELOAD_MODELS": "1"}), \
                patch.object(api.voice, "load_runtime", side_effect=RuntimeError("missing weights")), \
                patch("uvicorn.run") as serve:
            with self.assertRaisesRegex(RuntimeError, "missing weights"):
                api.main()
        serve.assert_not_called()

    def test_config_example_and_audio_use_server_owned_identifiers(self):
        config = self.client.get("/api/config").json()
        self.assertEqual(config["text_limit"], api.TEXT_LIMIT)
        self.assertEqual(config["example"]["id"], "example")
        self.assertEqual(self.client.get(config["example"]["url"]).content, api.voice.EXAMPLE_AUDIO.read_bytes())
        self.assertEqual(self.client.get("/api/audio/not-a-local-path").status_code, 404)

    def test_text_expansion_returns_proposal_without_generating(self):
        with patch.object(api, "expand_english_text", return_value="Pay twelve dollars.") as normalize, patch.object(api.voice, "synthesize") as generate:
            response = self.client.post("/api/text/normalize", json={"text": "Pay $12."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Pay twelve dollars."})
        normalize.assert_called_once_with("Pay $12.", max_length=api.TEXT_LIMIT)
        generate.assert_not_called()

    def test_text_expansion_rejects_oversized_input_and_transcript_fields(self):
        with patch.object(api, "expand_english_text") as normalize:
            for payload in ({"text": "x" * (api.TEXT_LIMIT + 1)}, {"text": "Hello", "transcript": "Reference words"}):
                response = self.client.post("/api/text/normalize", json=payload)
                self.assertEqual(response.status_code, 422)
        normalize.assert_not_called()

    def test_text_expansion_errors_preserve_actionable_messages(self):
        message = "Expanded text has 151 characters; shorten the original to fit the 150-character limit."
        with patch.object(api, "expand_english_text", side_effect=ValueError(message)):
            response = self.client.post("/api/text/normalize", json={"text": "123"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], message)
        with patch.object(api, "expand_english_text", side_effect=RuntimeError("internal details")), self.assertLogs(api.LOGGER, level="WARNING"):
            response = self.client.post("/api/text/normalize", json={"text": "123"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Text expansion is unavailable. Keep or edit your original text.")

    def test_config_sample_is_stable_and_audio_supports_byte_ranges(self):
        sample = self.root / "test-english.wav"
        sample.write_bytes(wav_bytes())
        with patch.object(api.voice, "OUTPUT_DIR", self.root):
            first = self.client.get("/api/config").json()
            self.assertEqual(first["sample_url"], self.client.get("/api/config").json()["sample_url"])
            self.assertEqual(len(api.audio_files), 0)
            result = self.client.get(first["sample_url"], headers={"Range": "bytes=0-43"})
        self.assertEqual(result.status_code, 206)
        self.assertEqual(result.headers["content-type"], "audio/wav")
        self.assertEqual(result.headers["accept-ranges"], "bytes")
        self.assertEqual(result.content, sample.read_bytes()[:44])

    def test_upload_retains_valid_audio_when_transcription_is_unavailable(self):
        with patch.object(api, "UPLOAD_DIR", self.root), patch.object(api.voice.stt, "transcribe_file", side_effect=RuntimeError("weights missing")):
            result = self.client.post("/api/reference", files={"file": ("../../escape.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(result.status_code, 200)
        reference = result.json()
        self.assertEqual(reference["transcript"], "")
        self.assertIn("Enter the exact words", reference["status"])
        self.assertEqual(self.client.get(reference["url"]).status_code, 200)
        self.assertTrue(all(item.name.startswith("reference-") for item in self.root.iterdir()))

    def test_invalid_upload_is_rejected_before_transcription(self):
        with patch.object(api, "UPLOAD_DIR", self.root), patch.object(api.voice.stt, "transcribe_file") as recognize:
            result = self.client.post("/api/reference", files={"file": ("silence.wav", wav_bytes(silent=True), "audio/wav")})
        self.assertEqual(result.status_code, 400)
        self.assertIn("silent", result.json()["detail"])
        recognize.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_oversized_upload_is_removed(self):
        with patch.object(api, "UPLOAD_DIR", self.root), patch.object(api, "MAX_UPLOAD_BYTES", 16):
            result = self.client.post("/api/reference", files={"file": ("big.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(result.status_code, 413)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_generation_rejects_unknown_audio_and_invalid_controls(self):
        request = {"text": "Hello", "reference_id": "/etc/passwd", "transcript": "Words"}
        self.assertEqual(self.client.post("/api/generate", json=request).status_code, 404)
        for extra in ({"steps": 0}, {"seed": -1}, {"text": "x" * (api.TEXT_LIMIT + 1)}, {"reference_audio": "/etc/passwd"}):
            self.assertEqual(self.client.post("/api/generate", json={**request, **extra}).status_code, 422)

    def test_generation_stream_exposes_audio_and_releases_slot(self):
        output = self.root / "speech-test.wav"
        output.write_bytes(wav_bytes())

        def synthesize(*args):
            yield gr.skip(), "Generating", gr.update(interactive=False), gr.update(interactive=True)
            yield str(output), "Generated", gr.update(interactive=True), gr.update(interactive=False)

        with patch.object(api.voice, "OUTPUT_DIR", self.root), patch.object(api.voice, "synthesize", side_effect=synthesize) as generate:
            result = self.client.post("/api/generate", json={"text": "Hello", "reference_id": "example", "transcript": "Example words"})
        self.assertEqual(result.status_code, 200)
        events = [json.loads(line) for line in result.text.splitlines()]
        self.assertFalse(events[0]["done"])
        self.assertTrue(events[-1]["done"])
        elapsed = [event["elapsed_ms"] for event in events]
        self.assertEqual(elapsed, sorted(elapsed))
        for event in events:
            timings = event["timings"]
            self.assertGreaterEqual(timings["reference_validated_ms"], 0)
            self.assertLessEqual(timings["reference_validated_ms"], timings["first_status_ms"])
            self.assertLessEqual(timings["first_status_ms"], event["elapsed_ms"])
        self.assertNotIn("completion_ms", events[0]["timings"])
        self.assertEqual(events[-1]["timings"]["completion_ms"], elapsed[-1])
        self.assertEqual(self.client.get(events[-1]["audio_url"]).content, output.read_bytes())
        self.assertEqual(generate.call_args.args[1], str(api.voice.EXAMPLE_AUDIO))
        self.assertFalse(api.generation_lock.locked())

    def test_stream_error_keeps_timing_milestones_and_releases_slot(self):
        def synthesize(*args):
            yield gr.skip(), "Generating", gr.update(interactive=False), gr.update(interactive=True)
            raise RuntimeError("decode failed")

        with patch.object(api.voice, "synthesize", side_effect=synthesize), self.assertLogs(api.LOGGER, level="ERROR"):
            result = self.client.post("/api/generate", json={"text": "Hello", "reference_id": "example", "transcript": "Words"})
        events = [json.loads(line) for line in result.text.splitlines()]
        self.assertEqual(result.status_code, 200)
        self.assertEqual(events[-1]["error"], "decode failed")
        self.assertTrue(events[-1]["done"])
        self.assertGreaterEqual(events[-1]["elapsed_ms"], events[0]["elapsed_ms"])
        self.assertEqual(events[-1]["timings"]["completion_ms"], events[-1]["elapsed_ms"])
        self.assertEqual(events[-1]["timings"]["reference_validated_ms"], events[0]["timings"]["reference_validated_ms"])
        self.assertFalse(api.generation_lock.locked())

    def test_running_generation_rejects_second_request(self):
        api.generation_lock.acquire()
        try:
            result = self.client.post("/api/generate", json={"text": "Hello", "reference_id": "example", "transcript": "Words"})
            self.assertEqual(result.status_code, 409)
        finally:
            api.generation_lock.release()

    def test_failed_generation_releases_slot(self):
        def fail(*args):
            raise ValueError("bad request")
            yield

        with patch.object(api.voice, "synthesize", side_effect=fail):
            result = self.client.post("/api/generate", json={"text": "Hello", "reference_id": "example", "transcript": "Words"})
        self.assertEqual(result.status_code, 400)
        self.assertFalse(api.generation_lock.locked())

    def test_stop_uses_existing_cancellation(self):
        with patch.object(api.voice, "stop_generation") as stop:
            self.assertEqual(self.client.post("/api/stop").status_code, 200)
        stop.assert_called_once()

    def test_stream_close_cancels_source_and_releases_generation_slot(self):
        closed = threading.Event()

        def synthesize(*args):
            try:
                while True:
                    yield gr.skip(), "Generating", gr.update(interactive=False), gr.update(interactive=True)
            finally:
                closed.set()

        async def disconnect():
            response = await api.generate(api.GenerateRequest(text="Hello", reference_id="example", transcript="Words"))
            await anext(response.body_iterator)
            await response.body_iterator.aclose()

        with patch.object(api.voice, "synthesize", side_effect=synthesize):
            asyncio.run(disconnect())
        self.assertTrue(closed.is_set())
        self.assertFalse(api.generation_lock.locked())

    def test_asgi_send_failure_closes_source_and_releases_slot(self):
        closed = threading.Event()

        def synthesize(*args):
            try:
                while True:
                    yield gr.skip(), "Generating", gr.update(interactive=False), gr.update(interactive=True)
            finally:
                closed.set()

        async def disconnect():
            response = await api.generate(api.GenerateRequest(text="Hello", reference_id="example", transcript="Words"))

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client disconnected")

            with self.assertRaises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        with patch.object(api.voice, "synthesize", side_effect=synthesize):
            asyncio.run(disconnect())
        self.assertTrue(closed.is_set())
        self.assertFalse(api.generation_lock.locked())

    def test_pcm_transport_finalizes_unsent_tail_and_serves_reference(self):
        class Session:
            text = "Partial words"

            def __init__(self):
                self.samples = []

            def feed(self, rate, samples):
                self.samples.append(samples.copy())

            def finish(self):
                return "Complete words"

            def close(self):
                pass

        session = Session()
        pcm = (np.sin(np.arange(32000) * .05) * 6000).astype("<i2").tobytes()
        service = transport.RecorderService(10)
        with patch.object(api, "recorder", service), patch.object(transport, "RECORDING_DIR", self.root), patch.object(transport, "create_session", return_value=session):
            started = self.client.post("/api/recordings/start", json={"sample_rate": 16000, "noise_reduction": False}).json()
            fed = self.client.post("/api/recordings/feed", json={"token": started["token"], "sequence": 0, "pcm": base64.b64encode(pcm[:16000]).decode()})
            self.assertEqual(fed.json()["text"], "Partial words")
            finished = self.client.post("/api/recordings/finish", json={"token": started["token"], "pcm": base64.b64encode(pcm).decode()})
        self.assertEqual(finished.status_code, 200)
        reference = finished.json()
        self.assertEqual(reference["transcript"], "Complete words")
        self.assertEqual(np.concatenate(session.samples).tobytes(), pcm)
        actual, rate = sf.read(io.BytesIO(self.client.get(reference["url"]).content), dtype="int16")
        self.assertEqual(rate, 16000)
        self.assertEqual(actual.tobytes(), pcm)
        self.assertEqual(service._recordings, {})


if __name__ == "__main__":
    unittest.main()
