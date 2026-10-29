"""Exercise paragraph handoffs with the real WAV writer and EOS observer."""

import io
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from t1000 import tts
from t1000.paragraphs import plan_text


TEXT = "One short sentence. Another short one. Last short line."
RATE = 16000


def tone(frames, value):
    return torch.full((1, frames), value, dtype=torch.float32)


def pcm16(samples):
    output = io.BytesIO()
    sf.write(output, np.asarray(samples, dtype=np.float32), RATE,
             format="WAV", subtype="PCM_16")
    return output.getvalue()


class FakeModel:
    def __init__(self, runtime):
        self.runtime = runtime

    def _decode(self, *, state, index):
        for action in self.runtime.parts[index]:
            if isinstance(action, BaseException):
                raise action
            chunk = action() if callable(action) else action
            yield chunk
        state.end_flag = self.runtime.eos[index]


class FakeRuntime:
    device = SimpleNamespace(type="cpu")
    sample_rate = RATE

    def __init__(self, parts, *, eos=None, after_close=None):
        self.parts = parts
        self.eos = [True] * len(parts) if eos is None else eos
        self.after_close = after_close
        self.calls = []
        self.closed = []
        self.model = FakeModel(self)

    def generate_stream(self, **kwargs):
        index = len(self.calls)
        self.calls.append({**kwargs, "model_lock_held": tts.MODEL_LOCK.locked(),
                           "reference_exists": Path(kwargs["prompt_audio_path"]).is_file()})
        state = SimpleNamespace(end_flag=False)
        try:
            yield from self.model._decode(state=state, index=index)
        finally:
            self.closed.append(index)
            if self.after_close is not None:
                self.after_close(index)


class ParagraphGenerationTests(unittest.TestCase):
    def run_generation(self, runtime, text=TEXT, *, segment_limit=20,
                       cancel=None, output_limit=600):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        with ExitStack() as stack:
            stack.enter_context(patch.object(tts, "OUTPUT_DIR", root))
            stack.enter_context(patch.object(tts, "CACHE", root / "cache"))
            stack.enter_context(patch.object(tts, "SEGMENT_TEXT_LIMIT", segment_limit, create=True))
            stack.enter_context(patch.object(tts, "MAX_OUTPUT_SECONDS", output_limit, create=True))
            stack.enter_context(patch.object(tts, "load_runtime", return_value=runtime))
            updates = list(tts.synthesize(text, str(tts.EXAMPLE_AUDIO),
                                          tts.EXAMPLE_TRANSCRIPT, 10, 42,
                                          cancel_event=cancel))
        output = updates[-1][0]
        self.assertIsInstance(output, str, updates[-1][1])
        samples, rate = sf.read(output, dtype="float32")
        self.assertEqual(rate, RATE)
        self.assertFalse(tts.MODEL_LOCK.locked())
        self.assertIsNone(tts.ACTIVE_STOP)
        self.assertEqual(list(root.glob("*.part")), [])
        self.assertEqual(list((root / "cache" / "requests").glob("*.wav")), [])
        return output, samples, updates[-1][1]

    def test_short_request_keeps_one_decode_and_unchanged_pcm(self):
        parts = [tone(37, .125), tone(19, -.25)]
        runtime = FakeRuntime([parts])
        output, samples, _ = self.run_generation(runtime, "Hello.")
        expected = torch.cat(parts, dim=1).reshape(-1).numpy()
        self.assertEqual(Path(output).read_bytes(), pcm16(expected))
        self.assertEqual([call["text"] for call in runtime.calls], ["Hello."])
        self.assertTrue(output.metadata["complete"])
        self.assertEqual(output.metadata["segments_total"], 1)
        self.assertEqual(output.metadata["segments_completed"], 1)
        self.assertEqual(len(samples), 56)

    def test_ordered_sentence_decodes_share_frozen_voice_and_one_exact_wav(self):
        segments = plan_text(TEXT, 20)
        parts = [[tone(37, .125)], [tone(19, .25), tone(11, -.25)], [tone(23, .375)]]
        runtime = FakeRuntime(parts)
        output, samples, _ = self.run_generation(runtime)
        expected = []
        for segment, chunks in zip(segments, parts):
            if segment.pause_before:
                expected.append(np.zeros(round(segment.pause_before * RATE), dtype=np.float32))
            expected.extend(chunk.reshape(-1).numpy() for chunk in chunks)
        expected = np.concatenate(expected)
        self.assertEqual(Path(output).read_bytes(), pcm16(expected))
        self.assertEqual(len(samples), len(expected))
        self.assertEqual([call["text"] for call in runtime.calls], [segment.text for segment in segments])
        self.assertTrue(all(call["model_lock_held"] and call["reference_exists"] for call in runtime.calls))
        self.assertEqual(len({call["prompt_audio_path"] for call in runtime.calls}), 1)
        self.assertTrue(all(call["prompt_text"] == tts.EXAMPLE_TRANSCRIPT
                            and call["language"] == "en" and call["num_steps"] == 10
                            and call["guidance_scale"] == 1.2 for call in runtime.calls))
        self.assertEqual(runtime.closed, [0, 1, 2])
        self.assertEqual(output.metadata["segments_total"], 3)
        self.assertEqual(output.metadata["segments_completed"], 3)
        self.assertTrue(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "eos")

    def test_join_padding_counts_existing_quiet_edges_without_trimming_samples(self):
        text = "One sentence. Another line."
        segments = plan_text(text, 14)
        first = torch.cat((tone(40, .125), tone(64, 0)), dim=1)
        second = torch.cat((tone(48, 0), tone(30, .25)), dim=1)
        runtime = FakeRuntime([[first], [second]])
        output, _, _ = self.run_generation(runtime, text, segment_limit=14)
        deficit = round(segments[1].pause_before * RATE) - 64 - 48
        self.assertGreater(deficit, 0)
        expected = np.concatenate((first.reshape(-1).numpy(), np.zeros(deficit, np.float32), second.reshape(-1).numpy()))
        self.assertEqual(Path(output).read_bytes(), pcm16(expected))

    def test_already_sufficient_quiet_gap_is_preserved_without_extra_padding(self):
        text = "One sentence. Another line."
        first = torch.cat((tone(40, .125), tone(3000, 0)), dim=1)
        second = torch.cat((tone(400, 0), tone(30, .25)), dim=1)
        output, _, _ = self.run_generation(FakeRuntime([[first], [second]]), text, segment_limit=14)
        expected = torch.cat((first, second), dim=1).reshape(-1).numpy()
        self.assertEqual(Path(output).read_bytes(), pcm16(expected))

    def test_schedule_exhaustion_stops_later_sentences_and_marks_prefix_partial(self):
        runtime = FakeRuntime([[tone(37, .125)], [tone(19, .25)], [tone(23, .375)]],
                              eos=[True, False, True])
        output, samples, status = self.run_generation(runtime)
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.closed, [0, 1])
        self.assertFalse(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "schedule_exhausted")
        self.assertEqual(output.metadata["segments_completed"], 1)
        self.assertEqual(output.metadata["segments_total"], 3)
        self.assertIn("partial", status.lower())
        self.assertTrue(np.allclose(samples[:37], .125))
        self.assertTrue(np.allclose(samples[-19:], .25))

    def test_later_inference_error_preserves_finished_prefix_and_releases_locks(self):
        runtime = FakeRuntime([[tone(37, .125)], [tone(19, .25), RuntimeError("second sentence failed")], [tone(23, .375)]])
        output, samples, status = self.run_generation(runtime)
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.closed, [0, 1])
        self.assertFalse(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "error")
        self.assertEqual(output.metadata["segments_completed"], 1)
        self.assertIn("second sentence failed", status)
        self.assertTrue(np.allclose(samples[:37], .125))
        self.assertTrue(np.allclose(samples[-19:], .25))

    def test_cancel_between_sentences_does_not_claim_complete_or_append_a_pause(self):
        cancel = threading.Event()
        runtime = FakeRuntime([[tone(37, .125)], [tone(19, .25)], [tone(23, .375)]],
                              after_close=lambda index: cancel.set() if index == 0 else None)
        output, samples, _ = self.run_generation(runtime, cancel=cancel)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(Path(output).read_bytes(), pcm16(np.full(37, .125, dtype=np.float32)))
        self.assertEqual(len(samples), 37)
        self.assertFalse(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "cancelled")
        self.assertEqual(output.metadata["segments_completed"], 1)

    def test_cancel_during_audio_step_retains_that_chunk_and_skips_remaining_sentences(self):
        cancel = threading.Event()
        def in_flight_chunk():
            cancel.set()
            return tone(17, .375)
        runtime = FakeRuntime([[tone(37, .125)], [tone(19, .25), in_flight_chunk], [tone(23, .5)]])
        output, samples, _ = self.run_generation(runtime, cancel=cancel)
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.closed, [0, 1])
        self.assertFalse(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "cancelled")
        self.assertEqual(output.metadata["segments_completed"], 1)
        self.assertTrue(np.allclose(samples[-36:-17], .25))
        self.assertTrue(np.allclose(samples[-17:], .375))

    def test_output_audio_cap_stops_paragraph_with_explicit_partial_prefix(self):
        runtime = FakeRuntime([[tone(300, .125)], [tone(19, .25)], [tone(23, .375)]])
        output, samples, status = self.run_generation(runtime, output_limit=200 / RATE)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.closed, [0])
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 200)
        self.assertFalse(output.metadata["complete"])
        self.assertEqual(output.metadata["reason"], "audio_limit")
        self.assertEqual(output.metadata["segments_completed"], 0)
        self.assertIn("partial", status.lower())
        self.assertTrue(np.allclose(samples, .125))


if __name__ == "__main__":
    unittest.main()
