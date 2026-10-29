import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "compare_tts_models.py"
spec = importlib.util.spec_from_file_location("compare_tts_models", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ModelComparisonTests(unittest.TestCase):
    def test_mf2_uses_artifact_sampling_without_overrides(self):
        case = {"text": "Hello.", "reference": "/voice.wav", "transcript": "Reference."}
        options = comparison.generation_options(case, "mf2")
        self.assertTrue({"num_steps", "guidance_scale", "ode_method"}.isdisjoint(options))
        self.assertEqual(comparison.generation_options(case, "soar")["num_steps"], 10)
        self.assertFalse(options["normalize_text"])

    def test_eos_is_observed_after_final_yield(self):
        state = SimpleNamespace(end_flag=False)

        def decode(*args, **kwargs):
            yield "final"
            kwargs["state"].end_flag = True

        observer = comparison.DecodeObserver(SimpleNamespace(_decode=decode))
        stream = observer(state=state)
        self.assertEqual(next(stream), "final")
        self.assertEqual(observer.stop_reason, "decoding")
        self.assertEqual(list(stream), [])
        self.assertEqual(observer.stop_reason, "eos")
        self.assertEqual(observer.decoded_patches, 1)

    def test_schedule_exhaustion_cannot_be_mislabelled_as_eos(self):
        observer = comparison.DecodeObserver(SimpleNamespace(_decode=lambda **kwargs: iter([1, 2])))
        self.assertEqual(list(observer(state=SimpleNamespace(end_flag=False))), [1, 2])
        self.assertEqual(observer.stop_reason, "schedule_exhausted")

    def test_cancellation_and_error_remain_distinct_from_exhaustion(self):
        observer = comparison.DecodeObserver(SimpleNamespace(_decode=lambda **kwargs: iter([1, 2])))
        stream = observer(state=SimpleNamespace(end_flag=False))
        next(stream)
        stream.close()
        self.assertEqual(observer.stop_reason, "closed")

        def broken(**kwargs):
            yield 1
            raise RuntimeError("failed decode")

        observer = comparison.DecodeObserver(SimpleNamespace(_decode=broken))
        with self.assertRaises(RuntimeError):
            list(observer(state=SimpleNamespace(end_flag=False)))
        self.assertEqual(observer.stop_reason, "error")

    def test_manifest_preserves_expected_words_and_resolves_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").touch()
            manifest = root / "manifest.json"
            case = {"id": "voice-1", "reference": "voice.wav", "transcript": "Exact ref.",
                    "text": "Pay $20.", "spoken_expected_text": "Pay twenty dollars.", "seed": 42}
            manifest.write_text(json.dumps({"metadata": {"source": "test"}, "cases": [case]}))
            result = comparison.load_manifest(manifest)
            self.assertEqual(result["cases"][0]["spoken_expected_text"], "Pay twenty dollars.")
            self.assertEqual(result["cases"][0]["reference"], str(root / "voice.wav"))
            self.assertEqual(result["metadata"], {"source": "test"})
            for bad_cases in ([case, case], [dict(case, id="../escape")], [dict(case, seed=True)]):
                manifest.write_text(json.dumps({"cases": bad_cases}))
                with self.assertRaises(ValueError):
                    comparison.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
