import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import t1000
from t1000 import container


def write_files(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"model fixture")


class ContainerStartupTests(unittest.TestCase):
    @contextmanager
    def workspace(self, **environment):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "ROOT": root,
                "MODELS": root / "data" / "models",
                "OUTPUT": root / "data" / "output",
                "CACHE": root / ".cache",
                "WEB": root / "apps" / "web" / "dist",
            }
            paths["WEB"].mkdir(parents=True)
            (paths["WEB"] / "index.html").write_text("<html>Voice lab</html>")
            defaults = {
                "T1000_DOWNLOAD_MODELS": "0", "T1000_PRELOAD_MODELS": "1",
                "T1000_CONTAINER_DEVICE": "cpu", "STT_BACKEND": "cpu",
            }
            defaults.update(environment)
            with patch.multiple(container, **paths), patch.dict(os.environ, defaults):
                os.environ.pop("STT_MODEL_DIR", None)
                yield SimpleNamespace(**paths)

    def test_complete_marked_soar_model_skips_downloader(self):
        with self.workspace() as paths:
            directory = paths.MODELS / "dots.tts-soar"
            write_files(directory, container.TTS_FILES)
            marker = directory / ".t1000-revision"
            marker.write_text(container.TTS_REVISION + "\n")
            with patch.object(container.subprocess, "run") as run:
                container.ensure_tts_model(directory, download=True)
            run.assert_not_called()
            self.assertEqual(marker.read_text().strip(), container.TTS_REVISION)

    def test_offline_complete_host_model_is_accepted_without_revision_marker(self):
        with self.workspace() as paths:
            directory = paths.MODELS / "dots.tts-soar"
            write_files(directory, container.TTS_FILES)
            with patch.object(container.subprocess, "run") as run:
                container.ensure_tts_model(directory, download=False)
            run.assert_not_called()
            self.assertFalse((directory / ".t1000-revision").exists())

    def test_offline_missing_or_empty_tokenizer_rejects_without_network(self):
        for state in ("missing", "empty"):
            with self.subTest(state=state), self.workspace() as paths:
                directory = paths.MODELS / "dots.tts-soar"
                write_files(directory, container.TTS_FILES)
                tokenizer = directory / "tokenizer.json"
                if state == "missing":
                    tokenizer.unlink()
                else:
                    tokenizer.write_bytes(b"")
                with patch.object(container.subprocess, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, "tokenizer.json"):
                        container.ensure_tts_model(directory, download=False)
                run.assert_not_called()

    def test_completed_download_writes_revision_and_overrides_child_offline_mode(self):
        with self.workspace(HF_HUB_OFFLINE="1") as paths:
            directory = paths.MODELS / "dots.tts-soar"

            def download(command, **kwargs):
                self.assertEqual(command[:2], [sys.executable, "-c"])
                self.assertEqual(command[3:], [container.TTS_REPOSITORY, container.TTS_REVISION,
                                               str(directory)])
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "0")
                self.assertTrue(kwargs["check"])
                self.assertFalse((directory / ".t1000-revision").exists())
                write_files(directory, container.TTS_FILES)

            with patch.object(container.subprocess, "run", side_effect=download) as run:
                container.ensure_tts_model(directory, download=True)
            run.assert_called_once()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual((directory / ".t1000-revision").read_text().strip(),
                             container.TTS_REVISION)
            self.assertFalse(list(directory.glob(".t1000-revision-*.tmp")))

    def test_incomplete_or_failed_download_does_not_write_revision_marker(self):
        for failure in (None, subprocess.CalledProcessError(1, "download")):
            with self.subTest(failure=type(failure).__name__), self.workspace() as paths:
                directory = paths.MODELS / "dots.tts-soar"
                with patch.object(container.subprocess, "run", side_effect=failure):
                    expected = subprocess.CalledProcessError if failure else RuntimeError
                    with self.assertRaises(expected):
                        container.ensure_tts_model(directory, download=True)
                self.assertFalse((directory / ".t1000-revision").exists())
                self.assertFalse(list(directory.glob(".t1000-revision-*.tmp")))

    def test_offline_stt_validation_does_not_run_download_script(self):
        with self.workspace() as paths:
            directory = paths.MODELS / container.STT_NAME
            write_files(directory, container.STT_FILES)
            with patch.object(container.subprocess, "run") as run:
                container.ensure_stt_model(directory, download=False)
                (directory / "tokens.txt").write_bytes(b"")
                with self.assertRaisesRegex(RuntimeError, "tokens.txt"):
                    container.ensure_stt_model(directory, download=False)
            run.assert_not_called()

    def test_online_stt_uses_pinned_existing_fetch_script_and_target_directory(self):
        with self.workspace() as paths:
            directory = paths.MODELS / container.STT_NAME

            def download(command, **kwargs):
                self.assertEqual(command, [sys.executable, str(paths.ROOT / "scripts" / "fetch_stt.py"),
                                           "--model-dir", str(directory)])
                self.assertTrue(kwargs["check"])
                write_files(directory, container.STT_FILES)

            with patch.object(container.subprocess, "run", side_effect=download) as run:
                container.ensure_stt_model(directory, download=True)
            run.assert_called_once()

    def test_cuda_unavailable_fails_with_container_setup_instructions(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=False)))
        with patch.dict(os.environ, {"T1000_CONTAINER_DEVICE": "cuda"}), \
                patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA Container Toolkit"):
                container.check_device()

    def test_cuda_validation_executes_a_small_operation(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=True),
                                 get_device_name=Mock(return_value="Test GPU")),
            ones=Mock(),
        )
        with patch.dict(os.environ, {"T1000_CONTAINER_DEVICE": "cuda"}), \
                patch.dict(sys.modules, {"torch": torch}):
            container.check_device()
        torch.ones.assert_called_once_with(1, device="cuda")
        torch.ones.return_value.add_.assert_called_once_with(1)
        torch.ones.return_value.add_.return_value.cpu.assert_called_once()

    def test_cpu_validation_hides_cuda_devices(self):
        with patch.dict(os.environ, {"T1000_CONTAINER_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "0"}), \
                patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            container.check_device()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_prepare_checks_device_then_downloads_then_preloads(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), self.workspace(
                    T1000_CONTAINER_DEVICE=device, T1000_DOWNLOAD_MODELS="1", HF_HUB_OFFLINE="0") as paths:
                calls = []

                def check():
                    self.assertEqual(os.environ["T1000_CONTAINER_DEVICE"], device)
                    calls.append("device")

                def tts(directory, *, download):
                    self.assertEqual(directory, paths.MODELS / "dots.tts-soar")
                    self.assertTrue(download)
                    self.assertTrue(all(path.is_dir() for path in (paths.MODELS, paths.OUTPUT, paths.CACHE)))
                    calls.append("tts")

                def stt(directory, *, download):
                    self.assertEqual(directory, paths.MODELS / container.STT_NAME)
                    self.assertTrue(download)
                    calls.append("stt")

                def preload():
                    self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                    calls.append("preload")

                with patch.object(container, "check_device", side_effect=check), \
                        patch.object(container, "ensure_tts_model", side_effect=tts), \
                        patch.object(container, "ensure_stt_model", side_effect=stt), \
                        patch.object(container, "preload_models", side_effect=preload):
                    container.prepare()
                self.assertEqual(calls, ["device", "tts", "stt", "preload"])

    def test_device_failure_prevents_downloads_and_preload(self):
        with self.workspace(), \
                patch.object(container, "check_device", side_effect=RuntimeError("CUDA unavailable")), \
                patch.object(container, "ensure_tts_model") as tts, \
                patch.object(container, "ensure_stt_model") as stt, \
                patch.object(container, "preload_models") as preload:
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                container.prepare()
        tts.assert_not_called()
        stt.assert_not_called()
        preload.assert_not_called()

    def test_preload_disabled_still_requires_complete_local_models(self):
        with self.workspace(T1000_PRELOAD_MODELS="0") as paths, \
                patch.object(container, "check_device"), \
                patch.object(container.subprocess, "run") as run, \
                patch.object(container, "preload_models") as preload:
            with self.assertRaisesRegex(RuntimeError, "SOAR model is incomplete"):
                container.prepare()
            write_files(paths.MODELS / "dots.tts-soar", container.TTS_FILES)
            with self.assertRaisesRegex(RuntimeError, "CPU speech model is incomplete"):
                container.prepare()
            write_files(paths.MODELS / container.STT_NAME, container.STT_FILES)
            container.prepare()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        run.assert_not_called()
        preload.assert_not_called()

    def test_non_cpu_asr_is_rejected_before_device_or_download_work(self):
        with self.workspace(STT_BACKEND="cuda"), \
                patch.object(container, "check_device") as device, \
                patch.object(container, "ensure_tts_model") as tts:
            with self.assertRaisesRegex(ValueError, "STT_BACKEND=cpu"):
                container.prepare()
        device.assert_not_called()
        tts.assert_not_called()

    def test_missing_web_build_stops_before_downloading(self):
        with self.workspace() as paths, patch.object(container, "check_device") as device:
            (paths.WEB / "index.html").unlink()
            with self.assertRaisesRegex(RuntimeError, "built interface is missing"):
                container.prepare()
        device.assert_not_called()

    def test_preload_loads_tts_before_cpu_asr(self):
        calls = []
        tts = SimpleNamespace(load_runtime=Mock(side_effect=lambda: calls.append("tts")))
        stt = SimpleNamespace(preload_cpu_model=Mock(side_effect=lambda: calls.append("stt")))
        with patch.object(t1000, "tts", tts, create=True), patch.object(t1000, "stt", stt, create=True):
            container.preload_models()
        self.assertEqual(calls, ["tts", "stt"])

    def test_main_prepares_before_serving_one_worker_on_all_interfaces(self):
        calls = []
        run = Mock(side_effect=lambda *args, **kwargs: calls.append("serve"))
        with patch.object(container, "prepare", side_effect=lambda: calls.append("prepare")), \
                patch.dict(sys.modules, {"uvicorn": SimpleNamespace(run=run)}):
            container.main()
        self.assertEqual(calls, ["prepare", "serve"])
        run.assert_called_once_with("t1000.api:app", host="0.0.0.0", port=7860, workers=1)

    def test_main_reports_startup_failure_without_starting_server(self):
        run = Mock()
        with patch.object(container, "prepare", side_effect=RuntimeError("incomplete model")), \
                patch.dict(sys.modules, {"uvicorn": SimpleNamespace(run=run)}):
            with self.assertRaisesRegex(SystemExit, "Container startup failed: incomplete model"):
                container.main()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
