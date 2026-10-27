"""Prepare persistent models, warm inference, and serve the Docker interface."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .paths import CACHE, MODELS, OUTPUT, ROOT, WEB


TTS_REPOSITORY = "dots-studio/dots.tts-soar"
TTS_REVISION = "2f9b3e18d70d670d4c701da2dc55ded5755815ce"
TTS_FILES = (
    "config.json", "llm_config.json", "latent_stats.pt", "model.safetensors",
    "vocoder.safetensors", "speaker_encoder.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json",
)
STT_NAME = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
STT_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def _flag(name, default="1"):
    value = os.environ.get(name, default).strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1.")
    return value == "1"


def _missing(directory, names):
    return [name for name in names if not (directory / name).is_file() or (directory / name).stat().st_size == 0]


def ensure_tts_model(directory, *, download):
    directory = Path(directory)
    marker = directory / ".t1000-revision"
    missing = _missing(directory, TTS_FILES)
    if not missing and (not download or (marker.is_file() and marker.read_text().strip() == TTS_REVISION)):
        print(f"Using local SOAR model: {directory}", flush=True)
        return
    if not download:
        raise RuntimeError(f"SOAR model is incomplete in {directory}: {', '.join(missing)}. Enable T1000_DOWNLOAD_MODELS=1.")

    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/resuming pinned SOAR model {TTS_REVISION} (about 5.2 GB)…", flush=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "0"
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    # A separate process prevents cached Hugging Face offline constants from
    # blocking downloads, even if a caller imported the application first.
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
         TTS_REPOSITORY, TTS_REVISION, str(directory)],
        env=environment, check=True,
    )
    missing = _missing(directory, TTS_FILES)
    if missing:
        raise RuntimeError(f"SOAR download is incomplete: {', '.join(missing)}")
    temporary = directory / f".t1000-revision-{uuid4().hex}.tmp"
    try:
        temporary.write_text(TTS_REVISION + "\n")
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_stt_model(directory, *, download):
    directory = Path(directory)
    if download:
        print("Preparing the verified CPU speech-recognition model…", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_stt.py"), "--model-dir", str(directory)],
            check=True,
        )
    missing = _missing(directory, STT_FILES)
    if missing:
        raise RuntimeError(f"CPU speech model is incomplete in {directory}: {', '.join(missing)}. Enable T1000_DOWNLOAD_MODELS=1.")


def check_device():
    expected = os.environ.get("T1000_CONTAINER_DEVICE", "cpu").strip()
    if expected not in {"cpu", "cuda"}:
        raise ValueError("T1000_CONTAINER_DEVICE must be cpu or cuda.")
    if expected == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if expected == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use the GPU Compose file with NVIDIA Container Toolkit and a compatible host driver.")
        # Fail before downloading gigabytes if the driver's CUDA execution fails.
        torch.ones(1, device="cuda").add_(1).cpu()
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Using CPU inference.", flush=True)


def preload_models():
    from . import stt, tts

    print("Loading the speech model; GPU compilation can take several minutes…", flush=True)
    tts.load_runtime()
    print("Loading CPU speech recognition…", flush=True)
    stt.preload_cpu_model()


def prepare():
    download = _flag("T1000_DOWNLOAD_MODELS")
    preload = _flag("T1000_PRELOAD_MODELS")
    if os.environ.get("STT_BACKEND", "cpu").strip().lower() != "cpu":
        raise ValueError("Docker startup uses CPU transcription. Set STT_BACKEND=cpu.")
    if not (WEB / "index.html").is_file():
        raise RuntimeError("The built interface is missing. Rebuild the Docker image.")
    check_device()
    for directory in (MODELS, OUTPUT, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_tts_model(MODELS / "dots.tts-soar", download=download)
    ensure_stt_model(Path(os.environ.get("STT_MODEL_DIR", MODELS / STT_NAME)), download=download)
    # All later model operations use the prepared local files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    if preload:
        preload_models()
    else:
        print("Preload disabled; the speech model will load on the first generation.", flush=True)


def main():
    try:
        prepare()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Container startup failed: {exc}") from exc
    import uvicorn

    print("Voice lab ready on port 7860.", flush=True)
    uvicorn.run("t1000.api:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
