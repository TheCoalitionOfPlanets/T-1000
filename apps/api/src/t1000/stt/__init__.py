"""Offline, cache-aware English speech recognition with Nemotron and sherpa-onnx.

Keep the recognizer in CPU memory and give every recording its own encoder and
decoder state. Audio is consumed once; growing recordings are never re-decoded.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from ..paths import MODELS
MODEL_NAME = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
MODEL_DIR = Path(os.environ.get("STT_MODEL_DIR", MODELS / MODEL_NAME))
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
SAMPLE_RATE = 16000
HEAD_PADDING_SECONDS = 0.5
TAIL_PADDING_SECONDS = 0.8
_LOAD_LOCK = threading.Lock()
_DECODE_LOCK = threading.Lock()
_recognizer = None


class STTError(RuntimeError):
    """An actionable transcription or recording error."""


def _load_recognizer():
    """Load weights exactly once, without changing the application's Torch stack."""
    global _recognizer
    with _LOAD_LOCK:
        if _recognizer is not None:
            return _recognizer
        missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
        if missing:
            raise STTError(
                "Speech recognition weights are missing. Run "
                "`.venv/bin/python scripts/fetch_stt.py` before recording "
                f"(model directory: {MODEL_DIR})."
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise STTError(
                "Speech recognition is not installed. Install the project's updated "
                "requirements, or run `uv pip install --python .venv/bin/python "
                "sherpa-onnx==2.0.1`."
            ) from exc
        try:
            num_threads = int(os.environ.get("STT_NUM_THREADS", min(4, os.cpu_count() or 1)))
            if num_threads < 1:
                raise ValueError("must be positive")
        except ValueError as exc:
            raise STTError("STT_NUM_THREADS must be a positive integer.") from exc
        try:
            _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(MODEL_DIR / "tokens.txt"),
                encoder=str(MODEL_DIR / "encoder.int8.onnx"),
                decoder=str(MODEL_DIR / "decoder.int8.onnx"),
                joiner=str(MODEL_DIR / "joiner.int8.onnx"),
                model_type="nemotron",
                sample_rate=SAMPLE_RATE,
                feature_dim=128,
                num_threads=num_threads,
                provider="cpu",
                decoding_method="greedy_search",
                enable_endpoint_detection=False,
            )
        except Exception as exc:
            raise STTError(f"Could not load speech recognition weights: {exc}") from exc
        return _recognizer


def _mono_float32(audio):
    """Convert browser PCM or normalized file audio without changing its loudness."""
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2) or (audio.ndim == 2 and not 1 <= audio.shape[1] <= 8):
        raise STTError("Audio must contain mono or stereo sample frames.")
    if np.issubdtype(audio.dtype, np.signedinteger):
        scale = float(-np.iinfo(audio.dtype).min)
        audio = audio.astype(np.float32) / scale
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    else:
        raise STTError("Audio must contain numeric PCM or floating-point samples.")
    if not np.isfinite(audio).all():
        raise STTError("The recording contains invalid audio samples.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


class STTSession:
    """One recording's incremental recognition state; safe to finalize twice."""

    def __init__(self, max_seconds=180.0):
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        self.max_seconds = float(max_seconds)
        self.text = ""
        self.audio_seconds = 0.0
        self.finished = False
        self.limit_reached = False
        self._input_rate = None
        self._input_samples = 0
        self._resampler = None
        self._recognizer = None
        self._stream = None
        self._lock = threading.Lock()

    def _ensure_stream(self):
        if self._stream is None:
            self._recognizer = _load_recognizer()
            with _DECODE_LOCK:
                self._stream = self._recognizer.create_stream()
                self._stream.accept_waveform(
                    SAMPLE_RATE, np.zeros(int(HEAD_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32)
                )

    def _accept(self, samples):
        # Bound each shared-recognizer lock hold even for uploaded recordings.
        for offset in range(0, samples.size, SAMPLE_RATE):
            with _DECODE_LOCK:
                self._stream.accept_waveform(SAMPLE_RATE, samples[offset : offset + SAMPLE_RATE])
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)
                self.text = self._recognizer.get_result(self._stream)

    def feed(self, sample_rate, audio):
        """Consume one new chunk and return this recording's current transcript."""
        with self._lock:
            if self.finished:
                raise STTError("This recording has finished. Start a new recording to transcribe again.")
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
                raise STTError("The recording's sample rate must be a positive integer.")
            if self._input_rate is not None and sample_rate != self._input_rate:
                raise STTError("The recording's sample rate changed. Start a new recording.")
            samples = _mono_float32(audio)
            remaining = max(0, int(self.max_seconds * sample_rate) - self._input_samples)
            samples = samples[:remaining]
            if not samples.size:
                return self.text
            self._ensure_stream()
            if self._input_rate is None:
                self._input_rate = int(sample_rate)
                if sample_rate != SAMPLE_RATE:
                    self._resampler = soxr.ResampleStream(
                        sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
                    )
            self._input_samples += samples.size
            self.audio_seconds = self._input_samples / sample_rate
            self.limit_reached = self._input_samples >= int(self.max_seconds * sample_rate)
            if self._resampler is not None:
                samples = self._resampler.resample_chunk(samples)
            self._accept(samples)
            return self.text

    def finish(self):
        """Flush resampling and model lookahead once, retaining the final words."""
        with self._lock:
            if self.finished:
                return self.text
            if self._stream is not None:
                if self._resampler is not None:
                    self._accept(self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
                self._accept(np.zeros(int(TAIL_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32))
                with _DECODE_LOCK:
                    self._stream.input_finished()
                    while self._recognizer.is_ready(self._stream):
                        self._recognizer.decode_stream(self._stream)
                    self.text = self._recognizer.get_result(self._stream)
            self.finished = True
            return self.text

    def close(self):
        """Discard this recording's native state without running any inference."""
        with self._lock:
            self.finished = True
            self._stream = None
            self._recognizer = None
            self._resampler = None


def create_session(max_seconds=180.0):
    """Choose explicit CPU streaming or optional CUDA buffered recognition."""
    backend = os.environ.get("STT_BACKEND", "cpu").lower().strip()
    if backend == "cpu":
        return STTSession(max_seconds=max_seconds)
    if backend == "cuda":
        from .gpu import GPUSTTSession

        return GPUSTTSession(max_seconds=max_seconds)
    raise STTError("STT_BACKEND must be 'cpu' or 'cuda'.")


def preload_cpu_model():
    """Move the default recognizer's cold load before the first microphone use."""
    if os.environ.get("STT_BACKEND", "cpu").lower().strip() == "cpu":
        _load_recognizer()


def release_gpu_model():
    """Free optional CUDA recognizer weights before loading the TTS model."""
    module = sys.modules.get(f"{__name__}.gpu")
    if module is not None:
        module.release_gpu_model()


def gpu_compute_lock():
    """Serialize optional CUDA recognition with TTS to bound peak GPU usage."""
    from .gpu import GPU_LOCK

    return GPU_LOCK


def transcribe_file(path, max_seconds=180.0):
    """Transcribe an uploaded reference, bounded like a microphone recording."""
    session = create_session(max_seconds=max_seconds)
    if hasattr(session, "previews_enabled"):
        session.previews_enabled = False  # Uploaded files need only a final pass.
    try:
        with sf.SoundFile(str(path)) as recording:
            while not session.limit_reached:
                samples = recording.read(recording.samplerate, dtype="float32", always_2d=True)
                if not samples.size:
                    break
                session.feed(recording.samplerate, samples)
        return session.finish()
    except (OSError, sf.LibsndfileError) as exc:
        raise STTError(f"Could not read the reference recording: {exc}") from exc
    finally:
        session.close()
