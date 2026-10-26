"""Optional CUDA Distil-Whisper transcription with bounded buffered previews.

Whisper is not cache-aware streaming: previews re-decode at most 30 seconds,
at most once per two seconds of new audio. Finalization decodes the whole bounded
reference for accuracy. The default Nemotron CPU backend avoids this extra work.
"""

import os
import re
import threading
from collections import deque
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE, STTError, STTSession
from ..paths import MODELS

MODEL_DIR = Path(os.environ.get("STT_GPU_MODEL_DIR", MODELS / "distil-large-v3.5-ct2"))
MODEL_FILES = ("config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json")
GPU_LOCK = threading.RLock()
_model = None


def _load_gpu_model():
    global _model
    with GPU_LOCK:
        if _model is not None:
            return _model
        if any(not (MODEL_DIR / name).is_file() for name in MODEL_FILES):
            raise STTError(
                "CUDA transcription weights are missing. Run "
                "`.venv/bin/python scripts/fetch_whisper.py` before recording "
                f"(model directory: {MODEL_DIR})."
            )
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError("Install requirements/stt-gpu.txt to enable CUDA speech recognition.") from exc
        try:
            if ctranslate2.get_cuda_device_count() < 1:
                raise STTError("CUDA speech recognition requires an NVIDIA CUDA GPU. Use STT_BACKEND=cpu on this machine.")
            if "int8_float16" not in ctranslate2.get_supported_compute_types("cuda"):
                raise STTError("This GPU does not support int8_float16 transcription. Use STT_BACKEND=cpu.")
            _model = WhisperModel(
                str(MODEL_DIR),
                device="cuda",
                compute_type="int8_float16",
                local_files_only=True,
                num_workers=1,
            )
        except STTError:
            raise
        except Exception as exc:
            raise STTError(
                "CUDA transcription could not start. Check available VRAM and the CUDA 12/cuDNN 9 libraries, "
                f"or use STT_BACKEND=cpu. Details: {exc}"
            ) from exc
        return _model


def release_gpu_model():
    """Unload CT2 weights while holding the same lock as every CUDA inference."""
    global _model
    with GPU_LOCK:
        if _model is not None:
            model = _model
            _model = None
            native_model = getattr(model, "model", None)
            if native_model is not None:
                native_model.unload_model()


def _decode(samples, final):
    with GPU_LOCK:
        try:
            model = _load_gpu_model()
            segments, _ = model.transcribe(
                samples,
                language="en",
                task="transcribe",
                beam_size=5 if final else 1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=final,
                vad_parameters={"min_silence_duration_ms": 500} if final else None,
                hallucination_silence_threshold=1.0 if final else None,
            )
            return list(segments)
        except STTError:
            release_gpu_model()
            raise
        except Exception as exc:
            release_gpu_model()
            raise STTError(
                f"CUDA transcription failed: {exc}. Free GPU memory or use STT_BACKEND=cpu."
            ) from exc


def _merge_draft(previous, latest):
    """Join overlapping rolling-window drafts; the final pass replaces this text."""
    if not previous:
        return latest
    old, new = previous.split(), latest.split()
    normalize = lambda word: re.sub(r"[^\w']", "", word).lower()
    old_keys, new_keys = list(map(normalize, old)), list(map(normalize, new))
    for count in range(min(len(old), len(new)), 1, -1):
        if old_keys[-count:] == new_keys[:count]:
            return " ".join(old + new[count:])
    # A draft can revise its opening words. Find a shared short phrase deeper
    # inside the overlap instead of growing duplicate complete windows.
    for old_start in range(max(0, len(old) - 200), len(old) - 2):
        phrase = old_keys[old_start : old_start + 3]
        for new_start in range(min(40, max(0, len(new) - 2))):
            if phrase == new_keys[new_start : new_start + 3]:
                return " ".join(old[:old_start] + new[new_start:])
    return latest  # The current window is safer than a fabricated join.


class GPUSTTSession(STTSession):
    """Explicit CUDA alternative sharing the CPU pipeline's input validation."""

    def __init__(self, max_seconds=180.0):
        super().__init__(max_seconds=max_seconds)
        self.previews_enabled = True
        self._chunks = []
        self._preview_chunks = deque()
        self._preview_samples = 0
        self._sample_count = 0
        self._last_preview_samples = 0

    def _ensure_stream(self):
        if self._stream is None:
            _load_gpu_model()
            self._stream = True

    def _accept(self, samples):
        if not samples.size:
            return
        samples = np.ascontiguousarray(samples, dtype=np.float32).copy()
        self._chunks.append(samples)
        self._preview_chunks.append(samples)
        self._sample_count += samples.size
        self._preview_samples += samples.size
        while self._preview_samples > 30 * SAMPLE_RATE:
            excess = self._preview_samples - 30 * SAMPLE_RATE
            first = self._preview_chunks.popleft()
            if first.size > excess:
                self._preview_chunks.appendleft(first[excess:])
                self._preview_samples -= excess
            else:
                self._preview_samples -= first.size
        if not self.previews_enabled or self._sample_count - self._last_preview_samples < 2 * SAMPLE_RATE:
            return
        self._last_preview_samples = self._sample_count
        window = np.concatenate(self._preview_chunks)
        # Avoid Whisper hallucinations on pure silence without discarding quiet
        # speech. Final inference also uses the model's speech activity filter.
        if not np.any(window):
            return
        segments = _decode(window, final=False)
        latest = " ".join(segment.text.strip() for segment in segments).strip()
        self.text = latest if self._sample_count <= 30 * SAMPLE_RATE else _merge_draft(self.text, latest)

    def finish(self):
        with self._lock:
            if self.finished:
                return self.text
            try:
                if self._resampler is not None:
                    # Flush fractional resampling state without triggering an
                    # unnecessary draft immediately before the final decode.
                    remaining = self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                    if remaining.size:
                        self._chunks.append(remaining)
                if self._chunks:
                    samples = np.concatenate(self._chunks)
                    if np.any(samples):
                        self.text = " ".join(segment.text.strip() for segment in _decode(samples, final=True)).strip()
                    else:
                        self.text = ""
                self.finished = True
                return self.text
            finally:
                release_gpu_model()
                self._chunks.clear()
                self._preview_chunks.clear()

    def close(self):
        """Discard buffered audio and release optional CUDA weights, without ASR."""
        super().close()
        with self._lock:
            self._chunks.clear()
            self._preview_chunks.clear()
            release_gpu_model()
