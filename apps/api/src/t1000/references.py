"""Preserve reviewed reference audio and cache its CPU-only preprocessing.

Upstream trims relative to the recording's loudest frame by default. That can
remove quiet opening or closing phonemes while retaining their transcript.
T1000_REFERENCE_TRIM=1 explicitly restores that behavior for comparisons; the
default keeps every input frame and leaves the transcript untouched.
"""

import hashlib
import os
import threading
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch


def _trim_enabled():
    value = os.environ.get("T1000_REFERENCE_TRIM", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("T1000_REFERENCE_TRIM must be 0 or 1.")
    return value == "1"


class _PromptAudioLoader:
    def __init__(self, sample_rate, *, trim=False, max_entries=8, max_bytes=16 * 2**20):
        for name, value in (("sample_rate", sample_rate), ("max_entries", max_entries),
                            ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.sample_rate = sample_rate
        self.trim = trim
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.Lock()

    def __call__(self, path):
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("The reference recording is empty or contains invalid audio.")
        # Match the application's existing mono conversion without changing gain.
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError("The reference recording contains invalid audio.")
        digest = hashlib.sha256(audio.tobytes()).digest()
        key = (digest, source_rate, self.sample_rate, self.trim)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                with torch.inference_mode(False):
                    return cached.clone()

        if self.trim:
            import librosa

            audio = librosa.effects.trim(audio, top_db=30)[0]
        from dots_tts.utils.audio import high_quality_resample

        # Normal CPU tensors remain safe across calls from inference-mode and
        # non-inference callers. No prompt copies are kept on the GPU here.
        with torch.inference_mode(False):
            waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            waveform = high_quality_resample(
                waveform, orig_sr=source_rate, target_sr=self.sample_rate,
            ).contiguous()
        if not waveform.numel() or not bool(torch.isfinite(waveform).all()):
            raise ValueError("Reference audio preprocessing produced invalid audio.")
        size = waveform.numel() * waveform.element_size()
        with self._lock:
            # Oversize entries still work, but never displace the bounded cache.
            if size <= self.max_bytes:
                previous = self._cache.pop(key, None)
                if previous is not None:
                    self._cache_bytes -= previous.numel() * previous.element_size()
                self._cache[key] = waveform
                self._cache_bytes += size
                while len(self._cache) > self.max_entries or self._cache_bytes > self.max_bytes:
                    _, removed = self._cache.popitem(last=False)
                    self._cache_bytes -= removed.numel() * removed.element_size()
            # Every caller gets independent storage, including the cache miss.
            with torch.inference_mode(False):
                return waveform.clone()


def configure_prompt_audio(runtime):
    """Override only this runtime's prompt loader; retain all other runtime APIs."""
    trim = _trim_enabled()
    current = getattr(runtime, "_load_prompt_audio", None)
    if (isinstance(current, _PromptAudioLoader) and current.trim == trim
            and current.sample_rate == runtime.sample_rate):
        return runtime
    runtime._load_prompt_audio = _PromptAudioLoader(runtime.sample_rate, trim=trim)
    return runtime
