"""Optional eager vocoder batching for the NF4 runtime.

The first two patches retain their usual cadence, then groups of four reduce
repeated decoder work. Each stream owns its pending latents and flushes its
final partial group. Grouping also means Stop can wait for up to four acoustic
patches between audio chunks. Compiled inference keeps the upstream adapter.
"""

import os
import weakref
from dataclasses import dataclass, field

import torch


def _validate_batch_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 4}:
        raise ValueError("Vocoder batch size must be 1 or 4.")
    return value


def read_vocoder_batch_size():
    """Read the opt-in before loading model weights or allocating CUDA memory."""
    value = os.environ.get("T1000_TTS_VOCODER_BATCH_SIZE", "1").strip()
    if value not in {"1", "4"}:
        raise ValueError("T1000_TTS_VOCODER_BATCH_SIZE must be 1 or 4.")
    return int(value)


@dataclass(slots=True)
class _BufferedStreamState:
    inner: object
    patches_seen: int = 0
    pending: list = field(default_factory=list)
    pending_start: int = 0


class BufferedVocoderInference:
    """Delegate upstream operations while batching only eager streaming calls."""

    def __init__(self, inner, *, batch_size=4, initial_unmerged=2, model=None):
        self.inner = inner
        self.batch_size = _validate_batch_size(batch_size)
        if (isinstance(initial_unmerged, bool) or not isinstance(initial_unmerged, int)
                or initial_unmerged < 0):
            raise ValueError("initial_unmerged must be a nonnegative integer.")
        self.initial_unmerged = initial_unmerged
        self._model_ref = weakref.ref(model) if model is not None else None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def init_stream_state(self, *, batch_size=1, chunk_size):
        model = self._model_ref() if self._model_ref is not None else None
        enabled = self.batch_size > 1 and not getattr(model, "_optimize_enabled", False)
        state = self.inner.init_stream_state(
            batch_size=batch_size,
            chunk_size=chunk_size * self.batch_size if enabled else chunk_size,
        )
        return _BufferedStreamState(state) if enabled else state

    def stream_step(self, latent_patch, stream_state, *, optimize,
                    profile_step=None, use_compiled=True):
        if not isinstance(stream_state, _BufferedStreamState):
            return self.inner.stream_step(
                latent_patch, stream_state, optimize=optimize,
                profile_step=profile_step, use_compiled=use_compiled,
            )
        if optimize:
            raise RuntimeError("Cannot enable optimization during a buffered vocoder stream.")
        window = stream_state.inner.decoder.window
        if (latent_patch.ndim != 3 or latent_patch.size(0) != window.size(0)
                or latent_patch.size(2) != window.size(1) or latent_patch.size(1) < 1):
            raise ValueError("Vocoder latents must have shape [batch, frames, latent_dim].")
        stream_state.patches_seen += 1
        if stream_state.patches_seen <= self.initial_unmerged:
            return self.inner.stream_step(
                latent_patch, stream_state.inner, optimize=False,
                profile_step=profile_step, use_compiled=False,
            )
        if not stream_state.pending:
            stream_state.pending_start = stream_state.patches_seen
        # The next acoustic step may overwrite the tensor that it yielded.
        stream_state.pending.append(latent_patch.detach().clone())
        if len(stream_state.pending) < self.batch_size:
            return window.new_empty((latent_patch.size(0), 1, 0))
        return self._emit_pending(stream_state)

    def _emit_pending(self, state):
        latents = torch.cat(state.pending, dim=1)
        result = self.inner.stream_step(
            latents, state.inner, optimize=False,
            profile_step=f"{state.pending_start}-{state.patches_seen}",
            use_compiled=False,
        )
        state.pending.clear()
        state.pending_start = 0
        return result

    def flush(self, stream_state):
        if not isinstance(stream_state, _BufferedStreamState):
            return self.inner.flush(stream_state)
        if not stream_state.pending:
            return self.inner.flush(stream_state.inner)
        audio = self._emit_pending(stream_state)
        return torch.cat((audio, self.inner.flush(stream_state.inner)), dim=-1)


def configure_vocoder_batching(runtime, batch_size):
    """Install once after placement; standard/compiled runtimes pass through."""
    _validate_batch_size(batch_size)
    if batch_size == 1 or runtime.optimize or runtime.model._optimize_enabled:
        return runtime
    model = runtime.model
    inner = model._get_vocoder_inference()
    if not isinstance(inner, BufferedVocoderInference):
        model._vocoder_inference = BufferedVocoderInference(
            inner, batch_size=batch_size,
            initial_unmerged=model.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES,
            model=model,
        )
        model._vocoder_inference_vocoder_id = id(model.vocoder)
    return runtime
