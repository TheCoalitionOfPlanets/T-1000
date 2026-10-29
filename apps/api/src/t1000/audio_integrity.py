"""Publish only closed WAV files and distinguish finished speech from a prefix.

The worker appends each host-ready mono chunk exactly once. The hidden working
file never crosses the API boundary; publication is one atomic rename after a
valid header, sample count and checksum are available. PCM16 quantization is
the existing output format, not a promise of bit-identical floating samples.
"""

import hashlib
import inspect
import os
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf


class GeneratedAudio(str):
    """A path compatible with existing players, carrying its transport contract."""

    def __new__(cls, path, metadata):
        instance = super().__new__(cls, str(path))
        instance.metadata = dict(metadata)
        return instance


class AtomicAudioWriter:
    """Append ordered mono float chunks to one bounded-memory PCM16 WAV.

    A failed chunk is rolled back to the preceding committed sample boundary.
    ``finish`` can then publish that valid prefix with ``complete=False``.
    Persistent filesystem failures leave the hidden file available for recovery
    and raise; they never publish a claimed-complete or malformed WAV.
    """

    def __init__(self, output_dir, sample_rate):
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate < 1:
            raise ValueError("The audio sample rate must be a positive integer.")
        self.sample_rate = int(sample_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        identity = uuid4().hex
        self._temporary = self.output_dir / f".speech-{identity}.wav.part"
        self._final = self.output_dir / f"speech-{identity}.wav"
        self.total_samples = 0
        self.chunks = 0
        self.clipped_sample_count = 0
        self._file = None
        self._write_error = None
        self._needs_repair = False
        self._sealed = False
        self._finished = False
        self._result = None
        self._lock = threading.RLock()

    @property
    def frames(self):
        return self.total_samples

    @property
    def chunk_count(self):
        return self.chunks

    def append(self, audio):
        """Return ``(sample_start, sample_count)`` after a whole chunk commits."""
        samples = np.asarray(audio)
        if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.floating):
            raise ValueError("Audio chunks must be one-dimensional floating-point mono samples.")
        with np.errstate(over="ignore", invalid="ignore"):
            samples = np.ascontiguousarray(samples, dtype=np.float32)
        if not np.isfinite(samples).all():
            raise ValueError("The audio chunk contains non-finite samples.")
        with self._lock:
            if self._sealed:
                raise RuntimeError("This audio writer has finished.")
            if self._write_error is not None:
                raise RuntimeError("An audio write failed; finalize the valid prefix before continuing.")
            start = self.total_samples
            if not samples.size:
                return start, 0
            if self._file is None:
                self._file = sf.SoundFile(
                    self._temporary, mode="x", samplerate=self.sample_rate,
                    channels=1, format="WAV", subtype="PCM_16",
                )
            try:
                self._file.write(samples)
                self._file.flush()
            except Exception as exc:
                self._write_error = f"{type(exc).__name__}: {exc}"
                try:
                    self._file.truncate(start)
                    self._file.flush()
                except Exception:
                    # Retry the truncation after closing, when libsndfile has
                    # updated the WAV header and no buffered writer remains.
                    self._needs_repair = True
                raise
            self.total_samples += int(samples.size)
            self.chunks += 1
            self.clipped_sample_count += int(np.count_nonzero((samples < -1) | (samples > 1)))
            return start, int(samples.size)

    def _close(self):
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        except Exception as exc:
            self._write_error = self._write_error or f"{type(exc).__name__}: {exc}"
            self._needs_repair = True
            # SoundFile.close flushes before closing. One retry handles a
            # transient failure without publishing a still-open mutable file.
            self._file.close()
        finally:
            if self._file.closed:
                self._file = None

    def finish(self, *, complete, reason):
        """Close and atomically publish once, or return ``None`` for no samples."""
        if not isinstance(complete, bool) or not isinstance(reason, str) or not reason:
            raise ValueError("Audio completion needs a boolean and a nonempty reason.")
        with self._lock:
            if self._finished:
                return self._result
            self._sealed = True
            self._close()
            if not self.total_samples:
                self._temporary.unlink(missing_ok=True)
                self._finished = True
                return None
            if self._needs_repair:
                with sf.SoundFile(self._temporary, mode="r+") as recovery:
                    if recovery.frames < self.total_samples:
                        raise RuntimeError("The partial WAV is missing previously committed audio.")
                    recovery.truncate(self.total_samples)
                    recovery.flush()
            info = sf.info(self._temporary)
            if (info.frames != self.total_samples or info.samplerate != self.sample_rate
                    or info.channels != 1 or info.subtype != "PCM_16"):
                raise RuntimeError("The saved WAV does not match its committed sample boundaries.")
            with self._temporary.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            metadata = {
                "sample_rate": self.sample_rate,
                "channels": 1,
                "total_samples": self.total_samples,
                "duration_seconds": self.total_samples / self.sample_rate,
                "chunks": self.chunks,
                "sha256": checksum,
                "byte_length": self._temporary.stat().st_size,
                "complete": complete and self._write_error is None,
                "reason": "write_error" if self._write_error else reason,
                "clipped_sample_count": self.clipped_sample_count,
                "clipped": self.clipped_sample_count > 0,
            }
            if self._write_error:
                metadata["error"] = self._write_error
            os.replace(self._temporary, self._final)
            self._result = GeneratedAudio(self._final, metadata)
            self._finished = True
            return self._result


class DecodeEndObserver:
    """Observe the existing decode state while the caller owns its model lock.

    A normal generator return is insufficient: the runtime can exhaust its
    acoustic schedule. The original CPU ``end_flag`` is read only after full
    exhaustion, including the final yielded patch. No sampler arithmetic or
    CUDA scalar reads are introduced. Fake runtimes without ``_decode`` remain
    ``unknown`` and are returned unchanged.
    """

    def __init__(self, runtime_or_model):
        self.model = getattr(runtime_or_model, "model", runtime_or_model)
        self.stop_reason = "unknown"
        self.decoded_patches = 0
        self._original = None
        self._owned_attribute = False
        self._streams = []
        self._entered = False
        self._exited = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("A decode observer cannot be entered twice.")
        self._entered = True
        self._original = getattr(self.model, "_decode", None)
        if callable(self._original):
            self._owned_attribute = "_decode" in getattr(self.model, "__dict__", {})
            self.model._decode = self._observe
        return self

    def _observe(self, *args, **kwargs):
        state = kwargs.get("state")
        if state is None and args:
            try:
                state = inspect.signature(self._original).bind_partial(*args, **kwargs).arguments.get("state")
            except (ValueError, TypeError):
                pass
        stream = None
        try:
            stream = self._original(*args, **kwargs)
            self._streams.append(stream)
            for patch in stream:
                self.decoded_patches += 1
                yield patch
        except GeneratorExit:
            self.stop_reason = "closed"
            raise
        except BaseException:
            self.stop_reason = "error"
            raise
        else:
            flag = getattr(state, "end_flag", None)
            if self._exited:
                self.stop_reason = "closed"
            elif isinstance(flag, (bool, np.bool_)):
                self.stop_reason = "eos" if flag else "schedule_exhausted"
            else:
                self.stop_reason = "unknown"
        finally:
            if stream is not None:
                if stream in self._streams:
                    self._streams.remove(stream)
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException:
                        self.stop_reason = "error"
                        raise

    def __exit__(self, exc_type, exc, traceback):
        self._exited = True
        close_error = None
        try:
            if self._streams:
                self.stop_reason = "closed"
            for stream in self._streams[:]:
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as error:
                        self.stop_reason = "error"
                        close_error = error
            self._streams.clear()
        finally:
            if callable(self._original):
                if self._owned_attribute:
                    self.model._decode = self._original
                else:
                    del self.model._decode
        if close_error is not None and exc_type is None:
            raise close_error
        return False
