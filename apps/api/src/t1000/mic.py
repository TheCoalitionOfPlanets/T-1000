"""Bounded microphone transport with a complete, validated final recording.

The browser keeps the full PCM recording independently of live ASR requests.
Tokens, sequence numbers, and final PCM reconciliation make retries and the last
audio frame safe without trusting browser-supplied local file paths.
"""

import asyncio
import base64
import binascii
import hashlib
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from .stt import STTError, create_session, transcribe_file
from .paths import CACHE


STATIC = Path(__file__).resolve().parent / "static"
RECORDING_DIR = CACHE / "recordings"
SESSION_TTL = 15 * 60
MAX_SESSIONS = 16
MAX_COMPLETED_RESULTS = 64
MAX_PREVIEW_CHUNKS = 4096
MANUAL_STATUS = "Automatic transcription is unavailable. Enter the recording’s words below."


@dataclass
class _Recording:
    sample_rate: int
    session: object
    updated: float = field(default_factory=time.monotonic)
    sequence: int = 0
    chunks: list = field(default_factory=list)
    sample_count: int = 0
    failed: bool = False
    error_reason: str = ""
    cancelled: bool = False
    closed: bool = False
    result: tuple | None = None
    final_digest: bytes | None = None
    published: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class RecorderService:
    def __init__(self, max_seconds):
        if not 0 < max_seconds <= 180:
            raise ValueError("Recording limit must be between 0 and 180 seconds.")
        self.max_seconds = float(max_seconds)
        self._recordings = {}
        # Final HTTP responses can be lost after ASR has finished. Keep a small
        # result ledger separately from active recognition sessions so retries
        # can recover the exact recording without using an active slot.
        self._completed = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _close_session(entry):
        """Called with entry.lock held, after all inference using it has ended."""
        if entry.closed:
            return
        try:
            close = getattr(entry.session, "close", None)
            if close is not None:
                close()
        except Exception:
            # Cleanup must not discard a complete recording or transcription.
            pass
        finally:
            entry.closed = True

    def _discard_entry(self, entry):
        # Never wait for an inference/session lock while holding the registry
        # lock: completion acquires the same locks in the opposite direction.
        with entry.lock:
            self._close_session(entry)
            entry.chunks.clear()
            if not entry.published and entry.result and entry.result[0]:
                Path(entry.result[0]).unlink(missing_ok=True)

    def _get(self, token):
        if not isinstance(token, str):
            raise ValueError("Invalid recording token.")
        expired = None
        with self._lock:
            entry = self._recordings.get(token) or self._completed.get(token)
            if entry is not None and time.monotonic() - entry.updated > SESSION_TTL:
                self._recordings.pop(token, None)
                self._completed.pop(token, None)
                entry.cancelled = True
                expired = entry
        if expired is not None:
            self._discard_entry(expired)
        if entry is None or entry.cancelled:
            raise ValueError("This recording expired. Start a new recording.")
        return entry

    def _pcm(self, data, sample_rate):
        maximum = int(self.max_seconds * sample_rate) * 2
        if not isinstance(data, str) or len(data) > ((maximum + 2) // 3) * 4:
            raise ValueError("The recording exceeds the recording limit.")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid microphone audio.") from exc
        if len(raw) % 2 or len(raw) > maximum:
            raise ValueError("Invalid microphone audio length.")
        return raw

    def _start(self, data):
        rate = data.get("sample_rate")
        if isinstance(rate, bool) or not isinstance(rate, int) or not 8000 <= rate <= 192000:
            raise ValueError("Unsupported microphone sample rate.")
        now = time.monotonic()
        expired = []
        with self._lock:
            for registry in (self._recordings, self._completed):
                for token, entry in list(registry.items()):
                    if now - entry.updated > SESSION_TTL:
                        entry.cancelled = True
                        registry.pop(token)
                        expired.append(entry)
        for entry in expired:
            self._discard_entry(entry)
        with self._lock:
            if len(self._recordings) >= MAX_SESSIONS:
                raise ValueError("Too many active recordings. Finish or cancel a recording first.")
            token = secrets.token_urlsafe(32)
            self._recordings[token] = _Recording(rate, create_session(max_seconds=self.max_seconds))
        return {"token": token, "max_seconds": self.max_seconds}

    def _feed(self, data):
        entry = self._get(data.get("token"))
        raw = self._pcm(data.get("pcm"), entry.sample_rate)
        sequence = data.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Invalid microphone sequence.")
        if not raw:
            raise ValueError("The microphone audio chunk is empty.")
        with entry.lock:
            if entry.cancelled or entry.result is not None:
                return {"cancelled": True}
            if sequence < entry.sequence:
                if raw != entry.chunks[sequence]:
                    raise ValueError("A microphone retry changed an acknowledged audio chunk.")
                entry.updated = time.monotonic()
                return {
                    "sequence": entry.sequence,
                    "sample_count": entry.sample_count,
                    "text": "" if entry.failed else entry.session.text,
                    "cancelled": entry.cancelled,
                }
            if sequence != entry.sequence:
                raise ValueError("Microphone audio arrived out of order.")
            if sequence >= MAX_PREVIEW_CHUNKS:
                raise ValueError("Too many live microphone chunks. Finish with the complete recording.")
            count = len(raw) // 2
            if entry.sample_count + count > int(self.max_seconds * entry.sample_rate):
                raise ValueError("The recording exceeds the recording limit.")
            entry.chunks.append(raw)
            entry.sample_count += count
            entry.sequence += 1
            entry.updated = time.monotonic()
            if not entry.failed:
                try:
                    entry.session.feed(entry.sample_rate, np.frombuffer(raw, dtype="<i2"))
                except Exception as exc:
                    # Finalization still saves the recording for manual transcription.
                    entry.failed = True
                    entry.error_reason = str(exc) if isinstance(exc, STTError) else ""
                    self._close_session(entry)
            return {
                "sequence": entry.sequence,
                "sample_count": entry.sample_count,
                "text": "" if entry.failed else entry.session.text,
                "status": f"{entry.error_reason} {MANUAL_STATUS}".strip() if entry.failed else "Listening…",
                "cancelled": entry.cancelled,
            }

    def _finish(self, data):
        token = data.get("token")
        entry = self._get(token)
        raw = self._pcm(data.get("pcm"), entry.sample_rate)
        digest = hashlib.sha256(raw).digest()
        with entry.lock:
            if entry.cancelled:
                return {"cancelled": True}
            if entry.result is not None:
                if digest != entry.final_digest:
                    raise ValueError("A final recording retry changed the saved audio. Start a new recording.")
                return {"token": token}
            if not raw:
                raise ValueError("The recording is empty. Record a clear sentence.")
            if len(raw) // 2 < entry.sample_count:
                raise ValueError("The complete recording is missing acknowledged audio. Retry with the complete recording.")
            RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            path = RECORDING_DIR / f"recording-{token}.wav"
            sf.write(path, np.frombuffer(raw, dtype="<i2"), entry.sample_rate, subtype="PCM_16")
            text = ""
            try:
                streamed = b"".join(entry.chunks)
                if entry.failed or not raw.startswith(streamed):
                    self._close_session(entry)
                    text = transcribe_file(str(path), max_seconds=self.max_seconds)
                else:
                    try:
                        tail = raw[len(streamed):]
                        if tail:
                            entry.session.feed(entry.sample_rate, np.frombuffer(tail, dtype="<i2"))
                        text = entry.session.finish()
                    except Exception as exc:
                        # A preview decoder can fail while flushing its last
                        # frames. The saved complete recording is authoritative;
                        # retry it in a fresh recognizer rather than publishing
                        # the earlier partial transcript or losing the tail.
                        if isinstance(exc, STTError):
                            entry.error_reason = str(exc)
                        self._close_session(entry)
                        text = transcribe_file(str(path), max_seconds=self.max_seconds)
                text = text.strip()
            except Exception as exc:
                # Preserve usable audio even when the optional ASR model is missing.
                entry.error_reason = str(exc) if isinstance(exc, STTError) else entry.error_reason
            finally:
                self._close_session(entry)
            status = "Recording transcribed. Check the words before generating speech." if text else f"{entry.error_reason} {MANUAL_STATUS}".strip()
            if entry.cancelled:
                path.unlink(missing_ok=True)
                return {"cancelled": True}
            entry.result = (str(path), text, status)
            entry.final_digest = digest
            entry.chunks.clear()
            entry.updated = time.monotonic()
            return {"token": token}

    def consume(self, token, *, retain=False):
        """Publish a complete result; HTTP callers can retain it for retries.

        The published WAV belongs to its caller, so expiry/eviction of the
        retry ledger never deletes it. Legacy Gradio consumption still removes
        the token immediately unless ``retain=True`` is explicitly requested.
        """
        entry = self._get(token)
        with entry.lock:
            if entry.result is None or entry.cancelled:
                raise ValueError("The recording has not finished.")
            self._close_session(entry)
            with self._lock:
                if entry.cancelled:
                    raise ValueError("The recording was cancelled before publication.")
                entry.published = True
                self._recordings.pop(token, None)
                if retain:
                    self._completed[token] = entry
                    self._completed.move_to_end(token)
                    while len(self._completed) > MAX_COMPLETED_RESULTS:
                        self._completed.popitem(last=False)
                else:
                    self._completed.pop(token, None)
            return entry.result

    async def start(self, data):
        return await asyncio.to_thread(self._start, data)

    async def feed(self, data):
        return await asyncio.to_thread(self._feed, data)

    async def finish(self, data):
        return await asyncio.to_thread(self._finish, data)

    async def cancel(self, data):
        with self._lock:
            entry = self._recordings.pop(data.get("token"), None)
            entry = self._completed.pop(data.get("token"), None) or entry
            if entry is not None:
                entry.cancelled = True
        if entry is not None:
            await asyncio.to_thread(self._discard_entry, entry)
        return {"cancelled": True}


_service = None


def build_recorder(max_seconds):
    global _service
    _service = RecorderService(max_seconds)
    service = _service

    # Gradio decorates registered callables by assigning attributes. Bound
    # methods do not permit that, so register ordinary async closures instead.
    async def start(data):
        return await service.start(data)

    async def feed(data):
        return await service.feed(data)

    async def finish(data):
        return await service.finish(data)

    async def cancel(data):
        return await service.cancel(data)

    return gr.HTML(
        value="ready",
        max_seconds=max_seconds,
        html_template='''<div class="microphone-recorder">
          <div class="recorder-actions">
            <button type="button" class="record-start">Record my voice</button>
            <button type="button" class="record-stop" disabled>Finish recording</button>
          </div>
          <p class="recorder-status" role="status" aria-live="polite">Record a clear, short English sentence. Words appear while you speak.</p>
          <p class="recorder-preview" aria-live="polite"></p>
        </div>''',
        css_template='''.recorder-actions { display: flex; gap: 8px; }
          button { border: 1px solid var(--border-color-primary, #aaa); border-radius: 8px; padding: 8px 12px; cursor: pointer; background: var(--button-secondary-background-fill, #eee); color: var(--body-text-color, #111); }
          button:disabled { cursor: default; opacity: .55; }
          p { margin-top: 8px; } .recorder-preview { min-height: 1.5em; font-weight: 500; }''',
        js_on_load=(STATIC / "mic.js").read_text(),
        server_functions=[start, feed, finish, cancel],
    )


def consume_complete(evt: gr.EventData):
    """Only return paths/transcripts produced by this server, never event fields."""
    data = evt._data if isinstance(evt._data, dict) else {}
    if not data.get("token"):
        return None, "", "Recording failed. Check microphone access and try again."
    try:
        return _service.consume(data["token"])
    except ValueError:
        return gr.skip(), gr.skip(), gr.skip()
