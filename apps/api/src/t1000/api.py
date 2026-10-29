"""Same-origin API and static host for the ElevenLabs UI voice lab.

Only server-owned audio IDs cross the API boundary. The existing inference and
bounded PCM recorder remain the source of truth for audio and cancellation.
"""

import json
import re
import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import anyio
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from . import tts as voice
from .text import expand_english_text
from .mic import RecorderService
from .jobs import GenerationJobs, JobBusy, JobConflict, JobNotFound
from .paths import CACHE, WEB


LOGGER = logging.getLogger(__name__)
UPLOAD_DIR = CACHE / "ui-references"
FRONTEND_DIR = WEB
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 40 * 1024 * 1024
MAX_AUDIO_ENTRIES = 256
TEXT_LIMIT = voice.TEXT_LIMIT
recorder = RecorderService(voice.REFERENCE_LIMIT)
generation_lock = threading.Lock()
audio_lock = threading.Lock()
audio_files = OrderedDict()
generation_jobs = GenerationJobs(shared_lock=generation_lock, disconnect_grace=60)
finished_references = OrderedDict()
finish_locks = {}
FINISH_CACHE_LIMIT = 64
FINISH_CACHE_TTL = 900


@dataclass(frozen=True)
class AudioFile:
    path: Path
    transcript: str = ""
    status: str = ""


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordingStart(Payload):
    sample_rate: int = Field(strict=True, ge=8000, le=192000)


class RecordingToken(Payload):
    token: str = Field(min_length=20, max_length=100)


class RecordingPCM(RecordingToken):
    pcm: str = Field(max_length=16 * 1024 * 1024)


class RecordingFeed(RecordingPCM):
    sequence: int = Field(strict=True, ge=0)


class StopRequest(Payload):
    generation_id: str | None = Field(default=None, min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class GenerateRequest(Payload):
    generation_id: str | None = Field(default=None, min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    after_sequence: int = Field(default=0, strict=True, ge=0)
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)
    reference_id: str = Field(min_length=1, max_length=64)
    transcript: str = Field(min_length=1, max_length=8000)
    steps: int = Field(default=10, strict=True, ge=4, le=32)
    seed: int = Field(default=42, strict=True, ge=0, le=4294967295)


class NormalizeTextRequest(Payload):
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)


def _register(path, transcript="", status=""):
    match = re.fullmatch(r"speech-([a-f0-9]{32})\.wav", Path(path).name)
    generated = not transcript and not status and match and Path(path).resolve().is_relative_to(voice.OUTPUT_DIR.resolve())
    audio_id = f"g-{match.group(1)}" if generated else uuid4().hex
    with audio_lock:
        audio_files[audio_id] = AudioFile(Path(path), transcript, status)
        while len(audio_files) > MAX_AUDIO_ENTRIES:
            audio_files.popitem(last=False)
    return {"id": audio_id, "url": f"/api/audio/{audio_id}", "transcript": transcript, "status": status}


def _lookup(audio_id):
    if audio_id == "example" and voice.EXAMPLE_AUDIO.is_file():
        return AudioFile(voice.EXAMPLE_AUDIO, voice.EXAMPLE_TRANSCRIPT, "Example voice ready.")
    if audio_id == "sample" and (voice.OUTPUT_DIR / "test-english.wav").is_file():
        return AudioFile(voice.OUTPUT_DIR / "test-english.wav")
    with audio_lock:
        audio = audio_files.get(audio_id)
    if audio is None and re.fullmatch(r"g-[a-f0-9]{32}", audio_id):
        candidate = voice.OUTPUT_DIR / f"speech-{audio_id[2:]}.wav"
        if candidate.resolve().is_relative_to(voice.OUTPUT_DIR.resolve()) and candidate.is_file():
            audio = AudioFile(candidate)
    if audio is None or not audio.path.is_file():
        raise HTTPException(404, "This audio is unavailable. Select or record the voice again.")
    return audio


def _validate_reference(path):
    try:
        return voice.read_reference(str(path))
    except (ValueError, RuntimeError, OSError, sf.LibsndfileError) as exc:
        raise HTTPException(400, str(exc)) from exc


api = FastAPI(title="T-1000 Voice Lab", docs_url=None, redoc_url=None)
app = api


@api.get("/api/health")
def health():
    return {"status": "ok", "device": voice.DEVICE, "speech_model_loaded": bool(voice.load_runtime.cache_info().currsize)}


@api.middleware("http")
async def request_size_limit(request: Request, call_next):
    try:
        size = int(request.headers.get("content-length", "0"))
    except ValueError:
        return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
    if size < 0 or size > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "The request is too large."}, status_code=413)
    return await call_next(request)


@api.get("/api/config")
def config():
    example = None
    if voice.EXAMPLE_AUDIO.is_file():
        example = {"id": "example", "url": "/api/audio/example", "transcript": voice.EXAMPLE_TRANSCRIPT, "status": "Example voice ready."}
    return {
        "device": voice.DEVICE,
        "resumable_generation": True,
        "reference_limit": voice.REFERENCE_LIMIT,
        "text_limit": TEXT_LIMIT,
        "segment_text_limit": voice.SEGMENT_TEXT_LIMIT,
        "example": example,
        "sample_url": "/api/audio/sample" if (voice.OUTPUT_DIR / "test-english.wav").is_file() else None,
    }


@api.post("/api/text/normalize")
async def normalize_text(payload: NormalizeTextRequest):
    try:
        normalized = await run_in_threadpool(expand_english_text, payload.text, max_length=TEXT_LIMIT)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("English text expansion failed: %s", exc)
        raise HTTPException(503, "Text expansion is unavailable. Keep or edit your original text.") from exc
    return {"text": normalized}


@api.get("/api/audio/{audio_id}")
def audio(audio_id: str):
    selected = _lookup(audio_id)
    return FileResponse(selected.path, media_type="audio/wav", filename=f"t1000-{audio_id}.wav")


@api.post("/api/reference")
async def reference(file: UploadFile = File(...)):
    """Normalize a bounded upload, validate it, then attempt transcription."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"upload-{uuid4().hex}"
    normalized_path = UPLOAD_DIR / f"reference-{uuid4().hex}.wav"
    try:
        size = 0
        with upload_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Upload an audio file smaller than 32 MB.")
                target.write(chunk)
        samples, sample_rate = await run_in_threadpool(_validate_reference, upload_path)
        await run_in_threadpool(sf.write, normalized_path, samples, sample_rate, subtype="PCM_16")
        try:
            transcript = (await run_in_threadpool(voice.stt.transcribe_file, str(normalized_path), max_seconds=voice.REFERENCE_LIMIT)).strip()
            status = "Recording transcribed. Check the words before generating speech." if transcript else "No speech was recognized. Enter the exact words in the recording."
        except Exception as exc:
            LOGGER.warning("Reference transcription failed: %s", exc)
            transcript = ""
            status = f"Automatic transcription is unavailable: {exc}. Enter the exact words in the recording."
        return _register(normalized_path, transcript, status)
    except BaseException:
        normalized_path.unlink(missing_ok=True)
        raise
    finally:
        upload_path.unlink(missing_ok=True)
        await file.close()


async def _recording_action(action, payload):
    try:
        return await action(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.post("/api/recordings/start")
async def recording_start(payload: RecordingStart):
    return await _recording_action(recorder.start, payload)


@api.post("/api/recordings/feed")
async def recording_feed(payload: RecordingFeed):
    return await _recording_action(recorder.feed, payload)


@dataclass
class _RecordingLease:
    lock: asyncio.Lock
    users: int = 0


def _forget_recording_lease(token):
    lease = finish_locks.get(token)
    if lease is not None and lease.users == 0 and token not in finished_references:
        finish_locks.pop(token, None)


def _prune_finished_references():
    now = time.monotonic()
    for token, cached in list(finished_references.items()):
        if now - cached[0] > FINISH_CACHE_TTL:
            finished_references.pop(token, None)
            _forget_recording_lease(token)
    while len(finished_references) > FINISH_CACHE_LIMIT:
        token, _ = finished_references.popitem(last=False)
        _forget_recording_lease(token)


@asynccontextmanager
async def _recording_lease(token):
    # Count both the holder and every waiter BEFORE the first await. A released
    # asyncio.Lock can be unlocked while a waiter is waking; locked() alone
    # cannot decide whether this token's serialization object is still in use.
    lease = finish_locks.get(token)
    if lease is None:
        lease = _RecordingLease(asyncio.Lock())
        finish_locks[token] = lease
    lease.users += 1
    try:
        async with lease.lock:
            yield
    finally:
        lease.users -= 1
        if (lease.users == 0 and token not in finished_references
                and finish_locks.get(token) is lease):
            finish_locks.pop(token, None)


@api.post("/api/recordings/finish")
async def recording_finish(payload: RecordingPCM):
    # A successful HTTP response may be lost. Serialize token completion and
    # retain the SAME published reference for an identical full-PCM retry.
    _prune_finished_references()
    digest = hashlib.sha256(payload.pcm.encode()).hexdigest()
    async with _recording_lease(payload.token):
        with anyio.CancelScope(shield=True):
            cached = finished_references.get(payload.token)
            if cached is not None:
                _, original_digest, published, path = cached
                if original_digest != digest:
                    raise HTTPException(400, "The completed recording cannot be changed on retry.")
                if not Path(path).is_file():
                    raise HTTPException(410, "The saved recording is unavailable. Upload your local copy.")
                with audio_lock:
                    audio_files[published["id"]] = AudioFile(Path(path), published["transcript"], published["status"])
                    while len(audio_files) > MAX_AUDIO_ENTRIES:
                        audio_files.popitem(last=False)
                return published
            result = await _recording_action(recorder.finish, payload)
            if result.get("cancelled"):
                return result
            try:
                path, transcript, status = await run_in_threadpool(recorder.consume, payload.token, retain=True)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            try:
                await run_in_threadpool(_validate_reference, path)
            except Exception:
                await _recording_action(recorder.cancel, RecordingToken(token=payload.token))
                Path(path).unlink(missing_ok=True)
                raise
            published = _register(path, transcript, status)
            finished_references[payload.token] = (time.monotonic(), digest, published, path)
            _prune_finished_references()
            return published


@api.post("/api/recordings/cancel")
async def recording_cancel(payload: RecordingToken):
    async with _recording_lease(payload.token):
        finished_references.pop(payload.token, None)
        result = await _recording_action(recorder.cancel, payload)
    return result


@api.post("/api/stop")
def stop(payload: StopRequest | None = None):
    if payload and payload.generation_id:
        try:
            generation_jobs.stop(payload.generation_id)
        except JobNotFound as exc:
            raise HTTPException(404, "This generation is unavailable.") from exc
    else:
        voice.stop_generation()
    return {"status": "Stop requested; finishing the current audio step."}


@api.get("/api/generations/{generation_id}")
def generation_state(generation_id: str):
    try:
        job = generation_jobs.get(generation_id)
    except JobNotFound as exc:
        raise HTTPException(404, "This generation is unavailable.") from exc
    return job.read(cursor=0, timeout=0) or {"generation_id": generation_id, "sequence": 0, "done": False, "status": "Preparing generation…"}


class _Generation:
    """Release a generation slot even when a response disconnects early."""

    def __init__(self, source):
        self.source = source
        self.closed = False
        self.lock = threading.Lock()

    def next(self):
        # Closing a disconnected response must wait for an in-flight next().
        with self.lock:
            return None if self.closed else _next(self.source)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.source.close()
            finally:
                generation_lock.release()


class _GenerationResponse(StreamingResponse):
    def __init__(self, content, generation):
        super().__init__(content, media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.generation = generation

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # ASGI 2.4 may raise at send() while the body iterator is suspended
            # at yield. Its generator finally and a background task would not
            # run reliably there, so explicitly close both under cancellation.
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await run_in_threadpool(self.generation.close)


def _next(source):
    # StopIteration cannot propagate through an asyncio Future.
    try:
        return next(source)
    except StopIteration:
        return None


def _event(update, *, started, timings):
    output, status, _, stop_button = update
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    done = not stop_button.get("interactive", False)
    event = {"status": status, "done": done, "elapsed_ms": elapsed_ms,
             "timings": {**timings, **({"completion_ms": elapsed_ms} if done else {})}}
    if done and status.startswith("Generation failed"):
        event["error"] = status
    if done and status.startswith("Stopped"):
        event["cancelled"] = True
    if isinstance(output, str):
        output_path = Path(output).resolve()
        if not output_path.is_relative_to(voice.OUTPUT_DIR.resolve()) or not output_path.is_file():
            raise RuntimeError("Generated audio is unavailable.")
        event["audio_url"] = _register(output_path)["url"]
        metadata = getattr(output, "metadata", None)
        if metadata is not None:
            event["audio"] = metadata
            event["partial"] = not metadata["complete"]
    return json.dumps(event) + "\n"


class _JobResponse(StreamingResponse):
    def __init__(self, content, job, subscription):
        super().__init__(content, media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.job, self.subscription = job, subscription

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    self.job.unsubscribe(self.subscription)


def _job_response(job, after_sequence):
    try:
        subscription = job.subscribe()
    except JobBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    async def events():
        cursor = after_sequence
        while True:
            event = await run_in_threadpool(job.read, cursor=cursor, timeout=0.5)
            if event is not None:
                cursor = event["sequence"]
                yield json.dumps(event) + "\n"
                if event.get("done"):
                    break
            elif job.done.is_set():
                # A terminal snapshot remains retrievable even if a client
                # reconnects after seeing its sequence but losing the response.
                terminal = job.read(cursor=0, timeout=0)
                if terminal is not None:
                    yield json.dumps(terminal) + "\n"
                break
    return _JobResponse(events(), job, subscription)


@api.post("/api/generate")
async def generate(payload: GenerateRequest):
    # All server milestones share one monotonic origin. These describe API
    # handling; first status is not the first PCM chunk or browser playback.
    started = time.perf_counter()
    if not payload.text.strip() or not payload.transcript.strip():
        raise HTTPException(400, "Enter text to speak and the reference recording's exact words.")
    if payload.generation_id:
        fingerprint = hashlib.sha256(json.dumps(payload.model_dump(exclude={"generation_id", "after_sequence"}), sort_keys=True).encode()).hexdigest()
        try:
            existing = generation_jobs.get(payload.generation_id)
        except JobNotFound:
            if payload.after_sequence:
                raise HTTPException(410, "This generation expired. Start a new generation explicitly.")
            existing = None
        if existing is not None:
            if existing.payload_digest != fingerprint:
                raise HTTPException(409, "A generation ID cannot be reused with different inputs.")
            return _job_response(existing, payload.after_sequence)
    elif payload.after_sequence:
        raise HTTPException(400, "Resuming requires a generation ID.")
    selected = _lookup(payload.reference_id)
    await run_in_threadpool(_validate_reference, selected.path)
    timings = {"reference_validated_ms": round((time.perf_counter() - started) * 1000, 3)}
    if payload.generation_id:
        def serialize(update):
            timings.setdefault("first_status_ms", round((time.perf_counter() - started) * 1000, 3))
            return json.loads(_event(update, started=started, timings=timings))
        try:
            job = generation_jobs.submit(payload.generation_id, fingerprint,
                lambda cancel: voice.synthesize(payload.text, str(selected.path), payload.transcript, payload.steps, payload.seed, cancel_event=cancel), serialize)
        except JobBusy as exc:
            raise HTTPException(409, "Speech generation is already running. Wait or press Stop.") from exc
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return _job_response(job, payload.after_sequence)
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(409, "Speech generation is already running. Wait or press Stop.")
    source = voice.synthesize(payload.text, str(selected.path), payload.transcript, payload.steps, payload.seed)
    generation = _Generation(source)
    try:
        first = await run_in_threadpool(generation.next)
        timings["first_status_ms"] = round((time.perf_counter() - started) * 1000, 3)
    except BaseException as exc:
        generation.close()
        if isinstance(exc, Exception):
            raise HTTPException(400, str(exc)) from exc
        raise

    async def events():
        try:
            update = first
            while update is not None:
                yield _event(update, started=started, timings=timings)
                update = await run_in_threadpool(generation.next)
        except Exception as exc:
            LOGGER.exception("Speech generation stream failed")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            yield json.dumps({"status": f"Generation failed: {exc}", "error": str(exc), "done": True,
                              "elapsed_ms": elapsed_ms, "timings": {**timings, "completion_ms": elapsed_ms}}) + "\n"
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(generation.close)

    return _GenerationResponse(events(), generation)


if FRONTEND_DIR.is_dir():
    api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @api.get("/")
    def missing_frontend():
        return JSONResponse({"detail": "Build the interface first: npm ci && npm run build"}, status_code=503)


def main():
    import uvicorn

    preload = os.environ.get("T1000_PRELOAD_MODELS", "0").strip()
    if preload not in {"0", "1"}:
        raise ValueError("T1000_PRELOAD_MODELS must be 0 or 1.")
    if preload == "1":
        LOGGER.info("Loading speech models before opening the interface.")
        voice.load_runtime()
        try:
            voice.stt.preload_cpu_model()
        except voice.stt.STTError as exc:
            LOGGER.warning("Automatic transcription is unavailable: %s", exc)
    uvicorn.run(api, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
