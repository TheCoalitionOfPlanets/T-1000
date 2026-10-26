"""Same-origin API and static host for the ElevenLabs UI voice lab.

Only server-owned audio IDs cross the API boundary. The existing inference and
bounded PCM recorder remain the source of truth for audio and cancellation.
"""

import json
import logging
import threading
from collections import OrderedDict
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
from .mic import RecorderService
from .paths import CACHE, WEB


LOGGER = logging.getLogger(__name__)
UPLOAD_DIR = CACHE / "ui-references"
FRONTEND_DIR = WEB
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 40 * 1024 * 1024
MAX_AUDIO_ENTRIES = 256
TEXT_LIMIT = 150 if voice.DEVICE == "CPU" else 500
recorder = RecorderService(voice.REFERENCE_LIMIT)
generation_lock = threading.Lock()
audio_lock = threading.Lock()
audio_files = OrderedDict()


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


class GenerateRequest(Payload):
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)
    reference_id: str = Field(min_length=1, max_length=64)
    transcript: str = Field(min_length=1, max_length=8000)
    steps: int = Field(default=10, strict=True, ge=4, le=32)
    seed: int = Field(default=42, strict=True, ge=0, le=4294967295)


def _register(path, transcript="", status=""):
    audio_id = uuid4().hex
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
        "reference_limit": voice.REFERENCE_LIMIT,
        "text_limit": TEXT_LIMIT,
        "example": example,
        "sample_url": "/api/audio/sample" if (voice.OUTPUT_DIR / "test-english.wav").is_file() else None,
    }


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


@api.post("/api/recordings/finish")
async def recording_finish(payload: RecordingPCM):
    result = await _recording_action(recorder.finish, payload)
    if result.get("cancelled"):
        return result
    try:
        path, transcript, status = await run_in_threadpool(recorder.consume, payload.token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        await run_in_threadpool(_validate_reference, path)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return _register(path, transcript, status)


@api.post("/api/recordings/cancel")
async def recording_cancel(payload: RecordingToken):
    return await _recording_action(recorder.cancel, payload)


@api.post("/api/stop")
def stop():
    voice.stop_generation()
    return {"status": "Stop requested; finishing the current audio step."}


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


def _event(update):
    output, status, _, stop_button = update
    event = {"status": status, "done": not stop_button.get("interactive", False)}
    if isinstance(output, str):
        output_path = Path(output).resolve()
        if not output_path.is_relative_to(voice.OUTPUT_DIR.resolve()) or not output_path.is_file():
            raise RuntimeError("Generated audio is unavailable.")
        event["audio_url"] = _register(output_path)["url"]
    return json.dumps(event) + "\n"


@api.post("/api/generate")
async def generate(payload: GenerateRequest):
    if not payload.text.strip() or not payload.transcript.strip():
        raise HTTPException(400, "Enter text to speak and the reference recording's exact words.")
    selected = _lookup(payload.reference_id)
    await run_in_threadpool(_validate_reference, selected.path)
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(409, "Speech generation is already running. Wait or press Stop.")
    source = voice.synthesize(payload.text, str(selected.path), payload.transcript, payload.steps, payload.seed)
    generation = _Generation(source)
    try:
        first = await run_in_threadpool(generation.next)
    except BaseException as exc:
        generation.close()
        if isinstance(exc, Exception):
            raise HTTPException(400, str(exc)) from exc
        raise

    async def events():
        try:
            update = first
            while update is not None:
                yield _event(update)
                update = await run_in_threadpool(generation.next)
        except Exception as exc:
            LOGGER.exception("Speech generation stream failed")
            yield json.dumps({"status": f"Generation failed: {exc}", "error": str(exc), "done": True}) + "\n"
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

    uvicorn.run(api, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
