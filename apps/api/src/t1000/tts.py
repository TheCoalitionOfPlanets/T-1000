"""Local English speech and voice-cloning demo using the downloaded SOAR model."""

import os
import logging
import hashlib
import queue
import threading
import time
from functools import lru_cache
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from . import stt
from .inference_settings import get_inference_settings
from .eager import configure_eager_cache, read_eager_cache_enabled
from .references import configure_prompt_audio
from .mic import consume_complete
from .paths import ASSETS, CACHE, MODELS, OUTPUT

MODEL_PATH = MODELS / "dots.tts-soar"
OUTPUT_DIR = OUTPUT
EXAMPLE_AUDIO = ASSETS / "ref.wav"
EXAMPLE_TRANSCRIPT = "We asked over twenty different people, and they all said it was his."
DEVICE = "CUDA GPU" if torch.cuda.is_available() else "CPU"
INFERENCE_SETTINGS = get_inference_settings()
REFERENCE_LIMIT = INFERENCE_SETTINGS.reference_limit
TEXT_LIMIT = INFERENCE_SETTINGS.text_limit
GENERATION_TIMEOUT = 600
MODEL_LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE_STOP = None
LOGGER = logging.getLogger(__name__)
if DEVICE == "CPU":
    torch.set_num_threads(min(6, os.cpu_count() or 1))


@lru_cache(maxsize=1)
def load_runtime():
    from dots_tts.runtime import DotsTtsRuntime

    if not (MODEL_PATH / "model.safetensors").is_file():
        raise RuntimeError("Model weights are missing. Follow the README download instructions.")
    settings = get_inference_settings()
    eager_cache = read_eager_cache_enabled()
    if settings.profile == "low-vram":
        from .low_vram import load_low_vram_runtime

        LOGGER.info("Loading experimental low-VRAM SOAR: NF4 language backbone, eager acoustic decoding.")
        runtime = load_low_vram_runtime(MODEL_PATH, precision=settings.precision)
        configure_prompt_audio(runtime)
        return configure_eager_cache(runtime, eager_cache)
    if settings.profile == "offload":
        from .offload import load_offloaded_runtime

        LOGGER.info("Loading experimental SOAR CPU offload with %s GPU-resident language blocks.", settings.gpu_layers)
        runtime = load_offloaded_runtime(MODEL_PATH, precision=settings.precision, gpu_layers=settings.gpu_layers)
        configure_prompt_audio(runtime)
        return configure_eager_cache(runtime, eager_cache)
    optimize = settings.optimize
    if optimize:
        LOGGER.info(
            "Loading SOAR with optimized CUDA inference. Initial compilation "
            "and warmup can take several minutes; later requests reuse the runtime."
        )
    runtime = DotsTtsRuntime.from_pretrained(
        str(MODEL_PATH),
        precision=settings.precision,
        optimize=optimize,
        max_generate_length=settings.max_generate_length,
        vocoder_merge_steps=1,
        warmup_on_optimize=False,
    )
    if optimize:
        # Match generate_stream's context so torch.compile can reuse the warmed
        # graphs instead of specializing again for inference-mode tensors.
        with torch.inference_mode():
            runtime.run_warmup()
    if runtime.device.type == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
    configure_prompt_audio(runtime)
    return configure_eager_cache(runtime, eager_cache)


def read_reference(path):
    """Reject unusable recordings before starting expensive inference."""
    info = sf.info(path)
    if info.duration > REFERENCE_LIMIT:
        raise ValueError(f"Use a reference recording of {REFERENCE_LIMIT} seconds or less.")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("The reference recording is empty or contains invalid audio.")
    audio = audio.mean(axis=1)
    duration = audio.size / sample_rate
    if duration < 1:
        raise ValueError(f"Record at least one second of speech; 3–{min(10, REFERENCE_LIMIT)} seconds works best.")
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if peak < 0.0001 or rms < 0.00001:
        raise ValueError(
            "The reference recording is silent or too quiet. Check your microphone, "
            "record a clear sentence, or click Use example voice."
        )
    return audio, sample_rate


class ReferenceState:
    """Invalidate late transcription results when the user replaces a reference."""

    def __init__(self):
        self.path = str(EXAMPLE_AUDIO) if EXAMPLE_AUDIO.exists() else None
        self.ready = EXAMPLE_AUDIO.exists()
        self.version = 0
        self.recording_token = None
        self.lock = threading.Lock()

    def __deepcopy__(self, memo):
        return ReferenceState()

    def replace(self, path=None, ready=False, recording_token=None):
        self.version += 1
        self.path = path
        self.ready = ready
        self.recording_token = recording_token


def prepare_upload(path, state):
    with state.lock:
        state.replace(path=path)
    return "", "Transcribing the reference recording…", gr.update(interactive=False), uuid4().hex


def transcribe_upload(path, state):
    with state.lock:
        version = state.version
        if path != state.path:
            return (gr.skip(),) * 3
    valid = False
    try:
        read_reference(path)
        valid = True
        words = stt.transcribe_file(path, max_seconds=REFERENCE_LIMIT).strip()
        message = (
            "Recording transcribed. Check the words before generating speech."
            if words else "No speech was recognized. Check the audio or enter its exact words below."
        )
    except Exception as exc:
        LOGGER.warning("Reference transcription failed: %s", exc)
        words = ""
        message = f"Automatic transcription failed: {exc}"
        if valid:
            message += " You can enter the exact words below."
    with state.lock:
        if version != state.version or path != state.path:
            return (gr.skip(),) * 3
        state.ready = valid
    return words, message, gr.update(interactive=valid)


def begin_recording(state, evt: gr.EventData):
    with state.lock:
        state.replace(recording_token=evt._data.get("token"))
    return "", "Recording… Your words appear beside the microphone controls.", gr.update(interactive=False)


def finish_recording(state, evt: gr.EventData):
    token = evt._data.get("token")
    with state.lock:
        if not token or token != state.recording_token:
            return (gr.skip(),) * 5
        if evt._data.get("error"):
            state.replace()
            return None, "", "Recording could not be saved. Try recording again or upload a clip.", gr.update(interactive=False), uuid4().hex
        version = state.version
    path, words, message = consume_complete(evt)
    if not isinstance(path, str):
        return (gr.skip(),) * 5
    valid = True
    try:
        read_reference(path)
    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
        valid = False
        words, message = "", str(exc)
    with state.lock:
        if version != state.version or token != state.recording_token:
            return (gr.skip(),) * 5
        state.replace(path=path, ready=valid)
    return path, words, message, gr.update(interactive=valid), uuid4().hex


def clear_reference(state):
    with state.lock:
        state.replace()
    return "", "Record or upload a reference voice.", gr.update(interactive=False), uuid4().hex


def restore_example(state):
    with state.lock:
        state.replace(path=str(EXAMPLE_AUDIO), ready=True)
    return str(EXAMPLE_AUDIO), EXAMPLE_TRANSCRIPT, "Example voice ready.", gr.update(interactive=True), uuid4().hex


def synthesize_ready(text, reference_audio, transcript, steps, seed, state):
    with state.lock:
        path, version = state.path, state.version
        if not state.ready or not path or not reference_audio:
            raise gr.Error("Finish recording and wait for its transcript before generating speech.")
    # Gradio serves copies of output audio from its cache. Validate their contents
    # against the selected reference, then freeze that server-owned source.
    if reference_audio != path:
        try:
            with open(reference_audio, "rb") as supplied, open(path, "rb") as selected:
                matches = hashlib.file_digest(supplied, "sha256").digest() == hashlib.file_digest(selected, "sha256").digest()
        except OSError:
            matches = False
        if not matches:
            raise gr.Error("The reference changed. Wait for the new recording's transcript.")
    with state.lock:
        if version != state.version or not state.ready:
            raise gr.Error("The reference changed. Wait for the new recording's transcript.")
    for update in synthesize(text, path, transcript, steps, seed):
        with state.lock:
            if not state.ready:
                update = (*update[:2], gr.update(interactive=False), update[3])
        yield update


def stop_generation():
    with ACTIVE_LOCK:
        if ACTIVE_STOP is not None:
            ACTIVE_STOP.set()


def synthesize(text, reference_audio, transcript, steps, seed):
    global ACTIVE_STOP

    text = (text or "").strip()
    transcript = (transcript or "").strip()
    if not text:
        raise gr.Error("Enter the text you want to hear.")
    if len(text) > TEXT_LIMIT:
        raise gr.Error(f"Keep each test under {TEXT_LIMIT} characters.")
    if not reference_audio:
        raise gr.Error("Add a reference recording, or keep the preloaded example voice.")
    if not transcript:
        raise gr.Error("Add the exact words spoken in the reference recording.")
    try:
        reference, reference_rate = read_reference(reference_audio)
    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
        raise gr.Error(str(exc)) from exc

    # Freeze the input so a new recording cannot change a running request.
    request_dir = CACHE / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    reference_path = request_dir / f"reference-{uuid4().hex}.wav"
    sf.write(reference_path, reference, reference_rate, subtype="PCM_16")
    messages = queue.Queue()
    cancel = threading.Event()
    with ACTIVE_LOCK:
        ACTIVE_STOP = cancel
    started = time.perf_counter()

    def run():
        global ACTIVE_STOP
        try:
            with MODEL_LOCK, stt.gpu_compute_lock():
                if cancel.is_set():
                    return
                stt.release_gpu_model()
                from dots_tts.utils.util import seed_everything

                runtime = load_runtime()
                if cancel.is_set():
                    return
                messages.put(("ready", None))
                if runtime.device.type == "cpu":
                    torch.set_num_threads(min(6, os.cpu_count() or 1))
                seed_everything(int(seed))
                with torch.inference_mode():
                    stream = runtime.generate_stream(
                        text=text,
                        prompt_audio_path=str(reference_path),
                        prompt_text=transcript,
                        language="en",
                        num_steps=int(steps),
                        guidance_scale=1.2,
                    )
                    try:
                        for chunk in stream:
                            if cancel.is_set():
                                break
                            audio = chunk.detach().float().cpu().reshape(-1).numpy().copy()
                            if not np.isfinite(audio).all():
                                raise RuntimeError("The model returned invalid audio.")
                            if audio.size:
                                messages.put(("audio", (audio, runtime.sample_rate)))
                    finally:
                        stream.close()
        except Exception as exc:
            LOGGER.exception("Speech generation failed")
            messages.put(("error", str(exc)))
        finally:
            reference_path.unlink(missing_ok=True)
            with ACTIVE_LOCK:
                if ACTIVE_STOP is cancel:
                    ACTIVE_STOP = None
            messages.put(("done", None))

    worker = threading.Thread(target=run, name="speech-generation", daemon=True)
    chunks = []
    sample_rate = 48000
    error = None
    timed_out = False
    runtime_ready = False
    worker.start()
    try:
        yield gr.skip(), "Loading the speech model and preparing inference…", gr.update(interactive=False), gr.update(interactive=True)
        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= GENERATION_TIMEOUT:
                timed_out = True
                cancel.set()
            try:
                kind, payload = messages.get(timeout=0.5)
                if kind == "done":
                    break
                if kind == "error":
                    error = payload
                elif kind == "ready":
                    runtime_ready = True
                elif kind == "audio":
                    audio, sample_rate = payload
                    chunks.append(audio)
            except queue.Empty:
                pass
            duration = sum(chunk.size for chunk in chunks) / sample_rate
            if cancel.is_set():
                reason = "Time limit reached" if timed_out else "Stop requested"
                status = f"{reason}; finishing the current audio step… Elapsed {elapsed:.0f}s."
            elif not runtime_ready:
                status = f"Loading and warming the speech model · elapsed {elapsed:.0f}s."
                if DEVICE != "CPU":
                    status += " First GPU use can take several minutes."
            else:
                status = f"Generating on {DEVICE} · elapsed {elapsed:.0f}s · produced {duration:.2f}s of audio."
            yield gr.skip(), status, gr.update(interactive=False), gr.update(interactive=True)

        elapsed = time.perf_counter() - started
        path = None
        duration = 0
        if chunks:
            audio = np.concatenate(chunks)
            duration = audio.size / sample_rate
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / f"speech-{uuid4().hex[:12]}.wav"
            sf.write(path, audio, sample_rate, subtype="PCM_16")
        if error:
            status = f"Generation failed: {error}"
        elif cancel.is_set():
            reason = f"Stopped at the {GENERATION_TIMEOUT / 60:g}-minute time limit" if timed_out else "Stopped"
            status = f"{reason} after {elapsed:.1f}s."
            if path:
                status += f" Saved {duration:.2f}s of partial audio. Check that the reference transcript matches the recording."
        elif path:
            status = f"Generated {duration:.2f}s of audio in {elapsed:.1f}s on {DEVICE}."
        else:
            status = "No audio was produced. Try another reference recording."
        yield str(path) if path else gr.skip(), status, gr.update(interactive=True), gr.update(interactive=False)
    finally:
        # Also stop computation when the browser/client closes the generator.
        cancel.set()
