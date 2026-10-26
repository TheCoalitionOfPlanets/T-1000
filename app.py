"""Local English speech and voice-cloning demo using the downloaded SOAR model."""

import os
import logging
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "dots.tts-soar"
OUTPUT_DIR = ROOT / "outputs"
EXAMPLE_AUDIO = ROOT / "examples" / "reference-en.wav"
EXAMPLE_TRANSCRIPT = "We asked over twenty different people, and they all said it was his."
DEVICE = "CUDA GPU" if torch.cuda.is_available() else "CPU"
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
    runtime = DotsTtsRuntime.from_pretrained(
        str(MODEL_PATH),
        precision="bfloat16" if torch.cuda.is_available() else "float32",
        optimize=False,
        max_generate_length=100 if DEVICE == "CPU" else 300,
        vocoder_merge_steps=1,
    )
    if runtime.device.type == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
    return runtime


def read_reference(path):
    """Reject unusable recordings before starting expensive inference."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("The reference recording is empty or contains invalid audio.")
    audio = audio.mean(axis=1)
    duration = audio.size / sample_rate
    if duration < 1:
        raise ValueError("Record at least one second of speech; 3–10 seconds works best.")
    if DEVICE == "CPU" and duration > 10:
        raise ValueError("Use a reference recording of 10 seconds or less for this CPU demo.")
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if peak < 0.0001 or rms < 0.00001:
        raise ValueError(
            "The reference recording is silent or too quiet. Check your microphone, "
            "record a clear sentence, or click Use example voice."
        )
    return audio, sample_rate


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
    text_limit = 150 if DEVICE == "CPU" else 500
    if len(text) > text_limit:
        raise gr.Error(f"Keep each test under {text_limit} characters.")
    if not reference_audio:
        raise gr.Error("Add a reference recording, or keep the preloaded example voice.")
    if not transcript:
        raise gr.Error("Add the exact words spoken in the reference recording.")
    try:
        reference, reference_rate = read_reference(reference_audio)
    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
        raise gr.Error(str(exc)) from exc

    # Freeze the input so a new recording cannot change a running request.
    request_dir = ROOT / ".cache" / "requests"
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
            with MODEL_LOCK:
                if cancel.is_set():
                    return
                from dots_tts.utils.util import seed_everything

                runtime = load_runtime()
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
    worker.start()
    try:
        yield gr.skip(), "Preparing your reference recording…", gr.update(interactive=False), gr.update(interactive=True)
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
                elif kind == "audio":
                    audio, sample_rate = payload
                    chunks.append(audio)
            except queue.Empty:
                pass
            duration = sum(chunk.size for chunk in chunks) / sample_rate
            if cancel.is_set():
                reason = "Time limit reached" if timed_out else "Stop requested"
                status = f"{reason}; finishing the current audio step… Elapsed {elapsed:.0f}s."
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


def build_app():
    with gr.Blocks(title="T-1000 · English Voice Lab") as demo:
        gr.Markdown("# English Voice Lab\nTest dots.tts SOAR locally. Enter a short sentence and press **Generate speech**.")
        if DEVICE == "CPU":
            gr.Markdown(f"Running on **CPU**. Synthesis can take several minutes. Start with one short sentence (up to 150 characters). Generation time limit: **{GENERATION_TIMEOUT / 60:g} minutes**.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text to speak",
                    value="Hello, this is a voice test.",
                    lines=4,
                )
                with gr.Accordion("Clone a voice", open=True):
                    gr.Markdown("An example English voice is preloaded. For your voice, record a clear 3–10 second sentence and enter exactly what you said. Changing the recording clears the old transcript.")
                    reference = gr.Audio(label="Reference voice", value=str(EXAMPLE_AUDIO) if EXAMPLE_AUDIO.exists() else None, sources=["upload", "microphone"], type="filepath")
                    transcript = gr.Textbox(label="Exact words in the recording", value=EXAMPLE_TRANSCRIPT if EXAMPLE_AUDIO.exists() else "", lines=2)
                    example = gr.Button("Use example voice")
                with gr.Accordion("Generation settings", open=False):
                    steps = gr.Slider(4, 32, value=10, step=1, label="Sampling steps", info="Higher values take longer. SOAR recommends 10–32.")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    generate = gr.Button("Generate speech", variant="primary")
                    stop = gr.Button("Stop", variant="stop", interactive=False)
            with gr.Column():
                sample = OUTPUT_DIR / "test-english.wav"
                output = gr.Audio(label="Generated speech", value=str(sample) if sample.exists() else None, type="filepath")
                status = gr.Textbox(label="Result", value="Ready. Use the example voice or add your own recording and transcript.", interactive=False)
        reference.input(lambda: "", outputs=[transcript], queue=False)
        example.click(lambda: (str(EXAMPLE_AUDIO), EXAMPLE_TRANSCRIPT), outputs=[reference, transcript], queue=False)
        generate.click(synthesize, [text, reference, transcript, steps, seed], [output, status, generate, stop], api_name="synthesize", concurrency_limit=1, show_progress="hidden")
        stop.click(stop_generation, queue=False, api_name="stop")
    return demo.queue(default_concurrency_limit=1, max_size=4)


if __name__ == "__main__":
    load_runtime()
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
