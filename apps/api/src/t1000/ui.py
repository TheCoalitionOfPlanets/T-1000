"""Optional Gradio interface for the local voice runtime."""

from . import tts as voice
from .mic import build_recorder

import gradio as gr


def build_app():
    with gr.Blocks(title="T-1000 · English Voice Lab") as demo:
        reference_state = gr.State(voice.ReferenceState())
        gr.Markdown(f"# English Voice Lab\nEnter a sentence or a few paragraphs (up to {voice.TEXT_LIMIT:,} characters), then press **Generate speech**.")
        if voice.DEVICE == "CPU":
            gr.Markdown(f"Running on **CPU**. Start with one short sentence; longer passages can take several minutes. Generation has a {voice.GENERATION_TIMEOUT / 60:g}-minute budget for one part and {voice.PARAGRAPH_TIMEOUT / 60:g} minutes for multiple parts. Stop saves any completed audio.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text to speak",
                    value="Hello, this is a voice test.",
                    lines=4,
                )
                with gr.Accordion("Clone a voice", open=True):
                    gr.Markdown(f"Record a clear 3–{min(10, voice.REFERENCE_LIMIT)} second English sentence. Your words appear while you speak and fill the transcript when you finish. You can also upload a recording. Review the transcript before generating speech.")
                    microphone = build_recorder(voice.REFERENCE_LIMIT)
                    reference = gr.Audio(label="Reference voice · upload or preview", value=str(voice.EXAMPLE_AUDIO) if voice.EXAMPLE_AUDIO.exists() else None, sources=["upload"], type="filepath", editable=False)
                    with gr.Accordion("Clean up and crop the reference", open=False):
                        gr.Markdown("Reduce steady background hiss and rumble, then listen to the result. Crop times refer to the original recording; each edit starts from that original. Keep at least one second of speech. Noise reduction does not isolate overlapping speakers or music.")
                        with gr.Row():
                            crop_start = gr.Number(label="Crop start · seconds in original", value=0, minimum=0)
                            crop_end = gr.Number(label="Crop end · seconds in original", value=None, minimum=0)
                        noise_reduction = gr.Checkbox(label="Reduce background noise", value=True)
                        auto_trim = gr.Checkbox(label="Trim quiet beginning and end automatically", value=False)
                        with gr.Row():
                            apply_edits = gr.Button("Apply cleanup and crop")
                            restore_original = gr.Button("Restore original recording")
                    transcript = gr.Textbox(label="Words in the recording · filled automatically, editable", value=voice.EXAMPLE_TRANSCRIPT if voice.EXAMPLE_AUDIO.exists() else "", lines=2)
                    asr_status = gr.Textbox(label="Transcription", value="Example voice ready. Record or upload your own voice for automatic transcription.", interactive=False)
                    example = gr.Button("Use example voice")
                with gr.Accordion("Generation settings", open=False):
                    steps = gr.Slider(4, 32, value=10, step=1, label="Sampling steps", info="Higher values take longer. SOAR recommends 10–32.")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    generate = gr.Button("Generate speech", variant="primary")
                    stop = gr.Button("Stop", variant="stop", interactive=False)
            with gr.Column():
                sample = voice.OUTPUT_DIR / "test-english.wav"
                output = gr.Audio(label="Generated speech", value=str(sample) if sample.exists() else None, type="filepath")
                status = gr.Textbox(label="Result", value="Ready. Use the example voice or record your own voice.", interactive=False)
        microphone.start_recording(voice.begin_recording, [reference_state], [transcript, asr_status, generate], queue=False)
        microphone.complete(voice.finish_recording, [reference_state], [reference, transcript, asr_status, generate, microphone], queue=False)
        reference.upload(voice.prepare_upload, [reference, reference_state], [transcript, asr_status, generate, microphone], queue=False).then(
            voice.transcribe_upload, [reference, reference_state], [transcript, asr_status, generate], concurrency_limit=1, concurrency_id="reference-transcription", show_progress="hidden"
        )
        reference.clear(voice.clear_reference, [reference_state], [transcript, asr_status, generate, microphone], queue=False)
        example.click(voice.restore_example, [reference_state], [reference, transcript, asr_status, generate, microphone], queue=False)
        edit_outputs = [reference, transcript, asr_status, generate, microphone]
        apply_edits.click(
            voice.edit_reference, [crop_start, crop_end, noise_reduction, auto_trim, reference_state], edit_outputs,
            concurrency_limit=1, concurrency_id="reference-transcription", show_progress="hidden",
        )
        restore_original.click(
            voice.restore_original_reference, [reference_state], edit_outputs,
            concurrency_limit=1, concurrency_id="reference-transcription", show_progress="hidden",
        )
        generate.click(voice.synthesize_ready, [text, reference, transcript, steps, seed, reference_state], [output, status, generate, stop], api_name="synthesize", concurrency_limit=1, show_progress="hidden")
        stop.click(voice.stop_generation, queue=False, api_name="stop")
    return demo.queue(default_concurrency_limit=1, max_size=4)


def main():
    voice.load_runtime()
    try:
        voice.stt.preload_cpu_model()
    except voice.stt.STTError as exc:
        voice.LOGGER.warning("Automatic transcription is unavailable: %s", exc)
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
