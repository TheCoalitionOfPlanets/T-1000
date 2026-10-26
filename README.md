# T-1000

For setup on another machine, start with [AGENT_HANDOFF.md](AGENT_HANDOFF.md).
It includes NVIDIA GPU installation, the pinned model download, verification,
and troubleshooting. See [CHANGELOG.md](CHANGELOG.md) for the current snapshot.

## Voice cloning model

- Model: [dots.tts SOAR](https://huggingface.co/dots-studio/dots.tts-soar)
- Upstream revision: `2f9b3e18d70d670d4c701da2dc55ded5755815ce`
- Local directory: `models/dots.tts-soar/`
- License: Apache 2.0 (as declared by upstream).
- Download size: approximately 5.2 GB, including the model, vocoder, speaker encoder, tokenizer, and configuration files.

Download or resume the pinned snapshot from this repository's root:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 \
  uvx --from huggingface-hub hf download dots-studio/dots.tts-soar \
  --revision 2f9b3e18d70d670d4c701da2dc55ded5755815ce \
  --local-dir models/dots.tts-soar
```

Model weights and download caches are excluded from Git. Downloading the model alone does not install its inference runtime.

See the [upstream runtime and fine-tuning instructions](https://github.com/studio-dots-ai/dots.tts) to run the checkpoint. Pass `models/dots.tts-soar` as the local model path. Voice cloning uses reference audio and its exact transcript.

## Test locally

### CPU setup

The local English demo uses the downloaded checkpoint and CPU PyTorch. Install its pinned dependencies once:

```sh
uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --torch-backend cpu --requirements requirements.txt
```

Start the interface from the repository root:

```sh
.venv/bin/python app.py
```

Open **http://127.0.0.1:7860**. An English reference recording and its transcript are preloaded. Enter a short English sentence and click **Generate speech**. To clone your own voice, replace the example with a clean 3–10 second reference recording and enter its exact transcript.

The server loads the model before opening the interface. CPU synthesis can take several minutes, so start with one short sentence. The CPU demo limits text to 150 characters and reference recordings to 10 seconds, with a bounded generation length. The model stays loaded for subsequent requests. Generated WAV files are saved in `outputs/`, which is excluded from Git. This cloning interface requires a reference recording; an example voice is included to start.

The Result field updates with elapsed time and the amount of audio generated. **Stop** cancels generation after the current audio step; long requests also stop at a ten-minute budget, with any partial audio saved. Cancellation can take a few seconds while the current CPU step finishes. Silent or invalid recordings are rejected before inference. Changing the recording clears the previous transcript, so enter the actual words in the new recording. **Use example voice** restores the included recording and matching transcript.

The runtime is pinned to upstream commit `b65ca7149fb43b8fb8d52916aa12738deda7b188`. `constraints.txt` follows upstream's recommended versions, using published Gradio `6.17.3` in place of the unavailable `6.17.0` pin.

### Verified sample

`outputs/test-english.wav` was generated locally from “Hello, this is a voice test.” using a [public English reference clip](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav). The output is 1.92 seconds of 48 kHz audio. The first request took about 94 seconds on this machine's Intel Core Ultra 9 CPU, including model loading. Generated files are local artifacts and are not committed to Git.
