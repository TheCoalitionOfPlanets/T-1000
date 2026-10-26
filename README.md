# T-1000

For setup on another machine, start with [the setup guide](docs/setup.md).
It includes NVIDIA GPU installation, the pinned model download, verification,
and troubleshooting. See [CHANGELOG.md](CHANGELOG.md) for the current snapshot.

## Layout

```text
apps/
  api/
    src/t1000/     Python API, synthesis, recognition, and microphone transport
    tests/         Python regression tests
  web/             React app and browser tests
assets/            Bundled reference audio and attribution
data/
  models/          Local model weights; ignored by Git
  output/          Generated audio; ignored by Git
docs/              Setup guide and research PDFs
requirements/      Shared pins and CPU/CUDA dependency profiles
scripts/           Model download tools
```

Run setup and workspace commands from the repository root. The npm workspace has
one root lockfile; the Python app uses an editable `src` package. Keep directory
names lowercase, Python modules in `snake_case`, React components in `PascalCase`,
and document names in `kebab-case`. Standard names such as `README.md`,
`CHANGELOG.md`, and license files retain their conventional spelling.

Runtime paths are defined in `t1000.paths`. Set `T1000_ROOT` to the checkout path
when installing the Python package outside this repository. Model weights and
generated audio belong under `data/`; caches and the virtual environment remain
in `.cache/` and `.venv/`.

Research: [training datasets](docs/datasets.pdf) and
[voice conversion and cloning](docs/voice-cloning.pdf).

## Voice cloning model

- Model: [dots.tts SOAR](https://huggingface.co/dots-studio/dots.tts-soar)
- Upstream revision: `2f9b3e18d70d670d4c701da2dc55ded5755815ce`
- Local directory: `data/models/dots.tts-soar/`
- License: Apache 2.0 (as declared by upstream).
- Download size: approximately 5.2 GB, including the model, vocoder, speaker encoder, tokenizer, and configuration files.

Download or resume the pinned snapshot from this repository's root:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 \
  uvx --from huggingface-hub hf download dots-studio/dots.tts-soar \
  --revision 2f9b3e18d70d670d4c701da2dc55ded5755815ce \
  --local-dir data/models/dots.tts-soar
```

Model weights and download caches are excluded from Git. Downloading the model alone does not install its inference runtime.

See the [upstream runtime and fine-tuning instructions](https://github.com/studio-dots-ai/dots.tts) to run the checkpoint. Pass `data/models/dots.tts-soar` as the local model path. Voice cloning uses reference audio and its exact transcript.

## Test locally

### CPU setup

The local English demo uses the downloaded checkpoint and CPU PyTorch. Install its pinned dependencies once:

```sh
uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --torch-backend cpu --requirements requirements/cpu.txt -e apps/api
.venv/bin/python scripts/fetch_stt.py
```

If the environment already contains the runtime dependencies, register the moved
Python package without reinstalling them:

```sh
uv pip install --python .venv/bin/python --no-deps -e apps/api
```

Build the ElevenLabs UI interface once, using Node.js 20.19+ or 22.12+:

```sh
npm ci
npm run build
```

Start the interface from the repository root:

```sh
.venv/bin/python -m t1000
```

Open **http://127.0.0.1:7860**. An English reference recording and its transcript are preloaded. Enter a short English sentence and click **Generate speech**. To clone your own voice, click **Record voice**, speak a clean 3–10 second sentence, then click **Finish recording**. Words appear while you speak and fill the editable transcript automatically. Uploading a reference also transcribes it automatically. Review any recognition mistakes before generating speech.

The interface opens before model loading; the first generation loads the local checkpoint. CPU synthesis can take several minutes, so start with one short sentence. The CPU demo limits text to 150 characters and reference recordings to 10 seconds, with a bounded generation length. The model stays loaded for subsequent requests. Generated WAV files are saved in `data/output/`, which is excluded from Git. This cloning interface requires a reference recording; an example voice is included to start.

Generation status updates with elapsed time and the amount of audio generated. **Stop generation** cancels generation after the current audio step; long requests also stop at a ten-minute budget, with any partial audio saved. Cancellation can take a few seconds while the current CPU step finishes. Silent, invalid, or overlong recordings are rejected before inference. Replacing a recording invalidates its previous transcript. **Use example voice** restores the included recording and matching transcript. If recognition is unavailable, the complete recording is retained and the transcription status explains how to fix setup or enter its words manually.

The runtime is pinned to upstream commit `b65ca7149fb43b8fb8d52916aa12738deda7b188`. `requirements/constraints.txt` follows upstream's recommended versions, using published Gradio `6.17.3` in place of the unavailable `6.17.0` pin.

### ElevenLabs UI frontend

The React interface uses the AudioPlayer and LiveWaveform components from
[ElevenLabs UI](https://github.com/elevenlabs/ui),
with their MIT license and source attribution recorded in
[apps/web/NOTICE.md](apps/web/NOTICE.md).
The component sources live in `apps/web/src/components/elevenlabs/` and are
bundled locally. No ElevenLabs account, API key, or hosted inference is required.
The browser calls the same-origin FastAPI adapter in `t1000.api`; synthesis and
recognition continue to use this project's local models. FastAPI, Uvicorn, and
multipart uploads are already dependencies of the pinned Gradio installation,
so both the existing CPU and CUDA environments can run the server.

For frontend development, run `.venv/bin/python -m t1000` in one terminal and the Vite development
server in another:

```sh
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:7860`. Rebuild with
`npm run build` after changing frontend sources before using
the Python-served interface. `node_modules/` directories and `apps/web/dist/` are
build artifacts excluded from Git. Commit the source and root package lockfile so a
receiving machine can repeat `npm ci` and the build.

With the server running, run the browser checks with:

```sh
npm exec --workspace @t1000/web -- playwright install chromium
npm run test:web
```

Set `PLAYWRIGHT_BASE_URL=http://127.0.0.1:7861` when testing a server on another
port. Browser checks mock inference and exercise the real microphone worklet;
the Python suite separately verifies the API and inference controls.

The original Gradio interface remains available with `.venv/bin/python -m t1000.ui`.
Run one interface at a time because both use port 7860. Unlike `t1000.api`, the
Gradio entry point preloads the speech model before opening its interface.

### Speech recognition: CPU and gaming GPU profiles

| Profile | Recognition model/runtime | Voice generation | Choose it for |
| --- | --- | --- | --- |
| CPU only | Nemotron English 0.6B, INT8 sherpa-onnx, `STT_BACKEND=cpu` (default) | CPU | Machines without NVIDIA CUDA |
| Gaming GPU, recommended default | Same CPU streaming recognizer | CUDA using `requirements/gpu.txt` | RTX 3050/4060 setups with limited VRAM |
| Optional CUDA recognition | Distil-Whisper large-v3.5, faster-whisper, `int8_float16`, `STT_BACKEND=cuda` | CUDA | Comparing recognition on a GPU with enough free VRAM |

The default [Nemotron model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
uses cache-aware streaming: each audio chunk is processed once. The pinned
[sherpa-onnx INT8 export](https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html)
has a 560 ms model chunk size; browser transport and inference add to that latency.
Four CPU threads are used by default (`STT_NUM_THREADS=2` can reduce contention
on smaller CPUs). One shared model stays loaded; recordings have separate decoder
state. The Gradio entry point preloads this recognizer; the React server initializes
it when first needed. Recognition takes no CUDA VRAM in this profile, leaving the GPU for SOAR.
The verified download is about 464 MB compressed and 662 MB extracted.

Local smoke measurement on the Core Ultra 9 185H, four threads: a 3.904-second
reference took about 0.90 seconds of warm recognition compute, with a first partial
after 1.0 second of input audio. Cold model initialization took 1.69 seconds.
All spoken words matched the included transcript; punctuation differed. These
are single-clip CPU measurements, not an accuracy benchmark or RTX latency claim.

For optional CUDA speech recognition, first follow the GPU handoff, then run:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --requirements requirements/stt-gpu.txt
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/fetch_whisper.py
STT_BACKEND=cuda .venv/bin/python -m t1000
```

[Distil-Whisper large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5)
is an English model. This optional backend makes buffered previews at most once
per two seconds of new audio, then rechecks the complete recording with beam search
and voice activity detection. It repeats decoding work; the default Nemotron path
provides true streaming. Uploads use a single final Whisper decode. The CUDA model
snapshot is pinned to `9793ccc07920e0f830e1dba0343efcdf0ef8c903`.

CUDA recognition requires [CUDA 12 cuBLAS and cuDNN 9](https://github.com/SYSTRAN/faster-whisper#gpu).
If their shared libraries are not found on Linux with this Python 3.11 environment,
add the CUDA Torch packages' library directories before launching:

```sh
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
STT_BACKEND=cuda .venv/bin/python -m t1000
```

CUDA recognition and synthesis share a compute lock, and recognition weights are
released after final transcription and before synthesis. SOAR remains resident,
so CUDA recognition still needs additional VRAM while recording. RTX 3050 models
vary in VRAM, and neither simultaneous model residency nor GPU latency has been
verified here. Use `STT_BACKEND=cpu` if CUDA recognition runs out of memory; a
four-GB card may also be insufficient for SOAR itself. No minimum VRAM is promised.

The microphone uses an AudioWorklet in a current browser on localhost or HTTPS.
It retains the complete recording separately from live recognition requests, so
stopping cannot discard the final words. Only one live request is sent at a time;
audio collected while inference is busy is coalesced. The final complete PCM repairs
any unsent network tail. Recordings are limited to 10 seconds with CPU TTS and
30 seconds with CUDA TTS. Raw microphone WAVs are stored in `.cache/recordings/`;
delete this directory when those references are no longer needed. Runtime inference
is local and downloads no weights automatically.

Run regression checks with `.venv/bin/python -m unittest discover -s apps/api/tests -v`.
The tests cover normalization, stateful resampling, complete recording transport,
stale results, manual fallback, and CUDA lifecycle using mocks. Real GPU validation
remains a receiving-machine task.

### Verified sample

`data/output/test-english.wav` was generated locally from “Hello, this is a voice test.” using a [public English reference clip](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav). The output is 1.92 seconds of 48 kHz audio. The first request took about 94 seconds on this machine's Intel Core Ultra 9 CPU, including model loading. Generated files are local artifacts and are not committed to Git.
