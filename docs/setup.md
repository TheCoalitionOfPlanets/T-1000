# T-1000 setup

Updated: 2026-10-28. Repository: https://github.com/TheCoalitionOfPlanets/T-1000

## Goal and current state

Set up the existing English voice-cloning demo on the receiving machine, download
the pinned SOAR checkpoint, and verify a short generation on its GPU. This is an
inference demo; training and fine-tuning are not implemented in this repository.
The [dataset survey](datasets.pdf) and [voice-cloning study guide](voice-cloning.pdf)
are background research, not executable training recipes.

The source machine has CPU-only PyTorch. A receiving RTX 3050 Laptop GPU has
also completed real CUDA synthesis, transcription and browser checks. See the
[receiving-machine measurements](#rtx-3050-validation--2026-10-28) below; fit and
performance still depend on the GPU, active workloads and input length.

## Files and pinned versions

| Item | Location / value |
| --- | --- |
| Main interface | `apps/web/` (React, ElevenLabs UI components, English only) |
| Local HTTP API | `t1000.api` (FastAPI, serves the built frontend) |
| Inference and fallback UI | `t1000.tts` (shared inference) and `t1000.ui` (Gradio fallback) |
| Frontend toolchain | Node.js 20.19+ or 22.12+; `package-lock.json` |
| Live microphone transport | `t1000.mic`, React client in `apps/web/src/`; Gradio client in `apps/api/src/t1000/static/mic.js` |
| CPU speech recognition | `t1000.stt`, Nemotron English 0.6B INT8 via sherpa-onnx 2.0.1 |
| Optional CUDA speech recognition | `t1000.stt.gpu`, Distil-Whisper large-v3.5 via faster-whisper 1.2.1 |
| Optional CUDA ASR dependencies | `requirements/stt-gpu.txt` (install after GPU requirements) |
| CPU dependencies | `requirements/cpu.txt` (CPU-only torch wheels) |
| NVIDIA GPU dependencies | `requirements/gpu.txt` (select CUDA backend at install) |
| Shared package pins | `requirements/constraints.txt` |
| Shared runtime dependencies | `requirements/base.txt` (included by CPU and GPU profiles) |
| Python | Use 3.11; upstream allows >=3.10 and <3.13 |
| PyTorch / torchaudio | 2.8.0, matching builds |
| Runtime source | `studio-dots-ai/dots.tts` at `b65ca7149fb43b8fb8d52916aa12738deda7b188` |
| Model | `dots-studio/dots.tts-soar` |
| Model revision | `2f9b3e18d70d670d4c701da2dc55ded5755815ce` |
| Model directory | `data/models/dots.tts-soar/` |
| Included voice | `assets/ref.wav`; provenance and transcript in `assets/README.md` |
| Tests | `apps/api/tests/` (inference, recording transport, CPU ASR, mocked CUDA ASR) |
| Generated audio | `data/output/` (ignored by Git) |

## 1. Check the machine and get the code

Commands below target Linux or WSL2 with an NVIDIA GPU and a working NVIDIA
driver. Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
first. On Ubuntu/Debian, `sudo apt-get install git ffmpeg libsndfile1` provides
Git and common audio dependencies. Run `nvidia-smi` inside the environment that
will run Python; fix driver/GPU visibility before proceeding if it fails.

```sh
nvidia-smi
git clone https://github.com/TheCoalitionOfPlanets/T-1000.git
cd T-1000
git log -1 --oneline
```

If already cloned, enter that checkout and run `git pull --ff-only origin main`.
Model weights, caches, the virtual environment, and generated WAVs are excluded
from Git, so pulling the repository alone will not transfer them. The model
snapshot is about 5.2 GB; allow additional disk space for CUDA packages and caches.

## 2. Install the NVIDIA GPU runtime

Use a fresh environment on this machine. Do not install `requirements/cpu.txt` into
the GPU environment: it explicitly selects CPU-only PyTorch.

```sh
uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --torch-backend cu128 --reinstall-package torch --reinstall-package torchaudio \
  --requirements requirements/gpu.txt -e apps/api
uv pip check --python .venv/bin/python
```

For a 4–6 GB GPU, replace `requirements/gpu.txt` in the installation command with
`requirements/gpu-low-vram.txt`. This adds bitsandbytes 0.50.2 and Accelerate 1.15.0
for the experimental quantization and CPU layer-offload profiles below.

CUDA 12.8 is one of the builds published for PyTorch 2.8.0 in the
[official version instructions](https://pytorch.org/get-started/previous-versions/#v280).
Use a driver compatible with that build. If the machine needs a different build,
select a supported backend from those instructions while keeping torch and
torchaudio at matching 2.8.0 versions. Do not assume the CUDA number displayed by
`nvidia-smi` means the Python environment already has CUDA-enabled PyTorch.
The reinstall flags ensure that an existing CPU build is replaced even though
it satisfies the same base version requirement.

Check the actual environment before downloading/loading the model:

```sh
.venv/bin/python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check driver, torch build, and GPU visibility'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM (GiB):', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('GPU precision:', 'BF16' if torch.cuda.is_bf16_supported() else 'FP16')
print('GPU tensor:', (torch.ones(1, device='cuda') + 1).item())
PY
```

The app selects BF16 on supported CUDA GPUs and otherwise FP16. FP16 numerical
behavior and audio quality need validation on the particular GPU. Its architecture
must also be supported by the chosen PyTorch and bitsandbytes CUDA wheels;
VRAM capacity alone does not establish compatibility. AMD/Intel GPU execution is
not covered by these CUDA instructions.
For native Windows PowerShell, use `.venv\Scripts\python.exe` instead of
`.venv/bin/python` and PowerShell environment-variable syntax instead of the
Unix inline assignments; Linux/WSL2 is the documented command sequence.

## 3. Download or resume the complete model snapshot

Run from the repository root, with network access:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 \
  uvx --from huggingface-hub hf download dots-studio/dots.tts-soar \
  --revision 2f9b3e18d70d670d4c701da2dc55ded5755815ce \
  --local-dir data/models/dots.tts-soar
```

Keep the entire snapshot: inference also needs the vocoder, speaker encoder,
latent statistics, tokenizer, and JSON configurations. Do not download only
`model.safetensors`. Re-run the same command after an interrupted download.
`t1000.tts` defaults to `HF_HUB_OFFLINE=1`, so download before launching; startup
does not automatically fetch missing files. The model license is declared as
Apache 2.0 upstream; its model card is included in the downloaded snapshot.

Download the additional CPU streaming recognizer (about 464 MB compressed):

```sh
.venv/bin/python scripts/fetch_stt.py
```

This pins the Nemotron English 0.6B 560 ms INT8 export dated 2026-04-25 and
verifies its upstream SHA-256 before extracting only the four runtime files.
Re-running resumes an interrupted download or skips a completed installation.
Both CPU and GPU TTS environments use this CPU recognizer by default. This is
the recommended profile for gaming GPUs such as the RTX 3050 and RTX 4060:
recognition uses CPU/RAM while SOAR uses CUDA/VRAM.

For optional GPU recognition, install `requirements/stt-gpu.txt`, run
`scripts/fetch_whisper.py` with `HF_HUB_OFFLINE=0`, and launch with
`STT_BACKEND=cuda`. It uses the pinned English Distil-Whisper large-v3.5 CT2
snapshot with INT8/FP16 computation. Follow the [README CUDA recognition
instructions](../README.md#speech-recognition-cpu-and-gaming-gpu-profiles) for
CUDA 12/cuDNN 9 shared library paths. Its previews are buffered every two
seconds; default Nemotron recognition is cache-aware streaming.

CUDA recognition unloads after finalization and before synthesis. TTS stays
loaded, so extra VRAM is still required while recording. If this fails on a
small card, use default `STT_BACKEND=cpu`. Four-GB GPUs may not fit SOAR itself;
no model/GPU combination is verified until tested on the receiving machine.

## 4. Build the interface, verify, and launch

Install Node.js 20.19+ or 22.12+, then build the React interface. The existing
CPU/GPU Python requirements already install FastAPI, Uvicorn, and multipart
upload support through Gradio; no separate Python UI requirements are needed.

```sh
npm ci
npm run build
.venv/bin/python -m unittest discover -s apps/api/tests -v
.venv/bin/python -m t1000
```

The tests use a fake runtime for generation controls and do not validate real
GPU inference. The interface opens before the model loads; the first generation
loads the real checkpoint. Open http://127.0.0.1:7860 on that machine. The server
listens on localhost with public sharing disabled. If using
a remote GPU host, an SSH tunnel such as
`ssh -L 7860:127.0.0.1:7860 user@gpu-host` lets the local browser reach it.

Use the included example voice and “Hello, this is a voice test.” with 10 steps
and seed 42 for the first generation. Confirm that the device and generation
status report `CUDA GPU`, listen to
the output, and check a WAV was saved under `data/output/`. Confirm the runtime log
reports `device=cuda` after the first request loads the checkpoint.
For a different voice, click **Record voice**, speak a clear 3–6 second
sentence, then **Finish recording**. Confirm words appear during recording and
the editable transcript fills automatically. Uploading a clip also transcribes
it. Check the recognized words before generating. Replacing audio invalidates
old work; **Use example voice** restores the included pair without recognition.
If ASR fails, its setup error is shown and complete audio remains available for
manual transcription. A current browser with AudioWorklet and microphone
permission is required, on localhost or HTTPS.

Standard GPU text is limited to 500 characters; CPU text to 150 characters.
References are limited to 10 seconds with CPU TTS and 30 seconds with standard
CUDA TTS. Experimental low-memory profiles use 150 characters and 6-second
references. Generation
has a 600-second budget. **Stop generation** and the
deadline request cancellation after the current audio step and preserve any
partial output; they do not immediately interrupt a running CUDA operation.

The frontend bundles the AudioPlayer and LiveWaveform components from
[ElevenLabs UI](https://github.com/elevenlabs/ui)
under the MIT license; see [apps/web/NOTICE.md](../apps/web/NOTICE.md)
for source attribution. This uses the open-source UI components without an
ElevenLabs account or API key. Speech generation and transcription remain local.
`node_modules/` directories and `apps/web/dist/` are not transferred by Git; run the
build commands again on every receiving machine.

For frontend editing, keep `.venv/bin/python -m t1000` running and use
`npm run dev` in another terminal; its development proxy sends
`/api` requests to port 7860. The original Gradio interface is available with
`.venv/bin/python -m t1000.ui` and preloads the model at startup. Stop `t1000.api`
before using it because both entry points bind port 7860.

## GPU inference performance

The `auto` profile selects experimental NF4 inference on CUDA GPUs with 6 GiB or
less, and standard inference otherwise. Standard inference automatically compiles
only on GPUs with at least 12 GiB, because the compiled flow-decoder caches and
warmup can use substantial memory. `T1000_TTS_OPTIMIZE=1` explicitly enables that
path in the standard profile; `0` disables it. CPU inference stays eager.

All profiles use the local SOAR checkpoint, full reference audio/text conditioning,
sampling controls and one-patch vocoder chunks by default. Standard and offload profiles keep
unquantized weights. Quantized or compiled arithmetic can change output, so
compare voice similarity, tone, pacing, pronunciation and completeness.

### Experimental 4–6 GB profiles

The current checkpoint contains approximately 4.795 GiB of tensor weights before
CUDA context, activations or decoding caches: 4.094 GiB in the BF16 core, 0.674 GiB
in the FP32 vocoder and 0.027 GiB in the speaker encoder. Those are measurements
of the local safetensors headers, not GPU peak-memory measurements. The full
resident checkpoint cannot fit on a 4 GiB GPU by merely disabling compilation.

| Profile | Placement | Tradeoff |
| --- | --- | --- |
| `low-vram` | NF4 linear projections in the language-model transformer blocks; other modules retain normal precision. | Approximately 2.97 GiB of raw weight storage before quantization metadata and inference overhead. Voice and delivery can change; speed depends on GPU kernels. |
| `offload`, GPU layers `0` | Acoustic modules and token embeddings stay on CUDA; language blocks are stored in CPU RAM and loaded onto CUDA for each forward call. | Preserves unquantized weights and lowers resident VRAM, but repeated transfers generally increase latency. |
| `offload`, GPU layers `14` | Keep half the bundled language model's 28 decoder blocks on CUDA and offload the rest. | Fewer transfers, more VRAM. This is not half of the complete TTS model, and may be too large for 4 GB. |

Both low-memory modes disable compilation, cap the runtime at 128 total audio
patches and a 512-position language context, and accept at most 150 text characters
and 6-second references. Long utterances can hit the audio cap; listen for complete
speech. Use the default CPU transcription backend so ASR does not compete for VRAM.
The loaders fuse acoustic inference weights on CPU before placement, avoiding
temporary duplicate weights on CUDA during the first decode.
The full checkpoint is still loaded into CPU RAM initially: these profiles do not
provide disk streaming or reduce the initial host-RAM requirement.

After installing the low-VRAM requirements, launch either profile explicitly:

```sh
T1000_TTS_PROFILE=low-vram .venv/bin/python -m t1000
T1000_TTS_PROFILE=offload T1000_TTS_GPU_LAYERS=0 .venv/bin/python -m t1000
```

For faster NF4 generation, opt into eager vocoder batching with
`T1000_TTS_VOCODER_BATCH_SIZE=4` (default `1`; other values are rejected).
It emits the first two patches separately, then decodes groups of four to
reduce repeated work. Model weights and sampling steps stay the same; CUDA
kernel rounding can produce small waveform differences. Stop can wait for up
to four acoustic patches between output chunks. This setting applies only to
the eager NF4 loader; standard and compiled inference retain upstream behavior.

Only run one server at a time. Restart to change profiles. Automatic selection on
a 4–6 GB GPU chooses `low-vram`; an explicitly enabled compile flag must be removed
or set to `0` for either low-memory profile. A GPU's total VRAM includes space
needed by the display and other programs, so short-request fit still needs testing.

### Measure memory, speed and audio quality

Run each approach in a separate process on the actual GPU:

```sh
.venv/bin/python scripts/benchmark_tts.py --mode low-vram --output-dir data/output/bench-nf4
.venv/bin/python scripts/benchmark_tts.py --mode offload --gpu-layers 0 --output-dir data/output/bench-offload
.venv/bin/python scripts/benchmark_tts.py --mode offload --gpu-layers 14 --output-dir data/output/bench-half
```

Start offload with zero resident decoder blocks on 4 GB. Increase the count only
after measuring headroom. Both profiles have completed the short receiving-machine
test below; that result does not guarantee every permitted input will fit.

The first model load runs upstream compilation and warmup, which can take several
minutes and uses extra GPU memory when standard compilation is enabled. In the web app it currently counts toward the
600-second request budget. Stop requests wait for this operation to return. Later
requests reuse the loaded runtime. Restart with optimization disabled if the GPU
encounters a compile failure, excessive memory use or a cold-start timeout:

```sh
T1000_TTS_OPTIMIZE=0 .venv/bin/python -m t1000
```

For larger GPUs, compare standard eager and compiled inference separately:

```sh
.venv/bin/python scripts/benchmark_tts.py --mode eager --output-dir data/output/bench-eager
.venv/bin/python scripts/benchmark_tts.py --mode optimized --output-dir data/output/bench-optimized
```

The benchmark keeps the bundled voice, transcript, text, ten sampling steps,
guidance 1.2 and seed 42 the same. It selects BF16 when supported, otherwise FP16;
use the same precision in every comparison. It separates model loading/warmup from warm
requests and reports time to the first audio chunk, complete generation time,
audio duration, real-time factor (generation seconds / audio seconds), and peak
CUDA memory, including separate peak-memory counters during loading. A real-time
factor below 1 means generation is faster than playback.
JSON reports and WAV samples are saved in each output directory. Listen to both
sets; speed alone does not establish preserved voice similarity or fluent delivery.

Use the same `--text`, `--reference` and `--transcript` arguments in both commands
to compare your own voice. Run `--help` for all options. Optional stage profiling
adds CUDA synchronization overhead, so keep it off when measuring normal latency.
Low-memory profiles fix their audio cap at 128 patches; pass
`--max-generate-length 128` to standard benchmark runs when comparing across
profiles. The benchmark enforces each profile's text and reference limits;
start with short inputs and check that output is complete.
The receiving-machine measurements below provide one hardware result.

The browser currently receives status updates during synthesis and plays a WAV
after generation completes. The benchmark's first-chunk measurement describes
the model output; it is not browser playback latency.

## Troubleshooting and next-agent completion record

- **CUDA unavailable / device says CPU:** verify `nvidia-smi` and torch inside the
  same environment. Install the GPU requirements with the CUDA backend, then
  restart the app; device selection is made when the module imports.
- **Frontend build missing:** run `npm ci` followed by
  `npm run build`, then start `t1000.api` from the repository.
- **Missing files or offline errors:** finish the full pinned snapshot download
  into the exact directory above. Do not rely on another machine's Hugging Face cache.
- **CUDA out of memory:** close other GPU workloads and restart. Try a short
  sentence/reference first. Minimum VRAM has not been established; report the
  GPU model, free VRAM, and traceback if it still fails.
- **Silent audio / slow or poor generation:** check the recording and exact
  transcript. Use the bundled example to separate setup issues from input issues.
- **Transcription unavailable:** run the recognition download script before
  launch and install the updated requirements. Check the transcription status
  for missing-weight, CUDA-library, or VRAM errors. Default CPU recognition
  needs no CUDA libraries. Microphone references are kept under `.cache/recordings/`.
- **Dependency conflicts:** keep `requirements/constraints.txt`, the runtime commit, and
  torch/torchaudio pins together. Gradio 6.17.3 replaces upstream's unpublished
  6.17.0 pin. Re-run `uv pip check` after changes.

After testing another machine, update this section (and [`CHANGELOG.md`](../CHANGELOG.md) for code changes)
with OS, GPU/VRAM, driver, Python/torch/CUDA versions, code commit, model revision,
test results, generation time, output duration, and any fixes.

### RTX 3050 validation — 2026-10-28

Tested Arch Linux, kernel 6.18.56-1-lts, NVIDIA driver 615.78.08,
RTX 3050 Laptop GPU advertised as 4096 MiB (3.73 GiB usable through PyTorch),
16 GB system RAM, Python 3.11.15, and torch/torchaudio 2.8.0+cu128.
The code was `0346bf2` plus the CPU inference-fusion changes above, with the
pinned SOAR revision in this guide. All three checkpoint hashes matched upstream.
CPU Nemotron recognition was used; CUDA recognition was not exercised.

Before CPU fusion, NF4 loaded but failed while allocating the first decoder's
220 MiB fused projection. Moving the same fusion to CPU resolved that failure.
The tested launch environment is:

```sh
STT_BACKEND=cpu T1000_TTS_PROFILE=low-vram T1000_TTS_OPTIMIZE=0 \
  T1000_TTS_VOCODER_BATCH_SIZE=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m t1000
```

For “Hello, this is a voice test.” with the bundled reference, ten steps and
seed 42, both profiles produced 1.92 seconds of finite, non-silent 48 kHz audio:

| Profile | First request, including load | Warm browser completion |
| --- | ---: | ---: |
| Offload, zero resident language blocks | 45.37 s | 11.55 s |
| NF4 with CPU inference fusion, batch size 1 | 58.99 s | 7.48 s |
| NF4 with CPU inference fusion, batch size 4 | 61.18 s | 5.97 s |

The warm comparison is one browser request per profile over an SSH tunnel,
including API/UI completion; it is not a repeated statistical benchmark.
Recognition recovered the requested words from the tested WAVs. 121 backend tests,
real upload and streaming-PCM recognition, real browser AudioWorklet recording
using prerecorded input, WAV decoding, download and player playback passed.
Cold loading used substantial host RAM and swap. Keep the model process running
to reuse its loaded weights. Subjective voice similarity and longer-input memory
headroom require separate evaluation. A 100-character sentence also completed
with batch size 4: 6.40 seconds of audio in 20.28 seconds, versus 26.1 seconds
with batch size 1. Recognition recovered every requested word, allowing
punctuation/case differences.

A separate two-run comparison in one resident NF4 process measured median
generation times of 7.00 seconds with one-patch vocoding and 5.80 seconds with
four-patch vocoding. Both produced 92,160 samples. Peak allocated CUDA memory
was 3449.2 and 3449.7 MiB respectively. Waveform RMS difference was 0.0575% of
the baseline RMS (64.8 dB signal-to-error ratio); maximum absolute difference
was 0.000484. A strict pointwise `allclose` check failed, so these results
establish small aggregate numerical error rather than bit-identical audio or
subjective perceptual equivalence. Separate synchronized profiles attributed
most of the gain to vocoding, which fell from 3.33 to 2.10 seconds. Profiling
requests were excluded from the timed medians.

Source CPU recognition smoke: Core Ultra 9 185H, four threads, included 3.904 s
reference; cold initialization 1.69 s, warm incremental recognition including
final flush about 0.90 s, first partial after 1.0 s of input. Spoken words matched;
punctuation differed. GPU ASR was checked with mocks and import/API validation
only; measure recording-to-partial latency and peak VRAM on the receiving GPU.
