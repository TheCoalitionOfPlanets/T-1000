# Changelog

## 2026-10-31 — Reference cleanup and trimming

- Enable browser microphone noise suppression and echo cancellation, with a
  recording/upload opt-out in the studio. Clean saved references locally with
  conservative background suppression while preserving their native rate.
- Add start/end inputs and sliders, range preview, optional automatic quiet-edge
  trimming, and original restoration. Always edit the original into a new WAV
  and re-transcribe the exact result before voice generation.
- Add equivalent cleanup/crop controls to the optional Gradio lab. Keep raw
  recordings available through recording retries and audio-registry eviction.
- Cover noise reduction, speech retention, exact crops, invalid selections,
  transcript consistency, original restoration and browser interactions.

Validation: 283 backend tests, 60 browser checks, and the production build pass.
A real API/browser smoke verifies microphone capture, upload cleanup, repeated
crops and restoration with matching WAV frame counts and no browser/API errors.
Speech recognition is stubbed at its boundary in this smoke; synthesis quality
and target-GPU inference are not measured by these checks.

## 2026-10-29 — Paragraph generation

- Raise the full request limit to 5,000 characters and label the editor counter
  explicitly as characters. Plan longer text into sentence-aware parts within
  the existing 150-character CPU/low-memory or 500-character standard CUDA cap.
- Use the same full reference and transcript for every part, preserving all
  input words and generated samples in one WAV. Add only missing quiet pauses
  at joins; require every part to reach EOS before declaring completion.
- Show part progress and preserve incomplete output on cancellation, inference
  failure or a model cap. Allow 30 minutes for multi-part requests and cap output
  at 10 minutes; expand verified WAV transfers to 64 MiB.

Validation: 255 backend tests pass locally and on the RTX 3050, alongside 53
browser checks and the production build. A real 290-character, two-paragraph
request completed three parts and produced 15.09 seconds of speech in 41.76
seconds. CPU ASR matched all 52 requested words after case and punctuation
normalization. The short seeded regression WAV remains byte-identical to the
previous deployment. These checks do not establish subjective naturalness or
cloning similarity.

## 2026-10-29 — Recording integrity and recoverable generation

- Batch microphone capture in 20 ms worklet chunks with ordered offsets, sample
  acknowledgements and final-tail flushing. Reconcile complete captured PCM on
  finish, retry immutable payloads and retain a local recovery WAV on save failure.
- Preserve the same completed reference across lost finish responses. Serialize
  concurrent finish/cancel operations and retry full-file ASR when previews fail.
- Move PCM ownership into a worker that writes an incremental, atomically
  published WAV with sample count, completion reason and SHA-256 metadata.
- Add bounded generation jobs with same-ID reconnects, progress cursors and
  request-scoped Stop. Resume WAV downloads with HTTP Range and verify the whole
  file before sharing one buffered Blob between playback and waveform rendering.

Validation: 233 backend tests pass locally and on the RTX 3050; 52 browser checks,
the production build and 16 live API/browser stages pass. The actual worklet
reference retained all 68,864 captured samples, including 8,064 after the last
feed acknowledgement. Reconnect, Range recovery, stale Stop isolation and explicit
partial cancellation pass. Two seeded before/after WAVs are byte-identical. This
change improves transfer reliability without establishing a new compute speedup.

## 2026-10-29 — Vocoder compilation and comparison experiments

- Add opt-in FP32 vocoder-only compilation through
  `T1000_TTS_COMPILE_VOCODER=1`, with bounded startup warmup and CUDA graphs
  disabled. Retain the existing SOAR sampling and four-patch buffering.
- Add reproducible SOAR/MF2 and isolated-vocoder comparison scripts with timing,
  memory, waveform and ASR scoring. Keep native LSTM, rotary reuse, reduced
  decoder precision and eight-patch buffering as disabled experimental variants.

Validation: eight unique English cases with two timed repeats per configuration
gave 16 complete requests each. The compiled FP32 candidate achieved a 1.099×
median paired speed ratio, or 9.0% lower matched generation latency; aggregate
median generation times were 11.062 seconds before and 9.983 seconds after.
Both unique eight-case sets had zero ASR word errors out of 80 words. The
compiled outputs have small waveform differences, so these measurements do not
establish audio identity, subjective quality or performance beyond the tested
RTX 3050 inputs.

## 2026-10-29 — Conditioning reuse and reference/text accuracy safeguards

- Add optional request-scoped reuse of eager Euler timestep conditioning via
  `T1000_TTS_EAGER_CACHE=1`. It preserves the original solver and sampling
  arithmetic, checks voice/timestep consistency, and defaults off.
- Preserve quiet reference boundaries instead of silently applying the upstream
  30 dB trimming threshold. Cache CPU prompt preprocessing by content, bounded
  to eight entries/16 MiB; `T1000_REFERENCE_TRIM=1` restores upstream trimming.
- Add explicit English text-expansion preview with Apply, Cancel and Undo.
  Ambiguous formatted phone numbers require manual spelling. Reference
  transcripts are never normalized.
- Add API/browser request timing milestones and opt-in local-server startup
  preload with `T1000_PRELOAD_MODELS=1`.

Validation: 150 backend tests pass locally and on the RTX 3050 laptop, along
with nine browser checks and both production builds. A two-run warm comparison
measured 5.709 seconds without conditioning reuse versus 5.578 seconds with it
(2.3% lower), with bit-identical audio. This small sample is not a statistical
performance guarantee. Four sentences with complete reference audio retained
every requested word in normalized ASR checks. Quiet-boundary tests reproduce
the old trimming loss; preserving the reference can change pacing and duration.
These checks do not establish general accuracy or subjective speaker quality.

## 2026-10-28 — RTX 3050 memory and generation optimization

- Fuse the acoustic decoder's inference projections on CPU before CUDA
  placement in both low-memory profiles. This avoids roughly 440 MiB of
  temporary CUDA copies during the first decode, allowing the NF4 profile to
  run on the tested 4 GB RTX 3050 Laptop GPU.
- Construct NF4 runtimes outside a caller's inference-mode context so mutable
  inference adapters retain normal tensors.
- Add optional four-patch eager vocoder batching for NF4 through
  `T1000_TTS_VOCODER_BATCH_SIZE=4`. The default remains one patch. Stream
  state, final partial groups, mutable latent buffers and compiled-mode
  passthrough are covered by real small-vocoder tests.

Validation: 121 Python tests pass locally and on the GPU laptop, including real
small-DiT/vocoder prediction comparisons and fusion placement checks. Real upload/streaming
transcription, browser recording, CUDA generation, WAV decoding and browser
playback pass. With the same example voice, ten steps, seed 42 and “Hello, this
is a voice test.”, warm browser completion fell from 11.55 seconds with CPU
layer offload to 5.97 seconds with resident NF4 and vocoder batching (1.92
seconds of audio in both). A longer sentence fell from 26.1 to 20.3 seconds
after batching, retaining all requested words in 6.40 seconds of audio. NF4 still changes
weight precision; these short checks do not establish subjective voice quality
or performance on other GPUs. See [the setup record](docs/setup.md#rtx-3050-validation--2026-10-28).

## 2026-10-27 — Separate CPU and GPU Docker deployments

- Added separate multi-stage CPU and CUDA Dockerfiles and Compose configurations.
  Each builds the React UI and installs its Python dependency profile; the GPU
  image includes quantization/offload dependencies and CUDA 12.8 Torch wheels.
- Added startup provisioning for pinned SOAR weights and verified CPU recognition,
  persistent model/cache volumes, model preload, device checks and a health endpoint.
  Completed marked snapshots skip repeat downloads; offline local-model reuse is
  explicit. One server worker serves the UI/API after preparation.
- Run as a nonroot user, bind the published port to localhost, and exclude local
  model files, recordings, dependencies and private files from the build context.
- Documented one-command startup, NVIDIA prerequisites, volume reuse and GPU
  benchmark commands.

Validation: 109 Python tests pass, including provisioning, download failure,
offline reuse, startup ordering, device validation and health status. Both images
build and both Compose configurations validate. The nonroot CPU container preloads
real local models, passes six Chromium checks and a live UI/API check, transcribes
the bundled clip, and generates a finite 48 kHz WAV (0.64 seconds in 29.0 seconds
for “Hi.” at four steps). Named-volume permissions and pinned-model download
access were checked. GPU image imports and its no-CUDA startup failure pass;
real CUDA synthesis and 4–6 GB memory fit still require hardware testing.

## 2026-10-27 — GPU inference and experimental low-memory profiles

- Added automatic profile selection: experimental LLM-only NF4 on CUDA GPUs with
  6 GiB or less; standard inference otherwise. Automatic compilation requires
  at least 12 GiB; explicit enable/disable flags remain available.
- Added unquantized CPU layer offload, with a configurable number of resident
  Qwen blocks. Initial placement avoids transferring the complete dense model
  to CUDA, and tied embeddings remain shared.
- Preserved SOAR reference conditioning, acoustic modules, sampling controls and
  one-patch vocoder chunks. Select BF16 when supported and FP16 otherwise. Bound
  low-memory inputs/context, and match compiled warmup to the inference context.
- Show model warmup in generation status and honor stop requests made during
  loading before starting audio synthesis.
- Added a GPU benchmark that separates startup and warm generation, measures
  first audio, total time, real-time factor and memory, and saves comparable WAVs
  and JSON reports. Optional stage profiling runs separately from speed timings.

Validation: 90 Python tests pass, including profile selection, placement,
quantization boundaries, runtime reuse and cancellation during loading. Small real
Qwen models verify bitsandbytes quantization order and Accelerate prefill/cached
decode hooks on CPU. Dependency checks, benchmark help, argument validation and
the no-CUDA failure path pass. Real 4–6 GB fit, GPU speed and voice/delivery
equivalence remain unverified on this CPU-only machine.

## 2026-10-26 — Project README

- Adopted the supplied README format with setup, usage, configuration, privacy,
  architecture, development, and troubleshooting sections.
- Added a local logo and a screenshot of the running voice studio.
- Documented the project's current status and third-party terms without
  introducing unsupported release, CI, or project-license badges.

Validation: local documentation links and anchors, shell example syntax,
template placeholders, and whitespace checks pass.

## 2026-10-26 — Monorepo structure

- Grouped the Python API and React app under `apps/api` and `apps/web`, with
  tests beside their owning app and one npm workspace lockfile at the root.
- Packaged Python code under `t1000`, separated the Gradio presentation from
  synthesis, and centralized runtime paths. Launch with `python -m t1000` or
  `python -m t1000.ui` after installing the editable package.
- Grouped dependency profiles under `requirements`, bundled audio under
  `assets`, and local models and generated WAVs under ignored `data` directories.
- Shortened model-download scripts, moved the setup guide to `docs/setup.md`,
  and named the research PDFs `datasets.pdf` and `voice-cloning.pdf`.
- Updated imports, asset references, setup commands, and naming conventions.

Validation: 63 Python tests and six browser checks pass after the moves. The
root npm workspace builds successfully, editable imports work outside the
checkout, and the Python wheel includes the microphone script. Real CPU
transcription, the relocated model download helper, and the Vite API proxy also
work with the new paths.

## 2026-10-26 — ElevenLabs UI voice lab

- Added a React interface with locally bundled, MIT-licensed ElevenLabs UI
  AudioPlayer and LiveWaveform components and source attribution.
- Added a same-origin FastAPI adapter for local reference uploads, microphone
  transport, transcription, generation status, playback, and cancellation.
- Kept the local SOAR and speech-recognition runtimes and the original Gradio
  interface; the React entry point loads speech weights on first generation.
- Added a reproducible npm build and CPU/GPU setup instructions. No ElevenLabs
  account or API key is required.

Validation: clean `npm ci` and production build passed, along with 63 Python
tests and six Chromium browser checks. A browser recording also completed real
local CPU recognition and populated the editable transcript. Desktop and mobile
layouts were inspected. Generation and stopping were verified with mocked inference;
real GPU validation remains a receiving-machine task.

## 2026-10-26 — Automatic live reference transcription

- Added Nemotron English 0.6B INT8 cache-aware speech recognition on CPU, with
  one cached recognizer, separate recording state, and stateful audio resampling.
- Added live microphone text, full final recording retention, bounded ordered
  transport, automatic upload transcription, editable results, and protection
  against stale transcripts when references change.
- Added optional CUDA Distil-Whisper large-v3.5 recognition using INT8/FP16,
  buffered previews, a final VAD/beam-search pass, and coordinated GPU cleanup.
- Added pinned model download scripts, CPU/gaming-GPU setup profiles, and
  input, transport, lifecycle, and reference-state regression tests.

Validation: real CPU recognition matched the spoken words in the 3.904-second
example, taking about 0.90 seconds of warm compute. Browser microphone recording,
live text, final transcript, reference replacement, and uploads were exercised.
CUDA import/API and mocked behavior were verified; RTX inference latency and
combined TTS/ASR VRAM remain unmeasured.

## 2026-10-26 — Initial repository snapshot and GPU handoff

- Added the English dots.tts SOAR Gradio voice-cloning demo with a bundled
  reference recording, transcript, and source attribution.
- Pinned the upstream runtime commit, model revision, CPU dependencies, and
  shared runtime constraints; documented model download and local launch.
- Included input validation, progress updates, stop/deadline cancellation,
  partial WAV saving, and five inference-control regression tests.
- Added separate NVIDIA GPU requirements and a setup guide (now `docs/setup.md`) with clone/pull,
  CUDA checks, model download, smoke testing, and troubleshooting instructions.
- Included the two background research PDFs under `docs/`.
- Excluded model weights, caches, environments, and generated outputs from Git.

Validation: all five existing tests pass on the source CPU environment. A previous
real CPU inference produced 1.92 seconds of 48 kHz audio. A uv dry run resolved
CUDA 12.8 torch and torchaudio 2.8.0 wheels with the GPU requirements. Real GPU installation,
inference performance, and minimum VRAM remain for the receiving machine to verify.
