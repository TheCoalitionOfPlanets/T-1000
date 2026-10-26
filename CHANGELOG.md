# Changelog

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
