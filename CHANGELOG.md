# Changelog

## 2026-10-26 — Initial repository snapshot and GPU handoff

- Added the English dots.tts SOAR Gradio voice-cloning demo with a bundled
  reference recording, transcript, and source attribution.
- Pinned the upstream runtime commit, model revision, CPU dependencies, and
  shared runtime constraints; documented model download and local launch.
- Included input validation, progress updates, stop/deadline cancellation,
  partial WAV saving, and five inference-control regression tests.
- Added separate NVIDIA GPU requirements and `AGENT_HANDOFF.md` with clone/pull,
  CUDA checks, model download, smoke testing, and troubleshooting instructions.
- Included the two background research PDFs under `docs/`.
- Excluded model weights, caches, environments, and generated outputs from Git.

Validation: all five existing tests pass on the source CPU environment. A previous
real CPU inference produced 1.92 seconds of 48 kHz audio. A uv dry run resolved
CUDA 12.8 torch and torchaudio 2.8.0 wheels with the GPU requirements. Real GPU installation,
inference performance, and minimum VRAM remain for the receiving machine to verify.
