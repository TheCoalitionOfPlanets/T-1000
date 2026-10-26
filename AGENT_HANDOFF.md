# T-1000 agent handoff

Updated: 2026-10-26. Repository: https://github.com/TheCoalitionOfPlanets/T-1000

## Goal and current state

Set up the existing English voice-cloning demo on the receiving machine, download
the pinned SOAR checkpoint, and verify a short generation on its GPU. This is an
inference demo; training and fine-tuning are not implemented in this repository.
The PDFs under `docs/` are background research, not executable training recipes.

The source machine has CPU-only PyTorch. Its five inference-control tests pass,
and a previous real CPU run produced 1.92 seconds of 48 kHz audio from
“Hello, this is a voice test.” GPU inference and GPU memory requirements have
not been measured. Record the receiving machine's GPU results below after testing.

## Files and pinned versions

| Item | Location / value |
| --- | --- |
| UI and inference | `app.py` (Gradio, English only) |
| CPU dependencies | `requirements.txt` (CPU-only torch wheels) |
| NVIDIA GPU dependencies | `requirements-gpu.txt` (select CUDA backend at install) |
| Shared package pins | `constraints.txt` |
| Python | Use 3.11; upstream allows >=3.10 and <3.13 |
| PyTorch / torchaudio | 2.8.0, matching builds |
| Runtime source | `studio-dots-ai/dots.tts` at `b65ca7149fb43b8fb8d52916aa12738deda7b188` |
| Model | `dots-studio/dots.tts-soar` |
| Model revision | `2f9b3e18d70d670d4c701da2dc55ded5755815ce` |
| Model directory | `models/dots.tts-soar/` |
| Included voice | `examples/reference-en.wav`; provenance and transcript in `examples/README.md` |
| Tests | `tests/test_inference_controls.py` |
| Generated audio | `outputs/` (ignored by Git) |

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

Use a fresh environment on this machine. Do not install `requirements.txt` into
the GPU environment: it explicitly selects CPU-only PyTorch.

```sh
uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --torch-backend cu128 --reinstall-package torch --reinstall-package torchaudio \
  --requirements requirements-gpu.txt
uv pip check --python .venv/bin/python
```

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
print('BF16 supported:', torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), 'This app selects BF16 on CUDA; review precision for this GPU before launching'
print('GPU tensor:', (torch.ones(1, device='cuda') + 1).item())
PY
```

The app selects BF16 whenever CUDA is available. Older GPUs without BF16 need a
reviewed precision change in `load_runtime()` before using this path; FP32 uses
more memory. AMD/Intel GPU execution is not covered by these CUDA instructions.
For native Windows PowerShell, use `.venv\Scripts\python.exe` instead of
`.venv/bin/python` and PowerShell environment-variable syntax instead of the
Unix inline assignments; Linux/WSL2 is the documented command sequence.

## 3. Download or resume the complete model snapshot

Run from the repository root, with network access:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 \
  uvx --from huggingface-hub hf download dots-studio/dots.tts-soar \
  --revision 2f9b3e18d70d670d4c701da2dc55ded5755815ce \
  --local-dir models/dots.tts-soar
```

Keep the entire snapshot: inference also needs the vocoder, speaker encoder,
latent statistics, tokenizer, and JSON configurations. Do not download only
`model.safetensors`. Re-run the same command after an interrupted download.
`app.py` defaults to `HF_HUB_OFFLINE=1`, so download before launching; startup
does not automatically fetch missing files. The model license is declared as
Apache 2.0 upstream; its model card is included in the downloaded snapshot.

## 4. Verify and launch

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python app.py
```

The tests use a fake runtime for generation controls and do not validate real
GPU inference. Startup loads the real checkpoint before the UI opens. Confirm
the runtime log reports `device=cuda`, then open http://127.0.0.1:7860 on that
machine. The server listens on localhost with public sharing disabled. If using
a remote GPU host, an SSH tunnel such as
`ssh -L 7860:127.0.0.1:7860 user@gpu-host` lets the local browser reach it.

Use the included example voice and “Hello, this is a voice test.” with 10 steps
and seed 42 for the first generation. Confirm that the Result field reports
`CUDA GPU`, listen to the output, and check a WAV was saved under `outputs/`.
For a different voice, use a clear 3–10 second reference and its exact transcript.
Changing the audio clears the previous transcript; **Use example voice** restores
the included pair. No separate ASR model or training dataset is needed for this demo.

GPU text is limited to 500 characters; CPU text to 150 characters and CPU
references to 10 seconds. Generation has a 600-second budget. **Stop** and the
deadline request cancellation after the current audio step and preserve any
partial output; they do not immediately interrupt a running CUDA operation.

## Troubleshooting and next-agent completion record

- **CUDA unavailable / Result says CPU:** verify `nvidia-smi` and torch inside the
  same environment. Install the GPU requirements with the CUDA backend, then
  restart the app; device selection is made when the module imports.
- **Missing files or offline errors:** finish the full pinned snapshot download
  into the exact directory above. Do not rely on another machine's Hugging Face cache.
- **CUDA out of memory:** close other GPU workloads and restart. Try a short
  sentence/reference first. Minimum VRAM has not been established; report the
  GPU model, free VRAM, and traceback if it still fails.
- **Silent audio / slow or poor generation:** check the recording and exact
  transcript. Use the bundled example to separate setup issues from input issues.
- **Dependency conflicts:** keep `constraints.txt`, the runtime commit, and
  torch/torchaudio pins together. Gradio 6.17.3 replaces upstream's unpublished
  6.17.0 pin. Re-run `uv pip check` after changes.

After successful setup, update this section (and `CHANGELOG.md` for code changes)
with OS, GPU/VRAM, driver, Python/torch/CUDA versions, code commit, model revision,
test results, generation time, output duration, and any fixes. GPU setup remains
unverified until the receiving agent completes the real generation above.
