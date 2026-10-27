# Run T-1000 with Docker

Run these commands from the repository root. Choose the CPU or NVIDIA GPU
Compose file; they build separate images and keep separate model volumes.
Docker builds the web interface and installs Python dependencies, so a host
Python environment and Node.js installation are unnecessary.

## CPU

Install Docker Engine or Docker Desktop with the Compose plugin, then run:

```bash
docker compose -f compose.cpu.yaml up --build -d
docker compose -f compose.cpu.yaml logs -f t1000
```

After model download and preload finish, open
[http://localhost:7860](http://localhost:7860). CPU synthesis can take several
minutes for a short sentence; the CPU image is useful for setup and functional
checks, and does not make the 2B SOAR model fast on a laptop.

## NVIDIA GPU

The host needs an NVIDIA GPU, a compatible NVIDIA driver, and Docker configured
with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
Follow NVIDIA's installation and Docker runtime configuration instructions for
your operating system. Check that `nvidia-smi` works on the host first.

The GPU image installs PyTorch 2.8.0 and torchaudio 2.8.0 CUDA 12.8 wheels. Use a
driver compatible with CUDA 12.8; NVIDIA documents corresponding driver versions
and the limits of compatibility in the
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
and [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/).
The image supplies its CUDA runtime libraries; a separate host CUDA Toolkit is
not required. Compose requests one NVIDIA GPU using Docker's
[GPU device reservation support](https://docs.docker.com/compose/how-tos/gpu-support/).

Allow sufficient host RAM too: SOAR is constructed in CPU memory before GPU
placement, and offloaded layers remain backed by RAM. A 4 GB VRAM target does
not establish a 4 GB system-memory requirement. Docker Desktop's memory limit
must accommodate model loading; minimum host RAM has not been benchmarked.

```bash
docker compose -f compose.gpu.yaml up --build -d
docker compose -f compose.gpu.yaml logs -f t1000
```

Open [http://localhost:7860](http://localhost:7860) after startup completes.
The GPU service requires CUDA to be available and fails startup if the GPU is
unavailable; it does not silently switch to CPU synthesis. Both images use the
CPU streaming recognizer, leaving GPU memory for TTS.

To build the image and check CUDA access independently of model downloads and
preload:

```bash
docker compose -f compose.gpu.yaml build
docker compose -f compose.gpu.yaml run --rm --no-deps --entrypoint python t1000 -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name())"
```

## First startup and subsequent starts

Model weights are downloaded at container startup, not baked into either image.
The initial download includes approximately:

| Model | Download and pin |
| :--- | :--- |
| SOAR TTS | About 5.2 GB, pinned to revision `2f9b3e18d70d670d4c701da2dc55ded5755815ce`. |
| CPU Nemotron ASR | About 464 MB compressed, roughly 662 MB installed; the downloader verifies the pinned archive's SHA-256. |

Allow additional disk space for the Docker images, download caches, and generated
audio. The first startup needs internet access. It downloads missing models,
checks the local files, and preloads TTS and recognition before starting the web
server. GPU compilation, when selected, adds a one-time warmup. Follow the logs
while Docker reports the service as `starting`.

The GPU image includes the host compiler needed by optimized inference. Its
TorchInductor, Triton and CUDA kernel caches live under `/app/.cache` and persist
with the cache volume.

The startup command is `python -m t1000.container`. Its controls are environment
variables, not command-line download flags:

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `T1000_DOWNLOAD_MODELS` | `1` | Download the pinned models when needed. A complete SOAR snapshot with the matching `.t1000-revision` marker skips repeat downloads. |
| `T1000_PRELOAD_MODELS` | `1` | Load and check TTS and CPU recognition before serving the UI. |
| `T1000_PORT` | `7860` | Change the host port; the container still listens on 7860. |
| `T1000_CONTAINER_DEVICE` | Set by Compose | `cpu` in the CPU file, `cuda` in the GPU file. Choose the matching file. |

Set `T1000_DOWNLOAD_MODELS=0` to use complete model files already in the mounted
volume without downloading. Startup checks the required files and fails if they
are missing. A copied snapshot without the revision marker can be used this way,
but file presence alone cannot establish that it matches the pinned revision.

For example, once the CPU volume contains both models:

```bash
T1000_DOWNLOAD_MODELS=0 docker compose -f compose.cpu.yaml up -d
```

`T1000_PRELOAD_MODELS=0` is for diagnosing startup problems. It permits the web
server to start without preloading; TTS still loads on the first generation, so
this moves the delay and any loading error into that request. It does not reduce
inference memory requirements.

```bash
T1000_PRELOAD_MODELS=0 docker compose -f compose.gpu.yaml up -d
```

Repeat your environment overrides on subsequent `up` commands, or put them in a
repository-root `.env` file for Compose to read. Inspect status with:

```bash
docker compose -f compose.gpu.yaml ps
```

The healthcheck calls `/api/health`. With default preload enabled, the server and
health endpoint become available after model preparation finishes. The images
allow a 30-minute healthcheck startup period; slow downloads can take longer.

## Persistent data and ports

The application runs as UID/GID `10001:10001`. Fresh named volumes receive the
image's writable directory ownership.

| Deployment | Persistent volume | Container path | Contents |
| :--- | :--- | :--- | :--- |
| CPU | `t1000-cpu_data` | `/app/data` | Downloaded models and generated WAVs. |
| CPU | `t1000-cpu_cache` | `/app/.cache` | Download/runtime caches and reference audio. |
| GPU | `t1000-gpu_data` | `/app/data` | Downloaded models and generated WAVs. |
| GPU | `t1000-gpu_cache` | `/app/.cache` | Download/runtime caches and reference audio. |

Those names assume the Compose files' default project names. CPU and GPU do not
share weights automatically; each deployment has its own volumes. Stopping,
rebuilding, and recreating containers preserves them:

```bash
docker compose -f compose.cpu.yaml down
docker compose -f compose.gpu.yaml down
```

Do not add `-v` to routine shutdown commands: `down -v` deletes the named volumes,
including downloaded models and saved audio.

Both deployments bind to localhost. The app has no authentication; keep that
binding. To run CPU and GPU on the same host, give one a different host port:

```bash
T1000_PORT=7861 docker compose -f compose.gpu.yaml up --build -d
```

That GPU instance is available at [http://localhost:7861](http://localhost:7861).

## Reuse models already downloaded on the host

If this checkout already has complete models under `data/models`, copy them into
the selected named volume before the first normal start. These commands use the
CPU deployment; replace every `compose.cpu.yaml` with `compose.gpu.yaml` to seed
the GPU deployment instead.

```bash
docker compose -f compose.cpu.yaml build
docker compose -f compose.cpu.yaml create t1000
docker compose -f compose.cpu.yaml cp ./data/models/. t1000:/app/data/models/
docker compose -f compose.cpu.yaml run --rm --no-deps --user 0:0 --entrypoint chown t1000 -R 10001:10001 /app/data
T1000_DOWNLOAD_MODELS=0 docker compose -f compose.cpu.yaml up -d
```

`create` makes a stopped container and its volumes without starting model
downloads. The copy includes the SOAR and CPU ASR directories, along with any
other models you keep under that host directory. The one-time ownership command
makes copied files writable by the normal application user; the service itself
continues to run as UID 10001. Do this before starting the service, or stop it
before replacing its model files. Docker documents the copy command in its
[Compose CLI reference](https://docs.docker.com/reference/cli/docker/compose/cp/).

Only disable downloads when both models are complete. See the pinned model and
recognizer instructions in [setup.md](setup.md) if the local directories are
incomplete or their provenance is uncertain.

## Experimental 4–6 GB GPU profiles

The GPU image includes the optional bitsandbytes and Accelerate dependencies.
`T1000_TTS_PROFILE=auto` selects experimental `low-vram` inference when the visible
GPU has 6 GiB or less. This quantizes the language backbone to NF4 and keeps
compilation disabled. Whether SOAR actually fits, runs quickly, and preserves
your voice and delivery on a particular 4–6 GB GPU remains unverified.

For unquantized weights with CPU-backed language-model layers, select offload:

```bash
T1000_TTS_PROFILE=offload T1000_TTS_GPU_LAYERS=0 docker compose -f compose.gpu.yaml up -d
```

Zero keeps all decoder blocks backed by CPU RAM and stages them through the GPU.
To retain half of SOAR's 28 decoder blocks on GPU:

```bash
T1000_TTS_PROFILE=offload T1000_TTS_GPU_LAYERS=14 docker compose -f compose.gpu.yaml up -d
```

Fourteen resident blocks need more VRAM and may not fit on 4 GB. Offloading
preserves the unquantized weights but adds transfers and can be slower. Both
low-memory profiles limit references to six seconds and text to 150 characters;
their audio-generation cap can truncate long utterances. Listen for complete
sentences and compare tone, fluency, and similarity. See
[inference profiles and limitations](setup.md#gpu-inference-performance) for details.

## Measure GPU inference inside the container

Prepare the GPU model volume through normal startup or the copy procedure above,
then stop the UI service so another loaded TTS model does not occupy VRAM. The
benchmark runs in a separate container using the same volumes:

```bash
docker compose -f compose.gpu.yaml stop t1000
docker compose -f compose.gpu.yaml run --rm --no-deps --entrypoint python t1000 scripts/benchmark_tts.py --mode low-vram --output-dir data/output/bench-nf4
docker compose -f compose.gpu.yaml run --rm --no-deps --entrypoint python t1000 scripts/benchmark_tts.py --mode offload --gpu-layers 0 --output-dir data/output/bench-offload
```

The explicit Python entrypoint bypasses model-download and preload startup;
these commands require the models to exist already. Reports and sample WAVs
persist under `/app/data/output` in the GPU data volume. Use the same reference,
transcript, text, precision, seed, and step count across runs. For comparisons
against `eager` or `optimized`, also pass `--max-generate-length 128` to match the
low-memory profiles. Stage profiling is a separate diagnostic run, excluded from
the reported warm timings.

Use `--gpu-layers 14` to benchmark partial offload when sufficient VRAM is
available. Runtime fit and speed need measurement on your GPU; these commands do
not establish a minimum VRAM requirement. The benchmark's first-audio measurement
is backend timing; the current browser plays the completed WAV after synthesis.

Restart the UI with the desired profile after benchmarking:

```bash
docker compose -f compose.gpu.yaml up -d
```
