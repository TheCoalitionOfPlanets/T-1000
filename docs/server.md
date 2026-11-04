# SSH server startup

This launcher requires Linux with a systemd user session, the existing CUDA
Python environment at `.venv/`, downloaded models, and a built web app at
`apps/web/dist/`. Complete the [GPU setup](setup.md) with the low-VRAM profile
before using it on a new installation.

On `prieyan@10.10.1.164`, start the installed RTX 3050 runtime with:

```sh
cd ~/weeb/T-1000 && ./scripts/start.sh
```

The script installs or refreshes `t1000-3050.service` in the current user's
systemd configuration, starts it in the background, and waits for the API.
Repeated starts reuse the running service. It uses the existing Python
environment and model files; no model downloads or Docker volume copies occur.

Open <http://127.0.0.1:7860> on the server. From another computer, forward the
port while keeping the app's existing loopback binding:

```sh
ssh -N -L 7860:127.0.0.1:7860 prieyan@10.10.1.164
```

Then open <http://127.0.0.1:7860> on that computer.

```sh
./scripts/start.sh status
./scripts/start.sh logs
./scripts/start.sh restart
./scripts/start.sh stop
```

Logs go to the system journal with a per-service rate limit instead of an
unbounded project log file. The service restarts after a failure. Startup is
manual; it does not enable a GPU workload automatically at login or boot.

The low-VRAM SOAR profile uses CPU recognition and loads the synthesis model
on the first generation. Vocoder compilation reuses the existing cache. The
first request after starting can take longer than subsequent requests.

T-1000 and T-2000 share a 4 GB GPU. Starting or restarting T-1000 stops the
old `t2000.service` and the Docker service identified by Compose project
`t2000` and service `studio`, releasing its model memory. Starting T-2000
through its startup script stops T-1000. Use one active studio at a time;
other Docker projects are unaffected.

Keep `data/`, voice references under `.cache/recordings` and
`.cache/ui-references`, and the installed `.venv/`; they contain models, audio,
and the working inference dependencies. The original service file is preserved
as `~/.config/systemd/user/t1000-3050.service.before-maintenance`.
