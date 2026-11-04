#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
service=t1000-3050.service
action="${1:-start}"

case "$action" in
  stop|status|logs)
    case "$action" in
      stop) exec systemctl --user stop "$service" ;;
      status) exec systemctl --user status "$service" --no-pager ;;
      logs) exec journalctl --user -u "$service" -n 100 -f ;;
    esac
    ;;
  start|restart) ;;
  *) printf 'Usage: %s [start|stop|restart|status|logs]\n' "$0" >&2; exit 2 ;;
esac

if [[ ! -x "$project_dir/.venv/bin/python" || ! -f "$project_dir/apps/web/dist/index.html" ]]; then
  printf 'Install the Python environment and build the web app first; see README.md.\n' >&2
  exit 1
fi

# Both projects share a 4 GB GPU. Only stop the other project's named runtimes.
if [[ "$(systemctl --user show t2000.service -p LoadState --value 2>/dev/null || true)" == loaded ]]; then
  systemctl --user stop t2000.service
fi
if command -v docker >/dev/null 2>&1; then
  other_containers="$(docker ps -q \
    --filter label=com.docker.compose.project=t2000 \
    --filter label=com.docker.compose.service=studio)"
  if [[ -n "$other_containers" ]]; then
    mapfile -t container_ids <<< "$other_containers"
    docker stop --time 30 "${container_ids[@]}"
  fi
fi

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
unit_tmp="$(mktemp)"
trap 'rm -f -- "$unit_tmp"' EXIT
cat > "$unit_tmp" <<EOF
[Unit]
Description=T-1000 Voice Studio on RTX 3050
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$project_dir
Environment=STT_BACKEND=cpu
Environment=STT_NUM_THREADS=4
Environment=OMP_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
Environment=T1000_TTS_PROFILE=low-vram
Environment=T1000_TTS_OPTIMIZE=0
Environment=T1000_TTS_EAGER_CACHE=1
Environment=T1000_TTS_COMPILE_VOCODER=1
Environment=T1000_TTS_VOCODER_BATCH_SIZE=4
Environment=T1000_REFERENCE_TRIM=0
Environment=T1000_PRELOAD_MODELS=0
Environment=HF_HUB_OFFLINE=1
Environment=MAX_JOBS=2
Environment=TORCHINDUCTOR_COMPILE_THREADS=1
Environment="TORCHINDUCTOR_CACHE_DIR=$project_dir/.cache/torchinductor-soar-tests"
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=PYTHONUNBUFFERED=1
ExecStart="$project_dir/.venv/bin/python" -m t1000
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=t1000
LogRateLimitIntervalSec=30s
LogRateLimitBurst=2000
EOF

if ! cmp -s "$unit_tmp" "$unit_dir/$service"; then
  if [[ -f "$unit_dir/$service" && ! -f "$unit_dir/$service.before-maintenance" ]]; then
    cp -- "$unit_dir/$service" "$unit_dir/$service.before-maintenance"
  fi
  install -m 644 "$unit_tmp" "$unit_dir/$service"
  systemctl --user daemon-reload
fi
systemctl --user "$action" "$service"
"$project_dir/.venv/bin/python" - <<'PY'
import time
import urllib.request

deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=2) as response:
            if response.status == 200:
                print('T-1000 is ready: http://127.0.0.1:7860')
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('T-1000 did not become ready. Inspect: ./scripts/start.sh logs')
PY
