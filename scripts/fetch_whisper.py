#!/usr/bin/env python3
"""Download the pinned Distil-Whisper snapshot for optional CUDA transcription."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "distil-whisper/distil-large-v3.5-ct2"
REVISION = "9793ccc07920e0f830e1dba0343efcdf0ef8c903"
FILES = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("STT_GPU_MODEL_DIR", ROOT / "data" / "models" / "distil-large-v3.5-ct2")))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        allow_patterns=FILES,
        local_dir=str(args.model_dir),
    )
    missing = [name for name in FILES if not (Path(path) / name).is_file()]
    if missing:
        parser.exit(1, f"Incomplete CUDA transcription snapshot: {', '.join(missing)}\n")
    print(f"Downloaded CUDA transcription model revision {REVISION}: {path}")


if __name__ == "__main__":
    main()
