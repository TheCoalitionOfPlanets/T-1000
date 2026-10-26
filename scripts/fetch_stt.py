#!/usr/bin/env python3
"""Download the pinned Nemotron INT8 export, verify SHA-256, and install locally."""

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
ARCHIVE_NAME = MODEL_NAME + ".tar.bz2"
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + ARCHIVE_NAME
# Published by the upstream GitHub Releases API for asset 405278878.
SIZE = 463945051
SHA256 = "78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def download(archive):
    """Resume interrupted transfers, then verify the complete immutable artifact."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            partial.unlink()
            offset = 0
        if offset < SIZE:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("The download server returned an unexpected byte range.")
                else:
                    offset = 0
                print(f"Downloading {ARCHIVE_NAME} ({SIZE / 1e6:.0f} MB; resuming at {offset / 1e6:.0f} MB)…", flush=True)
                with partial.open("ab" if offset else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != SIZE:
            raise RuntimeError("The model download is incomplete. Run this command again to resume.")
        partial.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if archive.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        archive.unlink()
        raise RuntimeError("Model checksum verification failed. The cached archive was removed; rerun to retry.")


def install(model_dir, archive):
    """Extract only known regular files into an atomic staging directory."""
    marker = model_dir / ".archive-sha256"
    if marker.is_file() and marker.read_text().strip() == SHA256 and all((model_dir / f).is_file() for f in FILES):
        print(f"Speech recognition model is already installed: {model_dir}")
        return
    if model_dir.exists():
        raise RuntimeError(f"Unverified model directory already exists: {model_dir}. Move it aside before downloading.")
    download(archive)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stt-download-", dir=model_dir.parent) as staging:
        extracted = Path(staging) / MODEL_NAME
        extracted.mkdir()
        with tarfile.open(archive, "r:bz2") as source:
            for filename in FILES:
                member = source.getmember(f"{MODEL_NAME}/{filename}")
                if not member.isfile():
                    raise RuntimeError(f"The verified archive has an unexpected model entry: {filename}")
                with source.extractfile(member) as input_file, (extracted / filename).open("wb") as output:
                    shutil.copyfileobj(input_file, output)
        (extracted / ".archive-sha256").write_text(SHA256 + "\n")
        extracted.replace(model_dir)
    print(f"Verified and installed speech recognition model: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("STT_MODEL_DIR", ROOT / "data" / "models" / MODEL_NAME)))
    args = parser.parse_args()
    try:
        install(args.model_dir, ROOT / ".cache" / "stt" / ARCHIVE_NAME)
    except (OSError, ValueError, RuntimeError, tarfile.TarError, KeyError) as exc:
        parser.exit(1, f"Speech recognition download failed: {exc}\n")


if __name__ == "__main__":
    main()
