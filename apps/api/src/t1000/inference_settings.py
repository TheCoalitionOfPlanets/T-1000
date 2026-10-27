"""Select inference placement without assuming that every CUDA GPU is large."""

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceSettings:
    profile: str
    precision: str
    optimize: bool
    reference_limit: int
    text_limit: int
    max_generate_length: int
    gpu_layers: int = 0


def get_inference_settings():
    use_cuda = torch.cuda.is_available()
    total_memory = torch.cuda.get_device_properties(0).total_memory if use_cuda else 0
    requested = os.environ.get("T1000_TTS_PROFILE", "auto").strip()
    if requested not in {"auto", "standard", "low-vram", "offload"}:
        raise ValueError("T1000_TTS_PROFILE must be auto, standard, low-vram, or offload.")
    profile = requested
    if profile == "auto":
        profile = "low-vram" if use_cuda and total_memory <= 6 * 2**30 else "standard"
    if profile != "standard" and not use_cuda:
        raise RuntimeError(f"The {profile} profile requires CUDA-enabled PyTorch and a visible GPU.")

    optimization = os.environ.get("T1000_TTS_OPTIMIZE", "auto").strip()
    if optimization not in {"auto", "0", "1"}:
        raise ValueError("T1000_TTS_OPTIMIZE must be auto, 0, or 1.")
    if profile != "standard" and optimization == "1":
        raise ValueError("Low-memory profiles require T1000_TTS_OPTIMIZE=0 or auto.")
    optimize = use_cuda and profile == "standard" and (
        optimization == "1" or (optimization == "auto" and total_memory >= 12 * 2**30)
    )

    try:
        gpu_layers = int(os.environ.get("T1000_TTS_GPU_LAYERS", "0"))
    except ValueError as exc:
        raise ValueError("T1000_TTS_GPU_LAYERS must be a nonnegative integer.") from exc
    if gpu_layers < 0:
        raise ValueError("T1000_TTS_GPU_LAYERS must be a nonnegative integer.")
    if gpu_layers and profile != "offload":
        raise ValueError("Nonzero T1000_TTS_GPU_LAYERS requires the offload profile.")

    precision = "float32"
    if use_cuda:
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    small = profile != "standard"
    return InferenceSettings(
        profile=profile,
        precision=precision,
        optimize=optimize,
        reference_limit=6 if small else (30 if use_cuda else 10),
        text_limit=150 if small or not use_cuda else 500,
        max_generate_length=128 if small else (300 if use_cuda else 100),
        gpu_layers=gpu_layers,
    )
