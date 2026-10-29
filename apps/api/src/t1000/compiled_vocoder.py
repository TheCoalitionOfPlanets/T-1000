"""Opt-in vocoder-only compilation and startup warmup for eager NF4 CUDA.

The acoustic decoder, sampling schedule, vocoder precision and buffering stay
unchanged. Compilation is the same isolated, no-CUDA-graph configuration tested
by the SOAR experiment runner. A throwaway stream warms the actual outer audio
adapter, including its final partial group, before the runtime becomes ready.
"""

import os
import logging
import time

import torch


def read_compiled_vocoder_enabled():
    value = os.environ.get("T1000_TTS_COMPILE_VOCODER", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("T1000_TTS_COMPILE_VOCODER must be 0 or 1.")
    return value == "1"


def configure_compiled_vocoder(runtime, *, enabled=None):
    """Compile/warm once; disabled runtimes are returned without inspection.

    Call only after loading the eager NF4 runtime and installing its buffering.
    Errors propagate: a failed runtime must be discarded rather than retried
    with a second compiler wrapper or silently used as an eager fallback.
    """
    if enabled is None:
        enabled = read_compiled_vocoder_enabled()
    if not isinstance(enabled, bool):
        raise ValueError("The compiled vocoder flag must be a boolean.")
    if not enabled:
        return runtime
    if (runtime.device.type != "cuda" or runtime.optimize
            or runtime.model._optimize_enabled):
        raise ValueError("Compiled vocoding requires an eager low-VRAM CUDA runtime.")
    previous = getattr(runtime, "_t1000_compiled_vocoder", None)
    if previous is not None:
        if previous["status"] == "ready":
            return runtime
        raise RuntimeError("Compiled vocoder setup did not complete; load a fresh runtime.")

    from .vocoder_experiments import configure_vocoder_experiment

    status = {"status": "warming"}
    runtime._t1000_compiled_vocoder = status
    started = time.perf_counter()
    try:
        status["configuration"] = configure_vocoder_experiment(runtime, "compiled-vocoder")
        adapter = runtime.model._get_vocoder_inference()
        # Match inference-mode tensor construction used by real generation;
        # warming only an inner decoder can otherwise leave a first-use graph
        # specialization or buffered flush path until the user's request.
        with torch.inference_mode():
            patch_frames = int(runtime.model.core.latent_patch_size)
            latent_dim = int(runtime.model.core.latent_dim)
            state = adapter.init_stream_state(batch_size=1, chunk_size=patch_frames)
            latent = torch.zeros(
                (1, patch_frames, latent_dim), device=runtime.device,
                dtype=torch.bfloat16,
            )
            # Nine patches cover first-two cadence, a full group of four and a
            # final group of three on the validated production batch size.
            for _ in range(9):
                adapter.stream_step(latent, state, optimize=False, use_compiled=True)
            adapter.flush(state)
        torch.cuda.synchronize(runtime.device)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    status.update(status="ready", warmup_seconds=time.perf_counter() - started,
                  warmup_latent_patches=9)
    logging.getLogger(__name__).info("Vocoder-only compilation ready after %.3fs startup warmup (FP32, CUDA graphs disabled).", status["warmup_seconds"])
    return runtime
