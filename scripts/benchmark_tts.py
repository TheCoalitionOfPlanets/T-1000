#!/usr/bin/env python3
"""Compare warm SOAR inference in separate CUDA runs, including low-VRAM modes.

Run this script twice with identical inputs, once per --mode. Model loading,
compile warmup, reference warmup, WAV writing, and optional stage profiling are
excluded from the reported warm generation times. Listen to the saved WAVs to
check voice similarity, delivery, and pronunciation alongside the speed results.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXAMPLE_TRANSCRIPT = "We asked over twenty different people, and they all said it was his."
DEFAULT_TEXT = "I didn't think you'd come. It's good to see you again."


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def seed_int(value):
    number = int(value)
    if not 0 <= number <= 4294967295:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("low-vram", "offload", "eager", "optimized"), default="low-vram", help="inference profile (default: low-vram)")
    parser.add_argument("--precision", choices=("auto", "float16", "bfloat16"), default="auto", help="auto selects BF16 when supported, otherwise FP16")
    parser.add_argument("--gpu-layers", type=nonnegative_int, help="offload only: Qwen blocks kept on GPU (default: 0; 14 is half of SOAR's 28 blocks)")
    parser.add_argument("--model-dir", type=Path, help="local SOAR directory (default: project data/models/dots.tts-soar)")
    parser.add_argument("--reference", type=Path, help="reference audio (default: project assets/ref.wav)")
    parser.add_argument("--transcript", help="exact reference words; required with --reference")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="new text to synthesize identically in each mode")
    parser.add_argument("--steps", type=positive_int, default=10, help="flow steps per audio patch (default: 10)")
    parser.add_argument("--seed", type=seed_int, default=42)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--warmup-runs", type=positive_int, default=1, help="unmeasured requests using the selected reference (default: 1)")
    parser.add_argument("--max-generate-length", type=positive_int, help="maximum audio patches (fixed at 128 for low-vram/offload; default: 300 for eager/optimized)")
    parser.add_argument("--vocoder-merge-steps", type=positive_int, help="optimized vocoder buffering (default: 1; other profiles require 1)")
    parser.add_argument("--output-dir", type=Path, help="WAVs and report.json directory (default: a new project data/output/benchmarks directory)")
    parser.add_argument("--profile", action="store_true", help="run an extra diagnostic request with synchronized stage profiling, excluded from speed measurements")
    return parser


def resolve_mode_options(args, parser):
    if args.mode in {"low-vram", "offload"}:
        if args.max_generate_length not in {None, 128}:
            parser.error(f"{args.mode} fixes --max-generate-length at 128")
        if args.vocoder_merge_steps not in {None, 1}:
            parser.error(f"{args.mode} requires --vocoder-merge-steps 1")
        args.max_generate_length = 128
    else:
        args.max_generate_length = args.max_generate_length or 300
    args.vocoder_merge_steps = args.vocoder_merge_steps or 1
    if args.mode == "eager" and args.vocoder_merge_steps != 1:
        parser.error("eager mode always uses one vocoder patch; use --vocoder-merge-steps 1")
    if args.mode != "offload" and args.gpu_layers is not None:
        parser.error("--gpu-layers is supported only with --mode offload")
    if args.mode == "offload" and args.gpu_layers is None:
        args.gpu_layers = 0


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _generate(runtime, args, torch, seed_everything, *, profile=False):
    """Time until host-ready PCM and stream completion, including preprocessing."""
    seed_everything(args.seed)
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    chunks = []
    first_chunk_seconds = None
    started = time.perf_counter()
    with torch.inference_mode():
        stream = runtime.generate_stream(
            text=args.text,
            prompt_audio_path=str(args.reference),
            prompt_text=args.transcript,
            language="en",
            num_steps=args.steps,
            guidance_scale=1.2,
            profile_inference=profile,
            log_profile_calls=False,
        )
        try:
            for chunk in stream:
                # The app also transfers each chunk to CPU. This synchronizes
                # the producing CUDA work and measures audio ready for serving.
                audio = chunk.detach().float().cpu().reshape(-1)
                if not audio.numel():
                    continue
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - started
                if not bool(torch.isfinite(audio).all()):
                    raise RuntimeError("The model produced non-finite audio.")
                chunks.append(audio)
        finally:
            stream.close()
    torch.cuda.synchronize(runtime.device)
    generation_seconds = time.perf_counter() - started
    if not chunks:
        raise RuntimeError("The model produced no audio.")
    audio_seconds = sum(chunk.numel() for chunk in chunks) / runtime.sample_rate
    result = {
        "first_chunk_seconds": first_chunk_seconds,
        "generation_seconds": generation_seconds,
        "audio_seconds": audio_seconds,
        "real_time_factor": generation_seconds / audio_seconds,
        "chunks": len(chunks),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
    }
    return result, torch.cat(chunks).numpy()


def run(args, parser):
    # Keep --help usable without importing torch, Gradio, or model dependencies.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import soundfile as sf
    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable. Run this benchmark on the GPU machine with CUDA-enabled PyTorch.\n")
    precision = args.precision
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.exit(1, "This GPU does not support BF16; use --precision auto or --precision float16.\n")

    from t1000.paths import ASSETS, MODELS, OUTPUT
    from dots_tts.utils.util import seed_everything

    args.model_dir = (args.model_dir or MODELS / "dots.tts-soar").expanduser().resolve()
    args.reference = (args.reference or ASSETS / "ref.wav").expanduser().resolve()
    if not (args.model_dir / "model.safetensors").is_file():
        parser.error(f"model weights are missing in {args.model_dir}; this script does not download models")
    if not args.reference.is_file():
        parser.error(f"reference audio is missing: {args.reference}")
    info = sf.info(args.reference)
    reference_limit = 6 if args.mode in {"low-vram", "offload"} else 30
    if not 1 <= info.duration <= reference_limit:
        parser.error(f"reference audio must be between 1 and {reference_limit} seconds in this mode")
    args.transcript = (args.transcript or EXAMPLE_TRANSCRIPT).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (args.output_dir or OUTPUT / "benchmarks" / f"{stamp}-{args.mode}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optimize = args.mode == "optimized"
    print(f"Loading local SOAR on {torch.cuda.get_device_name()} ({args.mode}, {precision}).", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.mode == "low-vram":
        from t1000.low_vram import load_low_vram_runtime

        runtime = load_low_vram_runtime(args.model_dir, precision=precision)
    elif args.mode == "offload":
        from t1000.offload import load_offloaded_runtime

        runtime = load_offloaded_runtime(args.model_dir, precision=precision, gpu_layers=args.gpu_layers)
    else:
        from dots_tts.runtime import DotsTtsRuntime

        runtime = DotsTtsRuntime.from_pretrained(
            str(args.model_dir),
            precision=precision,
            optimize=optimize,
            max_generate_length=args.max_generate_length,
            vocoder_merge_steps=args.vocoder_merge_steps,
            warmup_on_optimize=False,
        )
    torch.cuda.synchronize(runtime.device)
    model_load_seconds = time.perf_counter() - started
    load_peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated(runtime.device)
    load_peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved(runtime.device)
    compile_warmup_seconds = 0.0
    if optimize:
        print("Running compile warmup; this can take several minutes.", flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            runtime.run_warmup()
        torch.cuda.synchronize(runtime.device)
        compile_warmup_seconds = time.perf_counter() - started

    print(f"Running {args.warmup_runs} unmeasured request(s) to warm the selected voice and text.", flush=True)
    started = time.perf_counter()
    for _ in range(args.warmup_runs):
        _generate(runtime, args, torch, seed_everything)
    reference_warmup_seconds = time.perf_counter() - started

    runs = []
    for index in range(1, args.iterations + 1):
        measurement, audio = _generate(runtime, args, torch, seed_everything)
        audio_path = output_dir / f"{args.mode}-{index:02d}.wav"
        sf.write(audio_path, audio, runtime.sample_rate, subtype="PCM_16")
        measurement.update(iteration=index, wav_path=str(audio_path))
        runs.append(measurement)
        print(
            f"Run {index}: first PCM {measurement['first_chunk_seconds']:.3f}s; "
            f"generation {measurement['generation_seconds']:.3f}s; "
            f"audio {measurement['audio_seconds']:.3f}s; "
            f"RTF {measurement['real_time_factor']:.3f}.",
            flush=True,
        )

    profile_path = None
    if args.profile:
        from loguru import logger

        profile_path = output_dir / "profile.log"
        sink = logger.add(str(profile_path), level="INFO")
        try:
            print("Running a separate diagnostic profile; its timing is excluded from the benchmark.", flush=True)
            _generate(runtime, args, torch, seed_everything, profile=True)
        finally:
            logger.remove(sink)

    median = {
        name: statistics.median(run[name] for run in runs)
        for name in ("first_chunk_seconds", "generation_seconds", "audio_seconds", "real_time_factor")
    }
    report = {
        "mode": args.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(runtime.device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(runtime.device).total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dots_tts_version": _package_version("dots-tts"),
        "model_dir": str(args.model_dir),
        "reference": str(args.reference),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "transcript": args.transcript,
        "text": args.text,
        "language": "en",
        "precision": precision,
        "gpu_layers": args.gpu_layers,
        "steps": args.steps,
        "seed": args.seed,
        "guidance_scale": 1.2,
        "max_generate_length": args.max_generate_length,
        "max_sequence_length": runtime.max_sequence_length,
        "vocoder_merge_steps": args.vocoder_merge_steps if optimize else 1,
        "warmup_runs": args.warmup_runs,
        "model_load_seconds": model_load_seconds,
        "load_peak_cuda_allocated_bytes": load_peak_cuda_allocated_bytes,
        "load_peak_cuda_reserved_bytes": load_peak_cuda_reserved_bytes,
        "compile_warmup_seconds": compile_warmup_seconds,
        "reference_warmup_seconds": reference_warmup_seconds,
        "profile_log": str(profile_path) if profile_path else None,
        "runs": runs,
        "median": median,
        "measurement_notes": [
            "Each --mode invocation uses a separate process; match voice, text, precision, steps, seed, and length cap when comparing modes.",
            "Warm timings include reference preprocessing, synthesis, and per-chunk host transfer; exclude loading, warmup, file writing, and stage profiling.",
            "First PCM measures the backend; it excludes HTTP transport, browser buffering, and playback.",
            "Peak CUDA memory includes resident model and cached allocations, not only new request allocations.",
            "Load peak CUDA memory is captured before warmup, separately from each inference request's peak.",
            "RTF is generation seconds divided by output audio seconds; below 1 is faster than real time.",
            "A generation-length cap can truncate speech; listen to WAVs for completeness, pronunciation, similarity, and delivery.",
            "Identical seeds do not guarantee identical audio across compilation, quantization, or offload profiles.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Median: first PCM {median['first_chunk_seconds']:.3f}s; "
        f"generation {median['generation_seconds']:.3f}s; "
        f"RTF {median['real_time_factor']:.3f}.\nReport: {report_path}",
        flush=True,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must contain speech to synthesize")
    text_limit = 150 if args.mode in {"low-vram", "offload"} else 500
    if len(args.text.strip()) > text_limit:
        parser.error(f"--text must be at most {text_limit} characters in this mode")
    if args.reference is not None and not (args.transcript or "").strip():
        parser.error("--transcript with the exact reference words is required with --reference")
    if args.transcript is not None and not args.transcript.strip():
        parser.error("--transcript must not be empty")
    resolve_mode_options(args, parser)
    try:
        run(args, parser)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
