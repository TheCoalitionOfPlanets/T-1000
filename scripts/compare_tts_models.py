#!/usr/bin/env python3
"""Measure matched SOAR/MF2 requests in separate, eager NF4 CUDA processes.

Each invocation loads one checkpoint. Input reading/resampling/schedule assembly,
model loading, warmup and WAV writing are reported separately from generation.
Speaker conditioning and host-ready PCM transfer remain inside generation time.
128 total acoustic spans include reference prefill and a discarded prompt tail;
schedule exhaustion is explicitly recorded and must not count as a speed win.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=("soar", "mf2"))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=1,
                        help="unmeasured requests per unique reference (default: 1)")
    return parser


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    if isinstance(document, list):
        document = {"metadata": {}, "cases": document}
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError("Manifest must be an object with a cases array or an array of cases.")
    cases, ids = [], set()
    for source in document["cases"]:
        if not isinstance(source, dict):
            raise ValueError("Each manifest case must be an object.")
        case = dict(source)
        identity = case.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
            raise ValueError("Case id must be a safe, nonempty filename component.")
        if identity in ids:
            raise ValueError(f"Duplicate case id: {identity}")
        ids.add(identity)
        for key in ("reference", "transcript", "text"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Case {identity} requires nonempty {key}.")
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
            raise ValueError(f"Case {identity} requires a seed between 0 and 4294967295.")
        reference = Path(case["reference"]).expanduser()
        case["reference"] = str((path.parent / reference).resolve())
        if not Path(case["reference"]).is_file():
            raise ValueError(f"Reference is missing for {identity}: {case['reference']}")
        expected = case.get("spoken_expected_text", case["text"])
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Case {identity} requires nonempty spoken_expected_text.")
        case["spoken_expected_text"] = expected
        case.setdefault("reference_id", Path(case["reference"]).stem)
        cases.append(case)
    if not cases:
        raise ValueError("Manifest has no cases.")
    return {"metadata": document.get("metadata", {}), "cases": cases}


def generation_options(case, label):
    options = {
        "text": case["text"], "prompt_audio_path": case["reference"],
        "prompt_text": case["transcript"], "language": "en",
        "normalize_text": False, "profile_inference": False, "log_profile_calls": False,
    }
    if label == "soar":
        options.update(num_steps=10, guidance_scale=1.2, ode_method="euler")
    elif label != "mf2":
        raise ValueError(f"Unknown model label: {label}")
    # MF2 resolves its trained 2-step/CFG0/Euler/tau1.3 contract from config.
    return options


class DecodeObserver:
    """Observe existing CPU end_flag only after the decode generator exhausts."""

    def __init__(self, model):
        self.original = model._decode
        self.stop_reason = "not_started"
        self.decoded_patches = 0

    def reset(self):
        self.stop_reason = "not_started"
        self.decoded_patches = 0

    def __call__(self, *args, **kwargs):
        state = kwargs["state"]
        self.stop_reason = "decoding"
        stream = self.original(*args, **kwargs)
        try:
            for patch in stream:
                self.decoded_patches += 1
                yield patch
        except GeneratorExit:
            self.stop_reason = "closed"
            raise
        except BaseException:
            self.stop_reason = "error"
            raise
        else:
            self.stop_reason = "eos" if state.end_flag else "schedule_exhausted"
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def measure_request(runtime, case, label, observer, torch, seed_everything):
    options = generation_options(case, label)
    preparation_started = time.perf_counter()
    prepared = runtime._prepare_inputs(
        text=options["text"], prompt_audio_path=options["prompt_audio_path"],
        prompt_text=options["prompt_text"], template_name=None, language="en",
        normalize_text=False,
    )
    torch.cuda.synchronize(runtime.device)
    preparation_seconds = time.perf_counter() - preparation_started
    original_prepare = runtime._prepare_inputs
    runtime._prepare_inputs = lambda **unused: prepared
    observer.reset()
    chunks, first_chunk_seconds, error = [], None, None
    seed_everything(case["seed"])
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            stream = runtime.generate_stream(**options)
            try:
                for chunk in stream:
                    audio = chunk.detach().float().cpu().reshape(-1)
                    if not audio.numel():
                        continue
                    if first_chunk_seconds is None:
                        first_chunk_seconds = time.perf_counter() - started
                    chunks.append(audio)
                    if not bool(torch.isfinite(audio).all()):
                        raise RuntimeError("The model produced non-finite PCM.")
            finally:
                stream.close()
        torch.cuda.synchronize(runtime.device)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        runtime._prepare_inputs = original_prepare
    generation_seconds = time.perf_counter() - started
    audio = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
    finite = bool(torch.isfinite(audio).all())
    nonzero = bool(torch.count_nonzero(audio)) if audio.numel() else False
    if error is None and (not audio.numel() or not nonzero):
        error = "RuntimeError: The model produced empty or entirely zero audio."
    if error is None and observer.stop_reason not in {"eos", "schedule_exhausted"}:
        error = f"RuntimeError: Decode did not exhaust: {observer.stop_reason}."
    audio_seconds = audio.numel() / runtime.sample_rate
    result = {
        "status": "error" if error else "ok", "error": error,
        "stop_reason": "error" if error else observer.stop_reason,
        "observed_decode_stop_reason": observer.stop_reason,
        "decoded_patches_including_regenerated_tail": observer.decoded_patches,
        "input_preparation_seconds": preparation_seconds,
        "first_chunk_seconds": first_chunk_seconds,
        "generation_seconds": generation_seconds,
        "audio_seconds": audio_seconds,
        "real_time_factor": generation_seconds / audio_seconds if audio_seconds else None,
        "chunks": len(chunks), "sample_rate": runtime.sample_rate,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
        "finite": finite, "nonzero": nonzero,
        "peak_abs": float(audio.abs().max()) if audio.numel() and finite else None,
        "clipped_sample_count": int((audio.abs() >= 1).sum()) if finite else None,
    }
    return result, audio.numpy()


def run(args):
    # Explicit matched setup; do not inherit a different shell profile.
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      T1000_TTS_VOCODER_BATCH_SIZE="4", T1000_REFERENCE_TRIM="0",
                      T1000_TTS_EAGER_CACHE="1")
    import soundfile as sf
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This controlled benchmark requires CUDA and BF16 support.")
    from dots_tts.utils.util import seed_everything
    from t1000.low_vram import load_low_vram_runtime
    from t1000.references import configure_prompt_audio
    from t1000.eager import configure_eager_cache

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise ValueError(f"Refusing to overwrite an existing benchmark: {report_path}")
    for case in manifest["cases"]:
        path = Path(case["reference"])
        info = sf.info(path)
        if not 1 <= info.duration <= 6:
            raise ValueError(f"Reference must be a complete 1–6 second clip: {path} ({info.duration}s)")
        case.update(reference_seconds=info.duration,
                    reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    model_path = args.model_dir.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text())
    if args.label == "mf2":
        expected = {"solver": "scm", "ode_method": "euler", "num_steps": 2,
                    "guidance_scale": 0.0, "tau_mid": 1.3}
        if config.get("sampling") != expected:
            raise ValueError(f"MF2 sampling contract differs: {config.get('sampling')!r}")
    report = {
        "schema_version": 1, "label": args.label, "model_revision": args.model_revision,
        "model_dir": str(model_path), "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_metadata": manifest["metadata"], "cases": manifest["cases"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "precision": "bfloat16", "language_model_weights": "NF4",
            "acoustic_core_weights": "bfloat16", "optimize": False,
            "vocoder_batch_size": 4, "reference_trim": False,
            "eager_conditioning_cache": args.label == "soar", "max_generate_length": 128,
            "max_sequence_length": 512, "repeats": args.repeats,
            "warmup_runs_per_reference": args.warmup_runs,
            "sampling": config.get("sampling") if args.label == "mf2" else {
                "solver": "flow_matching", "ode_method": "euler", "num_steps": 10, "guidance_scale": 1.2},
            "generation_timing": "Input reading/resampling/schedule assembly, seeding, load, warmup and WAV writing excluded; speaker conditioning and CPU PCM transfer included.",
            "cap_note": "128 total acoustic spans include reference prefill and discarded regenerated prompt tail; schedule_exhausted is censored/incomplete.",
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            **{name: package_version(name) for name in ("dots-tts", "bitsandbytes", "torchaudio", "accelerate")},
        },
        "warmups": [], "runs": [], "status": "loading",
    }
    write_report(report_path, report)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        runtime = load_low_vram_runtime(model_path, precision="bfloat16")
        report["configuration"]["loaded_parameter_dtypes"] = {
            "acoustic_core": str(next(runtime.model.core.patch_encoder.parameters()).dtype),
            "speaker_encoder": str(next(runtime.model.xvector_extractor.parameters()).dtype),
            "vocoder": str(next(runtime.model.vocoder.parameters()).dtype),
        }
        configure_prompt_audio(runtime)
        configure_eager_cache(runtime, args.label == "soar")
        observer = DecodeObserver(runtime.model)
        runtime.model._decode = observer
        torch.cuda.synchronize(runtime.device)
        report["model_load"] = {
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
        }
        report["status"] = "running"
        write_report(report_path, report)
        warmed = set()
        for case in manifest["cases"]:
            # Same audio with different transcripts is a different conditioning input.
            voice = (case["reference_sha256"], case["transcript"])
            if voice not in warmed:
                for warmup in range(1, args.warmup_runs + 1):
                    result, _ = measure_request(runtime, case, args.label, observer, torch, seed_everything)
                    result.update(case_id=case["id"], warmup=warmup)
                    report["warmups"].append(result)
                    write_report(report_path, report)
                    print(f"Warmup {case['reference_id']} {warmup}: {result['status']} {result['stop_reason']}", flush=True)
                warmed.add(voice)
            for repeat in range(1, args.repeats + 1):
                try:
                    result, audio = measure_request(runtime, case, args.label, observer, torch, seed_everything)
                except Exception as exc:
                    result, audio = {"status": "error", "stop_reason": "error",
                                     "error": f"{type(exc).__name__}: {exc}"}, None
                result.update(case_id=case["id"], reference_id=case["reference_id"], repeat=repeat,
                              text=case["text"], spoken_expected_text=case["spoken_expected_text"], seed=case["seed"])
                if audio is not None and len(audio) and result.get("finite"):
                    name = f"{case['id']}-r{repeat:02d}.wav"
                    sf.write(output / name, audio, runtime.sample_rate, subtype="PCM_16")
                    result["wav_path"] = name
                report["runs"].append(result)
                write_report(report_path, report)
                print(json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "complete" if all(run["status"] == "ok" for run in report["runs"]) else "complete_with_errors"
    except BaseException as exc:
        report.update(status="failed", fatal_error=f"{type(exc).__name__}: {exc}")
        write_report(report_path, report)
        raise
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(report_path, report)
    return report


if __name__ == "__main__":
    run(build_parser().parse_args())
