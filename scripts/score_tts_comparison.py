#!/usr/bin/env python3
"""Score matched TTS WAVs on CPU; write objective proxies and listening links.

ASR errors measure recognizer agreement, not human-rated intelligibility. Speaker
cosine uses the TTS conditioning encoder itself, not an independent evaluator.
No random crops, text expansions, silence removal, or loudness adjustment.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


def normalized_text(text):
    """Lowercase, remove punctuation, collapse whitespace; preserve digits."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c for c in text if not unicodedata.category(c).startswith(("P", "S"))).split())


def edit_counts(reference, hypothesis):
    """Levenshtein counts with deterministic ties: match, substitute, delete, insert."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
                continue
            substitution = previous[j - 1]
            deletion = previous[j]
            insertion = current[j - 1]
            options = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            current.append(min(options, key=lambda x: x[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"errors": errors, "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "reference_units": len(reference),
            "rate": errors / len(reference) if reference else None}


def error_metrics(expected, recognized):
    reference = normalized_text(expected)
    hypothesis = normalized_text(recognized)
    if not reference:
        raise ValueError("Expected speech text must contain at least one scored unit.")
    return {"expected_normalized": reference, "recognized_normalized": hypothesis,
            "word": edit_counts(reference.split(), hypothesis.split()),
            "character": edit_counts(reference.replace(" ", ""), hypothesis.replace(" ", ""))}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible(row):
    """Capped, invalid, partial, or unscored requests are never speed successes."""
    return (row.get("status") == "ok" and row.get("stop_reason") == "eos"
            and row.get("finite") is True and row.get("nonzero") is True
            and row.get("score_status") == "ok")


def aggregate(rows):
    scored = [r for r in rows if r.get("score_status") == "ok"]
    completed = [r for r in rows if eligible(r)]
    result = {"requests": len(rows), "scored_requests": len(scored), "eligible_eos_requests": len(completed),
              "status_counts": dict(Counter(r.get("status", "unknown") for r in rows)),
              "stop_reason_counts": dict(Counter(r.get("stop_reason", "unknown") for r in rows))}
    # Include scoreable partials in ASR results rather than silently dropping errors.
    for unit in ("word", "character"):
        totals = {key: sum(r["asr"][unit][key] for r in scored)
                  for key in ("errors", "substitutions", "deletions", "insertions", "reference_units")}
        totals["rate"] = totals["errors"] / totals["reference_units"] if totals["reference_units"] else None
        result[unit] = totals
    for field in ("generation_seconds", "first_chunk_seconds", "audio_seconds", "real_time_factor",
                  "speaker_cosine", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        values = [float(r[field]) for r in completed if r.get(field) is not None and math.isfinite(float(r[field]))]
        result[field] = {"median": statistics.median(values), "mean": statistics.mean(values)} if values else None
    return result


def paired_comparisons(rows, labels):
    grouped = {}
    for row in rows:
        key = (row["case_id"], row["repeat"])
        group = grouped.setdefault(key, {})
        if row["label"] in group:
            raise ValueError(f"Duplicate result for {key}: {row['label']}")
        group[row["label"]] = row
    output = []
    for (case_id, repeat), group in sorted(grouped.items()):
        pair = {"case_id": case_id, "repeat": repeat, "labels": labels,
                "eligible": all(label in group and eligible(group[label]) for label in labels)}
        if pair["eligible"] and len(labels) == 2:
            first, second = (group[label] for label in labels)
            pair.update(generation_speedup_first_over_second=first["generation_seconds"] / second["generation_seconds"],
                        rtf_speedup_first_over_second=first["real_time_factor"] / second["real_time_factor"],
                        second_over_first_audio_duration=second["audio_seconds"] / first["audio_seconds"],
                        second_minus_first_speaker_cosine=second["speaker_cosine"] - first["speaker_cosine"])
        else:
            pair["exclusion_reason"] = "Missing, failed, unscored, invalid, or no observed EOS; inspect individual rows."
        output.append(pair)
    return output


class CpuSpeakerScorer:
    def __init__(self, model_dir):
        import torch
        from safetensors.torch import load_file
        from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures

        self.torch = torch
        config = json.loads((model_dir / "config.json").read_text())
        self.sample_rate = int(config["vocoder"]["sample_rate"])
        self.encoder = SpeakerXVectorFeatures(sample_rate=self.sample_rate,
                                             campplus_embedding_size=int(config["campplus_embedding_size"]),
                                             max_audio_seconds=0).cpu().eval()
        self.encoder.load_state_dict(load_file(model_dir / "speaker_encoder.safetensors", device="cpu"), strict=True)
        self.cache = {}

    def embedding(self, path):
        import numpy as np
        import soundfile as sf
        import soxr

        key = sha256(path)
        if key in self.cache:
            return self.cache[key]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
        if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError(f"Empty, silent, or invalid audio: {path}")
        if rate != self.sample_rate:
            audio = soxr.resample(audio, rate, self.sample_rate, quality="HQ")
        with self.torch.inference_mode():
            vector = self.encoder(self.torch.from_numpy(audio).unsqueeze(0)).float().reshape(-1)
            norm = vector.norm()
            if not self.torch.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid speaker embedding")
            vector = vector / norm
        self.cache[key] = vector
        return vector

    def cosine(self, reference, generated):
        return float(self.torch.dot(self.embedding(reference), self.embedding(generated)).clamp(-1, 1))


def relative_link(path, base):
    return Path(os.path.relpath(path, base)).as_posix().replace(" ", "%20")


def write_products(result, output_dir):
    (output_dir / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["label", "case_id", "reference_id", "repeat", "status", "stop_reason", "score_status", "text",
              "expected_text", "recognized_text", "word_errors", "reference_words", "wer", "cer", "speaker_cosine",
              "generation_seconds", "first_chunk_seconds", "audio_seconds", "real_time_factor", "wav_path", "score_error"]
    with (output_dir / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["rows"]:
            flat = dict(row)
            if "asr" in row:
                flat.update(word_errors=row["asr"]["word"]["errors"], reference_words=row["asr"]["word"]["reference_units"],
                            wer=row["asr"]["word"]["rate"], cer=row["asr"]["character"]["rate"])
            writer.writerow(flat)
    lines = ["# Matched speech model comparison", "", result["limitations"], "",
             "Only finite, non-silent, successfully scored requests with observed EOS enter latency summaries and speed ratios. "
             "ASR totals include all scoreable WAVs, including partial outputs. Unscoreable requests are counted separately.", "",
             "| Model | Scored / requests | EOS eligible | ASR word errors / words | WER | Median generation | Median RTF | Speaker cosine proxy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"
    for label, summary in result["summaries"].items():
        def median(field):
            return summary[field]["median"] if summary[field] else None
        lines.append(f"| {label} | {summary['scored_requests']} / {summary['requests']} | {summary['eligible_eos_requests']} | "
                     f"{summary['word']['errors']} / {summary['word']['reference_units']} | {fmt(summary['word']['rate'])} | "
                     f"{fmt(median('generation_seconds'))} s | {fmt(median('real_time_factor'))} | {fmt(median('speaker_cosine'))} |")
    accepted = [p for p in result["pairs"] if p["eligible"]]
    if accepted and len(result["labels"]) == 2:
        speedup = statistics.median(p["generation_speedup_first_over_second"] for p in accepted)
        rtf_speedup = statistics.median(p["rtf_speedup_first_over_second"] for p in accepted)
        duration = statistics.median(p["second_over_first_audio_duration"] for p in accepted)
        lines += ["", f"Across {len(accepted)} eligible pairs, median {result['labels'][0]} / {result['labels'][1]} generation ratio "
                  f"is {speedup:.3f}×; RTF ratio {rtf_speedup:.3f}×; second / first audio duration {duration:.3f}×."]
    lines += ["", "## Paired listening and transcripts", "", "Listen for voice match, naturalness, pronunciation, repeated or missing words, and artifacts. "
              "These files have not been assigned human quality ratings.", ""]
    groups = {}
    for row in result["rows"]:
        groups.setdefault((row["case_id"], row["repeat"]), []).append(row)
    for (case_id, repeat), rows in sorted(groups.items()):
        lines += [f"### {case_id}, repeat {repeat}", "", f"Expected: {rows[0]['expected_text']}", ""]
        reference = rows[0].get("reference_path")
        if reference:
            lines += [f"[Reference audio]({relative_link(reference, output_dir)})", ""]
        for row in rows:
            wav = row.get("wav_path")
            link = f"[{row['label']} WAV]({relative_link(wav, output_dir)})" if wav else row["label"]
            lines += [f"- {link}: {row.get('stop_reason', 'unknown')}; ASR: {row.get('recognized_text', '[not scored]')}. "
                      f"WER {fmt(row.get('asr', {}).get('word', {}).get('rate'))}; cosine proxy {fmt(row.get('speaker_cosine'))}."
                      + (f" Error: {row['score_error']}" if row.get("score_error") else "")]
        lines += [""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines))


def run(args):
    os.environ["STT_BACKEND"] = "cpu"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from t1000.stt import MODEL_DIR, MODEL_FILES, transcribe_file

    torch.set_num_threads(args.cpu_threads)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases") if isinstance(manifest, dict) else manifest
    case_map = {}
    for case in cases:
        if case["id"] in case_map:
            raise ValueError(f"Duplicate manifest case ID: {case['id']}")
        case_map[case["id"]] = case
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_dir = args.speaker_model_dir.expanduser().resolve()
    speaker = CpuSpeakerScorer(speaker_dir)
    reports = [(p.expanduser().resolve(), json.loads(p.expanduser().resolve().read_text())) for p in args.reports]
    labels = [report["label"] for _, report in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("Report labels must be unique")
    result = {"schema_version": 1, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "labels": labels, "reports": [{"path": str(path), "sha256": sha256(path), "label": report["label"],
                                              "model_revision": report.get("model_revision")} for path, report in reports],
              "scorers": {"asr": {"model_dir": str(MODEL_DIR), "files_sha256": {name: sha256(MODEL_DIR / name) for name in MODEL_FILES},
                                  "provider": "cpu", "backend": "Nemotron int8 / sherpa-onnx greedy streaming, full file"},
                          "speaker": {"checkpoint": str(speaker_dir / "speaker_encoder.safetensors"),
                                      "sha256": sha256(speaker_dir / "speaker_encoder.safetensors"), "provider": "cpu",
                                      "preprocessing": "Full mono clip; soxr HQ to encoder sample rate; upstream fbank. max_audio_seconds=0 disables random cropping.",
                                      "metric": "CAM++ conditioning-encoder cosine proxy"},
                          "normalization": "NFKC, casefold, remove Unicode punctuation/symbols, collapse whitespace; CER additionally removes whitespace. No number or spelling rewrites."},
              "limitations": "ASR word/character error rates measure one recognizer's agreement, not human-rated speech accuracy. "
              "CAM++ speaker cosine is a conditioning-encoder proxy using the same encoder as TTS; it is not an independent "
              "perceptual voice-quality metric and has no calibrated pass threshold. This small English set cannot establish general model quality.",
              "rows": [], "summaries": {}, "pairs": []}
    asr_cache = {}
    for report_path, report in reports:
        seen = set()
        for source in report["runs"]:
            key = (source["case_id"], source["repeat"])
            if key in seen:
                raise ValueError(f"Duplicate run in {report_path}: {key}")
            seen.add(key)
            case = case_map[source["case_id"]]
            expected = case.get("spoken_expected_text", case["text"])
            if source.get("text", case["text"]) != case["text"] or source.get("seed", case["seed"]) != case["seed"]:
                raise ValueError(f"Report input mismatch for {key}")
            reference = Path(case["reference"])
            if not reference.is_absolute():
                reference = manifest_path.parent / reference
            row = dict(source, label=report["label"], expected_text=expected,
                       reference_path=str(reference.resolve()), score_status="error")
            try:
                if not source.get("wav_path"):
                    raise ValueError("No generated WAV; request failed before any audio")
                wav = Path(source["wav_path"])
                if not wav.is_absolute():
                    wav = report_path.parent / wav
                row["wav_path"] = str(wav.resolve())
                key_sha = sha256(wav)
                row["wav_sha256"] = key_sha
                if key_sha not in asr_cache:
                    asr_cache[key_sha] = transcribe_file(wav, max_seconds=180)
                row["recognized_text"] = asr_cache[key_sha]
                row["asr"] = error_metrics(expected, row["recognized_text"])
                row["speaker_cosine"] = speaker.cosine(reference, wav)
                row["score_status"] = "ok"
            except Exception as exc:
                row["score_error"] = f"{type(exc).__name__}: {exc}"
            result["rows"].append(row)
            result["summaries"] = {label: aggregate([r for r in result["rows"] if r["label"] == label]) for label in labels}
            result["pairs"] = paired_comparisons(result["rows"], labels)
            write_products(result, output_dir)
            print(f"{row['label']} {row['case_id']} repeat {row['repeat']}: {row['score_status']}; ASR {row.get('recognized_text', '')}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--speaker-model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/models/dots.tts-soar")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
