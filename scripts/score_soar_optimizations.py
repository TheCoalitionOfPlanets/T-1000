#!/usr/bin/env python3
"""CPU quality checks for isolated SOAR replay and full-TTS experiments.

Replay timings measure the vocoder alone. They are never full-generation
speedups. Quality uses repeat 1 only; timing summaries retain all repetitions.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from score_tts_comparison import CpuSpeakerScorer, error_metrics, relative_link, sha256

ORDER = ('baseline', 'native-lstm', 'compiled-vocoder', 'rotary', 'decoder-bf16', 'decoder-fp16', 'batch8')
SCOPES = {'replay_runs': 'isolated streaming vocoder replay',
          'runs': 'full TTS excluding input prep/load/writing'}
LIMITATIONS = (
    'ASR agreement measures one recognizer, not human-rated intelligibility. CAM++ cosine uses the '
    'TTS conditioning encoder itself on full clips; it is an optimistic proxy, not independent '
    'speaker verification or a calibrated pass threshold. No human listening ratings were collected. '
    'Quality scores use repeat 1 only; timing repetitions are not independent quality examples. '
    'Replay reuses identical saved acoustic latents and measures vocoding alone. Full-TTS screening '
    'is a separate measurement and normally contains only two cases. Failed or incomplete phases '
    'are not accepted speed improvements.'
)


def atomic_json(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_timing(row):
    return (row.get('status') == 'ok' and row.get('stop_reason') == 'eos'
            and row.get('finite') is True and row.get('nonzero') is True
            and finite_number(row.get('generation_seconds')) and row['generation_seconds'] > 0
            and finite_number(row.get('audio_seconds')) and row['audio_seconds'] > 0)


def stats(values):
    values = [float(x) for x in values if finite_number(x)]
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'minimum': min(values), 'maximum': max(values)} if values else None


def summarize(quality_rows, timing_rows):
    scored = [r for r in quality_rows if r.get('score_status') == 'ok']
    timed = [r for r in timing_rows if valid_timing(r)]
    result = {'unique_quality_cases': len(quality_rows), 'scored_unique_cases': len(scored),
              'timed_requests': len(timing_rows), 'valid_timed_eos_requests': len(timed),
              'clipped_unique_cases': sum(bool(r.get('clipped_sample_count')) for r in quality_rows)}
    for unit in ('word', 'character'):
        total = {k: sum(r['asr'][unit][k] for r in scored)
                 for k in ('errors', 'substitutions', 'deletions', 'insertions', 'reference_units')}
        total['rate'] = total['errors'] / total['reference_units'] if total['reference_units'] else None
        result[unit] = total
    result['speaker_cosine'] = stats([r.get('speaker_cosine') for r in scored])
    for field in ('generation_seconds', 'first_chunk_seconds', 'audio_seconds', 'real_time_factor',
                  'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
        result[field] = stats([r.get(field) for r in timed])
    comparisons = [r.get('waveform_comparison', {}) for r in quality_rows]
    result['waveforms'] = {
        'comparisons': len([c for c in comparisons if c]),
        'same_shape_cases': sum(c.get('same_shape') is True for c in comparisons),
        'bit_exact_cases': sum(c.get('bit_exact') is True for c in comparisons),
        'maximum_absolute_difference': max((c['max_abs_difference'] for c in comparisons
                                           if finite_number(c.get('max_abs_difference'))), default=None),
        'rms_difference': stats([c.get('rms_difference') for c in comparisons]),
        'signal_to_error_db_nonexact': stats([c.get('signal_to_error_db') for c in comparisons]),
    }
    result['mismatches'] = [{'case_id': r['case_id'], 'expected': r['expected_text'],
                             'recognized': r['recognized_text'], 'word': r['asr']['word'],
                             'character': r['asr']['character']}
                            for r in scored if r['asr']['word']['errors'] or r['asr']['character']['errors']]
    return result


def paired_speedup(baseline, candidate, *, completed):
    if not completed:
        return {'eligible_pairs': 0, 'excluded_reason': 'Phase failed, incomplete, or baseline unavailable.'}
    base = {(r['case_id'], r['repeat']): r for r in baseline if valid_timing(r)}
    pairs = []
    for row in candidate:
        key = (row['case_id'], row['repeat'])
        if key in base and valid_timing(row):
            other = base[key]
            pairs.append({'case_id': key[0], 'repeat': key[1],
                          'wall_time_ratio_baseline_over_candidate': other['generation_seconds'] / row['generation_seconds'],
                          'rtf_ratio_baseline_over_candidate': other['real_time_factor'] / row['real_time_factor']
                          if finite_number(other.get('real_time_factor')) and finite_number(row.get('real_time_factor'))
                          and row['real_time_factor'] > 0 else None})
    return {'eligible_pairs': len(pairs), 'pairs': pairs,
            'wall_time_ratio': stats([p['wall_time_ratio_baseline_over_candidate'] for p in pairs]),
            'rtf_ratio': stats([p['rtf_ratio_baseline_over_candidate'] for p in pairs])}


def write_products(result, output):
    atomic_json(output / 'scores.json', result)
    fields = ['phase', 'scope', 'case_id', 'repeat', 'score_status', 'expected_text', 'recognized_text',
              'wer', 'cer', 'speaker_cosine', 'wav_path', 'score_error', 'bit_exact',
              'max_abs_difference', 'rms_difference', 'signal_to_error_db']
    with (output / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in result['quality_rows']:
            flat = dict(row, **row.get('waveform_comparison', {}))
            if row.get('asr'):
                flat.update(wer=row['asr']['word']['rate'], cer=row['asr']['character']['rate'])
            writer.writerow(flat)
    def fmt(value, digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    lines = ['# SOAR optimization quality and timing checks', '', LIMITATIONS, '']
    for key, title in SCOPES.items():
        lines += ['## ' + title, '',
                  '| Phase | Report status | Quality cases scored | ASR errors / words | Median time | Paired baseline/time ratio | Cosine proxy | Bit-exact cases |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for phase, report in result['phases'].items():
            summary = report['summaries'][key]
            median = lambda name: summary[name]['median'] if summary.get(name) else None
            ratio = report['paired'][key].get('wall_time_ratio')
            pair_count = report['paired'][key]['eligible_pairs']
            lines.append(f"| {phase} | {report['status']} | {summary['scored_unique_cases']} / {summary['unique_quality_cases']} | "
                         f"{summary['word']['errors']} / {summary['word']['reference_units']} | "
                         f"{fmt(median('generation_seconds'))} s | {fmt(ratio['median'] if ratio else None)}× ({pair_count} pairs) | "
                         f"{fmt(median('speaker_cosine'))} | {summary['waveforms']['bit_exact_cases']} / {summary['unique_quality_cases']} |")
        lines += ['']
    lines += ['## Failures and incomplete phases', '']
    failures = [(name, report) for name, report in result['phases'].items() if report['status'] != 'complete']
    lines += [f"- {name}: {report['status']}; {report.get('fatal_error') or 'No terminal success recorded.'}" for name, report in failures] or ['None in the available reports.']
    lines += ['', '## Per-case transcripts and listening', '']
    for row in result['quality_rows']:
        lines += [f"### {row['phase']} / {row['scope']} / {row['case_id']}", '',
                  f"Expected: {row['expected_text']}", '', f"ASR: {row.get('recognized_text', '[not scored]')}", '']
        if row.get('wav_path'):
            lines += [f"[Generated WAV]({relative_link(row['wav_path'], output)}) · "
                      f"[Reference WAV]({relative_link(row['reference_path'], output)})", '']
        if row.get('score_error'):
            lines += [f"Scoring failed: {row['score_error']}", '']
        elif row.get('asr'):
            lines += [f"Word errors {row['asr']['word']['errors']} / {row['asr']['word']['reference_units']}; "
                      f"conditioning cosine {fmt(row['speaker_cosine'])}. "
                      f"Waveform comparison: `{json.dumps(row.get('waveform_comparison', {}), sort_keys=True)}`", '']
    (output / 'SCORING.md').write_text('\n'.join(lines) + '\n')


def run(args):
    os.environ['STT_BACKEND'] = 'cpu'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from t1000.stt import MODEL_DIR, MODEL_FILES, transcribe_file
    torch.set_num_threads(args.cpu_threads)
    root, manifest_path, output = args.root.resolve(), args.manifest.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['cases'] if isinstance(manifest, dict) else manifest
    case_map = {c['id']: c for c in cases}
    if len(case_map) != len(cases):
        raise ValueError('Manifest contains duplicate case IDs')
    phases = args.phases
    if not phases:
        available = [p.name for p in root.iterdir() if p.is_dir() and (p / 'report.json').is_file()]
        phases = [p for p in ORDER if p in available] + sorted(set(available) - set(ORDER))
    if not phases:
        raise ValueError(f'No experiment reports found in {root}')
    if len(phases) != len(set(phases)) or any(Path(p).name != p or p in {'.', '..'} for p in phases):
        raise ValueError('Phases must be unique single directory names')
    speaker_dir = args.speaker_model_dir.resolve()
    identity = {'asr': {name: sha256(MODEL_DIR / name) for name in MODEL_FILES},
                'speaker': sha256(speaker_dir / 'speaker_encoder.safetensors'),
                'speaker_config': sha256(speaker_dir / 'config.json'),
                'preprocessing': 'fullclip/soxrHQ/CAMmax_audio_seconds0/v1'}
    cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = output / 'score-cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('identity') != cache_identity:
        cache = {'identity': cache_identity, 'asr': {}, 'cosine': {}}
    speaker = None
    result = {'schema_version': 1, 'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
              'root': str(root), 'limitations': LIMITATIONS, 'scorers': identity, 'phases': {}, 'quality_rows': []}
    raw = {}
    for phase in phases:
        path = root / phase / 'report.json'
        if not path.exists():
            document = {'status': 'missing_report', 'fatal_error': 'Requested report.json is absent.'}
        else:
            document = json.loads(path.read_text())
            if document.get('manifest_sha256') not in {None, result['manifest_sha256']}:
                raise ValueError(f'Manifest hash differs in {path}')
        raw[phase] = document
        for case in document.get('cases', []):
            if case['id'] not in case_map:
                raise ValueError(f"Unknown report case {case['id']} in {path}")
            reference = Path(case_map[case['id']]['reference'])
            reference = reference if reference.is_absolute() else manifest_path.parent / reference
            if case.get('reference_sha256') not in {None, sha256(reference)}:
                raise ValueError(f"Reference hash differs for {case['id']} in {path}")
        result['phases'][phase] = {'status': document.get('status', 'unknown'),
                                   'fatal_error': document.get('fatal_error'),
                                   'report_sha256': sha256(path) if path.exists() else None,
                                   'experiment': document.get('experiment'), 'summaries': {}, 'paired': {}}
        for scope in SCOPES:
            seen = set()
            for source in document.get(scope, []):
                if source.get('repeat') != 1:
                    continue
                identity_key = source['case_id']
                if identity_key in seen:
                    raise ValueError(f'Duplicate repeat 1 for {phase}/{scope}/{identity_key}')
                seen.add(identity_key)
                case = case_map[identity_key]
                if source.get('text', case['text']) != case['text'] or source.get('seed', case['seed']) != case['seed']:
                    raise ValueError(f'Inputs differ for {phase}/{scope}/{identity_key}')
                reference = Path(case['reference'])
                reference = reference if reference.is_absolute() else manifest_path.parent / reference
                row = dict(source, phase=phase, scope=scope,
                           expected_text=case.get('spoken_expected_text', case['text']),
                           reference_path=str(reference.resolve()), score_status='error')
                try:
                    wav = Path(source['wav_path'])
                    wav = wav if wav.is_absolute() else path.parent / wav
                    row['wav_path'] = str(wav.resolve())
                    digest = sha256(wav)
                    row['wav_sha256'] = digest
                    if digest not in cache['asr']:
                        cache['asr'][digest] = transcribe_file(wav, max_seconds=180)
                    cosine_key = sha256(reference) + ':' + digest
                    if cosine_key not in cache['cosine']:
                        if speaker is None:
                            speaker = CpuSpeakerScorer(speaker_dir)
                        cache['cosine'][cosine_key] = speaker.cosine(reference, wav)
                    row['recognized_text'] = cache['asr'][digest]
                    row['speaker_cosine'] = cache['cosine'][cosine_key]
                    row['asr'] = error_metrics(row['expected_text'], row['recognized_text'])
                    row['score_status'] = 'ok'
                    atomic_json(cache_path, cache)
                except Exception as exc:
                    row['score_error'] = f'{type(exc).__name__}: {exc}'
                result['quality_rows'].append(row)
                print(f"{phase} {scope} {identity_key}: {row['score_status']}; {row.get('recognized_text', row.get('score_error'))}", flush=True)
            quality = [r for r in result['quality_rows'] if r['phase'] == phase and r['scope'] == scope]
            result['phases'][phase]['summaries'][scope] = summarize(quality, document.get(scope, []))
    baseline = raw.get('baseline', {})
    for phase, document in raw.items():
        for scope in SCOPES:
            result['phases'][phase]['paired'][scope] = paired_speedup(
                baseline.get(scope, []), document.get(scope, []),
                completed=baseline.get('status') == 'complete' and document.get('status') == 'complete')
    write_products(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--phases', nargs='+')
    parser.add_argument('--speaker-model-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/models/dots.tts-soar')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
