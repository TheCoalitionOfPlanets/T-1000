#!/usr/bin/env python3
"""Isolated SOAR experiments: identical-latent replay plus full TTS screening.

Capture is unmeasured and baseline-only. Replay times are vocoder-only and MUST
NOT be presented as full TTS speedups. Full request timings use matched cases.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from compare_tts_models import DecodeObserver, load_manifest, measure_request, package_version, write_report

VARIANTS = ('baseline', 'native-lstm', 'compiled-vocoder', 'rotary', 'decoder-bf16', 'decoder-fp16', 'batch8', 'combined')


def compare_audio(torch, baseline, candidate):
    result = {'same_shape': tuple(baseline.shape) == tuple(candidate.shape),
              'baseline_samples': baseline.numel(), 'candidate_samples': candidate.numel()}
    if result['same_shape']:
        delta = (baseline - candidate).float()
        rms = float(delta.square().mean().sqrt())
        base_rms = float(baseline.float().square().mean().sqrt())
        result.update(bit_exact=bool(torch.equal(baseline, candidate)), max_abs_difference=float(delta.abs().max()),
                      rms_difference=rms, baseline_rms=base_rms,
                      signal_to_error_db=20 * __import__('math').log10(base_rms / rms) if rms and base_rms else None)
    return result


def replay(runtime, capture, torch):
    # CPU->GPU inputs are excluded, as are file I/O and waveform comparisons.
    patches = [patch.to(runtime.device) for patch in capture['patches']]
    adapter = runtime.model._get_vocoder_inference()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    chunks = []
    first = None
    with torch.inference_mode():
        state = adapter.init_stream_state(**capture['init_kwargs'])
        for index, patch in enumerate(patches):
            piece = adapter.stream_step(patch, state, optimize=False, use_compiled=False)
            pcm = piece.detach().float().cpu().reshape(-1)
            if pcm.numel():
                first = first if first is not None else time.perf_counter() - started
                chunks.append(pcm)
        piece = adapter.flush(state).detach().float().cpu().reshape(-1)
        if piece.numel():
            first = first if first is not None else time.perf_counter() - started
            chunks.append(piece)
    torch.cuda.synchronize(runtime.device)
    seconds = time.perf_counter() - started
    audio = torch.cat(chunks) if chunks else torch.empty(0)
    duration = audio.numel() / runtime.sample_rate
    result = {'generation_seconds': seconds, 'first_chunk_seconds': first, 'audio_seconds': duration,
              'real_time_factor': seconds / duration if duration else None,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(runtime.device),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(runtime.device),
              'finite': bool(torch.isfinite(audio).all()), 'nonzero': bool(torch.count_nonzero(audio)),
              'clipped_sample_count': int((audio.abs() >= 1).sum()),
              'peak_abs': float(audio.abs().max()) if audio.numel() else None,
              'timing_scope': 'isolated streaming vocoder replay', 'chunks': len(chunks)}
    if not audio.numel() or not result['finite'] or not result['nonzero']:
        raise RuntimeError('Invalid replayed waveform')
    return result, audio


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      T1000_REFERENCE_TRIM='0', T1000_TTS_VOCODER_BATCH_SIZE='4', T1000_TTS_EAGER_CACHE='1')
    import torch
    import soundfile as sf
    from dots_tts.utils.util import seed_everything
    from t1000.low_vram import load_low_vram_runtime
    from t1000.eager import configure_eager_cache
    from t1000.references import configure_prompt_audio
    from loguru import logger

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists():
        raise ValueError(f'Refusing to overwrite {report_path}')
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(str(output/'runtime.log'),level='DEBUG')
    manifest = load_manifest(args.manifest)
    for case in manifest['cases']:
        case['reference_sha256'] = hashlib.sha256(Path(case['reference']).read_bytes()).hexdigest()
        case['reference_seconds'] = sf.info(case['reference']).duration
        if not 1 <= case['reference_seconds'] <= 6:
            raise ValueError('Whole reference must be 1–6 seconds')
    report = {'schema_version': 1, 'variant': args.variant, 'label': args.variant,
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'loading',
              'model_revision': '2f9b3e18d70d670d4c701da2dc55ded5755815ce',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'manifest_metadata': manifest['metadata'], 'cases': manifest['cases'],
              'configuration': {'steps':10,'guidance_scale':1.2,'ode_method':'euler','profile':'NF4 language blocks/BF16 core/eager',
                                'reference_trim':False,'conditioning_cache':True,'max_generate_length':128},
              'environment': {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_version':torch.backends.cudnn.version(),
                              **{n:package_version(n) for n in ['dots-tts','bitsandbytes','torchaudio']}},
              'captures':[], 'replay_warmups':[], 'replay_runs':[], 'warmups':[], 'runs':[]}
    def save():write_report(report_path,report)
    save()
    try:
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        runtime=load_low_vram_runtime(args.model_dir.resolve(),precision='bfloat16')
        configure_prompt_audio(runtime)
        configure_eager_cache(runtime,True)
        if args.variant=='combined':
            settings=[]
            from t1000.vocoder_experiments import configure_vocoder_experiment
            if args.enable_native_lstm:settings.append(configure_vocoder_experiment(runtime,'native-lstm'))
            if args.enable_compiler:settings.append(configure_vocoder_experiment(runtime,'compiled-vocoder'))
            if args.enable_rotary:
                from t1000.rotary import configure_rotary_cache
                configure_rotary_cache(runtime,True)
                settings.append({'variant':'rotary','stats':runtime.model._t1000_rotary_reuse.stats})
            if args.decoder_precision!='float32':settings.append(configure_vocoder_experiment(runtime,'decoder-bf16' if args.decoder_precision=='bfloat16' else 'decoder-fp16'))
            if args.enable_batch8:settings.append(configure_vocoder_experiment(runtime,'batch8'))
            if not settings:raise ValueError('Combined variant requires at least one explicit option')
            metadata={'variant':'combined','settings':settings}
        elif args.variant=='rotary':
            from t1000.rotary import configure_rotary_cache
            configure_rotary_cache(runtime,True)
            metadata={'variant':'rotary','scope':'reuse position tensors across compatible layers within each DiT forward','stats':runtime.model._t1000_rotary_reuse.stats}
        elif args.variant!='baseline':
            from t1000.vocoder_experiments import configure_vocoder_experiment
            metadata=configure_vocoder_experiment(runtime,args.variant)
        else:
            metadata={'variant':'baseline'}
        torch.cuda.synchronize()
        report['experiment']=metadata
        report['model_load']={'seconds':time.perf_counter()-started,
                              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                              'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
        report['configuration']['loaded_parameter_dtypes']={name:str(next(module.parameters()).dtype) for name,module in
            [('acoustic_core',runtime.model.core.patch_encoder),('speaker_encoder',runtime.model.xvector_extractor),('decoder',runtime.model.vocoder.decoder)]}
        observer=DecodeObserver(runtime.model)
        runtime.model._decode=observer
        report['status']='running'
        save()
        capture_dir=args.capture_dir.resolve()
        if args.capture:
            if args.variant!='baseline':raise ValueError('Only baseline can capture')
            capture_dir.mkdir(parents=True,exist_ok=True)
            for case in manifest['cases']:
                adapter=runtime.model._get_vocoder_inference()
                original_step,original_init=adapter.stream_step,adapter.init_stream_state
                patches=[]
                init_kwargs={}
                def capture_step(latent_patch,*a,**kw):
                    patches.append(latent_patch.detach().clone().cpu())
                    return original_step(latent_patch,*a,**kw)
                def capture_init(*a,**kw):
                    if a:raise ValueError('Expected keyword stream init')
                    init_kwargs.update(kw)
                    return original_init(**kw)
                adapter.stream_step,adapter.init_stream_state=capture_step,capture_init
                try:
                    result,audio=measure_request(runtime,case,'soar',observer,torch,seed_everything)
                finally:
                    adapter.stream_step,adapter.init_stream_state=original_step,original_init
                if result['status']!='ok' or result['stop_reason']!='eos':raise RuntimeError(f'Capture failed: {result}')
                tensor=torch.from_numpy(audio)
                capture={'patches':patches,'init_kwargs':init_kwargs,'case_id':case['id'],'sample_rate':runtime.sample_rate,
                         'baseline_audio':tensor,'stop_reason':result['stop_reason'],'reference_sha256':case['reference_sha256']}
                torch.save(capture,capture_dir/(case['id']+'.pt'))
                sf.write(capture_dir/(case['id']+'.wav'),audio,runtime.sample_rate,subtype='FLOAT')
                result.update(case_id=case['id'],timing_scope='UNTIMED capture with CPU instrumentation; excluded from latency',patches=len(patches),init_kwargs=init_kwargs)
                report['captures'].append(result);save()
                print('Captured',case['id'],len(patches),flush=True)
        captures={c['id']:torch.load(capture_dir/(c['id']+'.pt'),map_location='cpu',weights_only=True) for c in manifest['cases']}
        replay_cases=[] if args.skip_replay else manifest['cases']
        # One explicit replay warmup; catches compilation/workspace OOM separately.
        for case in replay_cases[:1]:
            started=time.perf_counter()
            measurement,_=replay(runtime,captures[case['id']],torch)
            measurement.update(case_id=case['id'],wall_seconds=time.perf_counter()-started)
            report['replay_warmups'].append(measurement);save()
        for case in replay_cases:
            capture=captures[case['id']]
            if capture['reference_sha256']!=case['reference_sha256']:raise ValueError('Capture/reference mismatch')
            for repeat in range(1,args.repeats+1):
                measurement,audio=replay(runtime,capture,torch)
                measurement.update(case_id=case['id'],reference_id=case['reference_id'],repeat=repeat,status='ok',stop_reason='eos',
                                   text=case['text'],spoken_expected_text=case['spoken_expected_text'],seed=case['seed'])
                measurement['waveform_comparison']=compare_audio(torch,capture['baseline_audio'],audio)
                name=f"replay-{case['id']}-r{repeat:02d}.wav"
                sf.write(output/name,audio.numpy(),runtime.sample_rate,subtype='FLOAT')
                measurement['wav_path']=name
                report['replay_runs'].append(measurement);save()
                print('Replay',case['id'],repeat,round(measurement['generation_seconds'],3),measurement['waveform_comparison'],flush=True)
        selected=manifest['cases'] if args.full_evaluation or args.variant=='rotary' else [c for c in manifest['cases'] if c['id'] in {'voice-a-short','voice-b-names'}]
        warmed=set()
        for case in selected:
            voice=(case['reference_sha256'],case['transcript'])
            if voice not in warmed:
                measurement,_=measure_request(runtime,case,'soar',observer,torch,seed_everything)
                measurement['case_id']=case['id'];report['warmups'].append(measurement);save();warmed.add(voice)
                if measurement['status']!='ok':raise RuntimeError(f'Full request warmup failed: {measurement}')
            for repeat in range(1,args.repeats+1):
                measurement,audio=measure_request(runtime,case,'soar',observer,torch,seed_everything)
                measurement.update(case_id=case['id'],reference_id=case['reference_id'],repeat=repeat,text=case['text'],
                                   spoken_expected_text=case['spoken_expected_text'],seed=case['seed'],timing_scope='full TTS excluding input prep/load/writing')
                if measurement['status']!='ok' or measurement['stop_reason']!='eos':raise RuntimeError(f'TTS failed: {measurement}')
                comparison=compare_audio(torch,captures[case['id']]['baseline_audio'],torch.from_numpy(audio))
                measurement['waveform_comparison']=comparison
                name=f"tts-{case['id']}-r{repeat:02d}.wav";sf.write(output/name,audio,runtime.sample_rate,subtype='FLOAT')
                measurement['wav_path']=name
                report['runs'].append(measurement);save()
                print('TTS',json.dumps(measurement),flush=True)
        report['status']='complete'
        report['completed_at_utc']=datetime.now(timezone.utc).isoformat();save()
    except BaseException as exc:
        report.update(status='failed',fatal_error=f'{type(exc).__name__}: {exc}')
        save();raise
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('data/models/dots.tts-soar'))
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--capture',action='store_true')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--full-evaluation',action='store_true')
    p.add_argument('--skip-replay',action='store_true')
    p.add_argument('--enable-native-lstm',action='store_true')
    p.add_argument('--enable-compiler',action='store_true')
    p.add_argument('--enable-rotary',action='store_true')
    p.add_argument('--enable-batch8',action='store_true')
    p.add_argument('--decoder-precision',choices=('float32','bfloat16','float16'),default='float32')
    args=p.parse_args()
    if args.repeats<1:p.error('repeats must be positive')
    run(args)
