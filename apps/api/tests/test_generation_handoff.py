import hashlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from t1000 import tts


class GenerationHandoffTests(unittest.TestCase):
    def test_worker_spools_every_chunk_in_order_and_returns_verified_manifest(self):
        parts = [torch.linspace(-.2, .2, 137).reshape(1,-1), torch.full((1,311),.1), torch.full((1,19),-.15)]
        class Runtime:
            device=SimpleNamespace(type='cpu');sample_rate=16000
            def generate_stream(self,**kwargs):yield from parts
        with tempfile.TemporaryDirectory() as directory, patch.object(tts,'OUTPUT_DIR',Path(directory)), patch.object(tts,'load_runtime',return_value=Runtime()):
            result=list(tts.synthesize('Hello.',str(tts.EXAMPLE_AUDIO),tts.EXAMPLE_TRANSCRIPT,10,42))[-1][0]
            raw=Path(result).read_bytes()
            expected=io.BytesIO();sf.write(expected,torch.cat(parts,dim=1).reshape(-1).numpy(),16000,format='WAV',subtype='PCM_16')
            self.assertEqual(raw,expected.getvalue())
            self.assertEqual(result.metadata['total_samples'],467)
            self.assertEqual(result.metadata['chunks'],3)
            self.assertEqual(result.metadata['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertTrue(result.metadata['complete'])
            self.assertEqual(list(Path(directory).glob('*.part')),[])

    def test_disconnect_retains_the_already_computed_inflight_chunk_and_joins_worker(self):
        inflight,release=threading.Event(),threading.Event()
        class Runtime:
            device=SimpleNamespace(type='cpu');sample_rate=16000
            def generate_stream(self,**kwargs):
                yield torch.full((1,100),.1)
                inflight.set();release.wait(2)
                yield torch.full((1,200),.2)
                raise AssertionError('cancelled generation started another audio step')
        with tempfile.TemporaryDirectory() as directory, patch.object(tts,'OUTPUT_DIR',Path(directory)), patch.object(tts,'load_runtime',return_value=Runtime()):
            source=tts.synthesize('Hello.',str(tts.EXAMPLE_AUDIO),tts.EXAMPLE_TRANSCRIPT,10,42)
            next(source);self.assertTrue(inflight.wait(2))
            closer=threading.Thread(target=source.close);closer.start()
            self.assertTrue(tts.ACTIVE_STOP.wait(1));release.set();closer.join(2)
            self.assertFalse(closer.is_alive());self.assertFalse(tts.MODEL_LOCK.locked());self.assertIsNone(tts.ACTIVE_STOP)
            files=list(Path(directory).glob('speech-*.wav'));self.assertEqual(len(files),1)
            audio,rate=sf.read(files[0]);self.assertEqual(len(audio),300);self.assertEqual(rate,16000)
            self.assertTrue(np.allclose(audio[:100],.1,atol=1/32768));self.assertTrue(np.allclose(audio[100:],.2,atol=1/32768))

    def test_worker_import_failure_terminates_instead_of_stranding_a_job(self):
        with patch.dict('sys.modules',{'t1000.audio_integrity':None}):
            updates=list(tts.synthesize('Hello.',str(tts.EXAMPLE_AUDIO),tts.EXAMPLE_TRANSCRIPT,10,42))
        self.assertIn('Generation failed',updates[-1][1]);self.assertIsNone(tts.ACTIVE_STOP);self.assertFalse(tts.MODEL_LOCK.locked())

    def test_thread_start_failure_cleans_frozen_reference_and_active_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tts,'CACHE',Path(directory)), patch('threading.Thread.start',side_effect=RuntimeError('thread unavailable')):
            with self.assertRaisesRegex(RuntimeError,'thread unavailable'):
                next(tts.synthesize('Hello.',str(tts.EXAMPLE_AUDIO),tts.EXAMPLE_TRANSCRIPT,10,42))
            self.assertEqual(list((Path(directory)/'requests').iterdir()),[])
        self.assertIsNone(tts.ACTIVE_STOP)
