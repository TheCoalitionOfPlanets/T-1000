import asyncio
import base64
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from t1000 import api, mic
from t1000.audio_integrity import AtomicAudioWriter
from t1000.jobs import GenerationJobs


class FlowAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.manager = GenerationJobs(shared_lock=api.generation_lock, disconnect_grace=1)
        self.manager_patch = patch.object(api, 'generation_jobs', self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.addCleanup(self.directory.cleanup)
        self.client = TestClient(api.api)
        self.addCleanup(self.client.close)
        api.finished_references.clear(); api.finish_locks.clear()
        with api.audio_lock:
            api.audio_files.clear()

    def output(self, *, complete=True):
        writer = AtomicAudioWriter(self.root, 16000)
        writer.append(np.sin(np.arange(16000) * .05).astype(np.float32) * .2)
        return writer.finish(complete=complete, reason='eos' if complete else 'cancelled')

    def payload(self, identity='generation-identity-0001', text='Hello'):
        return dict(text=text, reference_id='example', transcript='Reference words', generation_id=identity)

    def test_retried_generation_returns_same_audio_and_manifest_without_rerunning(self):
        output = self.output()
        calls = []
        def source(*args, cancel_event=None):
            calls.append(cancel_event)
            yield gr.skip(), 'Generating', gr.update(interactive=False), gr.update(interactive=True)
            yield output, 'Generated', gr.update(interactive=True), gr.update(interactive=False)
        with patch.object(api.voice, 'OUTPUT_DIR', self.root), patch.object(api.voice, 'synthesize', side_effect=source):
            first = self.client.post('/api/generate', json=self.payload())
            events = [json.loads(line) for line in first.text.splitlines()]
            terminal = events[-1]
            repeated = self.client.post('/api/generate', json={**self.payload(), 'after_sequence': terminal['sequence'] - 1})
            final = json.loads(repeated.text.splitlines()[-1])
            state = self.client.get('/api/generations/' + self.payload()['generation_id']).json()
        self.assertEqual(len(calls), 1)
        self.assertEqual(final, terminal)
        self.assertEqual(state, terminal)
        self.assertEqual(terminal['audio']['total_samples'], 16000)
        self.assertTrue(terminal['audio']['complete'])
        raw = self.client.get(terminal['audio_url']).content
        self.assertEqual(hashlib.sha256(raw).hexdigest(), terminal['audio']['sha256'])
        self.assertEqual(len(raw), terminal['audio']['byte_length'])
        self.assertFalse(api.generation_lock.locked())

    def test_same_generation_identity_cannot_change_text(self):
        output = self.output()
        def source(*args, **kwargs):
            yield output, 'Done', gr.update(interactive=True), gr.update(interactive=False)
        with patch.object(api.voice, 'OUTPUT_DIR', self.root), patch.object(api.voice, 'synthesize', side_effect=source):
            self.assertEqual(self.client.post('/api/generate', json=self.payload()).status_code, 200)
            self.assertEqual(self.client.post('/api/generate', json=self.payload(text='Different words')).status_code, 409)

    def test_disconnect_reconnect_drains_once_and_retains_terminal_wav(self):
        output = self.output()
        gate = threading.Event()
        calls = []
        def source(*args, cancel_event=None):
            calls.append(1)
            yield gr.skip(), 'Generating', gr.update(interactive=False), gr.update(interactive=True)
            gate.wait(2)
            yield output, 'Generated', gr.update(interactive=True), gr.update(interactive=False)
        async def disconnect():
            response = await api.generate(api.GenerateRequest(**self.payload()))
            async def receive():return {'type': 'http.disconnect'}
            async def send(message):
                if message['type'] == 'http.response.body':raise OSError('connection lost')
            with self.assertRaises(ClientDisconnect):
                await response({'type':'http','asgi':{'spec_version':'2.4'}}, receive, send)
        with patch.object(api.voice,'OUTPUT_DIR',self.root), patch.object(api.voice,'synthesize',side_effect=source):
            asyncio.run(disconnect())
            job = self.manager.get(self.payload()['generation_id'])
            self.assertFalse(job.cancel_event.is_set())
            gate.set()
            self.assertTrue(job.done.wait(2))
            recovered = self.client.post('/api/generate',json=self.payload())
        final = json.loads(recovered.text.splitlines()[-1])
        self.assertTrue(final['audio']['complete'])
        self.assertEqual(calls,[1])
        self.assertEqual(self.client.get(final['audio_url']).content,Path(output).read_bytes())

    def test_stale_stop_does_not_cancel_new_job_and_partial_is_explicit(self):
        output = self.output(complete=False)
        running = threading.Event()
        events = {}
        def source(text,*args,cancel_event=None):
            events[text] = cancel_event
            if text == 'New':
                running.set()
                while not cancel_event.wait(.01):
                    yield gr.skip(),'Generating',gr.update(interactive=False),gr.update(interactive=True)
            yield output,'Stopped',gr.update(interactive=True),gr.update(interactive=False)
        with patch.object(api.voice,'OUTPUT_DIR',self.root), patch.object(api.voice,'synthesize',side_effect=source):
            self.client.post('/api/generate',json=self.payload(text='Old'))
            response=[]
            worker=threading.Thread(target=lambda:response.append(self.client.post('/api/generate',json=self.payload('generation-identity-0002','New'))))
            worker.start();self.assertTrue(running.wait(2))
            self.client.post('/api/stop',json={'generation_id':self.payload()['generation_id']})
            self.assertFalse(events['New'].is_set())
            self.client.post('/api/stop',json={'generation_id':'generation-identity-0002'})
            worker.join(2);self.assertFalse(worker.is_alive())
        final=json.loads(response[0].text.splitlines()[-1])
        self.assertTrue(final['partial']);self.assertFalse(final['audio']['complete'])

    def test_finish_lost_response_reuses_same_reference_id_and_cancel_invalidates_retry(self):
        class Session:
            text='Preview'
            finalized=0
            def feed(self,rate,audio):pass
            def finish(self):self.finalized+=1;return 'Final words'
            def close(self):pass
        session=Session();service=mic.RecorderService(10)
        pcm=(np.sin(np.arange(32000)*.05)*6000).astype('<i2').tobytes()
        with patch.object(api,'recorder',service), patch.object(mic,'RECORDING_DIR',self.root), patch.object(mic,'create_session',return_value=session):
            token=self.client.post('/api/recordings/start',json={'sample_rate':16000}).json()['token']
            data={'token':token,'pcm':base64.b64encode(pcm).decode()}
            first=self.client.post('/api/recordings/finish',json=data)
            second=self.client.post('/api/recordings/finish',json=data)
            self.assertEqual(first.status_code,200);self.assertEqual(first.json(),second.json())
            changed=self.client.post('/api/recordings/finish',json={**data,'pcm':base64.b64encode(pcm[:-2]).decode()})
            self.assertEqual(changed.status_code,400)
            self.assertEqual(session.finalized,1)
            raw=self.client.get(first.json()['url']).content
            self.assertEqual(sf.read(io.BytesIO(raw),dtype='int16')[0].tobytes(),pcm)
            self.client.post('/api/recordings/cancel',json={'token':token})
            self.assertEqual(self.client.post('/api/recordings/finish',json=data).status_code,400)
            self.assertEqual(self.client.get(first.json()['url']).status_code,200)

    def test_retained_job_audio_survives_the_small_audio_registry_being_evicted(self):
        output=self.output()
        def source(*args,**kwargs):yield output,'Done',gr.update(interactive=True),gr.update(interactive=False)
        with patch.object(api.voice,'OUTPUT_DIR',self.root),patch.object(api.voice,'synthesize',side_effect=source):
            result=self.client.post('/api/generate',json=self.payload())
            final=json.loads(result.text.splitlines()[-1])
            with api.audio_lock:api.audio_files.clear()
            recovered=self.client.post('/api/generate',json=self.payload())
            self.assertEqual(json.loads(recovered.text.splitlines()[-1]),final)
            self.assertEqual(self.client.get(final['audio_url']).content,Path(output).read_bytes())
            self.assertEqual(self.client.get('/api/audio/g-'+'f'*32).status_code,404)

    def test_an_expired_resume_cursor_never_silently_starts_new_inference(self):
        with patch.object(api.voice,'synthesize') as source:
            response=self.client.post('/api/generate',json={**self.payload(),'after_sequence':3})
        self.assertEqual(response.status_code,410);source.assert_not_called()
