// Complete PCM stays in the browser until finish. Live requests are incremental,
// ordered, and limited to one in flight; a final full copy reconciles any tail.
let generation = 0;
let current = null;
const query = (name) => element.querySelector(name);
const show = (message) => { const node = query('.recorder-status'); if (node) node.textContent = message; };
const buttons = (recording, busy = false) => {
    const start = query('.record-start'); const stop = query('.record-stop');
    if (start) start.disabled = recording || busy;
    if (stop) stop.disabled = !recording || busy;
};
const encode = (samples) => {
    const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 16384) binary += String.fromCharCode(...bytes.subarray(i, i + 16384));
    return btoa(binary);
};
const combined = (state, begin = 0, end = state.count) => {
    const result = new Int16Array(end - begin);
    let offset = 0; let target = 0;
    for (const chunk of state.chunks) {
        const left = Math.max(0, begin - offset); const right = Math.min(chunk.length, end - offset);
        if (right > left) { result.set(chunk.subarray(left, right), target); target += right - left; }
        offset += chunk.length;
        if (offset >= end) break;
    }
    return result;
};
async function release(state) {
    if (state.released) return;
    state.released = true;
    clearInterval(state.timer);
    state.stream?.getTracks().forEach((track) => track.stop());
    try { state.source?.disconnect(); state.node?.disconnect(); state.gain?.disconnect(); } catch (_) {}
    try { if (state.context?.state !== 'closed') await state.context.close(); } catch (_) {}
    if (state.moduleUrl) URL.revokeObjectURL(state.moduleUrl);
}
async function abort() {
    generation += 1;
    const state = current; current = null;
    if (state) {
        state.cancelled = true;
        await release(state);
        if (state.token) server.cancel({token: state.token}).catch(() => {});
    }
    buttons(false);
}
async function pump(state) {
    if (state.sending || state.streamFailed || state.cancelled) return;
    state.sending = true;
    try {
        // Coalesce all samples accumulated while recognition was busy. No chunk
        // is dropped, and no unbounded queue of recognition requests is created.
        // One request per timer tick also bounds transport work when round trips
        // exceed an audio quantum; do not chase an ever-growing buffer in a loop.
        if (state.sent < state.count && !state.cancelled) {
            const end = state.count;
            const result = await server.feed({token: state.token, sequence: state.sequence, pcm: encode(combined(state, state.sent, end))});
            state.sent = end; state.sequence += 1;
            if (state.id !== generation || state.cancelled || result.cancelled) return;
            const preview = query('.recorder-preview');
            if (preview) preview.textContent = result.text || '';
            if (result.status && result.status !== 'Listening…') show(result.status);
        }
    } catch (_) {
        state.streamFailed = true;
        if (state.id === generation) show('Live transcription paused. The complete recording will be transcribed when you finish.');
    } finally { state.sending = false; }
}
async function finish() {
    const state = current;
    if (!state || state.finishing || state.cancelled) return;
    state.finishing = true;
    buttons(false, true);
    show('Finishing recording and transcript…');
    // AudioWorklet acknowledges after all preceding sample messages, preserving
    // the final partial network chunk instead of discarding it on stop.
    if (state.node) {
        await new Promise((resolve) => {
            state.resolveStopped = resolve;
            state.node.port.postMessage({stop: true});
            setTimeout(resolve, 1000);
        });
    }
    await release(state);
    // An in-flight feed holds the session lock. Await it before finalization to
    // preserve order; final PCM also repairs a failed or unsent live request.
    if (state.pending) await state.pending;
    if (state.cancelled || state.id !== generation) return;
    try {
        const result = await server.finish({token: state.token, pcm: encode(combined(state))});
        if (state.cancelled || state.id !== generation || result.cancelled) return;
        // The complete listener acknowledges by changing this component's value.
        // Keep Start disabled until those reference/transcript outputs arrive.
        buttons(false, true);
        show('Recording complete. Your transcript is available below.');
        trigger('complete', {token: result.token});
    } catch (_) {
        if (state.id !== generation) return;
        server.cancel({token: state.token}).catch(() => {});
        buttons(false, true);
        show('Recording could not be saved. Check the connection and try again.');
        trigger('complete', {token: state.token, error: true});
    }
}
async function start() {
    if (current) return;
    const id = ++generation;
    const state = {id, chunks: [], count: 0, sent: 0, sequence: 0, sending: false, cancelled: false};
    current = state;
    buttons(false, true);
    show('Requesting microphone access…');
    try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access needs localhost or HTTPS.');
        state.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
        if (id !== generation) { await release(state); return; }
        state.context = new AudioContext({sampleRate: 16000});
        await state.context.resume();
        if (!state.context.audioWorklet) throw new Error('This browser needs AudioWorklet support. Use a current browser.');
        const code = `class VoiceCapture extends AudioWorkletProcessor {
            constructor() { super(); this.stopped = false; this.port.onmessage = (event) => { if (event.data.stop) { this.stopped = true; this.port.postMessage({stopped: true}); } }; }
            process(inputs) { if (this.stopped) return false; const audio = inputs[0]?.[0]; if (audio?.length) { const copy = new Float32Array(audio); this.port.postMessage({samples: copy}, [copy.buffer]); } return true; }
        } registerProcessor('voice-capture', VoiceCapture);`;
        state.moduleUrl = URL.createObjectURL(new Blob([code], {type: 'application/javascript'}));
        await state.context.audioWorklet.addModule(state.moduleUrl);
        if (id !== generation) { await release(state); return; }
        const started = await server.start({sample_rate: state.context.sampleRate});
        state.token = started.token;
        if (id !== generation) { await release(state); server.cancel({token: state.token}).catch(() => {}); return; }
        state.maximum = Math.floor(started.max_seconds * state.context.sampleRate);
        state.node = new AudioWorkletNode(state.context, 'voice-capture');
        state.node.port.onmessage = (event) => {
            if (event.data.stopped) { state.resolveStopped?.(); return; }
            if (state.cancelled || state.id !== generation || !event.data.samples) return;
            const samples = event.data.samples;
            const length = Math.min(samples.length, state.maximum - state.count);
            if (length > 0) {
                const chunk = new Int16Array(length);
                for (let i = 0; i < length; i++) {
                    const value = Math.max(-1, Math.min(1, samples[i]));
                    chunk[i] = Math.round(value * (value < 0 ? 32768 : 32767));
                }
                state.chunks.push(chunk); state.count += length;
            }
            if (state.count >= state.maximum && !state.finishing) finish();
        };
        state.source = state.context.createMediaStreamSource(state.stream);
        state.gain = state.context.createGain(); state.gain.gain.value = 0;
        state.source.connect(state.node); state.node.connect(state.gain); state.gain.connect(state.context.destination);
        state.timer = setInterval(() => {
            if (state.finishing || state.cancelled) return;
            if (!state.sending) state.pending = pump(state);
            if (!state.streamFailed) show(`Recording ${(state.count / state.context.sampleRate).toFixed(1)}s / ${started.max_seconds}s. Finish when your sentence is complete.`);
        }, 500);
        const preview = query('.recorder-preview'); if (preview) preview.textContent = '';
        buttons(true);
        show('Listening…');
        trigger('start_recording', {token: state.token});
    } catch (error) {
        await release(state);
        if (state.token) server.cancel({token: state.token}).catch(() => {});
        if (id !== generation) return;
        current = null;
        buttons(false);
        show(error.name === 'NotAllowedError' ? 'Microphone access was denied. Allow microphone access or upload a recording below.' : (error.message || 'Could not start the microphone. Try again.'));
    }
}
// Delegation survives template re-renders caused by a reset output.
element.addEventListener('click', (event) => {
    if (event.target.closest('.record-start')) start();
    if (event.target.closest('.record-stop')) finish();
});
watch('value', () => { abort(); });
window.addEventListener('pagehide', () => { abort(); }, {once: true});
