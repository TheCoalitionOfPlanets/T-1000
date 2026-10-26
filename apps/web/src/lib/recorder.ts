import { post, type Reference } from "./api"

interface Callbacks {
  onStream: (stream: MediaStream | null) => void
  onProgress: (seconds: number, text: string) => void
  onStatus: (status: string) => void
  onFinishing?: () => void
  onComplete: (reference: Reference) => void
  onError: (message: string) => void
}

function encode(chunks: Int16Array[], start = 0): string {
  const samples = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const bytes = new Uint8Array(Math.max(0, samples - start) * 2)
  let offset = 0
  let skipped = 0
  for (const chunk of chunks) {
    const from = Math.max(0, start - skipped)
    if (from < chunk.length) {
      const part = chunk.subarray(from)
      bytes.set(new Uint8Array(part.buffer, part.byteOffset, part.byteLength), offset)
      offset += part.byteLength
    }
    skipped += chunk.length
  }
  let binary = ""
  for (let index = 0; index < bytes.length; index += 8192) binary += String.fromCharCode(...bytes.subarray(index, index + 8192))
  return btoa(binary)
}

/** Keep final PCM independent of streaming requests, including the last worklet frame. */
export class VoiceRecorder {
  private context?: AudioContext
  private stream?: MediaStream
  private node?: AudioWorkletNode
  private source?: MediaStreamAudioSourceNode
  private gain?: GainNode
  private timer?: ReturnType<typeof setInterval>
  private token?: string
  private chunks: Int16Array[] = []
  private count = 0
  private sent = 0
  private sequence = 0
  private maximum = 0
  private pending?: Promise<void>
  private cancelled = false
  private finishing = false
  private finishRequested = false
  private streamPublished = false
  private streamFailed = false
  private text = ""
  private resolveStopped?: () => void

  constructor(private callbacks: Callbacks) {}

  async start() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access needs localhost or HTTPS.")
      this.callbacks.onStatus("Allow microphone access to start recording…")
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
      if (this.cancelled) { await this.release(); return }
      this.context = new AudioContext({ sampleRate: 16000 })
      await this.context.resume()
      const code = `class Capture extends AudioWorkletProcessor {
        constructor() { super(); this.stopped = false; this.port.onmessage = e => { if(e.data.stop) { this.stopped=true; this.port.postMessage({stopped:true}); } }; }
        process(inputs) { if(this.stopped) return false; const audio=inputs[0]?.[0]; if(audio?.length) { const copy=new Float32Array(audio); this.port.postMessage({samples:copy},[copy.buffer]); } return true; }
      } registerProcessor('voice-capture',Capture);`
      const url = URL.createObjectURL(new Blob([code], { type: "application/javascript" }))
      try { await this.context.audioWorklet.addModule(url) } finally { URL.revokeObjectURL(url) }
      if (this.cancelled) { await this.release(); return }
      const started = await post<{ token: string; max_seconds: number }>("/api/recordings/start", { sample_rate: this.context.sampleRate })
      this.token = started.token
      if (this.cancelled) { await this.cancel(); return }
      this.maximum = Math.floor(started.max_seconds * this.context.sampleRate)
      this.node = new AudioWorkletNode(this.context, "voice-capture")
      this.node.port.onmessage = event => {
        if (event.data.stopped) { this.resolveStopped?.(); return }
        if (this.cancelled || !event.data.samples) return
        const audio = event.data.samples as Float32Array
        const chunk = new Int16Array(Math.min(audio.length, this.maximum - this.count))
        for (let i = 0; i < chunk.length; i++) {
          const value = Math.max(-1, Math.min(1, audio[i]))
          chunk[i] = Math.round(value * (value < 0 ? 32768 : 32767))
        }
        this.chunks.push(chunk)
        this.count += chunk.length
        if (this.count >= this.maximum) void this.finish()
      }
      this.source = this.context.createMediaStreamSource(this.stream)
      this.gain = this.context.createGain()
      this.gain.gain.value = 0
      this.source.connect(this.node)
      this.node.connect(this.gain)
      this.gain.connect(this.context.destination)
      this.streamPublished = true
      this.callbacks.onStream(this.stream)
      this.callbacks.onStatus("Listening. Speak one clear sentence.")
      this.timer = setInterval(() => {
        if (this.finishing || this.cancelled) return
        this.callbacks.onProgress(this.count / this.context!.sampleRate, this.text)
        if (!this.pending && !this.streamFailed) {
          this.pending = this.pump().finally(() => { this.pending = undefined })
        }
      }, 500)
      if (this.finishRequested) await this.finish()
    } catch (error) {
      if (this.cancelled) { await this.release(); return }
      await this.cancel()
      this.callbacks.onError(error instanceof Error && error.name === "NotAllowedError" ? "Microphone access was denied. Allow it in your browser or upload a recording." : String(error instanceof Error ? error.message : error))
    }
  }

  private async pump() {
    const target = this.count
    if (target === this.sent || this.cancelled) return
    const data = { token: this.token, sequence: this.sequence, pcm: encode(this.chunks, this.sent) }
    try {
      let result: { text?: string; status?: string }
      try { result = await post("/api/recordings/feed", data) }
      catch { result = await post("/api/recordings/feed", data) }
      if (this.cancelled) return
      this.sent = target
      this.sequence++
      this.text = result.text || ""
      this.callbacks.onProgress(this.count / this.context!.sampleRate, this.text)
      if (result.status && result.status !== "Listening…") this.callbacks.onStatus(result.status)
    } catch {
      if (this.cancelled) return
      this.streamFailed = true
      this.callbacks.onStatus("Live transcription paused. Your complete recording will be checked when you finish.")
    }
  }

  async finish() {
    if (this.finishing || this.cancelled) return
    if (!this.token) {
      if (!this.finishRequested) this.callbacks.onFinishing?.()
      this.finishRequested = true
      return
    }
    this.finishing = true
    this.callbacks.onFinishing?.()
    clearInterval(this.timer)
    this.callbacks.onStatus("Saving your recording and finishing its transcript…")
    try {
      // Worklet messages are ordered: acknowledgement follows the final PCM frame.
      const stopped = new Promise<void>(resolve => { this.resolveStopped = resolve })
      this.node?.port.postMessage({ stop: true })
      await Promise.race([stopped, new Promise(resolve => setTimeout(resolve, 300))])
      await this.release()
      await this.pending
      if (this.cancelled) return
      const result = await post<Reference>("/api/recordings/finish", { token: this.token, pcm: encode(this.chunks) })
      if (!this.cancelled) this.callbacks.onComplete(result)
    } catch (error) {
      if (this.cancelled) return
      await this.cancel()
      this.callbacks.onError(error instanceof Error ? error.message : "The recording could not be saved.")
    } finally { this.chunks = [] }
  }

  private async release() {
    clearInterval(this.timer)
    this.source?.disconnect()
    this.node?.disconnect()
    this.gain?.disconnect()
    this.stream?.getTracks().forEach(track => track.stop())
    if (this.streamPublished) {
      this.streamPublished = false
      this.callbacks.onStream(null)
    }
    if (this.context && this.context.state !== "closed") await this.context.close()
  }

  async cancel() {
    this.cancelled = true
    await this.release()
    if (this.token) await post("/api/recordings/cancel", { token: this.token }).catch(() => {})
    this.chunks = []
  }
}
