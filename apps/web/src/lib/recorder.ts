import { post, type Reference } from "./api"

interface Callbacks {
  onStream: (stream: MediaStream | null) => void
  onProgress: (seconds: number, text: string) => void
  onStatus: (status: string) => void
  onFinishing?: () => void
  onComplete: (reference: Reference) => void
  onRecovery?: (file: File, message: string) => void
  onError: (message: string) => void
}

const FEED_TIMEOUT_MS = 4_000
const FINISH_TIMEOUT_MS = 60_000
const STOP_TIMEOUT_MS = 2_000

function pcmBytes(chunks: Int16Array[], start = 0): Uint8Array {
  const samples = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const bytes = new Uint8Array(Math.max(0, samples - start) * 2)
  const view = new DataView(bytes.buffer)
  let offset = 0
  let skipped = 0
  for (const chunk of chunks) {
    const from = Math.max(0, start - skipped)
    for (let index = from; index < chunk.length; index++) {
      view.setInt16(offset, chunk[index], true)
      offset += 2
    }
    skipped += chunk.length
  }
  return bytes
}

function encode(chunks: Int16Array[], start = 0): string {
  const bytes = pcmBytes(chunks, start)
  let binary = ""
  for (let index = 0; index < bytes.length; index += 8192) binary += String.fromCharCode(...bytes.subarray(index, index + 8192))
  return btoa(binary)
}

function recoveryFile(chunks: Int16Array[], sampleRate: number): File {
  const pcm = pcmBytes(chunks)
  const wav = new ArrayBuffer(44 + pcm.length)
  const bytes = new Uint8Array(wav)
  const view = new DataView(wav)
  const word = (at: number, value: string) => {
    for (let i = 0; i < value.length; i++) bytes[at + i] = value.charCodeAt(i)
  }
  word(0, "RIFF"); view.setUint32(4, 36 + pcm.length, true); word(8, "WAVE")
  word(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  word(36, "data"); view.setUint32(40, pcm.length, true); bytes.set(pcm, 44)
  return new File([wav], "voice-recording.wav", { type: "audio/wav" })
}

/** Coalesce render quanta without discarding samples, including the stop tail. */
function captureProcessorSource() {
  return `class Capture extends AudioWorkletProcessor {
    constructor(options) {
      super(); this.stopped=false; this.captured=0; this.filled=0;
      this.maximum=options.processorOptions.maximum;
      this.buffer=new Float32Array(options.processorOptions.chunkFrames);
      this.port.onmessage=e=>{if(e.data.stop)this.stop();};
    }
    flush() {
      if(!this.filled)return;
      const copy=this.buffer.slice(0,this.filled);
      this.port.postMessage({samples:copy,start_sample:this.captured-this.filled},[copy.buffer]);
      this.filled=0;
    }
    stop() {
      if(this.stopped)return;
      this.stopped=true; this.flush(); this.port.postMessage({stopped:true,captured_samples:this.captured});
    }
    process(inputs) {
      if(this.stopped)return false;
      const audio=inputs[0]?.[0];
      if(audio?.length) {
        let offset=0;
        const available=Math.min(audio.length,this.maximum-this.captured);
        while(offset<available) {
          const take=Math.min(available-offset,this.buffer.length-this.filled);
          this.buffer.set(audio.subarray(offset,offset+take),this.filled);
          offset+=take; this.filled+=take; this.captured+=take;
          if(this.filled===this.buffer.length)this.flush();
        }
        if(this.captured>=this.maximum){this.stop();return false;}
      }
      return true;
    }
  } registerProcessor('voice-capture',Capture);`
}

/** Final PCM remains independent of live STT requests and their acknowledgements. */
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
  private controllers = new Set<AbortController>()
  private feedController?: AbortController
  private cancelled = false
  private finishing = false
  private finishRequested = false
  private streamPublished = false
  private streamFailed = false
  private stoppedSamples?: number
  private captureError?: Error
  private text = ""
  private resolveStopped?: () => void

  constructor(private callbacks: Callbacks, private noiseReduction = true) {}

  private async send<T>(url: string, data: unknown, timeout: number, controller = new AbortController()): Promise<T> {
    this.controllers.add(controller)
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; controller.abort() }, timeout)
    try { return await post<T>(url, data, controller.signal) }
    catch (error) {
      if (timedOut) throw new Error("The connection timed out. Please try again.")
      throw error
    }
    finally { clearTimeout(timer); this.controllers.delete(controller) }
  }

  async start() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access needs localhost or HTTPS.")
      this.callbacks.onStatus("Allow microphone access to start recording…")
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: this.noiseReduction, noiseSuppression: this.noiseReduction, autoGainControl: false } })
      if (this.cancelled) { await this.release(); return }
      this.context = new AudioContext({ sampleRate: 16000 })
      await this.context.resume()
      const url = URL.createObjectURL(new Blob([captureProcessorSource()], { type: "application/javascript" }))
      try { await this.context.audioWorklet.addModule(url) } finally { URL.revokeObjectURL(url) }
      if (this.cancelled) { await this.release(); return }
      const started = await this.send<{ token: string; max_seconds: number }>("/api/recordings/start", { sample_rate: this.context.sampleRate, noise_reduction: this.noiseReduction }, 15_000)
      if (typeof started.token !== "string" || started.token.length < 20 || started.token.length > 100) throw new Error("The server did not confirm a recording session.")
      this.token = started.token
      if (this.cancelled) { await this.cancel(); return }
      this.maximum = Math.floor(started.max_seconds * this.context.sampleRate)
      if (!Number.isSafeInteger(this.maximum) || this.maximum <= 0 || started.max_seconds > 180) throw new Error("The server returned an invalid recording limit.")
      this.node = new AudioWorkletNode(this.context, "voice-capture", {
        processorOptions: { maximum: this.maximum, chunkFrames: Math.max(128, Math.round(this.context.sampleRate * 0.02)) },
      })
      this.node.port.onmessage = event => {
        if (event.data.stopped) {
          this.stoppedSamples = event.data.captured_samples
          this.resolveStopped?.()
          return
        }
        if (this.cancelled || !event.data.samples) return
        const audio = event.data.samples as Float32Array
        if (event.data.start_sample !== this.count || audio.length > this.maximum - this.count) {
          this.captureError = new Error("Recording audio arrived with a missing or repeated frame. Record again or upload a complete recording.")
          void this.finish()
          return
        }
        if (!audio.length) return
        const chunk = new Int16Array(audio.length)
        for (let i = 0; i < chunk.length; i++) {
          if (!Number.isFinite(audio[i])) {
            this.captureError = new Error("The microphone produced invalid audio. Record again or upload a recording.")
            void this.finish()
            return
          }
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
    if (target === this.sent || this.cancelled || this.finishing || this.streamFailed) return
    const data = { token: this.token, sequence: this.sequence, pcm: encode(this.chunks, this.sent) }
    try {
      let result: { sequence?: number; sample_count?: number; text?: string; status?: string; cancelled?: boolean } | undefined
      for (let attempt = 0; attempt < 2; attempt++) {
        const controller = new AbortController()
        this.feedController = controller
        try { result = await this.send("/api/recordings/feed", data, FEED_TIMEOUT_MS, controller); break }
        catch (error) { if (this.cancelled || this.finishing || attempt) throw error }
        finally { if (this.feedController === controller) this.feedController = undefined }
      }
      if (this.cancelled || this.finishing) return
      if (!result || result.cancelled) throw new Error("The recording session is unavailable.")
      if ((result.sequence !== undefined && result.sequence !== this.sequence + 1)
          || (result.sample_count !== undefined && result.sample_count !== target)) throw new Error("The recording acknowledgement did not match the audio sent.")
      this.sent = target
      this.sequence++
      this.text = result.text || ""
      this.callbacks.onProgress(this.count / this.context!.sampleRate, this.text)
      if (result.status && result.status !== "Listening…") this.callbacks.onStatus(result.status)
    } catch {
      if (this.cancelled || this.finishing) return
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
    let captureComplete = false
    try {
      // Stop flushes the processor's partial batch before the ordered ACK.
      if (this.stoppedSamples === undefined) {
        let timeout: ReturnType<typeof setTimeout> | undefined
        try {
          await new Promise<void>((resolve, reject) => {
            this.resolveStopped = resolve
            timeout = setTimeout(() => reject(new Error("The microphone did not confirm its final audio frame. The recording was not submitted; record again or upload audio.")), STOP_TIMEOUT_MS)
            this.node?.port.postMessage({ stop: true })
          })
        } finally { clearTimeout(timeout); this.resolveStopped = undefined }
      }
      if (this.captureError) throw this.captureError
      if (!Number.isSafeInteger(this.stoppedSamples) || this.stoppedSamples !== this.count) throw new Error("The microphone's final sample count did not match the recording. The recording was not submitted.")
      captureComplete = true
      await this.release()
      // The complete immutable final PCM repairs a failed, unacknowledged or
      // unsent feed. Aborting a slow preview does not discard local samples.
      this.feedController?.abort()
      await this.pending
      if (this.cancelled) return
      const data = { token: this.token, pcm: encode(this.chunks) }
      let result: Reference | undefined
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          result = await this.send<Reference>("/api/recordings/finish", data, FINISH_TIMEOUT_MS)
          if (!result || typeof result.id !== "string" || !result.id || typeof result.url !== "string" || !result.url
              || typeof result.transcript !== "string" || typeof result.status !== "string") throw new Error("The server did not confirm a complete saved recording.")
          break
        }
        catch (error) { if (this.cancelled || attempt) throw error }
      }
      if (!this.cancelled && result) this.callbacks.onComplete(result)
    } catch (error) {
      if (this.cancelled) return
      if (captureComplete && this.count && this.context) {
        this.callbacks.onRecovery?.(recoveryFile(this.chunks, this.context.sampleRate), "Your complete recording is preserved locally. Download it or retry uploading it.")
      }
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
    for (const controller of this.controllers) controller.abort()
    this.resolveStopped?.()
    await this.release()
    // Releasing microphone/UI never waits for a disconnected cancel request.
    if (this.token) void this.send("/api/recordings/cancel", { token: this.token }, FEED_TIMEOUT_MS).catch(() => {})
    this.chunks = []
  }
}
