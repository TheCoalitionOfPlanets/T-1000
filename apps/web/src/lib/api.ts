export interface Reference {
  id: string
  url: string
  transcript: string
  status: string
}

export interface Config {
  device: string
  reference_limit: number
  text_limit: number
  example: Reference | null
  sample_url: string | null
  resumable_generation?: boolean
}

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}). Please try again.`)
  }
  return response.json()
}

export function post<T>(url: string, data: unknown, signal?: AbortSignal) {
  return request<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data), signal })
}

export interface GenerationTimings {
  reference_validated_ms: number
  first_status_ms: number
  completion_ms?: number
}

export interface AudioMetadata {
  sample_rate: number
  channels: number
  total_samples: number
  duration_seconds: number
  chunks: number
  sha256: string
  byte_length: number
  complete: boolean
  reason: string
  clipped_sample_count: number
  clipped?: boolean
  error?: string
}

export interface GenerationUpdate {
  status: string
  audio_url?: string
  audio?: AudioMetadata
  partial?: boolean
  generation_id?: string
  sequence?: number
  done?: boolean
  error?: string
  elapsed_ms?: number
  timings?: GenerationTimings
  client_elapsed_ms?: number
}

export interface GenerationRequestTimings {
  response_headers_ms: number
  first_event_ms: number
  completion_ms: number
  total_ms: number
  server?: GenerationTimings
}

const MAX_EVENT_BYTES = 64 * 1024
const MAX_AUDIO_BYTES = 32 * 1024 * 1024
const HEADER_TIMEOUT_MS = 10_000
const PROGRESS_TIMEOUT_MS = 30_000

class TransportError extends Error {}
class ProtocolError extends Error {}
class HTTPError extends Error {
  constructor(message: string, readonly retryable: boolean) { super(message) }
}

function aborted(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new DOMException("Request cancelled.", "AbortError")
}

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function integer(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum && value <= maximum
}

function validateAudio(value: unknown): AudioMetadata {
  if (!object(value)
      || !integer(value.sample_rate, 8000, 192000) || value.channels !== 1
      || !integer(value.total_samples, 1, MAX_AUDIO_BYTES / 2)
      || !finite(value.duration_seconds) || value.duration_seconds <= 0
      || Math.abs(value.duration_seconds - value.total_samples / value.sample_rate) > 1 / value.sample_rate
      || !integer(value.chunks, 1, value.total_samples)
      || !integer(value.byte_length, 44 + value.total_samples * 2, MAX_AUDIO_BYTES)
      || typeof value.sha256 !== "string" || !/^[a-fA-F0-9]{64}$/.test(value.sha256)
      || typeof value.complete !== "boolean" || typeof value.reason !== "string" || !value.reason || value.reason.length > 100
      || !integer(value.clipped_sample_count, 0, value.total_samples)
      || (value.clipped !== undefined && (typeof value.clipped !== "boolean" || value.clipped !== (value.clipped_sample_count > 0)))
      || (value.complete && !["eos", "stream_complete"].includes(value.reason))
      || (!value.complete && ["eos", "stream_complete"].includes(value.reason))) {
    throw new ProtocolError("The server returned invalid audio integrity information.")
  }
  return value as unknown as AudioMetadata
}

function audioURL(value: unknown): string {
  if (typeof value !== "string" || !value) throw new ProtocolError("The generated audio address is invalid.")
  const url = new URL(value, location.href)
  if (url.origin !== location.origin || !url.pathname.startsWith("/api/audio/") || url.username || url.password || url.hash) {
    throw new ProtocolError("The generated audio address is invalid.")
  }
  return url.href
}

/** Abort the actual fetch as well as bounding transports that ignore abort. */
async function bounded<T>(operation: Promise<T>, controller: AbortController, signal: AbortSignal, milliseconds: number): Promise<T> {
  if (signal.aborted) { controller.abort(); throw aborted(signal) }
  let timer: ReturnType<typeof setTimeout> | undefined
  let cancel: (() => void) | undefined
  try {
    return await Promise.race([operation, new Promise<never>((_resolve, reject) => {
      cancel = () => { controller.abort(); reject(aborted(signal)) }
      signal.addEventListener("abort", cancel, { once: true })
      timer = setTimeout(() => {
        controller.abort()
        reject(new TransportError("The connection timed out. Reconnecting may recover the request."))
      }, Math.max(1, milliseconds))
    })])
  } finally {
    clearTimeout(timer)
    if (cancel) signal.removeEventListener("abort", cancel)
  }
}

async function headers(url: string, init: RequestInit, controller: AbortController, signal: AbortSignal): Promise<Response> {
  try { return await bounded(fetch(url, { ...init, signal: controller.signal }), controller, signal, HEADER_TIMEOUT_MS) }
  catch (error) {
    if (signal.aborted) throw aborted(signal)
    if (error instanceof TransportError) throw error
    throw new TransportError("The connection was interrupted before a response arrived.")
  }
}

async function responseError(response: Response, controller: AbortController, signal: AbortSignal): Promise<HTTPError> {
  let detail: unknown
  try { detail = (await bounded(response.json(), controller, signal, HEADER_TIMEOUT_MS))?.detail } catch { /* Keep status when the error body is unavailable. */ }
  const message = typeof detail === "string" ? detail : `The request failed (${response.status}). Please try again.`
  return new HTTPError(message, response.status === 408 || response.status === 429 || response.status >= 500)
}

async function read(reader: ReadableStreamDefaultReader<Uint8Array>, controller: AbortController, signal: AbortSignal, milliseconds: number) {
  if (milliseconds <= 0) throw new TransportError("The connection stopped making progress.")
  try { return await bounded(reader.read(), controller, signal, milliseconds) }
  catch (error) {
    if (signal.aborted) throw aborted(signal)
    if (error instanceof TransportError) throw error
    throw new TransportError("The connection closed before the transfer finished.")
  }
}

function retryDelay(signal: AbortSignal, milliseconds: number): Promise<void> {
  if (signal.aborted) return Promise.reject(aborted(signal))
  return new Promise((resolve, reject) => {
    const cancel = () => { clearTimeout(timer); signal.removeEventListener("abort", cancel); reject(aborted(signal)) }
    const timer = setTimeout(() => { signal.removeEventListener("abort", cancel); resolve() }, milliseconds)
    signal.addEventListener("abort", cancel, { once: true })
  })
}

function dispose(reader: ReadableStreamDefaultReader<Uint8Array> | undefined, controller: AbortController) {
  controller.abort()
  if (reader) {
    void reader.cancel().catch(() => {})
    try { reader.releaseLock() } catch { /* A cancelled read may still be settling. */ }
  }
}

function validateUpdate(value: unknown, resumable: boolean): GenerationUpdate {
  if (!object(value) || typeof value.status !== "string"
      || (value.done !== undefined && typeof value.done !== "boolean")
      || (resumable && typeof value.done !== "boolean")
      || (value.error !== undefined && typeof value.error !== "string")
      || (value.partial !== undefined && typeof value.partial !== "boolean")
      || (value.elapsed_ms !== undefined && (!finite(value.elapsed_ms) || value.elapsed_ms < 0))) {
    throw new ProtocolError("The server returned a malformed generation update.")
  }
  if (value.timings !== undefined) {
    const t = value.timings
    if (!object(t) || !finite(t.reference_validated_ms) || t.reference_validated_ms < 0
        || !finite(t.first_status_ms) || t.first_status_ms < 0
        || (t.completion_ms !== undefined && (!finite(t.completion_ms) || t.completion_ms < 0))) {
      throw new ProtocolError("The server returned malformed request timing information.")
    }
  }
  if (value.audio_url !== undefined) {
    audioURL(value.audio_url)
    if (!value.done) throw new ProtocolError("Audio was published before the generation finished.")
    if (resumable || value.audio !== undefined) {
      const metadata = validateAudio(value.audio)
      if ((resumable && typeof value.partial !== "boolean") || (value.partial !== undefined && value.partial !== !metadata.complete)) {
        throw new ProtocolError("The audio completion information is inconsistent.")
      }
    }
  } else if (value.audio !== undefined) throw new ProtocolError("The audio result is missing its download address.")
  return value as unknown as GenerationUpdate
}

/** Reconnect one immutable generation job; never regenerate a lost response. */
export async function generate(data: unknown, signal: AbortSignal, onUpdate: (update: GenerationUpdate) => void): Promise<GenerationRequestTimings> {
  if (signal.aborted) throw aborted(signal)
  const encoded = JSON.stringify(data)
  const payload: unknown = encoded === undefined ? undefined : JSON.parse(encoded)
  const id = object(payload) ? payload.generation_id : undefined
  if (id !== undefined && (typeof id !== "string" || !/^[A-Za-z0-9_-]{16,64}$/.test(id))) throw new ProtocolError("The generation identifier is invalid.")
  const resumable = typeof id === "string"
  const initialSequence = object(payload) && payload.after_sequence !== undefined ? payload.after_sequence : 0
  if (!integer(initialSequence, 0, Number.MAX_SAFE_INTEGER)) throw new ProtocolError("The generation cursor is invalid.")
  let sequence: number = initialSequence
  const started = performance.now()
  let responseHeadersMs: number | undefined
  let firstEventMs: number | undefined
  let completionMs: number | undefined
  let serverTimings: GenerationTimings | undefined
  for (let attempt = 0; ; attempt++) {
    if (signal.aborted) throw aborted(signal)
    const controller = new AbortController()
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
    try {
      const body = resumable ? JSON.stringify({ ...(payload as Record<string, unknown>), after_sequence: sequence }) : encoded
      const response = await headers("/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      }, controller, signal)
      responseHeadersMs ??= performance.now() - started
      if (!response.ok) throw await responseError(response, controller, signal)
      if (!response.body) throw new TransportError("Your browser could not read the generation response.")
      reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8", { fatal: true })
      let buffer = "", bufferedBytes = 0, complete = false
      let progressDeadline = performance.now() + PROGRESS_TIMEOUT_MS
      const consume = (line: string) => {
        if (signal.aborted) throw aborted(signal)
        if (!line.trim()) return
        let value: unknown
        try { value = JSON.parse(line) } catch { throw new ProtocolError("The generation response contains invalid JSON.") }
        if (resumable) {
          if (!object(value) || value.generation_id !== id || !integer(value.sequence, 1, Number.MAX_SAFE_INTEGER)) throw new ProtocolError("The generation response belongs to a different request or has an invalid sequence.")
          if (value.sequence <= sequence) return
        }
        const update = validateUpdate(value, resumable)
        if (resumable) sequence = update.sequence!
        update.client_elapsed_ms = performance.now() - started
        firstEventMs ??= update.client_elapsed_ms
        progressDeadline = performance.now() + PROGRESS_TIMEOUT_MS
        if (update.timings) serverTimings = update.timings
        if (update.done) { complete = true; completionMs = update.client_elapsed_ms }
        onUpdate(update)
      }
      while (!complete) {
        const next = await read(reader, controller, signal, progressDeadline - performance.now())
        if (next.done) {
          try { buffer += decoder.decode() }
          catch { throw new TransportError("The connection closed during a generation update.") }
          if (buffer.trim()) {
            try { consume(buffer) }
            catch (error) {
              if (error instanceof ProtocolError && resumable && error.message.includes("invalid JSON")) throw new TransportError("The connection closed during a generation update.")
              throw error
            }
          }
          if (!complete) throw new TransportError("The connection closed before generation finished. Please try again.")
          break
        }
        const value = next.value
        for (let offset = 0; offset < value.length && !complete;) {
          const newline = value.indexOf(10, offset)
          const end = newline < 0 ? value.length : newline + 1
          bufferedBytes += end - offset - (newline < 0 ? 0 : 1)
          if (bufferedBytes > MAX_EVENT_BYTES) throw new ProtocolError("A generation update exceeded the safe size limit.")
          try { buffer += decoder.decode(value.subarray(offset, end), { stream: true }) }
          catch { throw new ProtocolError("The generation response contains invalid UTF-8.") }
          if (newline >= 0) {
            consume(buffer.slice(0, -1))
            buffer = ""; bufferedBytes = 0
          }
          offset = end
        }
      }
      if (signal.aborted) throw aborted(signal)
      return { response_headers_ms: responseHeadersMs!, first_event_ms: firstEventMs!,
        completion_ms: completionMs!, total_ms: performance.now() - started, server: serverTimings }
    } catch (error) {
      if (signal.aborted) throw aborted(signal)
      const retryable = error instanceof TransportError || (error instanceof HTTPError && error.retryable)
      if (!resumable || !retryable || attempt >= 3) throw error
      onUpdate({ status: "Connection interrupted. Reconnecting to your existing generation…", generation_id: id, client_elapsed_ms: performance.now() - started })
    } finally { dispose(reader, controller) }
    await retryDelay(signal, 250 * (attempt + 1))
  }
}

function validateWav(bytes: Uint8Array, metadata: AudioMetadata) {
  if (bytes.length < 44) throw new ProtocolError("The downloaded WAV is incomplete.")
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const text = (offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4))
  if (text(0) !== "RIFF" || text(8) !== "WAVE" || view.getUint32(4, true) !== bytes.length - 8) throw new ProtocolError("The downloaded WAV header does not match its length.")
  let format = false, data = false
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const size = view.getUint32(offset + 4, true), start = offset + 8, end = start + size
    if (end > bytes.length) throw new ProtocolError("The downloaded WAV has a truncated audio chunk.")
    if (text(offset) === "fmt ") {
      if (size < 16 || view.getUint16(start, true) !== 1 || view.getUint16(start + 2, true) !== metadata.channels
          || view.getUint32(start + 4, true) !== metadata.sample_rate || view.getUint16(start + 14, true) !== 16
          || view.getUint16(start + 12, true) !== metadata.channels * 2
          || view.getUint32(start + 8, true) !== metadata.sample_rate * metadata.channels * 2) throw new ProtocolError("The downloaded WAV format does not match the generated audio.")
      format = true
    } else if (text(offset) === "data") {
      if (data || size !== metadata.total_samples * metadata.channels * 2) throw new ProtocolError("The downloaded WAV sample count does not match the generated audio.")
      data = true
    }
    if (end + (size % 2) > bytes.length) throw new ProtocolError("The downloaded WAV has incomplete chunk padding.")
    offset = end + (size % 2)
  }
  if (!format || !data) throw new ProtocolError("The downloaded WAV is missing its audio format or samples.")
}

/** Download exactly the committed WAV, resuming byte ranges without duplication. */
export async function downloadVerifiedAudio(url: string, value: AudioMetadata, signal: AbortSignal): Promise<Blob> {
  const metadata = validateAudio(value)
  const address = audioURL(url)
  if (!crypto.subtle) throw new Error("Verified audio downloads require localhost or HTTPS.")
  let chunks: Uint8Array[] = [], received = 0
  for (let attempt = 0; attempt < 3; attempt++) {
    if (signal.aborted) throw aborted(signal)
    const controller = new AbortController()
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
    try {
      const requestedOffset = received
      const response = await headers(address, { headers: requestedOffset ? { Range: `bytes=${requestedOffset}-` } : {} }, controller, signal)
      if (!response.ok) throw await responseError(response, controller, signal)
      let responseLimit = metadata.byte_length
      if (response.status === 206) {
        const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("Content-Range") || "")
        if (!range || Number(range[1]) !== requestedOffset || Number(range[3]) !== metadata.byte_length
            || Number(range[2]) < requestedOffset || Number(range[2]) >= metadata.byte_length) throw new ProtocolError("The audio resume range does not match the downloaded bytes.")
        responseLimit = Number(range[2]) - requestedOffset + 1
        const length = response.headers.get("Content-Length")
        if (length !== null && Number(length) !== responseLimit) throw new ProtocolError("The audio resume response has an invalid length.")
      } else if (response.status === 200) {
        // A server without range support restarts the whole download cleanly.
        chunks = []; received = 0
        const length = response.headers.get("Content-Length")
        if (length !== null && Number(length) !== metadata.byte_length) throw new ProtocolError("The audio download length does not match the saved file.")
      } else throw new ProtocolError("The server returned an unsupported audio transfer response.")
      if (!response.body) throw new TransportError("The audio response was empty.")
      reader = response.body.getReader()
      let responseReceived = 0, progressDeadline = performance.now() + PROGRESS_TIMEOUT_MS
      while (received < metadata.byte_length) {
        const piece = await read(reader, controller, signal, progressDeadline - performance.now())
        if (piece.done) throw new TransportError("The audio connection closed before every byte arrived.")
        responseReceived += piece.value.length
        if (responseReceived > responseLimit || received + piece.value.length > metadata.byte_length) throw new ProtocolError("The audio download contains unexpected extra bytes.")
        if (piece.value.length) { chunks.push(piece.value.slice()); received += piece.value.length; progressDeadline = performance.now() + PROGRESS_TIMEOUT_MS }
      }
      break
    } catch (error) {
      if (signal.aborted) throw aborted(signal)
      const retryable = error instanceof TransportError || (error instanceof HTTPError && error.retryable)
      if (!retryable || attempt === 2) throw error
    } finally { dispose(reader, controller) }
    await retryDelay(signal, 250 * (attempt + 1))
  }
  if (signal.aborted) throw aborted(signal)
  if (received !== metadata.byte_length) throw new ProtocolError("The complete generated audio could not be downloaded.")
  const bytes = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length }
  const checksum = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.buffer)), byte => byte.toString(16).padStart(2, "0")).join("")
  if (signal.aborted) throw aborted(signal)
  if (checksum !== metadata.sha256.toLowerCase()) throw new ProtocolError("The downloaded audio failed its integrity check. Please retry the download.")
  validateWav(bytes, metadata)
  return new Blob([bytes.buffer], { type: "audio/wav" })
}
