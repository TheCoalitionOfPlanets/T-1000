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

export interface GenerationUpdate {
  status: string
  audio_url?: string
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

export async function generate(data: unknown, signal: AbortSignal, onUpdate: (update: GenerationUpdate) => void): Promise<GenerationRequestTimings> {
  const started = performance.now()
  const response = await fetch("/api/generate", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data), signal,
  })
  const responseHeadersMs = performance.now() - started
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(typeof body.detail === "string" ? body.detail : "Generation could not start. Please try again.")
  }
  if (!response.body) throw new Error("Your browser could not read the audio response.")
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let completed = false
  let firstEventMs: number | undefined
  let completionMs: number | undefined
  let serverTimings: GenerationTimings | undefined
  const consume = (line: string) => {
    if (!line.trim()) return
    const update = JSON.parse(line) as GenerationUpdate
    update.client_elapsed_ms = performance.now() - started
    firstEventMs ??= update.client_elapsed_ms
    if (update.done) completionMs = update.client_elapsed_ms
    if (update.timings) serverTimings = update.timings
    completed ||= !!update.done
    onUpdate(update)
  }
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split("\n")
      buffer = lines.pop() || ""
      lines.forEach(consume)
      if (done) break
    }
    consume(buffer)
    if (!completed) throw new Error("The connection closed before generation finished. Please try again.")
    return {
      response_headers_ms: responseHeadersMs,
      first_event_ms: firstEventMs!,
      completion_ms: completionMs!,
      total_ms: performance.now() - started,
      server: serverTimings,
    }
  } finally {
    reader.releaseLock()
  }
}
