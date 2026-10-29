import { test, expect, type Page } from "@playwright/test"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { createHash } from "node:crypto"
import ts from "typescript"

const source = readFileSync(fileURLToPath(new URL("../src/lib/api.ts", import.meta.url)), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const job = "generation-test-0123456789"
const wav = Buffer.alloc(60)
wav.write("RIFF"); wav.writeUInt32LE(52, 4); wav.write("WAVEfmt ", 8)
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22)
wav.writeUInt32LE(48000, 24); wav.writeUInt32LE(96000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34)
wav.write("data", 36); wav.writeUInt32LE(16, 40)
;[0, 1000, -1000, 16384, -16384, 32767, -32768, 0].forEach((n, i) => wav.writeInt16LE(n, 44 + i * 2))
const metadata = { sample_rate: 48000, channels: 1, total_samples: 8, duration_seconds: 8 / 48000, chunks: 2,
  sha256: createHash("sha256").update(wav).digest("hex"), byte_length: wav.length,
  complete: true, reason: "eos", clipped_sample_count: 0 }

async function boot(page: Page) {
  await page.route("http://127.0.0.1:17863/**", route => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>Flow checks</title>" }))
  await page.goto("http://127.0.0.1:17863/")
  await page.evaluate(({ code, id, audio, bytes }) => {
    const exports: any = {}
    new Function("exports", code)(exports)
    const state: any = { api: exports, job: id, audio, bytes, requests: [], updates: [] }
    ;(window as any).flow = state
    state.encoder = new TextEncoder()
    state.event = (sequence: number, patch: any = {}) => ({ generation_id: id, sequence, status: "Checking your voice…", done: false, ...patch })
    state.terminal = (sequence: number, patch: any = {}) => state.event(sequence, { status: "Generated speech.", done: true,
      audio_url: "/api/audio/result", audio, partial: false, ...patch })
    state.response = (parts: Uint8Array[], settings: any = {}) => {
      let index = 0
      return new Response(new ReadableStream({
        pull(controller) {
          if (index < parts.length) controller.enqueue(parts[index++])
          else if (settings.broken) controller.error(new TypeError("Connection interrupted"))
          else if (!settings.open) controller.close()
        },
      }), { status: settings.status || 200, headers: settings.headers || { "Content-Type": "application/x-ndjson" } })
    }
    state.lines = (events: any[]) => state.encoder.encode(events.map(e => JSON.stringify(e)).join("\n") + "\n")
    state.input = { text: "Hello", transcript: "Reviewed reference.", reference_id: "example", seed: 42, steps: 10, generation_id: id }
  }, { code: compiled, id: job, audio: metadata, bytes: Array.from(wav) })
}

async function runGeneration(page: Page) {
  return page.evaluate(async () => {
    const s = (window as any).flow
    try {
      const timings = await s.api.generate(s.input, new AbortController().signal, (u: any) => s.updates.push(u))
      return { timings, requests: s.requests, updates: s.updates, error: null }
    } catch (error) { return { error: String((error as Error).message), requests: s.requests, updates: s.updates } }
  })
}

test("split UTF-8/NDJSON reconnects the immutable job at its last sequence and ignores duplicate snapshots", async ({ page }) => {
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async (_url, init) => {
      s.requests.push(JSON.parse(String(init?.body)))
      if (s.requests.length === 1) {
        s.input.text = "An edit after the request started must never change a retry."
        const bytes = s.lines([s.event(1), s.event(3, { status: "Maya’s voice 🎙️" })])
        return s.response(Array.from(bytes, (value: number) => Uint8Array.of(value)))
      }
      const bytes = s.lines([s.event(1), s.event(3), s.terminal(4)])
      return s.response([bytes], { open: true })
    }
  })
  const r = await runGeneration(page)
  expect(r.error).toBeNull()
  expect(r.requests).toHaveLength(2)
  expect(r.requests[0]).toMatchObject({ text: "Hello", generation_id: job, after_sequence: 0 })
  expect(r.requests[1]).toEqual({ ...r.requests[0], after_sequence: 3 })
  expect(r.updates.filter((u: any) => u.sequence).map((u: any) => u.sequence)).toEqual([1, 3, 4])
  expect(r.updates.filter((u: any) => u.done)).toHaveLength(1)
  expect(r.updates.some((u: any) => u.status.includes("Reconnecting"))).toBe(true)
})

for (const issue of ["wrong-id", "invalid-sequence", "wrong-done", "bad-metadata", "partial-mismatch", "invalid-utf8", "oversized-event"]) {
  test(`malformed generation event ${issue} fails without resubmitting the job`, async ({ page }) => {
    await boot(page)
    await page.evaluate(issue => {
      const s = (window as any).flow
      window.fetch = async () => {
        s.requests.push({})
        const event = s.terminal(1)
        if (issue === "wrong-id") event.generation_id = "different-generation-012345"
        if (issue === "invalid-sequence") event.sequence = 0
        if (issue === "wrong-done") event.done = "true"
        if (issue === "bad-metadata") event.audio = { ...s.audio, total_samples: 999 }
        if (issue === "partial-mismatch") event.partial = true
        if (issue === "oversized-event") event.status = "x".repeat(65536)
        return s.response([issue === "invalid-utf8" ? Uint8Array.of(255, 10) : s.lines([event])])
      }
    }, issue)
    const r = await runGeneration(page)
    expect(r.error).toBeTruthy()
    expect(r.requests).toHaveLength(1)
    expect(r.updates.filter((u: any) => u.done)).toEqual([])
  })
}

for (const status of [400, 409]) {
  test(`HTTP ${status} is fatal and never starts a replacement generation`, async ({ page }) => {
    await boot(page)
    await page.evaluate(status => {
      const s = (window as any).flow
      window.fetch = async () => { s.requests.push({}); return Response.json({ detail: "The request cannot be changed." }, { status }) }
    }, status)
    const r = await runGeneration(page)
    expect(r.error).toContain("cannot be changed")
    expect(r.requests).toHaveLength(1)
  })
}

test("legacy requests accept old terminal events and never retry a prematurely closed generation", async ({ page }) => {
  await boot(page)
  const r = await page.evaluate(async () => {
    const s = (window as any).flow
    delete s.input.generation_id
    window.fetch = async () => { s.requests.push({}); return s.response([s.lines([{ status: "Done", done: true, audio_url: "/api/audio/legacy" }])], { open: true }) }
    await s.api.generate(s.input, new AbortController().signal, (u: any) => s.updates.push(u))
    window.fetch = async () => { s.requests.push({}); return s.response([s.lines([{ status: "Working", done: false }])]) }
    let error = ""
    try { await s.api.generate(s.input, new AbortController().signal, () => {}) } catch (e) { error = String((e as Error).message) }
    return { calls: s.requests.length, terminal: s.updates.filter((u: any) => u.done).length, error }
  })
  expect(r.calls).toBe(2)
  expect(r.terminal).toBe(1)
  expect(r.error).toContain("closed before generation finished")
})

test("caller cancellation during headers never retries a generation", async ({ page }) => {
  await boot(page)
  const r = await page.evaluate(async () => {
    const s = (window as any).flow, controller = new AbortController()
    window.fetch = async () => { s.requests.push({}); return new Promise<Response>(() => {}) }
    const pending = s.api.generate(s.input, controller.signal, () => {})
    controller.abort()
    try { await pending; return { error: "", calls: s.requests.length } }
    catch (error) { return { error: (error as Error).name, calls: s.requests.length } }
  })
  expect(r.error).toBe("AbortError")
  expect(r.calls).toBe(1)
})

test("cancelled partial audio is explicit and terminal completion does not wait for HTTP EOF", async ({ page }) => {
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async () => {
      s.requests.push({})
      return s.response([s.lines([s.terminal(1, { partial: true, audio: { ...s.audio, complete: false, reason: "cancelled" } })])], { open: true })
    }
  })
  const r = await runGeneration(page)
  expect(r.error).toBeNull()
  expect(r.updates[0]).toMatchObject({ done: true, partial: true, audio: { complete: false, reason: "cancelled" } })
})

test("terminal cancellation before any audio is valid without an audio manifest", async ({ page }) => {
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async () => { s.requests.push({}); return s.response([s.lines([s.event(1, { done: true, status: "Stopped before any audio was produced." })])]) }
  })
  const r = await runGeneration(page)
  expect(r.error).toBeNull()
  expect(r.updates[0].done).toBe(true)
})

test("header timeouts have a bounded retry count and preserve the generation identifier", async ({ page }) => {
  await page.clock.install()
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async (_url, init) => { s.requests.push(JSON.parse(String(init?.body))); return new Promise<Response>(() => {}) }
    s.pending = s.api.generate(s.input, new AbortController().signal, () => {}).catch((e: Error) => { s.error = e.message })
  })
  for (let i = 0; i < 4; i++) await page.clock.runFor(11001)
  const r = await page.evaluate(async () => { const s = (window as any).flow; await s.pending; return { error: s.error, requests: s.requests } })
  expect(r.requests).toHaveLength(4)
  expect(r.requests.every((request: any) => request.generation_id === job && request.after_sequence === 0)).toBe(true)
  expect(r.error).toContain("timed out")
})

test("duplicate progress does not indefinitely extend the reconnect deadline", async ({ page }) => {
  await page.clock.install()
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async (_url, init) => {
      s.requests.push(JSON.parse(String(init?.body)))
      if (s.requests.length > 1) return s.response([s.lines([s.terminal(2)])])
      let timer: ReturnType<typeof setInterval>
      return new Response(new ReadableStream({ start(c) { c.enqueue(s.lines([s.event(1)])); timer = setInterval(() => c.enqueue(s.lines([s.event(1)])), 10000) }, cancel() { clearInterval(timer) } }))
    }
    s.pending = s.api.generate(s.input, new AbortController().signal, (u: any) => s.updates.push(u))
  })
  await page.clock.runFor(31001)
  const r = await page.evaluate(async () => { const s = (window as any).flow; await s.pending; return { requests: s.requests, updates: s.updates } })
  expect(r.requests).toHaveLength(2)
  expect(r.requests[1].after_sequence).toBe(1)
  expect(r.updates.filter((u: any) => u.sequence).map((u: any) => u.sequence)).toEqual([1, 2])
})

for (const mode of ["range", "restart", "clean-eof", "wrong-range"]) {
  test(`verified WAV download handles ${mode} recovery without duplicate bytes`, async ({ page }) => {
    await boot(page)
    const r = await page.evaluate(async mode => {
      const s = (window as any).flow, bytes = Uint8Array.from(s.bytes)
      window.fetch = async (_url, init) => {
        s.requests.push(new Headers(init?.headers).get("Range"))
        if (s.requests.length === 1) return s.response([bytes.slice(0, 17)], { broken: mode !== "clean-eof", headers: { "Content-Length": String(bytes.length) } })
        if (mode === "restart") return s.response([bytes], { headers: { "Content-Length": String(bytes.length) } })
        return s.response([bytes.slice(17, 35), bytes.slice(35)], { status: 206, headers: {
          "Content-Range": `bytes ${mode === "wrong-range" ? 16 : 17}-${bytes.length - 1}/${bytes.length}`,
          "Content-Length": String(bytes.length - 17),
        } })
      }
      try {
        const blob = await s.api.downloadVerifiedAudio("/api/audio/result", s.audio, new AbortController().signal)
        return { bytes: Array.from(new Uint8Array(await blob.arrayBuffer())), requests: s.requests, error: null }
      } catch (e) { return { bytes: null, requests: s.requests, error: String((e as Error).message) } }
    }, mode)
    expect(r.requests).toEqual([null, "bytes=17-"])
    if (mode === "wrong-range") { expect(r.error).toContain("resume range"); expect(r.bytes).toBeNull() }
    else { expect(r.error).toBeNull(); expect(r.bytes).toEqual(Array.from(wav)) }
  })
}

for (const mode of ["hash", "extra-bytes", "wrong-wav-frames", "oversized-manifest"]) {
  test(`verified download rejects ${mode} instead of publishing unverified audio`, async ({ page }) => {
    await boot(page)
    const r = await page.evaluate(async mode => {
      const s = (window as any).flow
      const bytes = Uint8Array.from(mode === "extra-bytes" ? [...s.bytes, 0] : s.bytes)
      const audio = { ...s.audio }
      if (mode === "hash") audio.sha256 = "f".repeat(64)
      if (mode === "wrong-wav-frames") { audio.total_samples = 7; audio.duration_seconds = 7 / audio.sample_rate }
      if (mode === "oversized-manifest") audio.byte_length = 32 * 1024 * 1024 + 1
      window.fetch = async () => { s.requests.push({}); return s.response([bytes]) }
      try { await s.api.downloadVerifiedAudio("/api/audio/result", audio, new AbortController().signal); return { error: "", calls: s.requests.length } }
      catch (e) { return { error: String((e as Error).message), calls: s.requests.length } }
    }, mode)
    expect(r.error).toBeTruthy()
    expect(r.calls).toBe(mode === "oversized-manifest" ? 0 : 1)
  })
}

test("abort from a progress callback suppresses later events already in the same network chunk", async ({ page }) => {
  await boot(page)
  const r = await page.evaluate(async () => {
    const s = (window as any).flow, controller = new AbortController()
    window.fetch = async () => { s.requests.push({}); return s.response([s.lines([s.event(1), s.terminal(2)])]) }
    try {
      await s.api.generate(s.input, controller.signal, (u: any) => { s.updates.push(u); controller.abort() })
      return { error: "", requests: s.requests.length, updates: s.updates }
    } catch (e) { return { error: (e as Error).name, requests: s.requests.length, updates: s.updates } }
  })
  expect(r.error).toBe("AbortError")
  expect(r.requests).toBe(1)
  expect(r.updates.map((u: any) => u.sequence)).toEqual([1])
})

test("EOF inside a UTF-8 codepoint resumes after the last complete accepted event", async ({ page }) => {
  await boot(page)
  await page.evaluate(() => {
    const s = (window as any).flow
    window.fetch = async (_url, init) => {
      s.requests.push(JSON.parse(String(init?.body)))
      if (s.requests.length > 1) return s.response([s.lines([s.terminal(3)])])
      const partial = s.encoder.encode('{"generation_id":"' + s.job + '","sequence":2,"status":"🎙️')
      return s.response([s.lines([s.event(1)]), partial.slice(0, partial.length - 4)])
    }
  })
  const r = await runGeneration(page)
  expect(r.error).toBeNull()
  expect(r.requests).toHaveLength(2)
  expect(r.requests[1].after_sequence).toBe(1)
  expect(r.updates.filter((u: any) => u.sequence).map((u: any) => u.sequence)).toEqual([1, 3])
})

test("caller cancellation of a verified download never retries or returns a partial Blob", async ({ page }) => {
  await boot(page)
  const r = await page.evaluate(async () => {
    const s = (window as any).flow, controller = new AbortController()
    window.fetch = async () => { s.requests.push({}); return new Promise<Response>(() => {}) }
    const pending = s.api.downloadVerifiedAudio("/api/audio/result", s.audio, controller.signal)
    controller.abort()
    try { await pending; return { error: "", calls: s.requests.length } }
    catch (e) { return { error: (e as Error).name, calls: s.requests.length } }
  })
  expect(r.error).toBe("AbortError")
  expect(r.calls).toBe(1)
})


test("cancelling while reconnecting prevents the next generation request", async ({ page }) => {
  await boot(page)
  const r = await page.evaluate(async () => {
    const s = (window as any).flow, controller = new AbortController()
    window.fetch = async () => { s.requests.push({}); return s.response([]) }
    try {
      await s.api.generate(s.input, controller.signal, (u: any) => { if (u.status.includes("Reconnecting")) controller.abort() })
      return { error: "", calls: s.requests.length }
    } catch (e) { return { error: (e as Error).name, calls: s.requests.length } }
  })
  expect(r.error).toBe("AbortError")
  expect(r.calls).toBe(1)
})
