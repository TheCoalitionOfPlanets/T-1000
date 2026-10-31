import { test, expect, type Page } from "@playwright/test"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import ts from "typescript"

// Run the real recorder and its actual generated worklet in a deterministic
// browser audio graph. Network faults are injected at the POST boundary.
const source = readFileSync(fileURLToPath(new URL("../src/lib/recorder.ts", import.meta.url)), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText

async function recorder(page: Page, options: Record<string, unknown> = {}) {
  await page.goto("about:blank")
  await page.evaluate(async ({ code, options }) => {
    const state: any = { options, statuses: [], progress: [], completed: [], errors: [], recovered: null,
      feedRequests: [], finishRequests: [], cancelled: 0, feedAborted: 0, micStopped: 0, accepted: new Map(), samples: [], expected: [], published: [] }
    ;(window as any).captureTest = state
    const decode = (pcm: string) => {
      const raw = atob(pcm), bytes = Uint8Array.from(raw, char => char.charCodeAt(0))
      const view = new DataView(bytes.buffer)
      return Array.from({ length: bytes.length / 2 }, (_, i) => view.getInt16(i * 2, true))
    }
    const abortable = (signal?: AbortSignal) => new Promise((_resolve, reject) => {
      if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"))
      signal?.addEventListener("abort", () => { state.feedAborted++; reject(new DOMException("Aborted", "AbortError")) }, { once: true })
    })
    const post = async (url: string, data: any, signal?: AbortSignal) => {
      if (url.endsWith("/start")) { state.startRequest = data; return { token: "test-recording-token-0123456789", max_seconds: options.maximum ? Number(options.maximum) / 16000 : 6 } }
      if (url.endsWith("/cancel")) { state.cancelled++; return { cancelled: true } }
      if (url.endsWith("/feed")) {
        state.feedRequests.push({ ...data })
        if (options.hangFeed) return abortable(signal)
        if (!state.accepted.has(data.sequence)) {
          state.accepted.set(data.sequence, data.pcm)
          state.samples.push(...decode(data.pcm))
        } else if (state.accepted.get(data.sequence) !== data.pcm) throw new Error("Conflicting retry")
        if (options.loseFeedAck && state.feedRequests.length === 1) throw new TypeError("Connection lost after accepting PCM")
        return { sequence: data.sequence + (options.badAck ? 2 : 1), sample_count: state.samples.length + (options.badSampleCount ? 1 : 0),
          text: "preview", status: "Listening…", cancelled: !!options.cancelledAck }
      }
      if (url.endsWith("/finish")) {
        state.finishRequests.push({ ...data })
        if (options.invalidFinish) return { cancelled: true }
        if (options.failFinish || (options.loseFinishAck && state.finishRequests.length === 1)) throw new TypeError("Connection lost while saving")
        return { id: "saved-recording", url: "/api/audio/saved-recording", transcript: "Complete transcript.", status: "Ready" }
      }
      throw new Error("Unexpected URL: " + url)
    }
    class Processor {
      port: any = { onmessage: undefined, postMessage: () => {} }
    }
    class Context {
      sampleRate = 16000
      state = "running"
      destination = {}
      audioWorklet = { addModule: async (url: string) => {
        const text = await (await fetch(url)).text()
        new Function("AudioWorkletProcessor", "registerProcessor", text)(Processor, (_name: string, Class: any) => { state.Processor = Class })
      } }
      async resume() {}
      async close() { this.state = "closed" }
      createMediaStreamSource() { return { connect() {}, disconnect() {} } }
      createGain() { return { gain: { value: 0 }, connect() {}, disconnect() {} } }
    }
    class Node {
      processor: any
      port: any = { onmessage: undefined, postMessage: (data: any) => {
        if (options.noStopAck) return
        queueMicrotask(() => this.processor.port.onmessage?.({ data }))
      } }
      constructor(_context: any, _name: string, settings: any) {
        state.node = this
        this.processor = new state.Processor(settings)
        this.processor.port.postMessage = (data: any) => {
          if (options.dropFrame && data.samples && data.start_sample === 320) return
          queueMicrotask(() => this.port.onmessage?.({ data }))
        }
      }
      connect() {}
      disconnect() {}
    }
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: async (constraints: unknown) => { state.constraints = constraints; return { getTracks: () => [{ stop() { state.micStopped++ } }] } } } })
    Object.defineProperty(window, "AudioContext", { configurable: true, value: Context })
    Object.defineProperty(window, "AudioWorkletNode", { configurable: true, value: Node })
    const exports: any = {}
    new Function("require", "exports", code)((_name: string) => ({ post }), exports)
    state.recorder = new exports.VoiceRecorder({
      onStream: (value: any) => state.published.push(!!value),
      onProgress: (seconds: number, text: string) => state.progress.push({ seconds, text }),
      onStatus: (value: string) => state.statuses.push(value),
      onComplete: (value: any) => state.completed.push(value),
      onRecovery: (file: File, message: string) => { state.recovered = file; state.recoveryMessage = message },
      onError: (message: string) => state.errors.push(message),
    }, options.noiseReduction === undefined ? true : options.noiseReduction)
    await state.recorder.start()
    clearInterval(state.recorder.timer)
    state.push = async (length: number) => {
      const start = state.expected.length
      const samples = Float32Array.from({ length }, (_, i) => Math.sin((i + start) / 13) * 0.5)
      const remaining = state.recorder.maximum - state.expected.length
      state.expected.push(...Array.from(samples.subarray(0, Math.max(0, remaining)), value => (Math.round(value * (value < 0 ? 32768 : 32767)) || 0)))
      // Render quanta deliberately do not align with either network or 20ms batches.
      for (let offset = 0; offset < samples.length; offset += 128) state.node.processor.process([[samples.subarray(offset, offset + 128)]])
      await new Promise(resolve => setTimeout(resolve, 0))
    }
    state.finalSamples = () => decode(state.finishRequests.at(-1).pcm)
  }, { code: compiled, options })
}

for (const noiseReduction of [true, false]) {
  test(`capture and server noise reduction follow the selected option (${noiseReduction})`, async ({ page }) => {
    await recorder(page, { noiseReduction })
    const settings = await page.evaluate(() => {
      const s = (window as any).captureTest
      return { constraints: s.constraints, start: s.startRequest }
    })
    expect(settings.constraints).toEqual({ audio: { channelCount: 1, echoCancellation: noiseReduction, noiseSuppression: noiseReduction, autoGainControl: false } })
    expect(settings.start).toEqual({ sample_rate: 16000, noise_reduction: noiseReduction })
    await page.evaluate(async () => { await (window as any).captureTest.recorder.cancel() })
  })
}

async function state(page: Page) {
  return page.evaluate(() => {
    const s = (window as any).captureTest
    return { feed: s.feedRequests, finish: s.finishRequests, complete: s.completed, errors: s.errors,
      statuses: s.statuses, micStopped: s.micStopped, expected: s.expected, sent: s.recorder.sent,
      sequence: s.recorder.sequence, progress: s.progress, aborted: s.feedAborted, published: s.published,
      finalSamples: s.finishRequests.length ? s.finalSamples() : null }
  })
}

test("lost feed and finish acknowledgements retry identical bytes and preserve final partial render tail", async ({ page }) => {
  await recorder(page, { loseFeedAck: true, loseFinishAck: true })
  await page.evaluate(async () => {
    const s = (window as any).captureTest
    await s.push(640)
    await s.recorder.pump()
    await s.push(173)
    await s.recorder.finish()
  })
  const r = await state(page)
  expect(r.feed).toHaveLength(2)
  expect(r.feed[0]).toEqual(r.feed[1])
  expect(r.sequence).toBe(1)
  expect(r.sent).toBe(640)
  expect(r.finish).toHaveLength(2)
  expect(r.finish[0]).toEqual(r.finish[1])
  expect(r.finalSamples).toEqual(r.expected)
  expect(r.finalSamples).toHaveLength(813)
  expect(r.complete).toHaveLength(1)
  expect(r.errors).toEqual([])
  expect(r.published).toEqual([true, false])
})

test("finishing aborts hung live STT and reconciles all PCM instead of waiting for its timeout", async ({ page }) => {
  await recorder(page, { hangFeed: true })
  await page.evaluate(async () => {
    const s = (window as any).captureTest
    await s.push(640)
    s.recorder.pending = s.recorder.pump()
    await s.push(177)
    const before = performance.now()
    await s.recorder.finish()
    s.finishElapsed = performance.now() - before
  })
  const r = await state(page)
  expect(r.aborted).toBe(1)
  expect(r.feed).toHaveLength(1)
  expect(r.finalSamples).toEqual(r.expected)
  expect(r.complete).toHaveLength(1)
  expect(await page.evaluate(() => (window as any).captureTest.finishElapsed)).toBeLessThan(1000)
})

for (const options of [{ badAck: true }, { badSampleCount: true }, { cancelledAck: true }]) {
  test(`an invalid feed acknowledgement ${JSON.stringify(options)} pauses live feeds but preserves complete PCM`, async ({ page }) => {
    await recorder(page, options)
    await page.evaluate(async () => {
      const s = (window as any).captureTest
      await s.push(640)
      await s.recorder.pump()
      await s.push(199)
      await s.recorder.pump()
      await s.recorder.finish()
    })
    const r = await state(page)
    expect(r.sent).toBe(0)
    expect(r.sequence).toBe(0)
    expect(r.feed).toHaveLength(1)
    expect(r.statuses.some((s: string) => s.startsWith("Live transcription paused"))).toBe(true)
    expect(r.finalSamples).toEqual(r.expected)
    expect(r.complete).toHaveLength(1)
  })
}

test("auto-stop caps audio in the processor and flushes an exact non-aligned final batch only once", async ({ page }) => {
  await recorder(page, { maximum: 16033 })
  await page.evaluate(async () => { await (window as any).captureTest.push(16256) })
  await expect.poll(async () => (await state(page)).complete.length).toBe(1)
  const r = await state(page)
  expect(r.finish).toHaveLength(1)
  expect(r.finalSamples).toEqual(r.expected)
  expect(r.finalSamples).toHaveLength(16033)
  expect(r.errors).toEqual([])
})

test("missing processor tail is detected and never submitted as a complete recording", async ({ page }) => {
  await recorder(page, { dropFrame: true })
  await page.evaluate(async () => { const s = (window as any).captureTest; await s.push(640); await s.recorder.finish() })
  const r = await state(page)
  expect(r.finish).toEqual([])
  expect(r.complete).toEqual([])
  expect(r.errors.join(" ")).toMatch(/missing or repeated|sample count/)
})

test("missing stop acknowledgement is an explicit error rather than a silently truncated final recording", async ({ page }) => {
  await recorder(page, { noStopAck: true })
  await page.evaluate(async () => { const s = (window as any).captureTest; await s.push(640); await s.recorder.finish() })
  const r = await state(page)
  expect(r.finish).toEqual([])
  expect(r.complete).toEqual([])
  expect(r.errors.join(" ")).toContain("did not confirm its final audio frame")
})

test("permanent save failure exposes a complete PCM16 WAV for local recovery", async ({ page }) => {
  await recorder(page, { failFinish: true })
  await page.evaluate(async () => { const s = (window as any).captureTest; await s.push(817); await s.recorder.finish() })
  const r = await state(page)
  expect(r.finish).toHaveLength(2)
  expect(r.complete).toEqual([])
  expect(r.errors).toHaveLength(1)
  const recovery = await page.evaluate(async () => {
    const s = (window as any).captureTest
    const file: File = s.recovered
    const buffer = await file.arrayBuffer(), view = new DataView(buffer)
    return { name: file.name, type: file.type, riff: new TextDecoder().decode(buffer.slice(0, 4)),
      sampleRate: view.getUint32(24, true), dataBytes: view.getUint32(40, true),
      samples: Array.from({ length: (buffer.byteLength - 44) / 2 }, (_, i) => view.getInt16(44 + i * 2, true)) }
  })
  expect(recovery).toMatchObject({ name: "voice-recording.wav", type: "audio/wav", riff: "RIFF", sampleRate: 16000, dataBytes: 817 * 2 })
  expect(recovery.samples).toEqual(r.expected)
})

test("cancellation aborts in-flight STT and publishes no late transcript or completed reference", async ({ page }) => {
  await recorder(page, { hangFeed: true })
  await page.evaluate(async () => {
    const s = (window as any).captureTest
    await s.push(640)
    s.recorder.pending = s.recorder.pump()
    await s.recorder.cancel()
    await s.recorder.pending
  })
  const r = await state(page)
  expect(r.aborted).toBe(1)
  expect(r.complete).toEqual([])
  expect(r.errors).toEqual([])
  expect(r.progress).toEqual([])
  expect(r.published).toEqual([true, false])
})


test("an incomplete final response never becomes a voice reference and preserves local audio", async ({ page }) => {
  await recorder(page, { invalidFinish: true })
  await page.evaluate(async () => { const s = (window as any).captureTest; await s.push(817); await s.recorder.finish() })
  const r = await state(page)
  expect(r.complete).toEqual([])
  expect(r.finish).toHaveLength(2)
  expect(r.errors.join(" ")).toContain("did not confirm a complete saved recording")
  expect(await page.evaluate(() => (window as any).captureTest.recovered?.size)).toBe(44 + 817 * 2)
})

test("unresponsive live STT times out after one identical retry while captured PCM remains available", async ({ page }) => {
  await page.clock.install()
  await recorder(page, { hangFeed: true })
  await page.evaluate(() => {
    const s = (window as any).captureTest
    // Bypass test push's zero-delay timer while controlling the browser clock.
    const samples = Float32Array.from({ length: 640 }, (_, i) => Math.sin(i / 13) * 0.5)
    s.expected = Array.from(samples, value => Math.round(value * (value < 0 ? 32768 : 32767)) || 0)
    s.node.processor.process([[samples]])
  })
  await page.evaluate(() => { const s = (window as any).captureTest; s.recorder.pending = s.recorder.pump() })
  await page.clock.runFor(4001)
  await page.clock.runFor(4001)
  await page.evaluate(async () => { const s = (window as any).captureTest; await s.recorder.pending; await s.recorder.finish() })
  const r = await state(page)
  expect(r.feed).toHaveLength(2)
  expect(r.feed[0]).toEqual(r.feed[1])
  expect(r.aborted).toBe(2)
  expect(r.statuses.some((s: string) => s.startsWith("Live transcription paused"))).toBe(true)
  expect(r.finalSamples).toEqual(r.expected)
  expect(r.complete).toHaveLength(1)
})
