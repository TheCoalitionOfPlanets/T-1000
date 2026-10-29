import { test, expect, type Page } from "@playwright/test"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

const audio = readFileSync(fileURLToPath(new URL("../../../assets/ref.wav", import.meta.url)))
const example = {
  id: "example",
  url: "/api/audio/example",
  transcript: "We asked over twenty different people, and they all said it was his.",
  status: "Example voice ready.",
}
const config = { device: "CPU", reference_limit: 10, text_limit: 150, example, sample_url: null }

async function mockAudio(page: Page) {
  await page.route("**/api/audio/**", route => route.fulfill({ status: 200, contentType: "audio/wav", body: audio }))
}

async function ready(page: Page, overrides = {}) {
  await mockAudio(page)
  await page.route("**/api/config", route => route.fulfill({ json: { ...config, ...overrides } }))
  await page.goto("/")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
}

test("example voice is editable, and a failed replacement disables generation", async ({ page }) => {
  await ready(page)
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await page.getByLabel("Reference transcript").fill("The words I corrected.")
  await page.route("**/api/reference", route => route.fulfill({ status: 400, json: { detail: "The reference recording is silent or too quiet." } }))
  await page.locator('input[type="file"]').setInputFiles({ name: "silent.wav", mimeType: "audio/wav", buffer: audio })
  await expect(page.getByRole("alert")).toContainText("silent or too quiet")
  await expect(page.getByLabel("Reference transcript")).toHaveValue("")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await page.getByRole("button", { name: "Use example voice" }).click()
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
})

test("generation sends settings, stays cancellable, and presents resulting audio", async ({ page }) => {
  await ready(page)
  await page.getByLabel("Text to speak").fill("Make these words speak.")
  await page.getByLabel("Generation seed").fill("123")
  let complete!: () => void
  const pending = new Promise<void>(resolve => { complete = resolve })
  let payload: Record<string, unknown> | undefined
  let stopped = false
  await page.route("**/api/generate", async route => {
    payload = route.request().postDataJSON()
    await pending
    await route.fulfill({ contentType: "application/x-ndjson", body: '{"status":"Synthesizing…","done":false}\n{"status":"Generated speech.","audio_url":"/api/audio/generated","done":true}\n' })
  })
  await page.route("**/api/stop", route => { stopped = true; return route.fulfill({ json: { status: "Stop requested." } }) })
  await page.getByRole("button", { name: "Generate speech" }).click()
  await expect(page.getByRole("button", { name: "Stop generation" })).toBeEnabled()
  await expect(page.getByLabel("Text to speak")).toBeDisabled()
  await page.getByRole("button", { name: "Stop generation" }).click()
  await expect(page.getByRole("button", { name: "Stopping…" })).toBeDisabled()
  expect(stopped).toBe(true)
  expect(payload).toEqual({ text: "Make these words speak.", reference_id: "example", transcript: example.transcript, steps: 10, seed: 123 })
  complete()
  await expect(page.getByRole("link", { name: "Download speech as WAV" })).toHaveAttribute("href", "/api/audio/generated")
  await expect(page.getByText("Your generated speech", { exact: true })).toBeVisible()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await expect(page.locator(".output-panel audio")).toHaveAttribute("src", /\/api\/audio\/generated$/)
})

test("number expansion is reviewed, applied, undone, and cancelled explicitly", async ({ page }) => {
  await ready(page)
  const original = "Pay $12.50."
  const expanded = "Pay twelve point five dollars."
  await page.getByLabel("Text to speak").fill(original)
  await page.route("**/api/text/normalize", route => {
    expect(route.request().postDataJSON()).toEqual({ text: original })
    return route.fulfill({ json: { text: expanded } })
  })
  await page.getByRole("button", { name: "Expand numbers", exact: true }).click()
  await expect(page.getByRole("region", { name: "Spoken text preview" })).toContainText(expanded)
  await expect(page.getByLabel("Text to speak")).toHaveValue(original)
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await page.getByRole("button", { name: "Apply", exact: true }).click()
  await expect(page.getByLabel("Text to speak")).toHaveValue(expanded)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await page.getByRole("button", { name: "Undo expansion" }).click()
  await expect(page.getByLabel("Text to speak")).toHaveValue(original)
  await page.getByRole("button", { name: "Expand numbers", exact: true }).click()
  await expect(page.getByRole("region", { name: "Spoken text preview" })).toBeVisible()
  await page.getByRole("button", { name: "Cancel", exact: true }).click()
  await expect(page.getByRole("region", { name: "Spoken text preview" })).toHaveCount(0)
  await expect(page.getByLabel("Text to speak")).toHaveValue(original)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
})

test("editing text invalidates a pending expansion and ignores its late response", async ({ page }) => {
  await ready(page)
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  let requested = false
  await page.route("**/api/text/normalize", async route => {
    requested = true
    await pending
    await route.fulfill({ json: { text: "Old expanded text." } }).catch(() => {})
  })
  await page.getByLabel("Text to speak").fill("Pay $12.")
  await page.getByRole("button", { name: "Expand numbers", exact: true }).click()
  await expect.poll(() => requested).toBe(true)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await page.getByLabel("Text to speak").fill("My newly edited words.")
  release()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await expect(page.getByRole("button", { name: "Expand numbers", exact: true })).toBeEnabled()
  await expect(page.getByRole("region", { name: "Spoken text preview" })).toHaveCount(0)
  await expect(page.getByLabel("Text to speak")).toHaveValue("My newly edited words.")
  await expect(page.getByRole("alert")).toHaveCount(0)
})

test("an overlong expansion preserves the original and mobile preview fits", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await ready(page)
  await page.getByLabel("Text to speak").fill("123")
  await page.route("**/api/text/normalize", route => route.fulfill({ status: 400, json: { detail: "Expanded text has 151 characters; shorten the original to fit the 150-character limit." } }))
  await page.getByRole("button", { name: "Expand numbers", exact: true }).click()
  await expect(page.getByRole("alert")).toContainText("150-character limit")
  await expect(page.getByLabel("Text to speak")).toHaveValue("123")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await page.unroute("**/api/text/normalize")
  await page.route("**/api/text/normalize", route => route.fulfill({ json: { text: "One hundred twenty three." } }))
  await page.getByRole("button", { name: "Expand numbers", exact: true }).click()
  await expect(page.getByRole("region", { name: "Spoken text preview" })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  const preview = page.getByRole("region", { name: "Spoken text preview" })
  expect(await preview.evaluate(element => element.getBoundingClientRect().right <= window.innerWidth)).toBe(true)
})

test("a missing configuration can be retried without reloading", async ({ page }) => {
  await mockAudio(page)
  let attempts = 0
  await page.route("**/api/config", route => ++attempts === 1
    ? route.fulfill({ status: 503, json: { detail: "Unavailable" } })
    : route.fulfill({ json: config }))
  await page.goto("/")
  await expect(page.getByRole("alert")).toContainText("Could not connect")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await page.getByRole("button", { name: "Retry connection" }).click()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await expect(page.getByRole("alert")).toHaveCount(0)
  expect(attempts).toBe(2)
})

test("mobile studio and help fit within the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await ready(page)
  await expect(page.getByRole("heading", { name: "Good words. Great voice." })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByRole("button", { name: "Quick guide" }).click()
  await expect(page.getByRole("dialog")).toBeVisible()
  expect(await page.getByRole("dialog").evaluate(element => element.getBoundingClientRect().right <= window.innerWidth)).toBe(true)
  await page.keyboard.press("Escape")
  await expect(page.getByRole("dialog")).toHaveCount(0)
})

test("microphone auto-finish sends all PCM once and shares one microphone stream", async ({ page }) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    Object.defineProperty(window, "microphoneRequests", { value: 0, writable: true })
    navigator.mediaDevices.getUserMedia = constraints => {
      const state = window as unknown as { microphoneRequests: number }
      state.microphoneRequests++
      return original(constraints)
    }
  })
  let sampleRate = 0
  let finalPCM: Buffer | undefined
  let finishCount = 0
  let save!: () => void
  const saving = new Promise<void>(resolve => { save = resolve })
  await page.route("**/api/recordings/start", route => {
    sampleRate = route.request().postDataJSON().sample_rate
    return route.fulfill({ json: { token: "test-recording-token-1234", max_seconds: 1.5 } })
  })
  await page.route("**/api/recordings/feed", route => route.fulfill({ json: { text: "We asked over twenty different people.", status: "Listening…" } }))
  await page.route("**/api/recordings/finish", async route => {
    finishCount++
    finalPCM = Buffer.from(route.request().postDataJSON().pcm, "base64")
    await saving
    await route.fulfill({ json: { id: "recorded", url: "/api/audio/recorded", transcript: example.transcript, status: "Recording transcribed. Check the words before generating speech." } })
  })
  await ready(page, { reference_limit: 1.5 })
  await page.getByRole("button", { name: "Record voice" }).click()
  await expect(page.getByText("Saving your recording and finishing its transcript…", { exact: true })).toBeVisible()
  await expect(page.getByRole("button", { name: "Finish recording" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Use example voice" })).toBeDisabled()
  await expect.poll(() => finishCount).toBe(1)
  expect(sampleRate).toBe(16000)
  expect(finalPCM?.length).toBe(1.5 * sampleRate * 2)
  expect(finalPCM?.some(byte => byte !== 0)).toBe(true)
  expect(await page.evaluate(() => (window as unknown as { microphoneRequests: number }).microphoneRequests)).toBe(1)
  save()
  await expect(page.getByText("My recorded voice", { exact: true })).toBeVisible()
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
})

test("a cancelled microphone permission request cannot overwrite the next recording", async ({ page }) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    let first = true
    navigator.mediaDevices.getUserMedia = constraints => {
      if (!first) return original(constraints)
      first = false
      return new Promise((_resolve, reject) => {
        Object.defineProperty(window, "rejectFirstMicrophone", {
          value: () => reject(new DOMException("Permission denied", "NotAllowedError")),
        })
      })
    }
  })
  await page.route("**/api/recordings/start", route => route.fulfill({ json: { token: "test-recording-token-5678", max_seconds: 10 } }))
  await page.route("**/api/recordings/feed", route => route.fulfill({ json: { text: "Listening", status: "Listening…" } }))
  await page.route("**/api/recordings/cancel", route => route.fulfill({ json: { cancelled: true } }))
  await ready(page)
  await page.getByRole("button", { name: "Record voice" }).click()
  await expect(page.locator(".recording-indicator")).toContainText("Connecting microphone")
  await page.getByRole("button", { name: "Discard recording" }).click()
  await page.getByRole("button", { name: "Record voice" }).click()
  await expect(page.getByRole("button", { name: "Finish recording" })).toBeEnabled()
  await page.evaluate(async () => {
    ;(window as unknown as { rejectFirstMicrophone: () => void }).rejectFirstMicrophone()
    await new Promise(resolve => setTimeout(resolve, 0))
  })
  await expect(page.getByRole("button", { name: "Finish recording" })).toBeEnabled()
  await expect(page.getByRole("alert")).toHaveCount(0)
  await page.getByRole("button", { name: "Discard recording" }).click()
  await expect(page.getByRole("button", { name: "Record voice" })).toBeEnabled()
})
