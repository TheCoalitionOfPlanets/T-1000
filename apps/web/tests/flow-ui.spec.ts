import { test, expect } from "@playwright/test"
import { createHash } from "node:crypto"

const audio = Buffer.alloc(44 + 32000)
audio.write("RIFF", 0); audio.writeUInt32LE(audio.length - 8, 4); audio.write("WAVEfmt ", 8)
audio.writeUInt32LE(16, 16); audio.writeUInt16LE(1, 20); audio.writeUInt16LE(1, 22)
audio.writeUInt32LE(16000, 24); audio.writeUInt32LE(32000, 28); audio.writeUInt16LE(2, 32); audio.writeUInt16LE(16, 34)
audio.write("data", 36); audio.writeUInt32LE(32000, 40)
for (let i = 0; i < 16000; i++) audio.writeInt16LE(Math.round(Math.sin(i * .05) * 5000), 44 + i * 2)
const metadata = { sample_rate: 16000, channels: 1, total_samples: 16000, duration_seconds: 1, chunks: 3,
  sha256: createHash("sha256").update(audio).digest("hex"), byte_length: audio.length, complete: true, reason: "eos", clipped_sample_count: 0 }
const example = { id: "example", url: "/api/audio/example", transcript: "A whole reference sentence.", status: "Ready" }
const config = { device: "CPU", reference_limit: 2, text_limit: 150, example, sample_url: null, resumable_generation: true }

test("the studio reconnects to one generation and verifies one shared WAV before playback", async ({ page }) => {
  await page.route("**/api/config", r => r.fulfill({ json: config }))
  const inputs: Record<string, unknown>[] = []
  let downloads = 0
  await page.route("**/api/audio/**", r => { if (r.request().url().endsWith("/generated")) downloads++; return r.fulfill({ contentType: "audio/wav", body: audio }) })
  await page.route("**/api/generate", r => {
    const input = r.request().postDataJSON(); inputs.push(input)
    const update = inputs.length === 1 ? { generation_id: input.generation_id, sequence: 1, status: "Generating…", done: false }
      : { generation_id: input.generation_id, sequence: 2, status: "Generated speech.", done: true, audio_url: "/api/audio/generated", audio: metadata, partial: false }
    return r.fulfill({ contentType: "application/x-ndjson", body: JSON.stringify(update) + "\n" })
  })
  await page.goto("/")
  await page.getByRole("button", { name: "Generate speech", exact: true }).click()
  await expect(page.getByRole("link", { name: "Download speech as WAV" })).toHaveAttribute("href", /^blob:/)
  expect(inputs).toHaveLength(2)
  expect(inputs[0].generation_id).toEqual(inputs[1].generation_id)
  expect(inputs[1].after_sequence).toBe(1)
  expect(downloads).toBe(1)
  await page.locator(".output-panel .play-button").click()
  await expect.poll(() => page.locator(".output-panel audio").evaluate((el: HTMLAudioElement) => el.currentTime)).toBeGreaterThan(0)
  await expect(page.getByRole("button", { name: "Generate speech", exact: true })).toBeEnabled()
})

test("corrupt bytes are never offered as speech and reconnect retrieves the same completed result", async ({ page }) => {
  await page.route("**/api/config", r => r.fulfill({ json: config }))
  const identities: unknown[] = []
  let downloads = 0
  await page.route("**/api/audio/**", r => {
    if (!r.request().url().endsWith("/generated")) return r.fulfill({ contentType: "audio/wav", body: audio })
    downloads++
    const body = Buffer.from(audio)
    if (downloads === 1) body[100] ^= 1
    return r.fulfill({ contentType: "audio/wav", body })
  })
  await page.route("**/api/generate", r => {
    const input = r.request().postDataJSON(); identities.push(input.generation_id)
    return r.fulfill({ contentType: "application/x-ndjson", body: JSON.stringify({ generation_id: input.generation_id, sequence: 2,
      status: "Generated speech.", done: true, audio_url: "/api/audio/generated", audio: metadata, partial: false }) + "\n" })
  })
  await page.goto("/")
  await page.getByRole("button", { name: "Generate speech", exact: true }).click()
  await expect(page.getByRole("button", { name: "Reconnect to generation" })).toBeVisible()
  await expect(page.getByRole("link", { name: "Download speech as WAV" })).toHaveCount(0)
  await expect(page.locator(".output-panel audio")).toHaveCount(0)
  await page.getByRole("button", { name: "Reconnect to generation" }).click()
  await expect(page.getByRole("link", { name: "Download speech as WAV" })).toHaveAttribute("href", /^blob:/)
  expect(identities).toHaveLength(2); expect(identities[0]).toEqual(identities[1])
  expect(downloads).toBe(2)
})


test("an expired job can be reset without losing the text or reference", async ({ page }) => {
  await page.route("**/api/config", r => r.fulfill({ json: config }))
  await page.route("**/api/audio/**", r => r.fulfill({ contentType: "audio/wav", body: audio }))
  await page.route("**/api/generate", r => r.fulfill({ status: 404, json: { detail: "This generation expired." } }))
  await page.route("**/api/stop", r => r.fulfill({ status: 404, json: { detail: "This generation is unavailable." } }))
  await page.goto("/")
  await page.getByLabel("Text to speak").fill("Keep my exact words.")
  await page.getByRole("button", { name: "Generate speech", exact: true }).click()
  await expect(page.getByRole("button", { name: "Reconnect to generation" })).toBeVisible()
  await page.getByRole("button", { name: "Stop and reset" }).click()
  await expect(page.getByRole("button", { name: "Generate speech", exact: true })).toBeEnabled()
  await expect(page.getByLabel("Text to speak")).toHaveValue("Keep my exact words.")
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByRole("button", { name: "Reconnect to generation" })).toHaveCount(0)
})
