import { test, expect, type Page } from "@playwright/test"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

const audio = readFileSync(fileURLToPath(new URL("../../../assets/ref.wav", import.meta.url)))
const example = { id: "example", url: "/api/audio/example", duration_seconds: 3.9,
  transcript: "The complete original sentence.", status: "Example voice ready." }

async function ready(page: Page) {
  await page.route("**/api/audio/**", route => route.fulfill({ contentType: "audio/wav", body: audio }))
  await page.route("**/api/config", route => route.fulfill({ json: {
    device: "CPU", reference_limit: 10, text_limit: 150, example, sample_url: null,
  } }))
  await page.goto("/")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  await page.getByText("Clean up and trim voice", { exact: true }).click()
  await expect(page.getByLabel("End (seconds)")).toBeEnabled()
}

test("trim and cleanup update the selected voice and exact transcript, always using the original", async ({ page }) => {
  const edits: unknown[] = []
  await page.route("**/api/reference/process", route => {
    edits.push(route.request().postDataJSON())
    return route.fulfill({ json: { id: `edited-${edits.length}`, url: "/api/audio/edited", duration_seconds: 1.5,
      original_id: "example", original_url: example.url, original_duration_seconds: 3.9,
      transcript: "The cropped sentence.", status: "Voice edited. Check the words." } })
  })
  await ready(page)
  await page.getByLabel("Start (seconds)").fill("0.5")
  await page.getByLabel("End (seconds)").fill("2")
  await page.getByLabel("Trim quiet edges automatically").check()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await expect(page.getByText("1.50 seconds selected")).toBeVisible()
  await page.getByRole("button", { name: "Apply voice edits" }).click()
  await expect(page.getByLabel("Reference transcript")).toHaveValue("The cropped sentence.")
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  expect(edits[0]).toEqual({ reference_id: "example", start_seconds: 0.5, end_seconds: 2, noise_reduction: true, auto_trim: true })
  await page.getByLabel("Start (seconds)").fill("0.75")
  await page.getByLabel("Reduce noise and background hiss").uncheck()
  await page.getByRole("button", { name: "Apply voice edits" }).click()
  await expect.poll(() => edits.length).toBe(2)
  expect(edits[1]).toEqual({ reference_id: "example", start_seconds: 0.75, end_seconds: 2, noise_reduction: false, auto_trim: true })
})

test("invalid crops cannot be applied and restore obtains the original transcript", async ({ page }) => {
  let restored: unknown
  await page.route("**/api/reference/process", route => {
    restored = route.request().postDataJSON()
    return route.fulfill({ json: example })
  })
  await ready(page)
  await page.getByLabel("Start (seconds)").fill("2")
  await page.getByLabel("End (seconds)").fill("2.5")
  await expect(page.getByRole("button", { name: "Apply voice edits" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Preview selection" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await page.getByRole("button", { name: "Restore original" }).click()
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByLabel("Start (seconds)")).toHaveValue("0")
  await expect(page.getByLabel("Reduce noise and background hiss")).not.toBeChecked()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
  expect(restored).toEqual({ reference_id: "example", start_seconds: 0, end_seconds: null, noise_reduction: false, auto_trim: false })
})

test("processing locks generation and reference controls; errors retain the previous voice", async ({ page }) => {
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  let requested = false
  await page.route("**/api/reference/process", async route => {
    requested = true
    await pending
    await route.fulfill({ status: 400, json: { detail: "The selected audio is silent or too quiet." } })
  })
  await ready(page)
  await page.getByLabel("Start (seconds)").fill("0.1")
  await page.getByRole("button", { name: "Apply voice edits" }).click()
  await expect.poll(() => requested).toBe(true)
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Record voice" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Use example voice" })).toBeDisabled()
  await expect(page.getByLabel("Start (seconds)")).toBeDisabled()
  release()
  await expect(page.getByRole("alert")).toContainText("silent or too quiet")
  await expect(page.getByLabel("Reference transcript")).toHaveValue(example.transcript)
  await expect(page.getByLabel("Start (seconds)")).toBeEnabled()
  await page.getByRole("button", { name: "Use example voice" }).click()
  await expect(page.getByRole("button", { name: "Generate speech" })).toBeEnabled()
})

test("upload cleanup choice reaches the API and the editor uses the raw uploaded reference", async ({ page }) => {
  let uploadBody = "", edit: unknown
  await page.route("**/api/reference", route => {
    uploadBody = route.request().postDataBuffer()?.toString() ?? ""
    return route.fulfill({ json: { id: "cleaned-upload", url: "/api/audio/cleaned-upload", transcript: "Uploaded words.", status: "Ready",
      duration_seconds: 3.9, original_id: "raw-upload", original_url: "/api/audio/raw-upload", original_duration_seconds: 3.9 } })
  })
  await page.route("**/api/reference/process", route => {
    edit = route.request().postDataJSON()
    return route.fulfill({ json: example })
  })
  await ready(page)
  await page.getByLabel("Reduce background noise on recording and upload").uncheck()
  await page.getByLabel("Upload reference audio").setInputFiles({ name: "my-voice.wav", mimeType: "audio/wav", buffer: audio })
  await expect(page.getByLabel("Reference transcript")).toHaveValue("Uploaded words.")
  expect(uploadBody).toMatch(/name="noise_reduction"\r\n\r\nfalse/)
  await page.getByText("Clean up and trim voice", { exact: true }).click()
  await page.getByRole("button", { name: "Restore original" }).click()
  await expect.poll(() => edit).toBeTruthy()
  expect(edit).toEqual({ reference_id: "raw-upload", start_seconds: 0, end_seconds: null, noise_reduction: false, auto_trim: false })
})

test("trim controls fit on a narrow screen", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await ready(page)
  await expect(page.getByRole("button", { name: "Apply voice edits" })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})
