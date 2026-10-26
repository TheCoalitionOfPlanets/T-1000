import { defineConfig } from "@playwright/test"
import { fileURLToPath } from "node:url"

const microphoneFile = fileURLToPath(new URL("../../assets/ref.wav", import.meta.url))

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 8_000 },
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:7860",
    browserName: "chromium",
    viewport: { width: 1440, height: 1000 },
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-audio-capture=${microphoneFile}`,
      ],
    },
    trace: "retain-on-failure",
  },
})
