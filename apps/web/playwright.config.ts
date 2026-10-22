import { defineConfig, devices } from "@playwright/test";

const captureVisual = process.env.CAPTURE_VISUAL === "1";
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:5173";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    ...(captureVisual
      ? [{ name: "firefox-visual", use: { ...devices["Desktop Firefox"] } }]
      : []),
  ],
  webServer: {
    command: "cd ../.. && make dev",
    url: `${baseURL}/api/health`,
    reuseExistingServer: true,
    timeout: 30_000,
  },
});