import { defineConfig, devices } from "@playwright/test";

declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  testDir: "./whep-e2e",
  testMatch: "whep-mediamtx.pw.ts",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.WHEP_FRONTEND_ORIGIN ?? "http://192.0.2.10:5173",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
