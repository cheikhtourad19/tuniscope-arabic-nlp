import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "../test-results",
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: process.env.APP_URL || "http://127.0.0.1:8080",
    browserName: "chromium",
    channel: "chromium",
    screenshot: "only-on-failure",
  },
});
