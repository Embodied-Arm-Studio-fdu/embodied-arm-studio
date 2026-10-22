import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

const run = { run_id: "analysis-run-a", user_id: "analysis-user", project_id: "teaching-project-without-workspace", class_id: "class", status: "completed", execution_mode: "simulation", run_kind: "train", parameters: { seed: 7, video_enabled: true }, project_config_snapshot: { environment_type: "s1_push", video_enabled: true, reward_extension: { schema: "reward-expression/v1", source: "def combine_reward(terms):\n    return 0\n", sha256: "a".repeat(64) } }, metrics_summary: { success_rate: 0.3 }, training_steps: 100, update_steps: 2, last_sequence: 2, code_sha256: "a".repeat(64), simulation_task_id: "task", progress: 1, created_at: "2026-10-17T00:00:00Z", updated_at: "2026-10-17T00:05:00Z", error: null };
const report = { evaluation_id: "evaluation-analysis", run_id: run.run_id, status: "assessed", score: 100, grade: "complete", evidence: {}, summary: {}, limitations: [], created_at: "2026-10-17T00:06:00Z" };
const longReply = "这段代码影响任务的奖励计算，需要根据实际运行证据检验。\n".repeat(65);

async function setup(page: Page) {
  const requests: { path: string; body: any }[] = [];
  await page.addInitScript(() => {
    sessionStorage.setItem("embodied-arm-access-token", "analysis-test");
    localStorage.setItem("embodied-arm-agent-provider", "deepseek");
    localStorage.setItem("embodied-arm-agent-level", "5");
    localStorage.setItem("embodied-arm:onboarding:analysis-user", JSON.stringify({ firstTimeOnboarding: { status: "completed", version: 1 } }));
  });
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST") requests.push({ path, body: request.postDataJSON() });
    let body: any = [];
    if (path === "/api/v1/me") body = { user_id: "analysis-user", username: "test", enabled: true, organizations: [] };
    else if (path === "/api/agent/providers") body = [{ id: "deepseek", name: "DeepSeek", configured: true, status: "healthy", models: ["deepseek-chat"], capabilities: { chat: true } }];
    else if (path === "/api/v1/runs") body = [run, { ...run, run_id: "analysis-run-b", parameters: { seed: 9 } }];
    else if (path === "/api/v1/runs/compare") body = { schema_version: "1.0", run_ids: [run.run_id, "analysis-run-b"], runs: [run, { ...run, run_id: "analysis-run-b" }], differences: {}, limitations: ["Different seed"], project_ids: [run.project_id], cross_project: false };
    else if (path.endsWith("/events")) body = [1, 2].map((index) => ({ event_id: `metric-${index}`, sequence: index, type: "metric", payload: { step: index * 50, metrics: { update_index: index, average_reward: index * 0.4, entropy: 0.8 - index * 0.01 } } }));
    else if (path.endsWith("/artifacts")) body = [{ artifact_id: "video", run_id: run.run_id, artifact_type: "training_video", path: "videos/train.mp4", sha256: "b".repeat(64) }];
    else if (path.endsWith("/download")) { await route.fulfill({ status: 200, contentType: "video/mp4", body: Buffer.from("\x00\x00\x00\x18ftypmp42") }); return; }
    else if (path.endsWith(".png")) { await route.fulfill({ status: 200, contentType: "image/png", body: Buffer.from("89504e470d0a1a0a", "hex") }); return; }
    else if (path === "/api/v1/evaluations") body = [report];
    else if (path.endsWith("/commentary")) body = { content: "DeepSeek 点评：证据完整不等同于策略成功。", tools_used: ["canonical_evaluation_read"] };
    else if (path.endsWith("/stream")) body = { run_id: run.run_id, task_id: "task", stream_id: "stream-analysis", status: "STARTING", camera_id: "overview", whep_url: "http://192.0.2.10:18889/stream-analysis/whep", width: 640, height: 480, fps: 15 };
    else if (path.startsWith("/api/v1/runs/")) body = { ...run, status: path.endsWith("active") ? "running" : "completed" };
    else if (path === "/api/v1/workbench/sources") body = { files: ["push_adapter.py"] };
    else if (path.includes("/workbench/sources/")) body = { content: "def reset():\n    return 0", sha256: "abc" };
    else if (path.endsWith("/diagnose")) body = { content: longReply, evidence: {}, provider: "deepseek" };
    else if (path.includes("/workbench/drafts/")) body = { source: "def combine_reward(terms):\n    return 0", starter: "def combine_reward(terms):\n    return 0", revision_id: "starter", editable_terms: ["precontact_progress"], protected_terms: ["success_bonus", "joint_limit_penalty"], history: [] };
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  return requests;
}

test("analysis navigation exposes standalone two-Run comparison", async ({ page }) => {
  const requests = await setup(page);
  await page.goto("/run-comparison");
  await expect(page.getByRole("heading", { name: "运行结果对比", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "运行结果对比 可用" })).toBeVisible();
  await page.locator(".comparison-run-grid button").nth(0).press("Enter");
  await page.locator(".comparison-run-grid button").nth(1).press("Enter");
  await page.getByRole("button", { name: "开始对比", exact: true }).press("Enter");
  await expect(page.locator(".comparison-limitation")).toContainText("Different seed");
  expect(requests.find((item) => item.path.endsWith("/compare"))?.body.run_ids).toEqual([run.run_id, "analysis-run-b"]);
});

test("personalized evaluation uses the report API rather than workspace sessions", async ({ page }) => {
  const requests = await setup(page);
  await page.goto("/evaluation");
  await page.getByRole("button", { name: "生成个性化点评", exact: true }).press("Enter");
  await expect(page.locator(".agent-evaluation-result")).toContainText("DeepSeek 点评");
  expect(requests.find((item) => item.path.endsWith("/commentary"))?.body).toMatchObject({ provider: "deepseek", assistance_level: 5 });
  expect(requests.some((item) => item.path === "/api/agent/sessions")).toBe(false);
});

test("artifacts export frozen code, real MP4, metric CSV and valid PNG SVG", async ({ page }) => {
  await setup(page);
  await page.goto("/export");
  await expect(page.getByRole("heading", { name: "运行制品", exact: true })).toBeVisible();
  for (const [button, suffix, marker] of [["本次奖励源码", ".py", "combine_reward"], ["指标 CSV", ".csv", "average_reward"], ["SVG", ".svg", "svg"], ["PNG", ".png", "PNG"]]) {
    const download = page.waitForEvent("download");
    await expect(page.getByRole("button", { name: button, exact: true })).toBeEnabled();
    await page.getByRole("button", { name: button, exact: true }).dispatchEvent("click");
    const result = await download;
    expect(result.suggestedFilename()).toContain(suffix);
    const contents = await readFile((await result.path())!);
    expect(contents.toString()).toContain(marker);
    if (suffix === ".svg") expect(contents.toString()).toContain("<path");
  }
  const downloading = page.waitForEvent("download");
  await page.locator(".artifact-file-table").getByRole("button", { name: "下载", exact: true }).press("Enter");
  expect((await downloading).suggestedFilename()).toBe("train.mp4");
});

test("long workbench Agent reply scrolls above the input on desktop and mobile", async ({ page }) => {
  await setup(page);
  await page.goto("/projects");
  await page.getByRole("button", { name: "分析代码", exact: true }).press("Enter");
  await expect(page.locator(".workbench-agent-copy > p")).toContainText("这段代码");
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 920 });
    const dimensions = await page.locator(".workbench-agent-copy").evaluate((element) => {
      const reply = element.querySelector(":scope > p")!;
      const input = element.querySelector("textarea")!;
      return { bottom: reply.getBoundingClientRect().bottom, top: input.getBoundingClientRect().top, scroll: reply.scrollHeight, height: reply.clientHeight };
    });
    expect(dimensions.bottom).toBeLessThanOrEqual(dimensions.top + 1);
    expect(dimensions.scroll).toBeGreaterThan(dimensions.height);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
});

test("active Run opens the live viewer without clicking open video", async ({ page }) => {
  await setup(page);
  const streamRequest = page.waitForRequest((request) => request.url().includes("/stream?"));
  await page.goto("/runs/active");
  await expect(page.getByRole("button", { name: "关闭实时视频", exact: true })).toBeVisible();
  await streamRequest;
});

test("real compatible live stream decodes through the same-origin proxy", async ({ page }) => {
  test.skip(process.env.TEST_LIVE_MEDIA !== "1", "Requires the explicit temporary media probe publisher");
  await setup(page);
  await page.route("**/api/v1/runs/analysis-run-a/stream?**", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify({ run_id: run.run_id, task_id: "task", stream_id: "stream-ui-media-probe", status: "READY", camera_id: "overview", whep_url: "http://192.0.2.10:18889/stream-ui-media-probe/whep", width: 640, height: 480, fps: 15 }) }));
  await page.goto("/runs/active", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("button", { name: "切换 WebRTC", exact: true })).toBeVisible();
  const video = page.locator(".run-video-panel video").first();
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.videoWidth), { timeout: 20000 }).toBe(640);
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime), { timeout: 20000 }).toBeGreaterThan(0);
  const range = await video.evaluate((element: HTMLVideoElement) => {
    const canvas = document.createElement("canvas"); canvas.width = 32; canvas.height = 24;
    const context = canvas.getContext("2d")!; context.drawImage(element, 0, 0, 32, 24);
    const values = context.getImageData(0, 0, 32, 24).data;
    return Math.max(...values) - Math.min(...values);
  });
  expect(range).toBeGreaterThan(40);
});

test("analysis desktop and mobile visual check", async ({ page }) => {
  test.skip(process.env.CAPTURE_VISUAL !== "1", "Explicit visual capture");
  await setup(page);
  await page.goto("/export");
  await expect(page.locator(".artifact-chart-preview path").first()).toBeVisible();
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `/tmp/analysis-artifacts-${width}.png`, fullPage: true, animations: "disabled" });
  }
});