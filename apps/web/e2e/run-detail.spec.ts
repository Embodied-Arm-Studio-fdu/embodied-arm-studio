import { expect, test, type Page, type Route } from "@playwright/test";

const runId = "run-e2e-completed-00000001";
const secondRunId = "run-e2e-completed-00000002";
const projectId = "project-e2e-s1-push-00000001";
const artifactId = "artifact-e2e-policy-00000001";
const timestamp = "2026-07-22T10:00:00Z";

const run = {
  run_id: runId,
  user_id: "user-e2e",
  project_id: projectId,
  class_id: "class-e2e",
  execution_mode: "simulation",
  run_kind: "train",
  parameters: { seed: 42 },
  project_config_snapshot: {
    expected_training_steps: 32,
    video_enabled: false,
    parameter_schema: {
      type: "object",
      required: ["seed"],
      properties: {
        seed: {
          type: "integer",
          title: "Seed",
          minimum: 0,
          maximum: 2147483647,
        },
      },
    },
  },
  code_sha256: "a".repeat(64),
  simulation_task_id: null,
  progress: 1,
  metrics_summary: {},
  error: null,
  last_sequence: 3,
  training_steps: 32,
  update_steps: 2,
  stop_reason: null,
  created_at: timestamp,
  started_at: timestamp,
  ended_at: "2026-07-22T10:01:00Z",
  updated_at: "2026-07-22T10:01:00Z",
  status: "completed",
};

const secondRun = {
  ...run,
  run_id: secondRunId,
  parameters: { seed: 44 },
  simulation_task_id: "task-e2e-completed-00000002",
};

const events = [
  {
    event_id: "event-e2e-metric-1",
    sequence: 1,
    run_id: runId,
    occurred_at: timestamp,
    schema_version: "1.0",
    type: "metric",
    payload: {
      step: 16,
      metrics: {
        update_index: 1,
        average_reward: 0.42,
        episode_length: 14,
        episodes: 3,
        policy_loss: -0.08,
        value_loss: 0.21,
        entropy: 1.7,
      },
    },
  },
  {
    event_id: "event-e2e-metric-2",
    sequence: 2,
    run_id: runId,
    occurred_at: "2026-07-22T10:00:30Z",
    schema_version: "1.0",
    type: "metric",
    payload: {
      step: 32,
      metrics: {
        update_index: 2,
        average_reward: 0.76,
        episode_length: 11,
        episodes: 6,
        policy_loss: -0.05,
        value_loss: 0.14,
        entropy: 1.5,
      },
    },
  },
  {
    event_id: "event-e2e-completed",
    sequence: 3,
    run_id: runId,
    occurred_at: "2026-07-22T10:01:00Z",
    schema_version: "1.0",
    type: "status",
    payload: { status: "completed" },
  },
];

const artifact = {
  artifact_id: artifactId,
  run_id: runId,
  artifact_type: "policy_checkpoint",
  path: "artifacts/checkpoints/s1_push_policy_final.pt",
  size_bytes: 4096,
  sha256: "b".repeat(64),
  is_primary_model: true,
  created_at: timestamp,
};

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function expectMetricChartsPainted(page: Page) {
  const expectedColors = [
    [224, 90, 63],
    [39, 123, 105],
    [176, 122, 22],
    [71, 111, 155],
    [154, 73, 62],
    [79, 111, 100],
  ];
  const stats = await page
    .getByRole("img", { name: /随 PPO 更新变化的折线图$/ })
    .evaluateAll((charts, colors) => charts.map((chart, index) => {
      const svg = chart.querySelector("svg");
      if (!svg) {
        return { width: 0, height: 0, paintedElements: 0, seriesElements: 0 };
      }
      const bounds = svg.getBoundingClientRect();
      const target = colors[index];
      const targetColor = `rgb(${target[0]}, ${target[1]}, ${target[2]})`;
      const paintedElements = svg.querySelectorAll("path, circle, rect, line, polyline, text").length;
      const seriesElements = [...svg.querySelectorAll("*")].filter((element) => {
        const style = window.getComputedStyle(element);
        return style.stroke === targetColor || style.fill === targetColor;
      }).length;
      return {
        width: bounds.width,
        height: bounds.height,
        paintedElements,
        seriesElements,
      };
    }), expectedColors);

  expect(stats).toHaveLength(6);
  for (const stat of stats) {
    expect(stat.width).toBeGreaterThan(100);
    expect(stat.height).toBeGreaterThan(100);
    expect(stat.paintedElements).toBeGreaterThan(10);
    expect(stat.seriesElements).toBeGreaterThan(0);
  }
}

async function installCanonicalRunRoutes(page: Page) {
  const unknownApiRequests: string[] = [];
  const mutationRequests: string[] = [];
  const streamRequests: string[] = [];
  const artifactDownloadRequests: string[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();

    if (method !== "GET") mutationRequests.push(`${method} ${url.pathname}`);
    if (url.pathname.includes("/stream")) streamRequests.push(url.pathname);

    if (method === "GET" && url.pathname === "/api/v1/me") {
      await fulfillJson(route, {
        user_id: "user-e2e",
        username: "e2e",
        display_name: "E2E User",
        enabled: true,
        organizations: [],
      });
      return;
    }
    const requestedRun = [run, secondRun].find(
      (candidate) =>
        url.pathname === `/api/v1/runs/${candidate.run_id}`,
    );
    if (method === "GET" && requestedRun) {
      await fulfillJson(route, requestedRun);
      return;
    }
    const eventRun = [run, secondRun].find(
      (candidate) =>
        url.pathname === `/api/v1/runs/${candidate.run_id}/events`,
    );
    if (method === "GET" && eventRun) {
      await fulfillJson(
        route,
        events.map((event) => ({ ...event, run_id: eventRun.run_id })),
      );
      return;
    }
    const artifactRun = [run, secondRun].find(
      (candidate) =>
        url.pathname === `/api/v1/runs/${candidate.run_id}/artifacts`,
    );
    if (method === "GET" && artifactRun) {
      await fulfillJson(
        route,
        [{ ...artifact, run_id: artifactRun.run_id }],
      );
      return;
    }
    if (
      method === "GET" &&
      url.pathname === `/api/v1/runs/${runId}/artifacts/${artifactId}/download`
    ) {
      artifactDownloadRequests.push(url.pathname);
      await route.fulfill({
        status: 200,
        contentType: "application/octet-stream",
        headers: { "Content-Disposition": 'attachment; filename="s1_push_policy_final.pt"' },
        body: "canonical-policy-bytes",
      });
      return;
    }
    if (method === "GET" && url.pathname === "/api/v1/runs") {
      await fulfillJson(route, [run, secondRun]);
      return;
    }

    unknownApiRequests.push(`${method} ${url.pathname}${url.search}`);
    await route.abort();
  });

  return { artifactDownloadRequests, unknownApiRequests, mutationRequests, streamRequests };
}

test("completed canonical Run is usable on desktop and mobile without starting work", async ({
  browserName,
  context,
  page,
}, testInfo) => {
  if (browserName === "chromium") {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  }
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-token");
  });
  await page.route("https://fonts.googleapis.com/**", (route) => route.abort());

  const apiRequests = await installCanonicalRunRoutes(page);
  const webSockets: string[] = [];
  page.on("websocket", (socket) => {
    if (new URL(socket.url()).pathname.startsWith("/api/v1/runs/")) {
      webSockets.push(socket.url());
    }
  });

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`/runs/${runId}`);

  await expect(page.locator(".run-status-card > strong")).toHaveText("已完成");
  await expect(page.getByRole("img", { name: /随 PPO 更新变化的折线图$/ })).toHaveCount(6);
  await expectMetricChartsPainted(page);
  await expect(page.locator('.run-video-panel[data-video-presentation="disabled"]')).toContainText(
    "本次运行未启用视频",
  );
  if (process.env.CAPTURE_VISUAL === "1") {
    await page.screenshot({
      path: "../../docs/reports/assets/s1-push-run-detail-desktop.png",
      fullPage: true,
    });
  }
  const runCopyButton = page.getByRole("button", { name: "复制完整Run ID" });
  const projectCopyButton = page.getByRole("button", { name: "复制完整Project ID" });
  await expect(runCopyButton).toBeVisible();
  await expect(runCopyButton).toBeEnabled();
  await expect(projectCopyButton).toBeVisible();
  await expect(projectCopyButton).toBeEnabled();
  expect(await runCopyButton.boundingBox()).toMatchObject({ width: 28, height: 28 });
  expect(await projectCopyButton.boundingBox()).toMatchObject({ width: 28, height: 28 });
  await runCopyButton.dispatchEvent("click");
  await projectCopyButton.dispatchEvent("click");
  await expect(page.getByText("已复制", { exact: true })).toHaveCount(2);

  const artifactDownloadButton = page
    .locator(".artifact-row")
    .getByRole("button", { name: "下载", exact: true });
  await expect(artifactDownloadButton).toBeVisible();
  await expect(artifactDownloadButton).toBeEnabled();
  await artifactDownloadButton.dispatchEvent("click");
  await expect(page.locator(".artifact-feedback.success")).toHaveText("下载已开始");
  expect(apiRequests.artifactDownloadRequests).toEqual([
    `/api/v1/runs/${runId}/artifacts/${artifactId}/download`,
  ]);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload();
  await expect(page.locator(".run-status-card > strong")).toHaveText("已完成");
  const overflow = await page.evaluate(() => {
    const viewportWidth = document.documentElement.clientWidth;
    return [...document.querySelectorAll<HTMLElement>("body *")]
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          element: element.tagName.toLowerCase(),
          className: element.className,
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
          scrollWidth: element.scrollWidth,
        };
      })
      .filter(({ right }) => right > viewportWidth + 1)
      .slice(0, 20);
  });
  expect(overflow).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await expect(page.getByRole("img", { name: /随 PPO 更新变化的折线图$/ })).toHaveCount(6);
  await expectMetricChartsPainted(page);
  if (process.env.CAPTURE_VISUAL === "1") {
    await page.screenshot({
      path: "../../docs/reports/assets/s1-push-run-detail-mobile-390.png",
      fullPage: true,
    });
  }
  await expect(page.getByRole("button", { name: "返回运行记录" })).toBeVisible();
  await page.getByRole("button", { name: "返回运行记录" }).dispatchEvent("click");
  await expect(page).toHaveURL(/\/runs$/);
  expect(apiRequests.mutationRequests).toEqual([]);
  expect(apiRequests.streamRequests).toEqual([]);
  expect(apiRequests.unknownApiRequests).toEqual([]);
  expect(webSockets).toEqual([]);
});

test("historical Run detail navigation uses full IDs and never creates a Run", async ({
  page,
}) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-token");
  });
  await page.route("https://fonts.googleapis.com/**", (route) => route.abort());
  const apiRequests = await installCanonicalRunRoutes(page);

  await page.goto("/runs");

  await expect(
    page.getByText(
      "已有运行可直接查看结果、制品和导出，无需重新训练。",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.getByText("SIMULATION MODE", { exact: true })).toBeVisible();
  await expect(
    page.getByText("GPU TRAINING ENABLED", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("REAL ROBOT CONTROL DISABLED", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("MOCK MODE", { exact: true })).toHaveCount(0);

  const detailLinks = page.locator(".run-record-detail");
  await expect(detailLinks).toHaveCount(2);
  await expect(detailLinks.nth(0)).toHaveAttribute("href", `/runs/${runId}`);
  await expect(detailLinks.nth(1)).toHaveAttribute(
    "href",
    `/runs/${secondRunId}`,
  );

  await expect(
    page.getByRole("button", { name: "查看完整运行详情" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "进入证据评估" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "下载 JSON" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "对比选中运行" }),
  ).toBeVisible();

  await page
    .getByRole("link", { name: `查看 Run ${runId} 完整详情` })
    .dispatchEvent("click");
  await expect(page).toHaveURL(`/runs/${runId}`);
  await expect(
    page.getByRole("heading", { name: "训练结果与导出" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "返回运行记录" }).dispatchEvent("click");
  const secondRecord = page
    .getByRole("link", { name: `查看 Run ${secondRunId} 完整详情` })
    .locator("..");
  await secondRecord.locator("button").first().dispatchEvent("click");
  await page.getByRole("button", { name: "查看完整运行详情" }).dispatchEvent("click");
  await expect(page).toHaveURL(`/runs/${secondRunId}`);

  expect(apiRequests.mutationRequests).not.toContain("POST /api/v1/runs");
});

test("video=false Run never fetches a recording or creates a replay Blob", async ({ page }) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-token");
    const original = URL.createObjectURL.bind(URL);
    (window as unknown as { replayBlobUrls: number }).replayBlobUrls = 0;
    URL.createObjectURL = (blob: Blob | MediaSource) => {
      (window as unknown as { replayBlobUrls: number }).replayBlobUrls += 1;
      return original(blob);
    };
  });
  const requests = await installCanonicalRunRoutes(page);
  await page.goto(`/runs/${runId}`);
  await expect(page.locator('[data-video-presentation="disabled"]'))
    .toHaveText("本次运行未启用视频");
  expect(await page.evaluate(() => (window as unknown as { replayBlobUrls: number }).replayBlobUrls))
    .toBe(0);
  expect(requests.streamRequests).toEqual([]);
  expect(requests.artifactDownloadRequests).toEqual([]);
});

test("terminal video Run loads one authenticated Blob and reuses it for replay and download", async ({ page }) => {
  const videoRun = {
    ...run,
    run_id: "run-e2e-video-00000001",
    project_config_snapshot: {
      ...run.project_config_snapshot,
      video_enabled: true,
    },
  };
  const videoArtifact = {
    ...artifact,
    artifact_id: "artifact-e2e-video-00000001",
    run_id: videoRun.run_id,
    artifact_type: "training_video",
    path: "video/overview.mp4",
  };
  let videoDownloads = 0;
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-token");
    let playCalls = 0;
    HTMLMediaElement.prototype.play = function () {
      playCalls += 1;
      return playCalls === 1
        ? Promise.reject(new DOMException("blocked", "NotAllowedError"))
        : Promise.resolve();
    };
  });
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/v1/me") {
      await fulfillJson(route, {
        user_id: "user-e2e", username: "e2e", display_name: "E2E User",
        enabled: true, organizations: [],
      });
    } else if (path === `/api/v1/runs/${videoRun.run_id}`) {
      await fulfillJson(route, videoRun);
    } else if (path === `/api/v1/runs/${videoRun.run_id}/events`) {
      await fulfillJson(route, []);
    } else if (path === `/api/v1/runs/${videoRun.run_id}/artifacts`) {
      await fulfillJson(route, [videoArtifact]);
    } else if (
      path === `/api/v1/runs/${videoRun.run_id}/artifacts/${videoArtifact.artifact_id}/download`
    ) {
      videoDownloads += 1;
      expect(request.headers()["authorization"]).toBe("Bearer e2e-token");
      await route.fulfill({
        status: 200,
        contentType: "video/mp4",
        headers: { "Content-Disposition": 'attachment; filename="overview.mp4"' },
        body: "fake-mp4",
      });
    } else {
      await route.abort();
    }
  });

  await page.goto(`/runs/${videoRun.run_id}`);
  const replay = page.locator(".replay-player");
  await expect(replay).toHaveAttribute("data-replay-state", "manual-play");
  const video = replay.locator("video");
  await expect(video).toHaveAttribute("controls", "");
  await expect(video).toHaveAttribute("autoplay", "");
  await expect(video).toHaveAttribute("muted", "");
  await expect(video).toHaveAttribute("playsinline", "");
  await expect(video).toHaveAttribute("loop", "");
  const loop = replay.getByRole("checkbox", { name: "循环播放" });
  await expect(loop).toBeChecked();
  await loop.uncheck();
  await expect(video).not.toHaveAttribute("loop");
  await replay.getByRole("button", { name: "点击播放" }).click();
  await expect(replay).toHaveAttribute("data-replay-state", "ready");
  await replay.getByRole("button", { name: "从头播放" }).click();
  await replay.getByRole("button", { name: "下载录像" }).click();
  expect(videoDownloads).toBe(1);

  await page.reload();
  await expect(page.locator(".replay-player video")).toBeVisible();
  expect(videoDownloads).toBe(2);
});

test("video Artifact query failure is distinct and retry reaches the no-recording state", async ({ page }) => {
  const videoRun = {
    ...run,
    run_id: "run-e2e-video-query-error",
    project_config_snapshot: {
      ...run.project_config_snapshot,
      video_enabled: true,
    },
  };
  let artifactQueries = 0;
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-token");
  });
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/v1/me") {
      await fulfillJson(route, {
        user_id: "user-e2e", username: "e2e", display_name: "E2E User",
        enabled: true, organizations: [],
      });
    } else if (path === `/api/v1/runs/${videoRun.run_id}`) {
      await fulfillJson(route, videoRun);
    } else if (path === `/api/v1/runs/${videoRun.run_id}/events`) {
      await fulfillJson(route, []);
    } else if (path === `/api/v1/runs/${videoRun.run_id}/artifacts`) {
      artifactQueries += 1;
      if (artifactQueries <= 4) {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "temporarily unavailable" }),
        });
      } else {
        await fulfillJson(route, []);
      }
    } else {
      await route.abort();
    }
  });

  await page.goto(`/runs/${videoRun.run_id}`);
  await expect(page.getByText("训练录像信息读取失败", { exact: true })).toBeVisible();
  await page.locator('[data-video-presentation="error"]').getByRole("button", { name: "重试" }).click();
  await expect(page.getByText("本次运行没有可用的训练录像", { exact: true })).toBeVisible();
  expect(artifactQueries).toBe(5);
});
