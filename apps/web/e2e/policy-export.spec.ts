import { expect, test, type Route } from "@playwright/test";

const runId = "run-e2e-export-completed-0001";
const projectId = "project-e2e-s1-push-00000001";

const completedRun = {
  run_id: runId,
  user_id: "user-e2e",
  project_id: projectId,
  class_id: "class-e2e",
  execution_mode: "simulation",
  run_kind: "train",
  parameters: { seed: 103 },
  project_config_snapshot: { expected_training_steps: 320, video_enabled: true },
  code_sha256: "a".repeat(64),
  simulation_task_id: null,
  progress: 1,
  metrics_summary: {},
  error: null,
  last_sequence: 22,
  training_steps: 320,
  update_steps: 20,
  stop_reason: null,
  created_at: "2026-07-22T10:00:00Z",
  started_at: "2026-07-22T10:00:01Z",
  ended_at: "2026-07-22T10:02:00Z",
  updated_at: "2026-07-22T10:02:00Z",
  status: "completed",
};

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

test("canonical Run JSON export reports failure and retry without enabling real control", async ({
  page,
}) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("embodied-arm-access-token", "e2e-export-token");
  });

  const exportRequests: Array<{ authorization: string | undefined; body: unknown }> = [];
  const unknownApiRequests: string[] = [];
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();

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
    if (method === "GET" && url.pathname === "/api/v1/runs") {
      await fulfillJson(route, [completedRun]);
      return;
    }
    if (method === "POST" && url.pathname === "/api/v1/runs/export") {
      exportRequests.push({
        authorization: request.headers().authorization,
        body: request.postDataJSON(),
      });
      if (exportRequests.length === 1) {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          headers: { "x-request-id": "request-e2e-export-retry" },
          body: JSON.stringify({ detail: "导出服务暂不可用" }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          headers: { "Content-Disposition": 'attachment; filename="s1-push-runs.json"' },
          body: JSON.stringify({ runs: [completedRun] }),
        });
      }
      return;
    }

    if (url.pathname.endsWith("/artifacts") || url.pathname.endsWith("/events") || url.pathname === "/api/v1/evaluations" || url.pathname === "/api/agent/providers") { await fulfillJson(route, []); return; }
    unknownApiRequests.push(`${method} ${url.pathname}${url.search}`);
    await route.abort();
  });

  await page.goto("/export");
  await expect(page.getByLabel("运行记录")).toHaveValue(runId);

  const exportButton = page.getByRole("button", { name: "权威运行 JSON" });
  await expect(exportButton).toBeVisible();
  await expect(exportButton).toBeEnabled();
  await exportButton.dispatchEvent("click");
  await expect(page.getByRole("alert")).toContainText("导出服务暂不可用");
  await expect(page.getByRole("alert")).toContainText("request-e2e-export-retry");

  const retryButton = page.getByRole("button", { name: "权威运行 JSON" });
  await retryButton.dispatchEvent("click");
  await expect(page.locator(".action-feedback.success")).toContainText(
    "下载已开始",
  );

  expect(exportRequests).toEqual([
    {
      authorization: "Bearer e2e-export-token",
      body: { run_ids: [runId], format: "json" },
    },
    {
      authorization: "Bearer e2e-export-token",
      body: { run_ids: [runId], format: "json" },
    },
  ]);
  expect(unknownApiRequests).toEqual([]);
});