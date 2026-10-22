import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";

const starter = 'def combine_reward(terms):\n    return terms["progress_reward"] + terms["action_penalty"]\n';
const user = { user_id: "workbench-test", username: "test", display_name: "Test", enabled: true, organizations: [{ organization_id: "org", name: "实验课程", slug: "test", enabled: true, role: "student", permissions: ["own_run:create"] }] };
const project = { project_id: "project", course_id: "course", title: "Reach Project", task_type: "training", environment_type: "s1_reach", training_profile: "standard", config_version: 4, enabled: true, expected_training_steps: 20000, permissions: { can_use: true, can_view: true, can_modify: false } };
const record = { run_id: "workbench-run", user_id: user.user_id, project_id: project.project_id, class_id: "class", run_kind: "train", execution_mode: "simulation", status: "failed", parameters: { seed: 7 }, project_config_snapshot: { environment_type: "s1_reach" }, metrics_summary: {}, progress: 0, training_steps: 0, update_steps: 0, last_sequence: 1, created_at: "2026-10-10T00:00:00Z", updated_at: "2026-10-10T00:00:00Z", error: { code: "reward_error", message: "Reward division denominator is zero", retryable: false } };

async function setup(page: Page, taskId = "s1_reach") {
  const selectedProject = { ...project, environment_type: taskId, video_enabled: true };
  let scene: Record<string, unknown> | null = null;
  const submissions: Record<string, unknown>[] = [];
  const questions: Record<string, unknown>[] = [];
  let source = starter;
  let revision = "starter";
  await page.addInitScript(() => {
    sessionStorage.setItem("embodied-arm-access-token", "isolated-workbench-token");
    localStorage.setItem("embodied-arm:onboarding:workbench-test", JSON.stringify({ firstTimeOnboarding: { status: "completed", version: 1 } }));
  });
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body: unknown = [];
    if (path === "/api/v1/me") body = user;
    else if (path === "/api/v1/courses") body = [{ course_id: "course", title: "实验课", code: "RL", enabled: true }];
    else if (path.endsWith("/classes")) body = [{ class_id: "class", name: "一班", enabled: true }];
    else if (path.endsWith("/projects")) body = [selectedProject];
    else if (path === "/api/simulation/task-catalog") body = { tasks: ["s1_push", "s1_reach"].map((id) => ({ id, display_name: id, task_type: "training", supports_training: true, parameters: [{ name: "seed", type: "integer", default: 7, minimum: 0 }, { name: "video_enabled", type: "boolean", default: true }, { name: "video_fps", type: "integer", default: 15, minimum: 1, maximum: 30 }, { name: "video_width", type: "integer", default: 640, minimum: 320, maximum: 1280 }, ...(id === "s1_reach" ? [{ name: "action_delta_m", type: "number", default: 0.005, minimum: 0.001, maximum: 0.01 }, { name: "target_center_xy_m", type: "number_pair", default: [0.36, 0] }, { name: "success_distance_m", type: "number", default: 0.02, minimum: 0.01, maximum: 0.05 }] : [])] })) };
    else if (path === "/api/agent/providers") body = [{ id: "mock", name: "内置基础 Agent", status: "healthy", configured: true, models: ["teaching-mock-v1"], capabilities: { chat: true, structured_output: false, tool_calls: false }, data_scope: "仅使用当前会话中提供的教学上下文，不发送外部请求。" }];
    else if (path === "/api/v1/workbench/sources") body = { files: ["reach_adapter.py", "ppo/trainer.py"] };
    else if (path.includes("/workbench/sources/")) body = { content: 'def reset(seed):\n    return seed\n', sha256: "abc" };
    else if (path === "/api/scenes" && request.method() === "POST") { scene = { ...request.postDataJSON().document, revision: 1 }; body = scene; }
    else if (path.startsWith("/api/scenes/")) body = path.endsWith("/preflight") ? { trainable_push: true } : scene;
    else if (path.endsWith("/preflight")) {
      const code = request.postDataJSON().source as string;
      body = { valid: true, extension: { schema: "reward-expression/v1", source: code, sha256: createHash("sha256").update(code).digest("hex") }, errors: [], warnings: ["未执行物理仿真"], samples: [] };
    } else if (path.endsWith("/diagnose")) {
      questions.push(request.postDataJSON());
      body = { content: "所选代码解释；低成功率原因尚待验证。", evidence: {}, provider: "mock", suggested_source: null };
    } else if (path.includes("/workbench/drafts/")) {
      if (request.method() === "PUT") { source = request.postDataJSON().source; revision = "revision-1"; }
      if (path.endsWith("/restore")) { source = starter; revision = "revision-2"; }
      body = { task_id: "s1_reach", source, starter, revision_id: revision, editable_terms: ["progress_reward", "action_penalty"], protected_terms: ["success_bonus", "joint_limit_penalty"], history: revision === "starter" ? [] : [{ revision_id: revision, sha256: "digest", created_at: "2026-10-10" }] };
    } else if (path === "/api/v1/runs" && request.method() === "POST") { submissions.push(request.postDataJSON()); body = { ...record, project_config_snapshot: { environment_type: taskId, ...(scene ? { scene: { document: scene, scene_ref: (submissions.at(-1) as Record<string, unknown>).scene_ref } } : {}) } }; }
    else if (path === "/api/v1/runs") body = [];
    else if (path.endsWith("/events")) body = [{ event_id: "error-1", sequence: 1, type: "error", payload: { message: "Reward division denominator is zero" } }];
    else if (path.startsWith("/api/v1/runs/")) body = { ...record, project_config_snapshot: { environment_type: taskId, ...(scene ? { scene: { document: scene, scene_ref: submissions.at(-1)?.scene_ref } } : {}) } };
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("/projects");
  await expect(page.getByRole("heading", { name: "实验工作台", exact: true })).toBeVisible();
  if (taskId === "s1_reach") await page.getByRole("button", { name: /S1 Reach · 平面到达/ }).press("Enter");
  await expect(page.locator(".cm-editor")).toBeVisible();
  await expect(page.locator(".cm-content")).toContainText("progress_reward");
  return { submissions, questions };
}

test("default Push geometry also gets an explicit frozen scene revision", async ({ page }) => {
  const { submissions } = await setup(page, "s1_push");
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).click();
  await expect.poll(() => submissions.length).toBe(1);
  expect(submissions[0].scene_ref).toMatchObject({ revision: 1, scene_id: expect.stringContaining("scene-push-") });
  await expect(page.locator(".frozen-scene-summary")).toContainText("0.6 / 0.15");
});

test("Push edits preview, require saving, freeze geometry and persist after reload", async ({ page }) => {
  const { submissions } = await setup(page, "s1_push");
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: /01 任务场景/ }).click();
  const ellipse = page.locator(".push-scene-preview ellipse");
  const before = await ellipse.getAttribute("cx");
  await page.getByRole("spinbutton", { name: "目标位置 X", exact: true }).fill("0.64");
  await page.getByRole("spinbutton", { name: "方块初始位置 Y", exact: true }).fill("-0.12");
  await page.getByRole("spinbutton", { name: "方块尺寸 Z", exact: true }).fill("0.1");
  expect(await ellipse.getAttribute("cx")).not.toBe(before);
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeDisabled();
  await expect(page.locator(".workbench-launch-reason")).toContainText("保存并校验场景");
  await page.getByRole("button", { name: "保存场景并校验", exact: true }).click();
  await expect(page.locator(".push-scene-summary")).toContainText("已保存并校验");
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).click();
  await expect.poll(() => submissions.length).toBe(1);
  expect(submissions[0].scene_ref).toMatchObject({ revision: 1, scene_id: expect.stringContaining("scene-push-") });
  await expect(page.locator(".frozen-scene-summary")).toContainText("0.64 / 0.15");
  await page.getByRole("spinbutton", { name: "目标位置 X", exact: true }).fill("0.65");
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeDisabled();
  await expect(page.locator(".frozen-scene-summary")).toContainText("0.64 / 0.15");
  await page.reload();
  await page.getByRole("button", { name: /01 任务场景/ }).click();
  await expect(page.getByRole("spinbutton", { name: "目标位置 X", exact: true })).toHaveValue("0.64");
  await page.setViewportSize({ width: 1600, height: 1050 });
  await page.screenshot({ path: "../../var/validation/scene-workbench-push-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "EN", exact: true }).click();
  expect(await page.locator(".push-scene-controls").innerText()).not.toMatch(/[\u4e00-\u9fff]/);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.screenshot({ path: "../../var/validation/scene-workbench-push-mobile.png", fullPage: true });
});

test("Push geometry rejects footprint overflow and recovers from failed server validation", async ({ page }) => {
  await setup(page, "s1_push");
  await page.getByRole("button", { name: /01 任务场景/ }).click();
  await page.getByRole("spinbutton", { name: "方块初始位置 X", exact: true }).fill("0.19");
  await expect(page.locator(".push-scene-controls [role=alert]")).toContainText("方块整体");
  await expect(page.getByRole("button", { name: "保存场景并校验", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "恢复默认场景", exact: true }).click();
  await page.getByRole("spinbutton", { name: "目标位置 X", exact: true }).fill("0.64");
  await page.route("**/api/scenes/*/preflight", (route) => route.fulfill({ json: { trainable_push: false } }));
  await page.getByRole("button", { name: "保存场景并校验", exact: true }).click();
  await expect(page.locator(".push-scene-controls [role=alert]")).toContainText("校验失败");
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeDisabled();
  await expect(page.getByRole("spinbutton", { name: "目标位置 X", exact: true })).toHaveValue("0.64");
  await page.unroute("**/api/scenes/*/preflight");
  await page.getByRole("button", { name: "保存场景并校验", exact: true }).click();
  await expect(page.locator(".push-scene-summary")).toContainText("已保存并校验");
});

test("advanced video controls stay collapsed, preserve edits and submit actual values", async ({ page }) => {
  const { submissions } = await setup(page);
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: /03 实验设置/ }).click();
  await expect(page.getByRole("spinbutton", { name: "视频帧率", exact: true })).not.toBeVisible();
  await page.locator(".workbench-recording-advanced summary").click();
  await page.getByRole("spinbutton", { name: "视频帧率", exact: true }).fill("20");
  await page.getByRole("checkbox", { name: "记录实验视频", exact: true }).uncheck();
  await expect(page.getByRole("spinbutton", { name: "视频帧率", exact: true })).toBeDisabled();
  await page.getByRole("checkbox", { name: "记录实验视频", exact: true }).check();
  await page.getByRole("button", { name: /01 任务场景/ }).click();
  await page.getByRole("button", { name: /03 实验设置/ }).click();
  await expect(page.getByRole("spinbutton", { name: "视频帧率", exact: true })).toHaveValue("20");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).click();
  await expect.poll(() => submissions.length).toBe(1);
  expect(submissions[0]).toMatchObject({ parameters: { video_enabled: true, video_fps: 20 } });
});

test("saved reward preflight launches a frozen Run and displays backend errors", async ({ page }) => {
  const { submissions } = await setup(page);
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: "启动训练", exact: true }).isDisabled();
  await page.getByRole("button", { name: "保存版本", exact: true }).press("Enter");
  await expect(page.getByRole("status")).toContainText("已保存");
  await page.getByRole("button", { name: "运行预检", exact: true }).press("Enter");
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).press("Enter");
  await expect(page.locator(".workbench-run-log")).toContainText("Reward division denominator is zero");
  expect(submissions).toHaveLength(1);
  expect(submissions[0]).toMatchObject({ environment_type: "s1_reach", code: starter, parameters: { seed: 7 }, reward_extension: { source: starter, sha256: createHash("sha256").update(starter).digest("hex") } });
  expect(submissions[0]).not.toHaveProperty("scene_ref");
});

test("code selection reaches Agent; edits invalidate preflight; source stays read only", async ({ page }) => {
  const { questions } = await setup(page);
  await page.getByRole("button", { name: "运行预检", exact: true }).press("Enter");
  await expect(page.locator(".preflight-success")).toBeVisible();
  await expect(page.getByRole("button", { name: "分析代码", exact: true })).toBeEnabled();
  const editor = page.locator(".workbench-editor .cm-editor");
  await page.getByRole("button", { name: "进阶代码编辑", exact: true }).click();
  await editor.getByRole("textbox").focus();
  await page.keyboard.press("Control+A");
  await page.getByRole("button", { name: "分析代码", exact: true }).press("Enter");
  await expect(page.locator(".workbench-agent-copy")).toContainText("所选代码解释");
  expect(questions[0].selection).toContain("combine_reward");
  await editor.getByRole("textbox").focus();
  await page.keyboard.press("Control+A");
  await page.keyboard.insertText('def combine_reward(terms):\n    return 2 * terms["progress_reward"]');
  await expect(editor.locator(".cm-content")).toContainText('2 * terms["progress_reward"]');
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeDisabled();
  await expect(page.locator(".workbench-commandbar")).toContainText("修改未保存");
  await page.getByRole("button", { name: "reach_adapter.py", exact: true }).press("Enter");
  await expect(editor.locator(".cm-content")).toContainText("reset");
  await expect(editor.getByRole("textbox")).toHaveAttribute("contenteditable", "false");
});

test("desktop and mobile remain inside viewport with nonblank editor", async ({ page }) => {
  await setup(page);
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.locator(".cm-content")).toContainText("combine_reward");
  }
});

test("task files stay focused while a full long source scrolls to its final line", async ({ page }) => {
  await setup(page);
  const longSource = Array.from({ length: 700 }, (_, index) => `# training_step_${index + 1}`).join("\n") + "\n# END_OF_FULL_SOURCE";
  await page.route("**/api/v1/workbench/sources", (route) => route.fulfill({ json: { files: ["reach_adapter.py", "ppo/trainer.py", "training_task_runner.py", "internal/reference.py"] } }));
  await page.route("**/api/v1/workbench/sources/training_task_runner.py", (route) => route.fulfill({ json: { content: longSource, read_only: true, sha256: "test" } }));
  await page.reload();
  await page.getByRole("button", { name: /S1 Reach · 平面到达/ }).click();
  await expect(page.getByRole("button", { name: "internal/reference.py", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "training_task_runner.py", exact: true }).click();
  await expect(page.locator(".workbench-scroll-tools")).toContainText("701 行");
  await expect(page.locator(".cm-content")).toContainText("training_step_1");
  const scroller = page.locator(".workbench-editor .cm-scroller");
  await expect.poll(() => scroller.evaluate((node) => node.scrollHeight > node.clientHeight)).toBe(true);
  await page.getByRole("button", { name: "文件末尾", exact: true }).click();
  await expect(page.locator(".cm-content")).toContainText("END_OF_FULL_SOURCE");
  await expect.poll(() => scroller.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
  await page.getByRole("button", { name: "顶部", exact: true }).click();
  await expect(page.locator(".cm-content")).toContainText("training_step_1");
  await page.getByRole("button", { name: "全部源码 (4)", exact: true }).click();
  await expect(page.getByRole("button", { name: "internal/reference.py", exact: true })).toBeVisible();
});

test("floating Agent drags without opening, stays in bounds and remembers its position", async ({ page }) => {
  await setup(page);
  const trigger = page.locator(".floating-agent-trigger");
  const initial = (await trigger.boundingBox())!;
  const actions = await page.locator(".workbench-actions").boundingBox();
  expect(initial.y + initial.height <= actions!.y || initial.x + initial.width <= actions!.x || initial.x >= actions!.x + actions!.width).toBe(true);
  await page.mouse.move(initial.x + 24, initial.y + 24);
  await page.mouse.down();
  await page.mouse.move(500, 300, { steps: 12 });
  await page.mouse.up();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  const moved = (await trigger.boundingBox())!;
  expect(moved.x).toBeLessThan(initial.x - 100);
  await page.reload();
  await expect(trigger).toBeVisible();
  expect((await trigger.boundingBox())!.x).toBeCloseTo(moved.x, 0);
  await trigger.click();
  await expect(page.locator(".floating-agent-panel")).toBeVisible();
  const panel = (await page.locator(".floating-agent-panel").boundingBox())!;
  expect(panel.x).toBeGreaterThanOrEqual(0);
  expect(panel.y).toBeGreaterThanOrEqual(0);
  expect(panel.x + panel.width).toBeLessThanOrEqual(1280);
  expect((await page.locator(".floating-agent-guide-tools").boundingBox())!.height).toBeLessThan(65);
  await page.setViewportSize({ width: 390, height: 844 });
  const clamped = (await trigger.boundingBox())!;
  expect(clamped.x + clamped.width).toBeLessThanOrEqual(390);
});

test("English workbench, page tour and full tutorial follow the current layout", async ({ page }) => {
  await setup(page);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.locator(".workbench-task-tabs")).toContainText("Transport and hold");
  const ui = await page.locator(".workbench-page").innerText();
  // Course/class names are user data; code source is deliberately untranslated.
  expect(ui.replaceAll("实验课程", "").replaceAll("实验课", "").replaceAll("一班", "")).not.toMatch(/[\u4e00-\u9fff]/);
  await page.getByRole("button", { name: "Page Tour", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Choose an experiment task");
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Select context and Project");
  await page.keyboard.press("Escape");
  await page.goto("/guide/project-workspace");
  await expect(page.locator(".guide-page")).toContainText("highlighted return expression");
  expect(await page.locator(".guide-page").innerText()).not.toMatch(/[\u4e00-\u9fff]/);
});

test("an unavailable workbench API shows a recovery action instead of a blank editable editor", async ({ page }) => {
  await setup(page);
  await page.route("**/api/v1/workbench/drafts/*", (route) => route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Not Found" }) }));
  await page.reload();
  await expect(page.locator(".workbench-load-state")).toContainText("奖励代码未加载");
  await expect(page.locator(".workbench-load-state").getByRole("button", { name: "重试加载" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "保存版本", exact: true })).toBeDisabled();
});

test("capture the workbench at desktop and mobile sizes", async ({ page }) => {
  await setup(page);
  await page.setViewportSize({ width: 1600, height: 1050 });
  await page.screenshot({ path: "../../var/validation/scene-workbench-reward-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "../../var/validation/scene-workbench-reward-mobile.png", fullPage: true });
});

test("provider configuration verifies a real request result and explains failures in English", async ({ page }) => {
  await setup(page);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  let configured = false;
  let checks = 0;
  const deepseek = () => ({ id: "deepseek", name: "DeepSeek", status: configured ? "unknown" : "unavailable", configured, models: configured ? ["test-model"] : [], capabilities: { chat: true, structured_output: false, tool_calls: false }, data_scope: "会将系统提示和当前会话消息发送到管理员或当前会话配置的 DeepSeek API。" });
  await page.route("**/api/agent/providers**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    let body: unknown = [deepseek()];
    if (path.endsWith("/configuration")) { configured = true; body = deepseek(); }
    if (path.endsWith("/check")) {
      checks++;
      body = { ...deepseek(), status: checks > 1 ? "healthy" : "degraded", connection: { status: checks > 1 ? "healthy" : "degraded", message: checks > 1 ? "Connection verified" : "API Key 无效或已撤销，请重新配置 (HTTP 401)" } };
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("/agent");
  await page.locator(".provider-list button").filter({ hasText: "DeepSeek" }).click();
  await page.locator('input[type="password"]').fill("test-key-never-persisted");
  await page.getByLabel("Model ID", { exact: true }).fill("test-model");
  await page.getByRole("button", { name: "Configure and verify", exact: true }).click();
  await expect(page.locator(".provider-config-message")).toContainText("API key is invalid");
  await expect(page.locator(".provider-list")).toContainText("Connection failed");
  expect(await page.locator(".agent-settings-page").innerText()).not.toMatch(/[\u4e00-\u9fff]/);
  await page.getByRole("button", { name: "Verify connection again", exact: true }).click();
  await expect(page.locator(".provider-config-message")).toContainText("Connection verified");
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain("test-key-never-persisted");
});

test("Run deletion confirms, recovers from a server error and removes terminal records", async ({ page }) => {
  await setup(page);
  let records = [
    { ...record, run_id: "completed-delete-test", status: "completed" },
    { ...record, run_id: "failed-delete-test", status: "failed" },
    { ...record, run_id: "active-keep-test", status: "running" },
  ];
  let deletions = 0;
  await page.route("**/api/v1/runs**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/v1/runs") return route.fulfill({ json: records });
    if (path === "/api/v1/runs/delete") {
      deletions++;
      if (deletions === 1) return route.fulfill({ status: 503, json: { detail: "Run deletion is temporarily unavailable" } });
      const ids = route.request().postDataJSON().run_ids as string[];
      expect(ids).toEqual(["completed-delete-test", "failed-delete-test"]);
      records = records.filter((item) => !ids.includes(item.run_id));
      return route.fulfill({ json: { deleted_count: ids.length, deleted_run_ids: ids } });
    }
    return route.fallback();
  });
  await page.goto("/runs");
  await expect(page.locator(".run-record")).toHaveCount(3);
  await expect(page.locator(".run-delete-option input").nth(2)).toBeDisabled();
  await page.locator(".run-delete-controls input").check();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "删除选中 (2)", exact: true }).click();
  expect(deletions).toBe(0);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "删除选中 (2)", exact: true }).click();
  await expect(page.locator(".run-delete-controls [role=alert]")).toContainText("temporarily unavailable");
  await expect(page.locator(".run-record")).toHaveCount(3);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "删除选中 (2)", exact: true }).click();
  await expect(page.locator(".run-record")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "删除选中 (0)", exact: true })).toBeDisabled();
  await expect(page.locator(".run-delete-controls [role=alert]")).toHaveCount(0);
});

test("teaching controls synchronize signed contributions, invalidate preflight and freeze edited source", async ({ page }) => {
  const { submissions, questions } = await setup(page);
  await expect(page.locator(".reward-sample-summary > strong")).toHaveText("0.030");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  await page.getByRole("spinbutton", { name: "动作幅度惩罚权重", exact: true }).fill("2");
  await expect(page.locator(".reward-sample-summary > strong")).toHaveText("0.020");
  await expect(page.locator(".cm-content")).toContainText('2 * terms["action_penalty"]');
  await expect(page.locator(".preflight-pending")).toBeVisible();
  await expect(page.locator(".cm-content")).toHaveAttribute("contenteditable", "false");
  await page.locator(".reward-term-card").filter({ hasText: "action_penalty" }).getByRole("button", { name: "请 Agent 解释这项" }).click();
  await expect.poll(() => questions.length).toBe(1);
  expect(questions[0].question).toContain("action_penalty");
  expect(questions[0].source).toContain('2 * terms["action_penalty"]');
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: "保存版本", exact: true }).click();
  await expect(page.locator(".workbench-commandbar")).toContainText("版本已保存");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).click();
  await expect.poll(() => submissions.length).toBe(1);
  expect(submissions[0].code).toEqual((submissions[0].reward_extension as { source: string }).source);
  expect(submissions[0].code).toContain('2 * terms["action_penalty"]');
});

test("scene and settings retain edits, reset independently and submit the real parameter values", async ({ page }) => {
  const { submissions } = await setup(page);
  await page.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class");
  await page.getByRole("combobox", { name: "训练 Project", exact: true }).selectOption("project");
  await page.getByRole("button", { name: /03 实验设置/ }).click();
  await expect(page.locator(".workbench-budget-card")).toContainText("20,000");
  await page.getByRole("spinbutton", { name: "随机种子", exact: true }).fill("11");
  await page.getByRole("button", { name: /01 任务场景/ }).click();
  await expect(page.getByRole("spinbutton", { name: "成功距离（课程固定）" })).toBeDisabled();
  await page.getByRole("spinbutton", { name: "单步最大位移（米）" }).fill("0.008");
  await page.getByRole("button", { name: "恢复默认", exact: true }).click();
  await expect(page.getByRole("spinbutton", { name: "单步最大位移（米）" })).toHaveValue("0.005");
  await page.getByRole("spinbutton", { name: "单步最大位移（米）" }).fill("0.007");
  await page.getByRole("button", { name: /03 实验设置/ }).click();
  await expect(page.getByRole("spinbutton", { name: "随机种子", exact: true })).toHaveValue("11");
  await page.getByRole("button", { name: /02 奖励设计/ }).click();
  await page.getByRole("button", { name: "保存版本", exact: true }).click();
  await expect(page.locator(".workbench-commandbar")).toContainText("版本已保存");
  await page.getByRole("button", { name: "运行预检", exact: true }).click();
  await expect(page.getByRole("button", { name: "启动训练", exact: true })).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "启动训练", exact: true }).click();
  await expect.poll(() => submissions.length).toBe(1);
  expect(submissions[0].parameters).toMatchObject({ seed: 11, action_delta_m: .007, target_center_xy_m: [.36, 0], success_distance_m: .02 });
});

test("custom expressions survive module switches and a cancelled preset replacement", async ({ page }) => {
  await setup(page);
  await page.getByRole("button", { name: "进阶代码编辑", exact: true }).click();
  const editor = page.locator(".workbench-editor .cm-content");
  await expect(editor).toHaveAttribute("contenteditable", "true");
  await editor.focus();
  await page.keyboard.press("Control+A");
  const custom = 'def combine_reward(terms):\n    return terms["progress_reward"] / 2 - terms["action_penalty"]\n';
  await editor.fill(custom);
  await expect(editor).toContainText('/ 2 - terms["action_penalty"]');
  await expect(page.locator(".reward-custom-note")).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "动作幅度惩罚权重", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: /03 实验设置/ }).click();
  await page.getByRole("button", { name: /02 奖励设计/ }).click();
  await page.locator(".workbench-generated-code").scrollIntoViewIfNeeded();
  await expect(editor).toContainText('/ 2 - terms["action_penalty"]');
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "默认方案", exact: true }).click();
  await page.locator(".workbench-generated-code").scrollIntoViewIfNeeded();
  await expect(editor).toContainText('/ 2 - terms["action_penalty"]');
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "默认方案", exact: true }).click();
  await expect(page.locator(".reward-custom-note")).toHaveCount(0);
  await expect(page.locator(".reward-sample-summary > strong")).toHaveText("0.030");
});

test("Push and Place use their own beginner terms and explain signed costs in English", async ({ page }) => {
  await setup(page);
  const cases = [
    { task: "s1_push", button: /S1 Push · Object pushing/, terms: ["precontact_progress", "wrong_side_or_alignment_term", "directional_push_progress", "lateral_push_penalty", "contact_sustain_bonus", "action_penalty"], penalty: "Action magnitude penalty weight", preset: "Emphasize smaller actions" },
    { task: "s1_place", button: /S1 Place · Transport and hold/, terms: ["xy_progress", "target_zone", "height_stability", "grasp_stability", "action_failure"], penalty: "Action and failure penalty weight", preset: "Emphasize action and failure costs" },
  ];
  for (const entry of cases) {
    const code = 'def combine_reward(terms):\n    return ' + entry.terms.map((term) => `terms["${term}"]`).join(" + ") + "\n";
    await page.route(`**/api/v1/workbench/drafts/${entry.task}`, (route) => route.fulfill({ json: { task_id: entry.task, starter: code, source: code, revision_id: "starter", editable_terms: entry.terms, protected_terms: ["success_bonus", "joint_limit_penalty"], history: [] } }));
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  for (const entry of cases) {
    await page.getByRole("button", { name: entry.button }).click();
    await expect(page.locator(".reward-term-card")).toHaveCount(entry.terms.length);
    expect(await page.locator(".reward-designer").innerText()).not.toMatch(/[\u4e00-\u9fff]/);
    await page.getByRole("button", { name: entry.preset, exact: true }).click();
    await expect(page.getByRole("spinbutton", { name: entry.penalty, exact: true })).toHaveValue("2");
    await expect(page.locator(".workbench-commandbar")).toContainText("Unsaved changes");
    // Avoid carrying the unsaved-expression confirmation into the next task.
    await page.getByRole("button", { name: "Baseline", exact: true }).click();
  }
});
