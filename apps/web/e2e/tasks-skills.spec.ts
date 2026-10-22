import { expect, test, type Page } from "@playwright/test";

const demoIds = ["demo_s1_target_pose", "demo_s1_line_tracking", "demo_s1_circle_tracking", "demo_s1_joint_position", "demo_s1_joint_sweep", "demo_s1_scripted_pick_place"];
const user = { user_id: "skills-test", username: "skills", display_name: "Skills", enabled: true, organizations: [{ organization_id: "org-skills", name: "Skills Lab", slug: "skills", enabled: true, role: "student", permissions: ["own_run:create"] }] };
const project = { project_id: "short-project", course_id: "course-skills", title: "S1 Short", task_type: "training", environment_type: "s1_push", training_profile: "short", config_version: 2, enabled: true, permissions: { can_use: true, can_view: true, can_modify: false } };
const catalog = { catalog_version: "task-catalog/v1", tasks: ["s1_push", "s1_reach", ...demoIds].map((id) => ({
  id, display_name: id, task_type: id.startsWith("demo_") ? "demo" : "training", supports_training: !id.startsWith("demo_"), supports_video: true,
  parameters: [{ name: "seed", type: "integer", default: 7, minimum: 0 }, { name: "duration_seconds", type: "number", default: 30, minimum: 5, maximum: 300 }, { name: "video_enabled", type: "boolean", default: true }],
})) };
const record = { run_id: "skills-demo-run", user_id: user.user_id, project_id: project.project_id, class_id: "class-skills", run_kind: "play", execution_mode: "simulation", status: "completed", parameters: { seed: 7 }, project_config_snapshot: { environment_type: "demo_s1_circle_tracking", task_type: "demo", video_enabled: false }, metrics_summary: {}, progress: 1, training_steps: 0, update_steps: 0, last_sequence: 0, created_at: "2026-10-07T00:00:00Z", started_at: null, ended_at: "2026-10-07T00:00:10Z", updated_at: "2026-10-07T00:00:10Z", error: null, simulation_task_id: null };

async function setup(page: Page, catalogUnavailable = false, canCreate = true) {
  const requests: { body: Record<string, unknown>; key: string | undefined }[] = [];
  await page.addInitScript(() => {
    sessionStorage.setItem("embodied-arm-access-token", "skills-test-token");
    localStorage.setItem("embodied-arm:onboarding:skills-test", JSON.stringify({ firstTimeOnboarding: { status: "completed", version: 1 } }));
  });
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body: unknown = [];
    if (path === "/api/v1/me") body = { ...user, organizations: user.organizations.map((organization) => ({ ...organization, permissions: canCreate ? ["own_run:create"] : [] })) };
    else if (path === "/api/simulation/task-catalog") {
      if (catalogUnavailable) { await route.fulfill({ status: 404, contentType: "application/json", body: "{}" }); return; }
      body = catalog;
    } else if (path === "/api/v1/courses") body = [{ course_id: "course-skills", title: "S1 Course", enabled: true }];
    else if (path.endsWith("/classes")) body = [{ class_id: "class-skills", name: "Class A", enabled: true }];
    else if (path.endsWith("/projects")) body = [project, { ...project, project_id: "standard-project", training_profile: "standard", title: "Standard excluded" }];
    else if (path === "/api/v1/runs" && request.method() === "POST") {
      requests.push({ body: request.postDataJSON() as Record<string, unknown>, key: request.headers()["idempotency-key"] });
      body = record;
    } else if (path === "/api/v1/runs") body = [];
    else if (path === `/api/v1/runs/${record.run_id}`) body = record;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("http://127.0.0.1:4173/skills");
  await expect(page.getByRole("heading", { name: "任务与技能", exact: true })).toBeVisible();
  return requests;
}

async function configure(page: Page) {
  await page.getByRole("tab", { name: "控制演示", exact: true }).focus();
  await page.keyboard.press("Enter");
  const context = page.locator(".skills-demo-context");
  await context.getByRole("combobox", { name: "课程", exact: true }).selectOption("course-skills");
  await context.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class-skills");
  await context.getByRole("combobox", { name: "Short Push Project" }).selectOption("short-project");
}

test("Demo launch uses the canonical play contract and existing Run details", async ({ page }) => {
  const requests = await setup(page);
  await configure(page);
  await expect(page.locator(".skills-demo-selector button")).toHaveCount(6);
  await expect(page.locator(".skills-demo-context option[value='standard-project']")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "启动控制演示" })).toBeEnabled();
  await page.getByRole("button", { name: "启动控制演示" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/runs\/skills-demo-run$/);
  expect(requests).toHaveLength(1);
  expect(requests[0].body).toMatchObject({ project_id: "short-project", class_id: "class-skills", project_config_version: 2, run_kind: "play", environment_type: "demo_s1_circle_tracking", execution_mode: "simulation", parameters: { seed: 7, duration_seconds: 30, video_enabled: true } });
  expect(requests[0].body).not.toHaveProperty("scene_ref");
  expect(requests[0].key).toBeTruthy();
});

test("fallback catalog blocks Demo submission", async ({ page }) => {
  const requests = await setup(page, true);
  await configure(page);
  await expect(page.getByRole("button", { name: "启动控制演示" })).toBeDisabled();
  await expect(page.locator(".skills-source")).toContainText("LOCAL FALLBACK");
  expect(requests).toHaveLength(0);
});

test("missing creation permission blocks Demo submission", async ({ page }) => {
  const requests = await setup(page, false, false);
  await configure(page);
  await expect(page.getByRole("button", { name: "启动控制演示" })).toBeDisabled();
  expect(requests).toHaveLength(0);
});

test("two actionable tabs merge task details and keep experiments non-executable", async ({ page }) => {
  const requests = await setup(page);
  await expect(page.getByRole("tab")).toHaveCount(2);
  await expect(page.getByRole("tab", { name: "强化学习任务" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tab", { name: "操作技能" })).toHaveCount(0);
  await expect(page.getByRole("tab", { name: "组合任务" })).toHaveCount(0);
  await page.locator(".skills-experimental-reference > summary").focus();
  await page.keyboard.press("Enter");
  const grasp = page.locator(".skills-experimental-reference .skill-row").filter({ hasText: "s1_grasp" });
  await expect(grasp).toContainText("未开放");
  await expect(grasp.getByRole("button", { name: "选择教学 Project" })).toHaveCount(0);
  await grasp.locator(".skill-task-details summary").focus();
  await page.keyboard.press("Enter");
  await expect(grasp.getByText("起始条件", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "添加阶段" })).toHaveCount(0);
  expect(requests).toHaveLength(0);
});

test("mobile workspace and tutorial remain within the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await page.locator(".skill-task-details summary").first().focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "阅读任务与技能教程" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/guide\/task-skills$/);
  await expect(page.locator(".guide-article h2")).toHaveText("任务与技能");
  await expect(page.locator(".guide-principles")).toContainText("准备姿态 → 接近物体");
});

test("newly registered manipulation is discoverable without enabling unregistered Place", async ({ page }) => {
  await setup(page);
  await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...catalog, tasks: [...catalog.tasks, { id: "s1_grasp", display_name: "S1 Grasp", task_type: "training", supports_training: true, supports_video: true, parameters: [] }] }) }));
  await page.reload();
  await expect(page.locator(".skill-row").filter({ hasText: "s1_grasp" })).toContainText("CHECKPOINT");
  const grasp = page.locator(".skills-list > .skill-row").filter({ hasText: "s1_grasp" });
  await expect(grasp).toContainText("先选择组织和课程");
  await expect(grasp.getByRole("button")).toHaveCount(0);
  await grasp.locator(".skill-task-details summary").focus();
  await page.keyboard.press("Enter");
  await expect(grasp.getByText("起始条件", { exact: true })).toBeVisible();
  await expect(page.locator(".skills-list > .skill-row").filter({ hasText: "s1_place" })).toHaveCount(0);
});

test("sandbox task and parameters follow only the selected teaching Project", async ({ page }) => {
  await setup(page);
  const tasks = ["s1_grasp", "s1_lift", "s1_lower_release"];
  await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tasks: tasks.map((id, index) => ({ id, display_name: id, task_type: "training", supports_training: true, parameters: [{ name: "seed", type: "integer", default: 10 + index, minimum: 0 }] })) }) }));
  await page.route("**/api/v1/courses/course-skills/projects", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(tasks.map((id) => ({ ...project, project_id: id, title: id, environment_type: id, algorithm: "PPO", expected_training_steps: 20000 }))) }));
  await page.route("**/api/scenes", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "[]" }));
  await page.evaluate(() => {
    sessionStorage.setItem("s1-draft-course", "course-skills");
    sessionStorage.setItem("s1-draft-project", "s1_grasp");
    sessionStorage.setItem("s1-draft-simulation-task-v1", JSON.stringify({ taskId: "s1_lower_release", parameters: { seed: 99 } }));
  });
  await page.goto("http://127.0.0.1:4173/training");
  const projectSelect = page.locator(".run-config").getByRole("combobox", { name: "教学 Project", exact: true });
  const form = page.locator(".s1-task-configuration");
  await expect(form.locator(".simulation-task-heading h3")).toHaveText("s1_grasp");
  await expect(form.getByRole("spinbutton")).toHaveValue("10");
  await expect(form.locator(".simulation-task-selector, .simulation-task-overview")).toHaveCount(0);
  await form.getByRole("spinbutton").fill("55");
  await projectSelect.selectOption("s1_lift");
  await expect(form.locator(".simulation-task-heading h3")).toHaveText("s1_lift");
  await expect(form.getByRole("spinbutton")).toHaveValue("11");
  await projectSelect.selectOption("s1_lower_release");
  await expect(form.locator(".simulation-task-heading h3")).toHaveText("s1_lower_release");
  await expect(form.getByRole("spinbutton")).toHaveValue("99");
  await expect(form).not.toContainText("任务与 Project 不匹配");
  await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tasks: [catalog.tasks[0]] }) }));
  await page.reload();
  await expect(form).toContainText("catalog 没有可用任务");
  await expect(form.getByRole("button", { name: "进入启动实验" })).toBeDisabled();
});

test("training details link authorized training evidence rather than Demo records", async ({ page }) => {
  await setup(page);
  await page.route("**/api/v1/runs", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([
    { ...record, run_id: "training-evidence", run_kind: "train", project_config_snapshot: { environment_type: "s1_push" } },
    { ...record, run_id: "demo-not-training", run_kind: "play", project_config_snapshot: { environment_type: "s1_push" } },
  ]) }));
  await page.reload();
  const push = page.locator(".skills-list > .skill-row").filter({ hasText: "s1_push" });
  await push.locator(".skill-task-details summary").focus();
  await page.keyboard.press("Enter");
  await expect(push.getByRole("button", { name: /evidence/ })).toBeVisible();
  await expect(push.locator(".skills-evidence-links button")).toHaveCount(1);
  await push.getByRole("button", { name: /evidence/ }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/runs\/training-evidence$/);
});

test("sandbox contains only RL Projects and explains the introductory Short Push profile", async ({ page }) => {
  const requests = await setup(page);
  const shortTrainingProject = { ...project, title: "入门训练与演示", algorithm: "PPO", expected_training_steps: 32 };
  await page.route("**/api/v1/courses/course-skills/projects", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([
    { ...shortTrainingProject, project_id: "demo-project", task_type: "demo", environment_type: "demo_s1_circle_tracking" },
    { ...shortTrainingProject, project_id: "mislabelled-demo-project", environment_type: "demo_s1_joint_position" },
    shortTrainingProject,
  ]) }));
  await page.route("**/api/scenes", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "[]" }));
  await page.evaluate(() => {
    sessionStorage.setItem("s1-draft-course", "course-skills");
    sessionStorage.setItem("s1-draft-project", "demo-project");
    sessionStorage.setItem("s1-draft-simulation-task-v1", JSON.stringify({ taskId: "demo_s1_circle_tracking", parameters: { seed: 99 } }));
  });
  await page.goto("http://127.0.0.1:4173/training");
  const context = page.locator(".s1-parameter-page .run-config");
  const projectSelect = context.getByRole("combobox", { name: "教学 Project", exact: true });
  await expect(projectSelect).toHaveValue("short-project");
  await expect(projectSelect.locator("option[value='short-project']")).toHaveText("S1 推动 · 入门短程");
  await expect(projectSelect.locator("option[value='demo-project'], option[value='mislabelled-demo-project']")).toHaveCount(0);
  await expect(context).toContainText("32 步短程 PPO");
  await expect(context).toContainText("不代表策略充分训练");
  const form = page.locator(".s1-task-configuration");
  await expect(form.locator(".simulation-task-heading h3")).toHaveText("s1_push");
  await expect(form.locator(".simulation-parameter-group h4")).toHaveText(["训练参数", "视频记录参数"]);
  await expect(form.locator(".simulation-parameter-group").filter({ has: page.getByRole("heading", { name: "视频记录参数", exact: true }) })).toContainText("video_enabled");
  await expect(form.locator(".simulation-task-selector, .simulation-task-overview")).toHaveCount(0);
  expect(requests).toHaveLength(0);
  await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tasks: [{ ...catalog.tasks[0], task_type: "demo", supports_training: false }] }) }));
  await page.reload();
  await expect(form).toContainText("catalog 没有可用任务");
  await expect(form.getByRole("button", { name: "进入启动实验" })).toBeDisabled();
});

test("scripted pick and place is configurable and uses a play Run with video enabled", async ({ page }) => {
  const requests = await setup(page);
  await configure(page);
  await page.locator(".skills-demo-selector").getByRole("button", { name: "demo_s1_scripted_pick_place", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".simulation-task-heading")).toContainText("不进行 PPO 学习");
  await expect(page.getByRole("button", { name: "启动控制演示" })).toBeEnabled();
  await page.getByRole("button", { name: "启动控制演示" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/runs\/skills-demo-run$/);
  expect(requests).toHaveLength(1);
  expect(requests[0].body).toMatchObject({ run_kind: "play", environment_type: "demo_s1_scripted_pick_place", parameters: { duration_seconds: 30, seed: 7, video_enabled: true } });
  expect(requests[0].body).not.toHaveProperty("scene_ref");
  expect(requests[0].key).toBeTruthy();
});

for (const taskId of ["s1_grasp", "s1_lift", "s1_lower_release"]) {
  test(`${taskId} opens its authorized Project and submits the matching train Run`, async ({ page }) => {
    const requests = await setup(page);
    const trainingTask = { id: taskId, display_name: taskId, task_type: "training", supports_training: true, supports_video: true, parameters: [{ name: "seed", type: "integer", default: 7, minimum: 0 }, { name: "video_enabled", type: "boolean", default: true }] };
    await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tasks: [...catalog.tasks, trainingTask] }) }));
    const manipulationProject = { ...project, project_id: `project-${taskId}`, title: `Training ${taskId}`, environment_type: taskId, training_profile: "standard", config_version: 4, algorithm: "PPO", video_enabled: true, expected_training_steps: 20000, description: "Manipulation training", instructions: "Check task conditions", parameter_schema: { type: "object", properties: { seed: { type: "integer", minimum: 0, maximum: 2147483647 } }, required: ["seed"] }, evaluation_rubric: {} };
    await page.route("**/api/v1/courses/course-skills/projects", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([manipulationProject]) }));
    await page.route("**/api/scenes", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "[]" }));
    await page.reload();
    const context = page.locator(".skills-training-context");
    await context.getByRole("combobox", { name: "课程", exact: true }).selectOption("course-skills");
    await context.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class-skills");
    const entry = page.locator(".skills-list > .skill-row").filter({ hasText: taskId }).getByRole("button", { name: `Training ${taskId} · V4` });
    await expect(entry).toBeEnabled();
    await entry.focus();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/courses$/);
    await expect(page.getByRole("heading", { name: `Training ${taskId}`, exact: true })).toBeVisible();
    await expect(page.locator("#formal-scene")).toHaveValue("");
    await expect(page.getByRole("button", { name: "开始训练", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "开始训练", exact: true }).focus();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/runs\/skills-demo-run$/);
    expect(requests).toHaveLength(1);
    expect(requests[0].body).toMatchObject({ run_kind: "train", project_id: manipulationProject.project_id, class_id: "class-skills", project_config_version: 4, environment_type: taskId, parameters: { seed: 7, video_enabled: true } });
    expect(requests[0].body).not.toHaveProperty("scene_ref");
  });
}

test("missing manipulation Project is explained and Full Place remains an experimental reference", async ({ page }) => {
  const requests = await setup(page);
  await page.route("**/api/simulation/task-catalog", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tasks: [...catalog.tasks, { id: "s1_grasp", display_name: "S1 Grasp", task_type: "training", supports_training: true, parameters: [] }] }) }));
  await page.reload();
  const context = page.locator(".skills-training-context");
  await context.getByRole("combobox", { name: "课程", exact: true }).selectOption("course-skills");
  await context.getByRole("combobox", { name: "教学班", exact: true }).selectOption("class-skills");
  const grasp = page.locator(".skills-list > .skill-row").filter({ hasText: "s1_grasp" });
  await expect(grasp).toContainText("没有匹配且可使用的训练 Project");
  await expect(grasp.getByRole("button")).toHaveCount(0);
  await page.locator(".skills-experimental-reference > summary").focus();
  await page.keyboard.press("Enter");
  const place = page.locator(".skills-experimental-reference .skill-row").filter({ hasText: "s1_place" });
  await place.locator(".skill-task-details summary").focus();
  await page.keyboard.press("Enter");
  await expect(place).toContainText("worker 训练与评估接线");
  await expect(place).toContainText("分阶段策略切换和状态保留验证");
  await expect(place.getByRole("button")).toHaveCount(0);
  expect(requests).toHaveLength(0);
});