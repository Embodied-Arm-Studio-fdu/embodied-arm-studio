import { expect, test, type Page } from "@playwright/test";

const user = { user_id: "english-test", username: "english", display_name: "English Test", enabled: true, organizations: [{ organization_id: "org-english", name: "English Lab", slug: "english", enabled: true, role: "student", permissions: ["own_run:create"] }] };
const taskNames: Record<string, string> = {
  s1_push: "S1 推动", s1_reach: "S1 平面到达", s1_grasp: "S1 抓取", s1_lift: "S1 抓取抬起", s1_lower_release: "S1 抓取放下", s1_place: "S1 搬运放置",
  demo_s1_target_pose: "S1 目标位姿", demo_s1_line_tracking: "S1 直线跟踪", demo_s1_circle_tracking: "S1 圆轨迹跟踪", demo_s1_joint_position: "S1 关节定位", demo_s1_joint_sweep: "S1 关节扫描", demo_s1_scripted_pick_place: "S1 脚本抓放",
};

async function setup(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    sessionStorage.setItem("embodied-arm-access-token", "english-test-token");
    localStorage.setItem("embodied-arm-studio-locale", "en");
    localStorage.setItem("embodied-arm:onboarding:english-test", JSON.stringify({ firstTimeOnboarding: { status: "completed", version: 1 } }));
  });
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    let body: unknown = [];
    if (path === "/api/v1/me") body = user;
    else if (path === "/api/simulation/task-catalog") body = { catalog_version: "task-catalog/v1", tasks: Object.entries(taskNames).map(([id, display_name]) => ({ id, display_name, task_type: id.startsWith("demo_") ? "demo" : "training", supports_training: !id.startsWith("demo_"), supports_video: true, parameters: [{ name: "seed", type: "integer", default: 7, minimum: 0 }, { name: "video_enabled", type: "boolean", default: true }] })) };
    else if (path.includes("/providers")) body = [{ id: "mock", name: "内置基础 Agent", configured: true, models: ["teaching-mock-v1"], capabilities: { chat: true, structured_output: false, tool_calls: false } }];
    else if (path.includes("/workbench/drafts/")) body = { task_id: path.split("/").pop(), source: 'def combine_reward(terms):\n    return terms["progress_reward"] + terms["action_penalty"]\n', starter: 'def combine_reward(terms):\n    return terms["progress_reward"] + terms["action_penalty"]\n', revision_id: "starter", editable_terms: ["progress_reward", "action_penalty"], protected_terms: ["success_bonus", "joint_limit_penalty"], history: [] };
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  return errors;
}

async function untranslated(page: Page) {
  return page.evaluate(() => {
    const residuals: string[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.parentElement?.closest("script,style") && /[\u3400-\u9fff]/u.test(node.textContent ?? "")) residuals.push(node.textContent!.trim());
    }
    for (const element of document.querySelectorAll("[aria-label],[title],[placeholder],[alt]")) {
      for (const attribute of ["aria-label", "title", "placeholder", "alt"]) {
        const value = element.getAttribute(attribute);
        if (value && /[\u3400-\u9fff]/u.test(value)) residuals.push(`${attribute}: ${value}`);
      }
    }
    return [...new Set(residuals)];
  });
}

async function expectEnglish(page: Page) {
  await expect.poll(() => untranslated(page)).toEqual([]);
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
}

test("all twelve learning chapters render English text, examples, diagrams and accessible labels", async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await setup(page);
  for (let chapter = 1; chapter <= 12; chapter += 1) {
    await page.goto(`/learn/${chapter}`);
    await expect(page.getByRole("heading", { name: "Introduction to Embodied AI", exact: true })).toBeVisible();
    await expect(page.locator(".learning-article > h2")).toContainText(`Chapter ${chapter}`);
    await expectEnglish(page);
  }
  expect(errors).toEqual([]);
});

test("animated stages and round-trip language switching retain translated state and chapter links", async ({ page }) => {
  const errors = await setup(page);
  await page.goto("/learn/1");
  await expect(page.locator(".embodied-loop-stage strong")).toContainText("Environment state");
  for (let step = 0; step < 5; step += 1) {
    await page.getByRole("button", { name: "Step", exact: true }).click();
    await expectEnglish(page);
  }
  await expect(page.locator(".embodied-loop-stage strong")).toContainText("New observation");
  await page.locator(".learning-chapter-sections button").nth(2).click();
  const sectionHash = new URL(page.url()).hash;
  await page.locator(".language-switcher").getByRole("button", { name: "ZH", exact: true }).click();
  await expect(page.locator(".learning-page h1")).toHaveText("入门学习");
  await expect(page.locator(".embodied-loop-stage strong")).toContainText("新观测");
  await page.locator(".language-switcher").getByRole("button", { name: "EN", exact: true }).click();
  await expectEnglish(page);
  expect(new URL(page.url()).hash).toBe(sectionHash);
  await expect(page.locator(".embodied-loop-stage strong")).toContainText("New observation");
  await page.reload();
  await expect(page.locator(".learning-page h1")).toHaveText("Introduction to Embodied AI");
  await expectEnglish(page);
  expect(errors).toEqual([]);
});

test("robot-part descriptions and kinematics stay English after interaction", async ({ page }) => {
  const errors = await setup(page);
  await page.goto("/learn/2");
  await expect(page.locator(".robotics-part-picker button")).toHaveCount(10);
  for (const button of await page.locator(".robotics-part-picker button").all()) {
    await button.click();
    await expectEnglish(page);
  }
  await page.getByRole("slider", { name: "First joint angle q1" }).fill("45");
  await expect(page.locator(".kinematics-controls output").first()).toHaveText("45°");
  await page.getByRole("button", { name: "Restore example" }).click();
  await expect(page.locator(".kinematics-readout")).toContainText("x = 0.346 + 0.000 = 0.346 m");
  await expectEnglish(page);
  expect(errors).toEqual([]);
});

test("all task and demo names are English when the server returns Chinese names", async ({ page }) => {
  const errors = await setup(page);
  await page.goto("/skills");
  await expect(page.getByRole("heading", { name: "S1 Grasp and Lift", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "S1 Lower and Release", exact: true })).toBeVisible();
  await expectEnglish(page);
  await page.getByRole("tab", { name: "Control demos", exact: true }).click();
  await expect(page.locator(".skills-demo-selector button")).toHaveCount(6);
  for (const button of await page.locator(".skills-demo-selector button").all()) {
    await button.click();
    await expectEnglish(page);
  }
  expect(errors).toEqual([]);
});

test("practice answers and copy-code feedback are translated", async ({ page }) => {
  const errors = await setup(page);
  await page.goto("/learn/11");
  const answer = page.locator(".learning-answer").first();
  await answer.locator("summary").click();
  await expect(answer).toHaveAttribute("open", "");
  const copy = page.getByRole("button", { name: "Copy code", exact: true }).first();
  await copy.evaluate(button => {
    for (let parent = button.parentElement; parent; parent = parent.parentElement) if (parent instanceof HTMLDetailsElement) parent.open = true;
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: async () => undefined } });
  });
  await copy.click();
  await expect(copy).toHaveText(/Copied/);
  await expectEnglish(page);
  expect(errors).toEqual([]);
});

test("remaining platform pages and sign-in screen render in English", async ({ page }) => {
  test.setTimeout(90_000);
  const errors = await setup(page);
  for (const path of ["/roadmap", "/guide/overview", "/courses", "/runs", "/evaluation", "/export", "/agent", "/projects"]) {
    await page.goto(path);
    await expect(page.locator("main h1").first()).toBeVisible();
    await expectEnglish(page);
  }
  await page.locator(".nav-logout").click();
  await expect(page.getByRole("heading", { name: "Sign in to Research Studio" })).toBeVisible();
  await expectEnglish(page);
  expect(errors).toEqual([]);
});
