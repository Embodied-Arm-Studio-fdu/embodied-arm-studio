import { describe, expect, it, vi } from "vitest";
import {
  applyLocale,
  LOCALE_STORAGE_KEY,
  readStoredLocale,
  translateUiText,
  translateUiTextToChinese,
} from "./i18n";

describe("i18n", () => {
  it("uses Chinese by default and restores English when saved", () => {
    expect(readStoredLocale({ getItem: () => null })).toBe("zh-CN");
    expect(readStoredLocale({ getItem: () => "en" })).toBe("en");
    expect(readStoredLocale({ getItem: () => "unsupported" })).toBe("zh-CN");
  });

  it("persists the locale and updates the document language", () => {
    const setItem = vi.fn();
    const documentElement = { lang: "zh-CN" };

    applyLocale("en", { setItem }, documentElement);
    expect(setItem).toHaveBeenCalledWith(LOCALE_STORAGE_KEY, "en");
    expect(documentElement.lang).toBe("en");

    applyLocale("zh-CN", { setItem }, documentElement);
    expect(setItem).toHaveBeenLastCalledWith(LOCALE_STORAGE_KEY, "zh-CN");
    expect(documentElement.lang).toBe("zh-CN");
  });

  it("translates exact UI labels while preserving whitespace", () => {
    expect(translateUiText("  登录研究平台\n")).toBe("  Sign in to Research Studio\n");
  });

  it("translates representative UI from the main workflows", () => {
    expect(translateUiText("项目与文件工作区")).toBe("Project and file workspace");
    expect(translateUiText("运行记录与对比")).toBe("Run records and comparison");
    expect(translateUiText("运行证据导出")).toBe("Run evidence export");
    expect(translateUiText("姿态滑杆只调整网页预览 不写入 Run 参数，不执行 IK、碰撞检测或真实机器人控制。"))
      .toBe("Pose sliders only adjust the web preview They do not write Run parameters or execute IK, collision checks, or real robot control.");
  });

  it("translates the current English experiment labels", () => {
    expect(translateUiText("学习路线与实验流程")).toBe("Learning path and experiment flow");
    expect(translateUiText("准备六轴场景")).toBe("Prepare the six-axis scene");
    expect(translateUiText("任务选择")).toBe("Task selection");
    expect(translateUiText("触达训练")).toBe("Reach training");
    expect(translateUiText("推动训练")).toBe("Push training");
    expect(translateUiText("进入启动实验")).toBe("Continue to launch experiment");
    expect(translateUiText("补齐两套任务草稿（不覆盖）")).toBe("Add both task drafts without overwriting");
  });

  it("translates multiline JSX text after normalizing whitespace", () => {
    expect(translateUiText("API 仅返回 configured 状态。密钥值、Authorization header\n                和服务端 secret reference 不进入此页面。"))
      .toBe("The API returns only the configured status. Key values, Authorization headers, and server-side secret references never enter this page.");
  });

  it("translates server-provided Provider metadata", () => {
    expect(translateUiText("内置基础 Agent")).toBe("Built-in basic Agent");
    expect(translateUiText("仅使用当前会话中提供的教学上下文，不发送外部请求。"))
      .toBe("Uses only the teaching context provided in the current session and sends no external requests.");
    expect(translateUiText("Mock Pipeline 是教学近似，不代表物理精确仿真"))
      .toBe("The Mock Pipeline is an instructional approximation and does not represent physically accurate simulation");
  });

  it("translates learning navigation and evaluation messages without mixed languages", () => {
    expect(translateUiText("入门学习")).toBe("Introduction to Embodied AI");
    expect(translateUiText("可用")).toBe("Available");
    for (const text of [
      "认识平台并按步骤理解一次六轴 S1 仿真实验",
      "策略评估证据",
      "该运行未生成正式评估。Run completed 不代表 Policy passed，也不会显示为 0%。",
      "正在读取实验工作台…",
    ]) expect(translateUiText(text)).not.toMatch(/[\u3400-\u9fff]/u);
  });

  it("translates mixed status phrases", () => {
    expect(translateUiText("正在保存，失败后重试")).toBe("Saving; retry if it fails");
  });

  it("leaves non-Chinese content unchanged", () => {
    expect(translateUiText("Run #42")).toBe("Run #42");
  });

  it("localizes legacy English UI labels without translating technical terms", () => {
    expect(translateUiTextToChinese("RUN EVIDENCE")).toBe("Run 证据");
    expect(translateUiTextToChinese("STEP 03 / 06")).toBe("步骤 03 / 06");
    expect(translateUiTextToChinese("PPO")).toBe("PPO");
    expect(translateUiTextToChinese("run_id")).toBe("run_id");
    expect(translateUiTextToChinese("Run lookup is temporarily unavailable")).toBe("Run 查询暂时不可用");
    expect(translateUiText("运行或评估报告不存在")).toBe("Run or evaluation report not found");
  });
});
