import { describe, expect, it } from "vitest";
import { teachingProjectLabel } from "./projectPresentation";

describe("teaching Project labels", () => {
  it("names Push profiles consistently in both languages", () => {
    const project = { title: "入门训练与演示", environment_type: "s1_push", training_profile: "short" };
    expect(teachingProjectLabel(project, "zh-CN")).toBe("S1 推动 · 入门短程");
    expect(teachingProjectLabel(project, "en")).toBe("S1 Push · introductory short run");
    expect(teachingProjectLabel({ ...project, training_profile: "standard" }, "zh-CN")).toBe("S1 推动 · 标准训练");
    expect(teachingProjectLabel({ ...project, training_profile: "standard" }, "en")).toBe("S1 Push · standard training");
  });

  it("preserves other tasks and unknown profile titles", () => {
    const project = { title: "抓取训练", environment_type: "s1_grasp", training_profile: "standard" };
    expect(teachingProjectLabel(project, "zh-CN")).toBe(project.title);
    expect(teachingProjectLabel({ ...project, environment_type: "s1_push", training_profile: "custom" }, "en")).toBe(project.title);
  });
});