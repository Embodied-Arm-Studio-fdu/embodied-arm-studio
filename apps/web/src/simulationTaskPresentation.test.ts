import { describe, expect, it } from "vitest";
import { simulationTaskTitle } from "./simulationTaskPresentation";
import { defaultSimulationCatalog } from "./simulationCatalog";

// Server catalog display names can remain Chinese even when the UI locale is English.
describe("simulationTaskTitle", () => {
  it("localizes server-provided task names by their stable IDs", () => {
    for (const task of defaultSimulationCatalog().tasks) {
      const serverTask = { ...task, title: "服务端中文任务名称" };
      expect(simulationTaskTitle(serverTask, "en")).not.toMatch(/[\u3400-\u9fff]/u);
      expect(simulationTaskTitle(serverTask, "zh-CN")).toBe(serverTask.title);
    }
    expect(simulationTaskTitle({ id: "s1_lift", title: "S1 抓取抬起" }, "en")).toBe("S1 Grasp and Lift");
    expect(simulationTaskTitle({ id: "s1_lower_release", title: "S1 抓取放下" }, "en")).toBe("S1 Lower and Release");
  });

  it("preserves custom English titles for unknown tasks", () => {
    expect(simulationTaskTitle({ id: "custom_task", title: "Custom task" }, "en")).toBe("Custom task");
  });
});
