import { describe, expect, it } from "vitest";
import { platformFirstTimeTour, tourRegistry } from "./registry";

describe("tour registry", () => {
  it("resolves all canonical routes, query parameters and run details", () => {
    const paths = ["/roadmap", "/agent", "/projects", "/training-code", "/environments", "/training", "/courses", "/runs", "/evaluation", "/export"];
    for (const path of paths) {
      expect(tourRegistry.resolve(path)?.steps.length).toBeGreaterThan(0);
      expect(tourRegistry.resolve(`${path}?mode=preview`)?.id).toBe(tourRegistry.resolve(path)?.id);
    }
    expect(tourRegistry.resolve("/runs/some-run-id")?.id).toBe("run-detail");
    expect(tourRegistry.resolve("/unregistered")).toBeUndefined();
  });

  it("uses the workbench tour for both current and legacy routes", () => {
    expect(tourRegistry.resolve("/training")?.id).toBe("experiment-workbench");
    expect(tourRegistry.resolve("/projects")?.steps.map((item) => item.id)).toContain("workbench-editor");
    expect(tourRegistry.resolve("/agent")?.steps.map((item) => item.id)).toContain("agent-connection-check");
  });

  it("defines five platform steps and a real J2 value-change step", () => {
    expect(platformFirstTimeTour.steps.map(({ id }) => id)).toEqual([
      "brand", "primary-navigation", "secondary-navigation", "system-status", "guide-actions",
    ]);
    const scene = tourRegistry.resolve("/environments")!;
    expect(scene.steps.find(({ id }) => id === "joint-j2")).toMatchObject({
      target: '[data-tour="joint-j2"]', waitFor: "valueChange",
    });
    for (const tour of [platformFirstTimeTour, scene, tourRegistry.resolve("/roadmap")!]) {
      expect(tour.steps.every(({ target }) => /^\[data-tour="[a-z0-9-]+"\]$/.test(target))).toBe(true);
    }
  });
});
