import { describe, expect, it } from "vitest";
import { buildAgentPageContext } from "./agentContext";

describe("Agent page context", () => {
  it("prefers the canonical Run detail route over an older active Run", () => {
    expect(buildAgentPageContext(
      "/runs/run%2Fproduction-01",
      "workspace-01",
      "legacy-run",
      "scene-s1-v2-visual",
    )).toEqual({
      path: "/runs/run%2Fproduction-01",
      project_id: "workspace-01",
      run_id: "run/production-01",
      scene_id: "scene-s1-v2-visual",
    });
  });

  it("omits identifiers that are not known", () => {
    expect(buildAgentPageContext("/roadmap", "", undefined, "")).toEqual({
      path: "/roadmap",
    });
  });

  it("carries the selected canonical evaluation with the Run context", () => {
    expect(buildAgentPageContext(
      "/evaluation",
      "workspace-01",
      "run-01",
      "scene-s1-v2-visual",
      "evaluation-01",
    )).toEqual({
      path: "/evaluation",
      project_id: "workspace-01",
      run_id: "run-01",
      scene_id: "scene-s1-v2-visual",
      evaluation_id: "evaluation-01",
    });
  });
});