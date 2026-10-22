import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "./i18n";
import { SimulationTaskConfigurator, isSceneParameter } from "./SimulationTaskConfigurator";
import { normalizeSimulationCatalog } from "./simulationCatalog";

const catalog = normalizeSimulationCatalog({ tasks: [
  { id: "s1_push", display_name: "S1 Push", task_type: "training", supports_training: true, parameters: [
    { name: "seed", type: "integer", default: 7 },
    { name: "object_size", type: "number_triplet", default: null, nullable: true },
  ] },
  { id: "s1_reach", display_name: "S1 Reach", task_type: "training", supports_training: true, parameters: [{ name: "seed", type: "integer", default: 7 }] },
  { id: "demo_s1_target_pose", display_name: "Target Pose", task_type: "demo", supports_training: false, parameters: [] },
] });

function render(client: QueryClient, initialTaskId = "s1_push") {
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <I18nProvider><SimulationTaskConfigurator initialTaskId={initialTaskId} /></I18nProvider>
    </QueryClientProvider>,
  );
}

describe("SimulationTaskConfigurator", () => {
  it("groups task controls separately from randomness and recording", () => {
    expect(isSceneParameter("action_delta_m")).toBe(true);
    expect(isSceneParameter("target_center_xy_m")).toBe(true);
    expect(isSceneParameter("seed")).toBe(false);
    expect(isSceneParameter("video_enabled")).toBe(false);
    expect(isSceneParameter("camera_position")).toBe(false);
  });
  it("lists only formal training tasks even when the server offers demos", () => {
    const client = new QueryClient();
    client.setQueryData(["simulation-task-catalog"], catalog);
    const markup = render(client);
    expect(markup).toContain("s1_push");
    expect(markup).toContain("s1_reach");
    expect(markup).not.toContain("object_size");
    expect(markup).not.toContain("demo_s1_target_pose");
    expect(markup).not.toContain("Target Pose");
  });

  it("does not select a demo supplied as an initial task", () => {
    const client = new QueryClient();
    client.setQueryData(["simulation-task-catalog"], catalog);
    expect(render(client, "demo_s1_target_pose")).not.toContain("demo_s1_target_pose");
  });

  it("shows a loading gate while the authenticated catalog is pending", () => {
    expect(render(new QueryClient())).toContain("正在读取 Simulation task catalog");
  });

});
