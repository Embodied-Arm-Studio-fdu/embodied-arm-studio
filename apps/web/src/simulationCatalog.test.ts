import { describe, expect, it } from "vitest";

import {
  clampTaskParameters,
  defaultSimulationCatalog,
  fetchSimulationCatalog,
  getTaskDefaults,
  groupSimulationTasks,
  normalizeSimulationCatalog,
  resolveTaskParameters,
  serializeTaskParameters,
  taskParameterPayload,
  validateTaskParameters,
} from "./simulationCatalog";

describe("simulation task catalog", () => {
  it("contains the six training tasks and six demo tasks", () => {
    const catalog = defaultSimulationCatalog();
    const groups = groupSimulationTasks(catalog);

    expect(catalog.source).toBe("fallback");
    expect(catalog.tasks).toHaveLength(12);
    expect(groups.training.map((task) => task.id)).toEqual([
      "s1_push", "s1_reach", "s1_grasp", "s1_lift", "s1_lower_release", "s1_place",
    ]);
    expect(groups.demo.map((task) => task.id)).toEqual([
      "demo_s1_target_pose",
      "demo_s1_line_tracking",
      "demo_s1_circle_tracking",
      "demo_s1_joint_position",
      "demo_s1_joint_sweep",
      "demo_s1_scripted_pick_place",
    ]);
  });

  it("marks formal training tasks and checkpoint-producing tasks", () => {
    const catalog = defaultSimulationCatalog();

    expect(catalog.tasks.filter((task) => task.supportsFormalRun).map((task) => task.id)).toEqual([
      "s1_push", "s1_grasp", "s1_lift", "s1_lower_release", "s1_place",
    ]);
    expect(catalog.tasks.every((task) => task.mode === "demo" ? !task.producesCheckpoint : true)).toBe(true);
    expect(catalog.tasks.filter((task) => task.mode === "demo").every((task) => !task.producesCheckpoint)).toBe(true);
  });

  it("provides array defaults and validates array item bounds", () => {
    const task = defaultSimulationCatalog().tasks.find((candidate) => candidate.id === "demo_s1_target_pose");
    if (!task) {
      throw new Error("target pose task missing from fallback catalog");
    }

    expect(getTaskDefaults(task).target_offset_xy_m).toEqual([0.08, 0.05]);
    expect(validateTaskParameters(task, { target_offset_xy_m: [0.2, 0.05] }).valid).toBe(false);
    expect(validateTaskParameters(task, { target_offset_xy_m: [0.1, -0.1] }).valid).toBe(true);
  });

  it("clamps scalar and array numeric values without changing unknown fields", () => {
    const task = defaultSimulationCatalog().tasks.find((candidate) => candidate.id === "demo_s1_circle_tracking");
    if (!task) {
      throw new Error("circle task missing from fallback catalog");
    }

    expect(clampTaskParameters(task, { radius_m: 10, ignored: true })).toMatchObject({ radius_m: 0.08 });
  });

  it("serializes allowed array parameters as JSON arrays", () => {
    const task = defaultSimulationCatalog().tasks.find((candidate) => candidate.id === "demo_s1_target_pose");
    if (!task) {
      throw new Error("target pose task missing from fallback catalog");
    }

    const serialized = serializeTaskParameters(task, { target_offset_xy_m: [0.1, 0.1] });
    expect(serialized).toContain('"target_offset_xy_m":[0.1,0.1]');
    expect(serialized).not.toContain("ignored");
  });

  it("normalizes server task catalogs and preserves server source", () => {
    const catalog = normalizeSimulationCatalog({
      schema_version: "1.0",
      tasks: [
        {
          task_id: "server_task",
          task_type: "demo",
          parameters: [
            { name: "values", type: "array", default: [1, 2], items: { type: "integer" } },
          ],
        },
      ],
    });

    expect(catalog.source).toBe("server");
    expect(catalog.tasks[0]).toMatchObject({ id: "server_task", mode: "demo" });
    expect(catalog.tasks[0].parameters[0].items).toEqual({ type: "integer" });
  });

  it("normalizes the Simulation Service enum and number_pair schema", () => {
    const catalog = normalizeSimulationCatalog({
      tasks: [
        {
          id: "demo_s1_joint_sweep",
          display_name: "S1 Joint Sweep",
          task_type: "demo",
          supports_training: false,
          supports_video: true,
          parameters: [
            { name: "joint_index", type: "enum", default: "all_arm", enum: ["all_arm", 0, 1] },
            { name: "target_xy", type: "number_pair", default: [0.36, 0], minimum: -0.5, maximum: 0.5 },
          ],
        },
      ],
    });

    expect(catalog.tasks[0].title).toBe("S1 Joint Sweep");
    expect(catalog.tasks[0].parameters[0]).toMatchObject({ type: "string", enum: ["all_arm", 0, 1] });
    expect(catalog.tasks[0].parameters[1]).toMatchObject({ type: "array", items: { type: "number" } });
    expect(validateTaskParameters(catalog.tasks[0], { joint_index: 0, target_xy: [0.1, -0.1] }).valid).toBe(true);
  });

  it("accepts backend-resolved number_triplet parameters without exposing or submitting them", () => {
    const catalog = normalizeSimulationCatalog({
      tasks: [{
        id: "s1_push",
        display_name: "S1 Push",
        task_type: "training",
        supports_training: true,
        parameters: [
          { name: "seed", type: "integer", default: 7 },
          { name: "object_size", type: "number_triplet", default: null, nullable: true },
        ],
      }],
    });
    const task = catalog.tasks[0];

    expect(task.parameters[1]).toMatchObject({
      type: "array",
      items: { type: "number" },
      fixedLength: 3,
    });
    expect(getTaskDefaults(task)).toEqual({ seed: 7 });
    expect(taskParameterPayload(task, { seed: 11 })).toEqual({ seed: 11 });
  });

  it("falls back to catalog defaults when a saved draft is stale or invalid", () => {
    const task = defaultSimulationCatalog().tasks.find((candidate) => candidate.id === "s1_lift");
    if (!task) throw new Error("lift task missing from fallback catalog");

    expect(resolveTaskParameters(task, { seed: -1, removed_parameter: true })).toEqual(
      getTaskDefaults(task),
    );
    expect(resolveTaskParameters(task, { seed: 19 })).toMatchObject({ seed: 19 });
  });

  it("uses fallback only for an explicitly unsupported catalog route", async () => {
    const fetcher = async () => new Response("not found", { status: 404 });
    const catalog = await fetchSimulationCatalog(fetcher);

    expect(catalog.source).toBe("fallback");
    expect(catalog.tasks).toHaveLength(12);
  });
});
