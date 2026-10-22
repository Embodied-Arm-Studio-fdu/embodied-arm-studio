import type { SceneDocument } from "./api";

/** Display coordinates are physics XY; SceneDocument uses Y-up (X, height, Y). */
export type PushGeometry = {
  objectX: number; objectY: number; goalX: number; goalY: number;
  sizeX: number; sizeY: number; sizeZ: number;
};
export const PUSH_BASELINE: PushGeometry = {
  objectX: 0.45, objectY: 0, goalX: 0.60, goalY: 0.15,
  sizeX: 0.08, sizeY: 0.08, sizeZ: 0.08,
};
export const geometryKey = (geometry: PushGeometry) => JSON.stringify(Object.keys(PUSH_BASELINE).map((key) => geometry[key as keyof PushGeometry]));
export function validPushGeometry(g: PushGeometry): boolean {
  return Object.values(g).every(Number.isFinite)
    && [g.sizeX, g.sizeY, g.sizeZ].every((size) => size >= 0.06 && size <= 0.10)
    && g.objectX - g.sizeX / 2 >= 0.18 && g.objectX + g.sizeX / 2 <= 0.72
    && g.objectY - g.sizeY / 2 >= -0.32 && g.objectY + g.sizeY / 2 <= 0.32
    && g.goalX >= 0.18 && g.goalX <= 0.72 && Math.abs(g.goalY) <= 0.32;
}
export function pushSceneDocument(g: PushGeometry, sceneId: string): SceneDocument {
  return {
    schema_version: "1.0", scene_id: sceneId, revision: 0,
    name: "S1 Push experiment", capability: "simulation",
    robot: { id: "yhrg-s1-v2", dof: 6, control_mode: "joint_position_preview" },
    objects: [
      { object_id: "plane", type: "plane", label: "Table", position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1], color: "#c9d1c8", static: true },
      { object_id: "push-object", type: "box", label: "Push block", position: [g.objectX, g.sizeZ / 2, g.objectY], rotation: [0, 0, 0], scale: [g.sizeX, g.sizeZ, g.sizeY], color: "#287f78", static: false },
      { object_id: "target-01", type: "target", label: "Goal", position: [g.goalX, 0.01, g.goalY], rotation: [0, 0, 0], scale: [0.1, 0.02, 0.1], color: "#e79e30", static: true },
    ],
    observation: [], action: { type: "unavailable", shape: [], range: [] }, reward: [], termination: [],
    safety: { real_robot_control_enabled: false, sim2real: "blocked" },
  };
}
export function readPushGeometry(scene: SceneDocument): PushGeometry | null {
  const box = scene.objects?.find((item) => item.type === "box" && !item.static);
  const goal = scene.objects?.find((item) => item.type === "target");
  if (!box || !goal) return null;
  return { objectX: box.position[0], objectY: box.position[2], goalX: goal.position[0], goalY: goal.position[2], sizeX: box.scale[0], sizeY: box.scale[2], sizeZ: box.scale[1] };
}
