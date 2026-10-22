import { describe, expect, it } from "vitest";
import { PUSH_BASELINE, geometryKey, pushSceneDocument, readPushGeometry, validPushGeometry } from "./pushScene";

describe("Push geometry boundary", () => {
  it("matches the actual Worker baseline and maps physics XY into the Y-up document", () => {
    const g = { ...PUSH_BASELINE, objectY: -0.12, goalX: 0.62, sizeZ: 0.10 };
    const document = pushSceneDocument(g, "scene-test");
    expect(document.objects[1].position).toEqual([0.45, 0.05, -0.12]);
    expect(document.objects[1].scale).toEqual([0.08, 0.10, 0.08]);
    expect(document.objects[2].position).toEqual([0.62, 0.01, 0.15]);
    expect(readPushGeometry(document)).toEqual(g);
    expect(validPushGeometry(g)).toBe(true);
  });
  it("rejects nonfinite, excessive dimensions and block footprints outside the table", () => {
    for (const g of [
      { ...PUSH_BASELINE, objectX: 0.19 },
      { ...PUSH_BASELINE, objectY: 0.31 },
      { ...PUSH_BASELINE, goalY: 0.33 },
      { ...PUSH_BASELINE, sizeZ: 0.11 },
      { ...PUSH_BASELINE, sizeY: NaN },
    ]) expect(validPushGeometry(g)).toBe(false);
  });
  it("a draft edit cannot mutate a saved document or keep the same identity", () => {
    const g = { ...PUSH_BASELINE };
    const document = pushSceneDocument(g, "scene-saved");
    const savedKey = geometryKey(g);
    g.goalX = 0.64;
    expect(geometryKey(g)).not.toBe(savedKey);
    expect(readPushGeometry(document)).toEqual(PUSH_BASELINE);
  });
});
