import type { SceneDocument } from "./api";
import { useI18n } from "./i18n";
import { PUSH_BASELINE, readPushGeometry, type PushGeometry } from "./pushScene";

export function FrozenSceneSummary({ snapshot }: { snapshot: Record<string, unknown> }) {
  const { t } = useI18n();
  if (snapshot.environment_type !== "s1_push") return null;
  const frozen = snapshot.scene as { document?: SceneDocument; scene_ref?: { scene_id: string; revision: number } } | undefined;
  const geometry = frozen?.document ? readPushGeometry(frozen.document) : PUSH_BASELINE;
  if (!geometry) return <p>{t("冻结场景无法解读", "Frozen scene could not be read")}</p>;
  const rows: Array<{ label: [string, string]; keys: Array<keyof PushGeometry> }> = [
    { label: ["方块起点 XY", "Block start XY"], keys: ["objectX", "objectY"] },
    { label: ["目标位置 XY", "Goal XY"], keys: ["goalX", "goalY"] },
    { label: ["尺寸 XYZ", "Dimensions XYZ"], keys: ["sizeX", "sizeY", "sizeZ"] },
  ];
  return <section className="frozen-scene-summary" data-tour="detail-scene"><h3>{t("本次训练的冻结场景", "Frozen scene used for this Run")}</h3><p>{frozen?.scene_ref ? `${frozen.scene_ref.scene_id} · REV ${frozen.scene_ref.revision}` : t("任务默认场景", "Task default scene")}</p><dl>{rows.map((row) => <div key={row.keys[0]}><dt>{t(...row.label)}</dt><dd>{row.keys.map((key) => geometry[key]).join(" / ")} m{row.keys.some((key) => geometry[key] !== PUSH_BASELINE[key]) && <small>{t("默认", "Default")} {row.keys.map((key) => PUSH_BASELINE[key]).join(" / ")} m</small>}</dd></div>)}</dl><small>{t("来自 Run 冻结快照；后续编辑不会改变本次训练。", "Read from the Run snapshot; later edits cannot change this training scene.")}</small></section>;
}
