import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { RotateCcw, Save, Check } from "lucide-react";
import { api, type SceneDocument } from "./api";
import { createBrowserId } from "./browserId";
import { useI18n } from "./i18n";
import { PUSH_BASELINE, geometryKey, pushSceneDocument, readPushGeometry, validPushGeometry, type PushGeometry } from "./pushScene";

export type PushSceneState = { ready: boolean; dirty: boolean; busy: boolean; valid?: boolean; geometry?: PushGeometry; sceneRef?: { scene_id: string; revision: number } };
export type PushSceneHandle = { saveAndValidate: () => Promise<void> };
type Gate = { trainable_push: boolean };
export const PushSceneControls = forwardRef<PushSceneHandle, {
  storageKey: string; disabled: boolean; onStateChange: (state: PushSceneState) => void;
}>(function PushSceneControls({ storageKey, disabled, onStateChange }, ref) {
  const { t } = useI18n();
  const [geometry, setGeometry] = useState<PushGeometry>({ ...PUSH_BASELINE });
  const [saved, setSaved] = useState<{ key: string; scene: SceneDocument } | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadFailed, setLoadFailed] = useState(false);
  const callback = useRef(onStateChange);
  callback.current = onStateChange;
  const key = geometryKey(geometry);
  const baseline = key === geometryKey(PUSH_BASELINE);
  const valid = validPushGeometry(geometry);
  const ready = !loading && !busy && valid && !error && (baseline || saved?.key === key);
  useEffect(() => {
    let current = true;
    async function load() {
      try {
        const stored = localStorage.getItem(storageKey);
        if (!stored) return;
        const ref = JSON.parse(stored) as { scene_id: string; revision: number };
        const document = await api<SceneDocument>(`/api/scenes/${encodeURIComponent(ref.scene_id)}`);
        const g = readPushGeometry(document);
        const gate = await api<Gate>(`/api/scenes/${encodeURIComponent(ref.scene_id)}/preflight`);
        if (document.revision !== ref.revision || !g || !validPushGeometry(g) || !gate.trainable_push) throw new Error("scene_revision_invalid");
        if (current) { setGeometry(g); setSaved({ key: geometryKey(g), scene: document }); }
      } catch {
        if (current) { setLoadFailed(true); setError(t("上次场景读取失败，请重试或明确恢复默认场景。", "Could not load the saved scene. Retry or explicitly restore the default scene.")); }
      } finally { if (current) setLoading(false); }
    }
    void load();
    return () => { current = false; };
  }, [storageKey]);
  useEffect(() => {
    callback.current({ ready, dirty: !baseline && saved?.key !== key, busy: busy || loading, valid, geometry,
      ...(ready && !baseline && saved ? { sceneRef: { scene_id: saved.scene.scene_id, revision: saved.scene.revision } } : {}) });
  }, [ready, baseline, key, saved, busy, loading]);
  useImperativeHandle(ref, () => ({ saveAndValidate: async () => {
    if (loadFailed || !valid || loading || busy) throw new Error(t("请在任务场景中修正参数或重新读取场景。", "Correct the scene parameters or reload the scene under Task scene."));
    if (!ready) await saveScene();
  } }));
  async function saveScene() {
    if (!valid || loading || busy) return;
    setBusy(true); setError("");
    try {
      // Every save has its own ID: later edits cannot overwrite a Run's referenced revision.
      const scene = await api<SceneDocument>("/api/scenes", { method: "POST", body: JSON.stringify({ document: pushSceneDocument(geometry, `scene-push-${createBrowserId()}`) }) });
      const gate = await api<Gate>(`/api/scenes/${encodeURIComponent(scene.scene_id)}/preflight`);
      if (!gate.trainable_push) throw new Error("scene_preflight_failed");
      localStorage.setItem(storageKey, JSON.stringify({ scene_id: scene.scene_id, revision: scene.revision }));
      setSaved({ key, scene });
    } catch {
      const message = t("场景保存或校验失败，请重试；当前修改不会用于训练。", "Scene save or validation failed. Retry; these edits cannot be used for training yet.");
      setError(message);
      throw new Error(message);
    } finally { setBusy(false); }
  }
  const x = (value: number) => 98 + (value - 0.18) * 600;
  const y = (value: number) => 434 - (value + 0.32) * 600;
  const fields: Array<{ title: [string, string]; lesson: [string, string]; keys: Array<keyof PushGeometry>; labels: string[] }> = [
    { title: ["方块初始位置", "Block starting position"], lesson: ["移动起点，改变需要推动的方向与距离。", "Move the starting point to change the direction and distance of the push."], keys: ["objectX", "objectY"], labels: ["X", "Y"] },
    { title: ["目标位置", "Goal position"], lesson: ["改变目标中心；成功距离固定为 5 厘米。", "Move the goal center; the success tolerance stays at 5 cm."], keys: ["goalX", "goalY"], labels: ["X", "Y"] },
    { title: ["方块尺寸", "Block dimensions"], lesson: ["各边 6–10 厘米，实际改变碰撞几何与质量；高度变化也会影响接触。", "Each edge is 6–10 cm. This changes collision geometry and mass; height also affects contact."], keys: ["sizeX", "sizeY", "sizeZ"], labels: ["X", "Y", "Z"] },
  ];
  return <section className="push-scene-controls" data-tour="push-scene-controls">
    <div className="push-scene-summary"><span className={ready ? "scene-ready" : "scene-pending"}>{ready ? <Check size={15} /> : <Save size={15} />}{loading ? t("读取场景中…", "Loading scene…") : ready ? baseline ? t("默认场景 · 可训练", "Default scene · ready") : t("已保存并校验", "Saved and validated") : t("修改未保存或未通过校验", "Changes need saving or validation")}</span><div className="push-scene-tools"><button className="primary" disabled={disabled || busy || loading || !valid || ready} onClick={() => void saveScene().catch(() => {})}><Save size={15} />{busy ? t("保存与校验中…", "Saving and validating…") : t("保存场景并校验", "Save and validate scene")}</button><button className="secondary" disabled={disabled || busy || loading} onClick={() => { setGeometry({ ...PUSH_BASELINE }); setError(""); setLoadFailed(false); localStorage.removeItem(storageKey); setSaved(null); }}><RotateCcw size={14} />{t("恢复默认场景", "Restore default scene")}</button></div></div>
    <div className="push-scene-layout"><div className="push-scene-visual">
    <div className="push-scene-preview"><svg viewBox="0 0 520 485" role="img" aria-label={t("Push 桌面俯视示意，随参数即时更新", "Push tabletop diagram, updated instantly from parameters")}>
      <defs><pattern id="push-scene-grid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M 30 0 L 0 0 0 30" fill="none" stroke="#d9e4dc" strokeWidth="1" /></pattern></defs>
      <rect x="98" y="50" width="324" height="384" rx="8" fill="#eff5ef" stroke="#aabfb0" /><rect x="98" y="50" width="324" height="384" fill="url(#push-scene-grid)" />
      {Object.values(geometry).every(Number.isFinite) && <><line x1={x(geometry.objectX)} y1={y(geometry.objectY)} x2={x(geometry.goalX)} y2={y(geometry.goalY)} stroke="#879d90" strokeDasharray="6 5" /><ellipse cx={x(geometry.goalX)} cy={y(geometry.goalY)} rx={0.05 * 600} ry={0.05 * 600} fill="#f4bf4930" stroke="#be8219" strokeWidth="2" /><text x={x(geometry.goalX)} y={y(geometry.goalY) - 29} textAnchor="middle" fill="#785711">{t("目标", "Goal")}</text><rect x={x(geometry.objectX) - geometry.sizeX * 300} y={y(geometry.objectY) - geometry.sizeY * 300} width={Math.max(0, geometry.sizeX * 600)} height={Math.max(0, geometry.sizeY * 600)} rx="3" fill="#287f78" /><text x={x(geometry.objectX)} y={y(geometry.objectY) + 5} textAnchor="middle" fill="white">{t("方块", "Block")}</text></>}
      <text x="98" y="30" fill="#435e50">Y ↑</text><text x="390" y="460" fill="#435e50">X →</text><text x="98" y="460" fill="#65766a">{t("桌面工作区 · 单位：米", "Table workspace · meters")}</text>
    </svg><p>{t("俯视示意用于确认位置与尺寸；不代表碰撞、可达性或学习效果已验证。", "This diagram shows positions and dimensions; collision safety, reachability and learning outcomes require execution evidence.")}</p></div>
    <p className="push-scene-distance">{t("起点到目标", "Start to goal")} <strong>{(Math.hypot(geometry.goalX - geometry.objectX, geometry.goalY - geometry.objectY) * 100).toFixed(1)} {t("厘米", "cm")}</strong> · {t("建议先只改变目标 X：0.60 → 0.62 米，保持奖励、种子和预算一致。", "Try changing only goal X: 0.60 → 0.62 m, keeping rewards, seed and budget unchanged.")}</p>
    {valid && Math.hypot(geometry.goalX - geometry.objectX, geometry.goalY - geometry.objectY) <= 0.05 && <p className="workbench-fixed-note">{t("起点已经在成功距离内，可能无需推动；这类场景不适合比较推物学习效果。", "The block starts within the success tolerance and may need no push. This scene is unsuitable for comparing push learning.")}</p>}
    </div><div className="push-scene-fields">{fields.map((group) => <fieldset key={group.keys[0]} disabled={disabled || busy || loading}><legend>{t(...group.title)}</legend><p>{t(...group.lesson)}</p><div>{group.keys.map((name, i) => <label key={name} className={geometry[name] !== PUSH_BASELINE[name] ? "scene-field-changed" : ""}>{group.labels[i]} {name.startsWith("size") ? t("（厘米）", "(cm)") : t("（米）", "(m)")}<input type="number" aria-label={`${t(...group.title)} ${group.labels[i]}`} min={name.startsWith("size") ? 6 : name.endsWith("Y") ? -0.32 : 0.18} max={name.startsWith("size") ? 10 : name.endsWith("Y") ? 0.32 : 0.72} step={name.startsWith("size") ? 1 : 0.01} value={Number.isFinite(geometry[name]) ? name.startsWith("size") ? Number((geometry[name] * 100).toFixed(4)) : geometry[name] : ""} onChange={(event) => { setError(""); setGeometry((current) => ({ ...current, [name]: event.target.value === "" ? NaN : Number(event.target.value) / (name.startsWith("size") ? 100 : 1) })); }} /><small>{t("默认", "Default")} {name.startsWith("size") ? PUSH_BASELINE[name] * 100 : PUSH_BASELINE[name]} {name.startsWith("size") ? "cm" : "m"}</small></label>)}</div></fieldset>)}</div>
    </div>
    {!valid && <p className="workbench-error" role="alert">{t("尺寸须为 0.06–0.10 米；方块整体须在 X 0.18–0.72、Y −0.32–0.32 米内，目标中心也须在此范围。", "Dimensions must be 0.06–0.10 m. The entire block and the goal center must lie within X 0.18–0.72 and Y −0.32–0.32 m.")}</p>}
    {error && <p className="workbench-error" role="alert">{error}<button className="secondary" onClick={() => window.location.reload()}>{t("重新读取", "Reload")}</button></p>}
    <div className="push-scene-save"><small>{t("训练会使用已保存的同一场景版本；修改后需再次保存。", "Training uses this saved scene revision. Save again after editing.")}</small></div>
  </section>;
});
