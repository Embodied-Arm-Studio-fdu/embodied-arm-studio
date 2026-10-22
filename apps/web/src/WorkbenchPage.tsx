import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { WorkbenchEditor } from "./WorkbenchEditor";
import { PushSceneControls, type PushSceneState } from "./PushSceneControls";
import { FrozenSceneSummary } from "./FrozenSceneSummary";
import { PUSH_BASELINE, pushSceneDocument } from "./pushScene";
import { RewardDesigner } from "./RewardDesigner";
import { Bot, Check, CircleAlert, FileCode2, History, Play, RotateCcw, ShieldCheck, Settings2, GitCompare, Save, Send, X } from "lucide-react";
import { api, type CurrentUserContext, type TeachingClass, type TeachingCourse, type TeachingProject, type RunRecord, type AgentProviderInfo } from "./api";
import { createBrowserId } from "./browserId";
import { createCanonicalRunSubmission } from "./canonicalRunRequest";
import { translateUiText, useI18n } from "./i18n";
import { SimulationTaskConfigurator, type SimulationTaskConfigurationState } from "./SimulationTaskConfigurator";
import { taskParameterPayload } from "./simulationCatalog";
import { normalizeRunEventsResponse } from "./runEvents";
import { runStatusLabel } from "./runPresentation";

type TaskId = "s1_push" | "s1_reach" | "s1_place";

type DraftResponse = {
  task_id: TaskId;
  starter: string;
  source: string;
  revision_id: string;
  editable_terms: string[];
  protected_terms: string[];
  history: Array<{ revision_id: string; sha256: string; created_at: string }>;
};

type PreflightResponse = {
  valid: boolean;
  sha256?: string;
  extension?: { schema: string; source: string; sha256: string };
  errors: Array<{ code: string; message: string }>;
  warnings: string[];
  samples: Array<{ input: Record<string, number>; output: number }>;
};

type DiagnosisResponse = {
  content: string;
  provider: string;
  evidence: { run?: { status: string; error?: unknown; metrics?: Record<string, unknown> } };
  suggested_source?: string | null;
};

export function WorkbenchPage({ user, navigate }: { user: CurrentUserContext; navigate: (path: string) => void }) {
  const { locale, t } = useI18n();
  const TASKS: Array<{ id: TaskId; title: string; goal: string; observation: string; action: string }> = [
    { id: "s1_push", title: t("S1 Push · 推物", "S1 Push · Object pushing"), goal: t("推动桌面方块进入目标区域", "Push the block into the target zone"), observation: t("24 维观测", "24 observations"), action: t("二维 XY 位移", "2D XY motion") },
    { id: "s1_reach", title: t("S1 Reach · 平面到达", "S1 Reach · Planar reaching"), goal: t("末端到达目标区域并保持", "Reach and hold the end effector in the target zone"), observation: t("18 维观测", "18 observations"), action: t("二维 XY 位移", "2D XY motion") },
    { id: "s1_place", title: t("S1 Place · 搬运放置", "S1 Place · Transport and hold"), goal: t("从已抓取抬起状态搬运到目标并稳定保持一秒", "Transport from a validated lifted grasp and hold at the target for one second"), observation: t("固定观测契约", "Fixed observation contract"), action: t("XY 搬运，Z 与夹爪锁定", "XY transport; Z and gripper locked") },
  ];
  const uiText = (value: string) => locale === "en" ? translateUiText(value) : value;
  const [reloadDraft, setReloadDraft] = useState(0);
  const [fileFilter, setFileFilter] = useState("");
  const [showAllSources, setShowAllSources] = useState(false);
  const [draftLoading, setDraftLoading] = useState(true);
  const [taskId, setTaskId] = useState<TaskId>("s1_push");
  const [organizationId, setOrganizationId] = useState(user.organizations[0]?.organization_id ?? "");
  const [courseId, setCourseId] = useState(() => sessionStorage.getItem("s1-draft-course") ?? "");
  const [classId, setClassId] = useState(() => sessionStorage.getItem("s1-draft-class") ?? "");
  const [projectId, setProjectId] = useState(() => sessionStorage.getItem("s1-draft-project") ?? "");
  const [source, setSource] = useState("");
  const [baseRevision, setBaseRevision] = useState("starter");
  const [draft, setDraft] = useState<DraftResponse | null>(null);
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResponse | null>(null);
  const [busy, setBusy] = useState<"save" | "check" | "run" | "diagnose" | "restore" | "">("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [activeFile, setActiveFile] = useState("reward.py");
  const [selection, setSelection] = useState("");
  const [question, setQuestion] = useState("");
  const [provider, setProvider] = useState("mock");
  const [configuration, setConfiguration] = useState<SimulationTaskConfigurationState | null>(null);
  const [workspaceSection, setWorkspaceSection] = useState<"scene" | "reward" | "settings" | "source">("reward");
  const [pushScene, setPushScene] = useState<PushSceneState>({ ready: false, dirty: false, busy: true });
  const [advancedCode, setAdvancedCode] = useState(false);
  const [showDiff, setShowDiff] = useState(false);
  const [runId, setRunId] = useState("");
  const [baselineRunId, setBaselineRunId] = useState("");
  const [bottomTab, setBottomTab] = useState<"preflight" | "logs" | "metrics">("preflight");
  const [proposalOpen, setProposalOpen] = useState(false);
  const [readOnlySelection, setReadOnlySelection] = useState(false);
  const codePreview = useRef<HTMLDivElement | null>(null);
  const runSubmission = useRef(createCanonicalRunSubmission());
  const baselineSceneRef = useRef<{ scene_id: string; revision: number }>();
  const courses = useQuery({
    queryKey: ["workbench-courses", organizationId],
    queryFn: () => api<TeachingCourse[]>(`/api/v1/courses?organization_id=${encodeURIComponent(organizationId)}`),
    enabled: Boolean(organizationId),
  });
  const classes = useQuery({
    queryKey: ["workbench-classes", courseId],
    queryFn: () => api<TeachingClass[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/classes`),
    enabled: Boolean(courseId),
  });
  const projects = useQuery({
    queryKey: ["workbench-projects", courseId],
    queryFn: () => api<TeachingProject[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/projects`),
    enabled: Boolean(courseId),
  });
  const providers = useQuery({ queryKey: ["agent-providers"], queryFn: () => api<AgentProviderInfo[]>("/api/agent/providers") });
  const runs = useQuery({ queryKey: ["workbench-runs"], queryFn: () => api<RunRecord[]>("/api/v1/runs"), refetchInterval: runId ? 4000 : false });
  const runQuery = useQuery({ queryKey: ["workbench-run", runId], queryFn: () => api<RunRecord>(`/api/v1/runs/${encodeURIComponent(runId)}`), enabled: Boolean(runId), refetchInterval: (query) => ["completed", "failed", "stopped"].includes(query.state.data?.status ?? "") ? false : 2000 });
  const events = useQuery({ queryKey: ["workbench-events", runId], queryFn: async () => normalizeRunEventsResponse(await api<unknown>(`/api/v1/runs/${encodeURIComponent(runId)}/events?limit=100`)), enabled: Boolean(runId), refetchInterval: runId && !["completed", "failed", "stopped"].includes(runQuery.data?.status ?? "") ? 2000 : false });
  const baseline = (runs.data ?? []).find((item) => item.run_id === baselineRunId);
  const sources = useQuery({
    queryKey: ["workbench-sources"],
    queryFn: () => api<{ files: string[] }>("/api/v1/workbench/sources"),
  });
  const selectedProject = (projects.data ?? []).find((item) => item.project_id === projectId && item.enabled && item.permissions.can_use && item.environment_type === taskId);
  const selectedClass = (classes.data ?? []).find((item) => item.class_id === classId && item.enabled);
  const task = TASKS.find((item) => item.id === taskId) ?? TASKS[0];
  const referenceFiles = [
    { path: taskId === "s1_place" ? "experimental/manipulation/place_env.py" : taskId === "s1_reach" ? "reach_adapter.py" : "push_adapter.py", description: t("当前任务 · 观测、动作与奖励", "Current task · observations, actions and rewards") },
    ...(taskId === "s1_place" ? [{ path: "experimental/manipulation/place_schema.py", description: t("奖励分项计算", "Reward term calculations") }] : []),
    { path: "training_task_runner.py", description: t("训练执行 · 从配置到环境", "Training execution · config to environment") },
    { path: "ppo/trainer.py", description: t("PPO 训练流程", "PPO training loop") },
  ];
  const completeSourceList = sources.data?.files ?? [];
  const browsingAll = showAllSources || Boolean(fileFilter.trim());
  const visibleSourceFiles = (browsingAll ? completeSourceList : referenceFiles.map((item) => item.path).filter((path) => completeSourceList.includes(path)))
    .filter((path) => path.toLowerCase().includes(fileFilter.trim().toLowerCase()));
  const permitted = user.organizations.find((item) => item.organization_id === organizationId)?.permissions.includes("own_run:create");
  const canSubmit = Boolean(selectedProject && selectedClass && permitted && preflight?.valid && preflight.extension?.source === source && draft?.source === source && configuration?.task?.id === taskId && configuration.catalog?.source === "server" && configuration.validation?.valid && configuration.task.supportsFormalRun && (taskId !== "s1_push" || pushScene.ready) && !busy);

  useEffect(() => {
    if (!courseId && courses.data?.[0]) setCourseId(courses.data[0].course_id);
  }, [courseId, courses.data]);

  useEffect(() => {
    let current = true;
    setDraftLoading(true);
    setShowDiff(false);
    setPreflight(null);
    setDraft(null);
    setSource("");
    setBaseRevision("starter");
    setDiagnosis(null);
    setActiveFile("reward.py");
    setWorkspaceSection("reward");
    setAdvancedCode(false);
    setSelection("");
    setRunId("");
    setBaselineRunId("");
    setProposalOpen(false);
    setError("");
    setReadOnlySelection(false);
    void api<DraftResponse>(`/api/v1/workbench/drafts/${taskId}`).then((value) => {
      if (!current) return;
      setDraft(value);
      setSource(value.source);
      setBaseRevision(value.revision_id);
    }).catch((caught) => { if (current) setError(caught instanceof Error ? caught.message : t("读取奖励草稿失败", "Failed to load reward draft")); }).finally(() => { if (current) setDraftLoading(false); });
    return () => { current = false; };
  }, [taskId, reloadDraft]);

  async function runPreflight() {
    setBusy("check");
    setError("");
    try {
      const result = await api<PreflightResponse>(`/api/v1/workbench/drafts/${taskId}/preflight`, { method: "POST", body: JSON.stringify({ source }) });
      setPreflight(result);
      setMessage(result.valid ? t("预检通过：可以创建训练版本。", "Preflight passed: a training revision can be created.") : t("预检未通过：请修正奖励函数。", "Preflight failed: fix the reward function."));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("预检失败", "Preflight failed"));
    } finally { setBusy(""); }
  }

  async function saveDraft() {
    setBusy("save");
    setError("");
    try {
      const value = await api<DraftResponse>(`/api/v1/workbench/drafts/${taskId}`, { method: "PUT", body: JSON.stringify({ source, base_revision: baseRevision }) });
      setDraft(value);
      setBaseRevision(value.revision_id);
      setPreflight(null);
      setMessage(t(`已保存 ${value.revision_id}，请重新运行预检。`, `Saved ${value.revision_id}; run preflight again.`));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("保存失败", "Save failed"));
    } finally { setBusy(""); }
  }

  async function restoreRevision(revisionId: string) {
    if (!window.confirm(t("回档将保存为新版本，并保留已有历史。继续？", "Restore as a new revision while retaining history?"))) return;
    setBusy("restore");
    setError("");
    try {
      const value = await api<DraftResponse>(`/api/v1/workbench/drafts/${taskId}/restore`, { method: "POST", body: JSON.stringify({ revision_id: revisionId, base_revision: baseRevision }) });
      setDraft(value);
      setSource(value.source);
      setBaseRevision(value.revision_id);
      setPreflight(null);
      setMessage(t("已创建回档后的新版本。", "A new revision was created from the restored version."));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("回档失败", "Restore failed"));
    } finally { setBusy(""); }
  }

  async function askAgent(lessonQuestion?: string) {
    setBusy("diagnose");
    setError("");
    try {
      const result = await api<DiagnosisResponse>(`/api/v1/workbench/drafts/${taskId}/diagnose`, { method: "POST", body: JSON.stringify({ source, selection, path: readOnlySelection ? activeFile : "reward.py", question: lessonQuestion || question || t("请解释选区和当前奖励修改。若有 Run，请分析实际错误或低成功率的可能原因，区分已证实和假设。", "Explain the selected code and reward changes. If a Run is selected, analyze its recorded errors or low success rate, separating evidence from hypotheses."), locale, provider, ...(runId ? { run_id: runId } : {}) }) });
      setDiagnosis(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("Agent 诊断失败", "Agent diagnosis failed"));
    } finally { setBusy(""); }
  }

  async function startRun() {
    if (!canSubmit || !selectedProject || !selectedClass || !preflight?.extension || !configuration?.task) return;
    if (!window.confirm(t("将冻结当前奖励、场景、参数和 Project 版本并排队运行训练。继续？", "Freeze this reward, scene, parameters and Project revision and queue training?"))) return;
    setBusy("run");
    setError("");
    try {
      let sceneRef = taskId === "s1_push" ? pushScene.sceneRef ?? baselineSceneRef.current : undefined;
      if (taskId === "s1_push" && !sceneRef) {
        // Freeze the baseline too, rather than relying on a future Worker's defaults.
        const saved = await api<{ scene_id: string; revision: number }>("/api/scenes", { method: "POST", body: JSON.stringify({ document: pushSceneDocument(PUSH_BASELINE, `scene-push-${createBrowserId()}`) }) });
        const gate = await api<{ trainable_push: boolean }>(`/api/scenes/${encodeURIComponent(saved.scene_id)}/preflight`);
        if (!gate.trainable_push) throw new Error(t("默认场景校验失败，请重试。", "Default scene validation failed; retry."));
        sceneRef = { scene_id: saved.scene_id, revision: saved.revision };
        baselineSceneRef.current = sceneRef;
      }
      const run = await runSubmission.current.submit({
        project_id: selectedProject.project_id,
        class_id: selectedClass.class_id,
        project_config_version: selectedProject.config_version,
        run_kind: "train",
        environment_type: taskId,
        parameters: taskParameterPayload(configuration.task, configuration.parameters),
        ...(sceneRef ? { scene_ref: sceneRef } : {}),
        code: source,
        reward_extension: preflight.extension,
      });
      setRunId(run.run_id);
      setBottomTab("logs");
      setMessage(t("已创建正式 Run，正在等待 Worker。", "Formal Run created; waiting for the Worker."));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("启动训练失败", "Training launch failed"));
    } finally { setBusy(""); }
  }

  const readOnlySource = useQuery({
    queryKey: ["workbench-source", activeFile],
    queryFn: () => api<{ path: string; content: string; sha256: string }>(`/api/v1/workbench/sources/${activeFile.split("/").map(encodeURIComponent).join("/")}`),
    enabled: activeFile !== "reward.py",
  });

  function chooseRun(id: string) {
    setRunId(id);
    setBottomTab("logs");
    setDiagnosis(null);
  }

  async function applyProposal() {
    if (!diagnosis?.suggested_source || !draft) return;
    setBusy("save");
    try {
      const value = await api<DraftResponse>(`/api/v1/workbench/drafts/${taskId}`, { method: "PUT", body: JSON.stringify({ source: diagnosis.suggested_source, base_revision: baseRevision }) });
      setDraft(value);
      setBaseRevision(value.revision_id);
      setSource(value.source);
      setPreflight(null);
      setProposalOpen(false);
      setShowDiff(false);
      setMessage(t("提案已审批并保存为新版本，请重新预检。", "Proposal approved and saved as a new revision; run preflight again."));
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("提案保存失败", "Failed to save proposal")); }
    finally { setBusy(""); }
  }

  function openSection(section: "scene" | "reward" | "settings") {
    setWorkspaceSection(section);
    setActiveFile("reward.py");
    setShowDiff(false);
    setProposalOpen(false);
    setSelection("");
    setReadOnlySelection(false);
  }

  function openSource(path: string) {
    setWorkspaceSection("source");
    setActiveFile(path);
    setShowDiff(false);
    setSelection("");
    setReadOnlySelection(true);
  }

  function changeReward(value: string) {
    setSource(value);
    setPreflight(null);
    setSelection("");
    setDiagnosis(null);
    setMessage("");
    setError("");
  }

  return <main className="workbench-page">
    <header className="workbench-header">
      <div><span className="eyebrow">{t("S1 / 实验工作台", "S1 / EXPERIMENT WORKBENCH")}</span><h1>{t("实验工作台", "Experiment workbench")}</h1><p>{task.goal} · {task.observation} · {task.action}</p></div>
      <div className="workbench-boundary"><ShieldCheck size={18} /><span><strong>{t("仅限仿真", "SIMULATION ONLY")}</strong><small>{t("PPO 核心、动作/观测契约和真实控制已锁定", "PPO core, action/observation contracts and real control are locked")}</small></span></div>
    </header>
    <div className="workbench-context" data-tour="workbench-context">
      <label>{t("组织", "Organization")}<select value={organizationId} onChange={(event) => { setOrganizationId(event.target.value); setCourseId(""); setClassId(""); setProjectId(""); }}><option value="">{t("选择组织", "Select organization")}</option>{user.organizations.map((item) => <option key={item.organization_id} value={item.organization_id}>{uiText(item.name)}</option>)}</select></label>
      <label>{t("课程", "Course")}<select value={courseId} disabled={!organizationId} onChange={(event) => { setCourseId(event.target.value); setClassId(""); setProjectId(""); }}><option value="">{t("选择课程", "Select course")}</option>{(courses.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.course_id} value={item.course_id}>{item.code} · {uiText(item.title)}</option>)}</select></label>
      <label>{t("教学班", "Class")}<select value={classId} disabled={!courseId} onChange={(event) => setClassId(event.target.value)}><option value="">{t("选择教学班", "Select class")}</option>{(classes.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.class_id} value={item.class_id}>{uiText(item.name)}</option>)}</select></label>
      <label>{t("训练 Project", "Training Project")}<select value={projectId} disabled={!courseId} onChange={(event) => setProjectId(event.target.value)}><option value="">{t("选择匹配 Project", "Select matching Project")}</option>{(projects.data ?? []).filter((item) => item.enabled && item.permissions.can_use && item.task_type === "training" && item.environment_type === taskId).map((item) => <option key={item.project_id} value={item.project_id}>{uiText(item.title)} · V{item.config_version}</option>)}</select></label>
    </div>
    <nav data-tour="workbench-tasks" className="workbench-task-tabs" aria-label={t("训练任务", "Training tasks")}>{TASKS.map((item) => <button key={item.id} disabled={Boolean(busy)} className={item.id === taskId ? "active" : ""} onClick={() => { if ((draft && source !== draft.source || taskId === "s1_push" && pushScene.dirty) && !window.confirm(t("当前奖励或场景修改尚未保存，切换任务？", "Switch tasks with unsaved changes?"))) return; setTaskId(item.id); }}><strong>{item.title}</strong><small>{item.goal}</small></button>)}</nav>
    <div className="workbench-commandbar">
      <span className="workbench-learning-path">{t("改一项 → 保存 → 预检 → 训练 → 对比", "Change one thing → Save → Preflight → Train → Compare")}</span>
      <button className="secondary" onClick={() => { openSection("reward"); setShowDiff(!showDiff); }} aria-pressed={showDiff}><GitCompare size={14} /> {t("对比 Starter", "Compare Starter")}</button>
      <span>{source === draft?.source ? t("版本已保存", "Revision saved") : t("修改未保存", "Unsaved changes")}</span>
      <label>{t("关联 Run", "Run")}<select value={runId} onChange={(event) => chooseRun(event.target.value)}><option value="">{t("选择运行", "Select Run")}</option>{(runs.data ?? []).filter((item) => item.project_config_snapshot.environment_type === taskId).map((item) => <option key={item.run_id} value={item.run_id}>{item.run_id.slice(-8)} · {runStatusLabel(item.status, locale)}</option>)}</select></label>
      {runId && <button className="secondary" onClick={() => navigate(`/runs/${encodeURIComponent(runId)}`)}>{t("运行详情", "Run details")}</button>}
    </div>
    {!draft && <div className="workbench-load-state" role={draftLoading ? "status" : "alert"}>
      <CircleAlert size={18} /><div><strong>{draftLoading ? t("正在加载奖励代码…", "Loading reward code…") : t("奖励代码未加载", "Reward code is unavailable")}</strong><p>{draftLoading ? t("正在读取你的任务版本", "Reading your task revision") : t("请确认 API 已更新并提供 /api/v1/workbench 路由；源码服务恢复后才能编辑与训练。", "Check that the API is up to date and exposes /api/v1/workbench routes. Editing and training require the source service.")}</p>{error && <small>{uiText(error)}</small>}</div><button className="secondary" disabled={draftLoading} onClick={() => { setReloadDraft((value) => value + 1); void sources.refetch(); }}>{t("重试加载", "Retry loading")}</button>
    </div>}
    <section className="workbench-grid">
      <aside data-tour="workbench-files" className="workbench-files">
        <div className="workbench-pane-title"><span><Settings2 size={14} /> {t("实验修改入口", "Experiment controls")}</span><small>{t("一次改一项", "One change at a time")}</small></div>
        <nav className="workbench-module-nav" aria-label={t("实验编辑模块", "Experiment editing modules")}>
          {(["scene", "reward", "settings"] as const).map((section, index) => <button key={section} data-tour={section === "scene" ? "workbench-scene" : section === "settings" ? "workbench-parameters" : "workbench-modules"} aria-pressed={workspaceSection === section} className={workspaceSection === section ? "active" : ""} onClick={() => openSection(section)}><span className="workbench-module-number">0{index + 1}</span><span><strong>{section === "scene" ? taskId === "s1_place" ? t("场景说明 · 当前固定", "Scene overview · fixed") : t("任务场景", "Task scene") : section === "reward" ? t("奖励设计", "Reward design") : t("实验设置", "Experiment settings")}</strong><small>{section === "scene" ? taskId === "s1_push" ? t("方块起点 · 尺寸 · 目标", "Block start · size · goal") : taskId === "s1_place" ? t("固定起点 · XY 搬运 · 保持", "Fixed start · XY transport · hold") : t("目标采样 · 动作步长", "Target sampling · motion step") : section === "reward" ? t("行为解释 · 权重 · reward.py", "Behavior · weights · reward.py") : t("训练方案 · 种子 · 录像", "Training plan · seed · video")}</small></span></button>)}
        </nav>
        <div className="workbench-source-heading"><span>{t("② 参考源码 · 只读", "② Reference source · read only")}</span><button aria-expanded={showAllSources} onClick={() => { setShowAllSources((current) => !current); setFileFilter(""); }}>{showAllSources ? t("任务相关", "Task files") : t(`全部源码 (${completeSourceList.length})`, `All source (${completeSourceList.length})`)}</button></div>
        <label className="workbench-file-search"><input value={fileFilter} onChange={(event) => setFileFilter(event.target.value)} aria-label={t("查找源码文件", "Find source files")} placeholder={t("查找全部源码…", "Search all source…")} /></label>
        <div className="workbench-file-list">{sources.isError && <div className="workbench-source-error" role="alert"><p>{t("源码服务不可用", "Source service unavailable")}</p><button onClick={() => void sources.refetch()}>{t("重试加载", "Retry loading")}</button></div>}{visibleSourceFiles.map((path) => <button key={path} className={path === activeFile ? "active" : ""} onClick={() => openSource(path)} title={path} aria-label={path}><FileCode2 size={13} /><span>{path.split("/").pop()}<small>{!browsingAll ? referenceFiles.find((item) => item.path === path)?.description : path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : t("运行核心", "Runtime core")}</small></span><ShieldCheck size={11} /></button>)}{!sources.isPending && !sources.isError && visibleSourceFiles.length === 0 && <p className="workbench-source-error">{t("没有匹配的文件", "No matching files")}</p>}</div>
      </aside>
      <section data-tour="workbench-editor" className="workbench-editor workbench-editor-unified">
        <div className="workbench-editor-bar"><span>{showDiff ? t("Starter / 当前版本", "Starter / current revision") : workspaceSection === "scene" ? t("任务场景", "Task scene") : workspaceSection === "settings" ? t("实验设置", "Experiment settings") : workspaceSection === "reward" ? t("奖励设计 · reward.py", "Reward design · reward.py") : activeFile}</span><small>{showDiff ? t("版本对比 · 只读", "Revision comparison · Read only") : workspaceSection === "source" ? t("平台源码 · 只读", "Platform source · Read only") : t("受控实验", "Controlled experiment")}</small></div>
        <div className="workbench-edit-guide"><strong>{workspaceSection === "scene" ? t("改变任务条件，观察学习差异", "Change task conditions and observe learning") : workspaceSection === "settings" ? t("让实验可以重复、可以比较", "Make experiments repeatable and comparable") : workspaceSection === "source" ? t("完整运行源码 · 只读", "Full runtime source · Read only") : t("理解行为，再修改奖励", "Understand behavior before changing rewards")}</strong><span>{workspaceSection === "scene" ? taskId === "s1_push" ? t("预览即时更新；保存并校验后，同一场景版本会进入训练。", "Preview updates immediately; save and validate to train with that exact scene revision.") : taskId === "s1_place" ? t("此任务场景固定：从已抓取抬起状态开始，XY 搬运并保持一秒。", "This task has a fixed scene: start from a lifted grasp, transport in XY and hold for one second.") : t("目标采样范围与动作步长实际进入训练；成功标准保持锁定。", "Target sampling and motion step changes reach the training environment; success criteria remain locked.") : workspaceSection === "settings" ? t("种子影响随机过程；正式训练预算来自课程 Project。", "The seed controls randomness; the course Project defines the formal training budget.") : workspaceSection === "source" ? t("选中代码请 Agent 解读；在奖励设计中修改可执行奖励。", "Select source for Agent explanations; edit executable rewards under Reward design.") : t("教学权重与代码同步；原有自定义表达式保留在进阶编辑中。", "Teaching weights and code stay synchronized; custom expressions are preserved in advanced editing.")}</span><button className="workbench-guide-action" disabled={workspaceSection !== "source" && !completeSourceList.includes(referenceFiles[0].path)} onClick={() => workspaceSection === "source" ? openSection("reward") : openSource(referenceFiles[0].path)}>{workspaceSection === "source" ? t("← 返回奖励编辑", "← Back to reward editing") : t("查看当前任务完整实现 →", "View the full task implementation →")}</button>{workspaceSection === "reward" && !showDiff && <button className="workbench-guide-action" aria-pressed={advancedCode} disabled={!draft || Boolean(busy)} onClick={() => { setAdvancedCode(!advancedCode); codePreview.current?.scrollIntoView({ block: "nearest" }); }}>{advancedCode ? t("结束代码编辑", "Finish code editing") : t("进阶代码编辑", "Advanced code editing")}</button>}</div>
        <div className="workbench-code-view">
          <div hidden={workspaceSection !== "scene" && workspaceSection !== "settings"} className="workbench-parameter-panel">
            <div className="workbench-parameter-intro"><h2>{workspaceSection === "scene" ? task.goal : t("从一次可重复的实验开始", "Start with a repeatable experiment")}</h2><p>{workspaceSection === "scene" ? taskId === "s1_push" ? t("起点、尺寸和目标会进入实际训练环境；每次只改一个条件，然后保存场景并校验。", "Start, dimensions and goal reach the actual training environment. Change one condition, then save and validate the scene.") : taskId === "s1_place" ? t("从已抓取抬起状态开始，夹爪与 Z 保持锁定，XY 搬运后稳定保持一秒。本任务可通过奖励设计进行实验。", "Start from a lifted grasp with the gripper and Z locked, transport in XY and hold for one second. Use Reward design to experiment with this task.") : t("目标中心、采样范围和动作步长会实际改变训练条件；每次只改一项并比较同一任务的结果。", "Target center, sampling extents and motion steps change actual training conditions. Change one at a time and compare Runs of the same task.") : t("先沿用默认种子，比较奖励方案；之后再换种子检验结果是否稳定。", "Keep the default seed when comparing rewards; later vary the seed to check consistency.")}</p></div>
            {workspaceSection === "settings" && <div className="workbench-budget-card"><label>{t("训练方案", "Training plan")}<select aria-label={t("训练方案", "Training plan")} value={selectedProject?.project_id ?? ""} disabled={!courseId || Boolean(busy)} onChange={(event) => setProjectId(event.target.value)}><option value="">{t("选择训练方案", "Select a training plan")}</option>{(projects.data ?? []).filter((item) => item.enabled && item.permissions.can_use && item.task_type === "training" && item.environment_type === taskId).map((item) => <option key={item.project_id} value={item.project_id}>{uiText(item.title)} · {item.expected_training_steps.toLocaleString()} {t("步", "steps")}</option>)}</select></label><span>{t("正式训练预算 · 课程管理", "Formal training budget · course controlled")}</span><strong>{selectedProject ? t(`${selectedProject.expected_training_steps.toLocaleString()} 步`, `${selectedProject.expected_training_steps.toLocaleString()} steps`) : t("选择匹配 Project 后显示", "Select a matching Project to view")}</strong><p>{selectedProject ? t(`${selectedProject.expected_training_steps <= 32 ? "入门方案用于检查流程；预算很短，不适合判断学习效果。" : "正式训练方案用于观察策略学习。"} Project V${selectedProject.config_version} 冻结预算；比较时保持一致。`, `${selectedProject.expected_training_steps <= 32 ? "The introductory plan checks the workflow; its short budget cannot demonstrate learning." : "The formal plan is intended for policy learning."} Project V${selectedProject.config_version} freezes this budget; match it when comparing.`) : t("预算、PPO 参数由课程版本管理，当前工作台不能单独覆盖。", "The course revision controls the budget and PPO parameters; this workbench does not override them.")}</p></div>}
            {taskId === "s1_push" && <div hidden={workspaceSection !== "scene"}><PushSceneControls key={user.user_id} storageKey={`s1-push-scene:${user.user_id}`} disabled={Boolean(busy)} onStateChange={setPushScene} /></div>}
            <div hidden={workspaceSection === "scene" && taskId === "s1_push"}>
            <SimulationTaskConfigurator key={`${taskId}:${projectId}`} initialTaskId={taskId} initialParameters={{ video_enabled: true }} allowedTaskIds={[taskId]} showTaskSelection={false} compact groupParameters workbenchSection={workspaceSection === "scene" ? "scene" : "settings"} requiredTaskMode="training" disabled={Boolean(busy)} onStateChange={setConfiguration} />
            </div>
          </div>
          <div hidden={workspaceSection !== "reward" && workspaceSection !== "source"} className={activeFile === "reward.py" && !showDiff ? "workbench-reward-content" : "workbench-source-content"}>
            {activeFile === "reward.py" && !showDiff && <><RewardDesigner source={source} terms={draft?.editable_terms ?? []} disabled={Boolean(busy) || !draft} onChange={changeReward} onExplain={(value) => { setQuestion(value); void askAgent(value); }} onViewSource={() => openSource(taskId === "s1_place" ? "experimental/manipulation/place_schema.py" : referenceFiles[0].path)} /><div className="workbench-code-mode"><strong>{advancedCode ? t("进阶：编辑 return 表达式", "Advanced: edit the return expression") : t("同步代码预览", "Synchronized code preview")}</strong><small>{advancedCode ? t("保留函数签名和单个 return", "Keep the signature and one return") : t("调整权重自动更新此代码", "Weights update this code automatically")}</small></div></>}
            {activeFile !== "reward.py" && (readOnlySource.isPending || readOnlySource.isError) ? <div className="workbench-source-placeholder" role="status">{readOnlySource.isError ? t("源码读取失败", "Failed to read source") : t("正在加载完整源码…", "Loading full source…")}<button className="secondary" onClick={() => void readOnlySource.refetch()}>{t("重试加载", "Retry loading")}</button></div> : <div ref={codePreview} className={activeFile === "reward.py" && !showDiff ? "workbench-generated-code" : "workbench-full-code"}><WorkbenchEditor key={showDiff ? "diff" : activeFile} original={showDiff ? proposalOpen ? source : draft?.starter ?? "" : undefined} locale={locale} value={showDiff ? proposalOpen ? diagnosis?.suggested_source ?? "" : source : activeFile === "reward.py" ? source : readOnlySource.data?.content ?? ""} readOnly={showDiff || activeFile !== "reward.py" || !advancedCode || Boolean(busy) || !draft} onChange={activeFile === "reward.py" && !showDiff ? changeReward : undefined} onSelection={(value) => { setSelection(value); setReadOnlySelection(activeFile !== "reward.py"); }} /></div>}
          </div>
        </div>
        <div className="workbench-code-contract"><ShieldCheck size={14} /><span>{draft?.protected_terms.join(" + ") ?? "success_bonus + joint_limit_penalty"}</span><strong>{t("平台保留", "Platform protected")}</strong></div>
      </section>
      <aside data-tour="workbench-agent" className="workbench-agent">
        <div className="workbench-pane-title"><span><Bot size={15} /> {t("代码解读", "Code explanation")}</span><small>{selection ? `${selection.length} chars` : t("选择代码以解读", "Select code to explain")}</small></div>
        <div className="workbench-agent-copy">
          {selection && <pre className="workbench-selection">{selection.slice(0, 700)}</pre>}
          <p>{diagnosis?.content ?? t("等待代码解读", "Waiting for code analysis")}</p>
          <label className="workbench-agent-question"><textarea aria-label={t("询问 Agent", "Ask Agent")} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={t("这段代码如何影响奖励？", "How does this code affect reward?")} /></label>
          <div className="workbench-agent-controls"><select aria-label="Agent Provider" value={provider} onChange={(event) => setProvider(event.target.value)}>{(providers.data ?? [{ id: "mock", name: t("内置基础 Agent", "Built-in basic Agent"), configured: true } as AgentProviderInfo]).filter((item) => item.configured).map((item) => <option key={item.id} value={item.id}>{uiText(item.name)}</option>)}</select><button title={t("发送选区与问题", "Send selection and question")} aria-label={t("分析代码", "Analyze code")} className="icon-button" disabled={Boolean(busy) || !draft} onClick={() => void askAgent()}><Send size={15} /></button></div>
          {diagnosis?.suggested_source && <button className="secondary" onClick={() => { openSection("reward"); setProposalOpen(true); setShowDiff(true); }}><GitCompare size={14} /> {t("审查 Agent 提案", "Review Agent proposal")}</button>}
          {proposalOpen && <div className="workbench-proposal-actions"><button className="primary" disabled={Boolean(busy)} onClick={() => void applyProposal()}><Check size={14} /> {t("审批为新版本", "Approve as revision")}</button><button className="icon-button" title={t("拒绝提案", "Reject proposal")} onClick={() => { setProposalOpen(false); setShowDiff(false); }}><X size={14} /></button></div>}
        </div>
        <div className="workbench-diagnosis"><h2>{t("运行诊断", "Run diagnosis")}</h2><strong>{runQuery.data ? runStatusLabel(runQuery.data.status, locale) : t("尚未关联 Run", "No Run selected")}</strong>{runQuery.data?.error && <p role="alert">{runQuery.data.error.message}</p>}<small>{runQuery.data?.code_sha256?.slice(0, 16) ?? ""}</small>{runQuery.data && <FrozenSceneSummary snapshot={runQuery.data.project_config_snapshot} />}</div>
      </aside>
    </section>
    <section data-tour="workbench-evidence" className="workbench-bottom">
      <div className="workbench-preflight"><div className="workbench-bottom-tabs" role="tablist">{(["preflight", "logs", "metrics"] as const).map((tab) => <button key={tab} role="tab" aria-selected={bottomTab === tab} onClick={() => setBottomTab(tab)}>{tab === "preflight" ? t("预检", "Preflight") : tab === "logs" ? t("后端日志", "Backend logs") : t("指标对比", "Metric comparison")}</button>)}</div>
        {bottomTab === "preflight" && <div className={preflight?.valid ? "preflight-success" : "preflight-pending"}><Check size={16} /><span><strong>{preflight?.valid ? t("契约与样例值通过", "Contract and samples passed") : (preflight?.errors[0]?.message ? uiText(preflight.errors[0].message) : undefined) ?? t("等待预检", "Awaiting preflight")}</strong><small>{(preflight?.warnings[0] ? uiText(preflight.warnings[0]) : undefined) ?? t("预检不保证学习成功率", "Preflight does not guarantee learning success")}</small></span></div>}
        {bottomTab === "logs" && <div className="workbench-run-log">{events.isError || runQuery.isError ? <p role="alert">{t("运行证据读取失败", "Run evidence unavailable")}</p> : (events.data ?? []).map((event) => <pre key={event.sequence}>#{event.sequence} {event.type} {JSON.stringify(event.payload)}</pre>)}{!events.data?.length && <p>{runId ? t("等待后端事件", "Waiting for backend events") : t("尚未关联 Run", "No Run selected")}</p>}</div>}
        {bottomTab === "metrics" && <div className="workbench-metric-comparison"><select aria-label={t("基线 Run", "Baseline Run")} value={baselineRunId} onChange={(event) => setBaselineRunId(event.target.value)}><option value="">{t("选择基线 Run", "Choose baseline Run")}</option>{(runs.data ?? []).filter((item) => item.project_config_snapshot.environment_type === taskId && item.run_id !== runId).map((item) => <option key={item.run_id} value={item.run_id}>{item.run_id.slice(-8)} · {item.status}</option>)}</select><div><pre>{JSON.stringify(baseline?.metrics_summary ?? {}, null, 2)}</pre><pre>{JSON.stringify(runQuery.data?.metrics_summary ?? {}, null, 2)}</pre></div><small>{t("仅供观察；场景、预算或 seed 不一致不能作因果结论。", "Observational comparison only; mismatched scene, budget or seed does not support causal claims.")}</small></div>}
        {(courses.isError || projects.isError || classes.isError) && <p className="workbench-error" role="alert">{t("课程上下文不可用", "Course context unavailable")}</p>}{error && <p className="workbench-error" role="alert">{uiText(error)}</p>}{message && <p className="workbench-message" role="status">{message}</p>}
      </div>
      <div className="workbench-history"><div className="workbench-bottom-title"><span><History size={15} /> {t("版本历史", "Revision history")}</span><small>{draft?.history.length ?? 0}</small></div><div className="workbench-history-actions">{(draft?.history ?? []).map((item) => <button key={item.revision_id} disabled={Boolean(busy) || item.revision_id === baseRevision} onClick={() => void restoreRevision(item.revision_id)} title={item.created_at}>{item.revision_id.slice(-8)}</button>)}<button disabled={Boolean(busy) || !draft} onClick={() => void restoreRevision("starter")}><RotateCcw size={13} /> Starter</button></div></div>
      <div data-tour="workbench-actions" className="workbench-action-group"><p className="workbench-launch-reason">{!draft ? t("等待奖励代码加载", "Waiting for reward code") : !selectedClass ? t("请选择教学班", "Select a class") : !selectedProject ? t("请选择匹配当前任务的 Project", "Select a Project matching this task") : source !== draft.source ? t("请先保存修改", "Save your changes first") : !preflight?.valid ? t("请运行奖励预检", "Run reward preflight") : taskId === "s1_push" && !pushScene.ready ? t("请在任务场景中保存并校验场景", "Save and validate under Task scene") : !configuration?.validation?.valid || configuration.catalog?.source !== "server" ? t("等待服务端参数校验", "Waiting for server parameter validation") : !permitted ? t("缺少创建 Run 权限", "Run creation permission required") : t("版本、预检与上下文已就绪", "Revision, preflight and context ready")}</p><div className="workbench-actions"><button className="secondary" disabled={Boolean(busy) || !draft} onClick={() => void saveDraft()}><Save size={15} /> {t("保存版本", "Save revision")}</button><button className="secondary" disabled={Boolean(busy) || !draft} onClick={() => void runPreflight()}><Check size={15} /> {t("运行预检", "Run preflight")}</button><button className="primary" disabled={!canSubmit} onClick={() => void startRun()}><Play size={15} /> {busy === "run" ? t("启动中…", "Starting…") : t("启动训练", "Start training")}</button></div></div>
    </section>
    <section className="workbench-status"><span>{t("当前任务", "Task")} <strong>{task.title}</strong></span><span>{t("场景", "Scene")} <strong>{taskId === "s1_push" && pushScene.sceneRef ? t(`自定义场景 · REV ${pushScene.sceneRef.revision}`, `Custom scene · REV ${pushScene.sceneRef.revision}`) : taskId === "s1_push" && pushScene.dirty ? t("场景修改未保存", "Unsaved scene changes") : t("任务基线场景", "Task baseline scene")}</strong></span><span>{t("执行状态", "Execution")} <strong>{canSubmit ? t("可提交", "Ready") : t("需要预检和上下文", "Preflight and context required")}</strong></span></section>
  </main>;
}
