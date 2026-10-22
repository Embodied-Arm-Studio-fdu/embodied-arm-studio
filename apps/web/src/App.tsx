import { useFloatingAgentPosition } from "./useFloatingAgentPosition";
import { lazy, Suspense, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Editor, { DiffEditor } from "@monaco-editor/react";
import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  Bot,
  BookOpen,
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleStop,
  Code2,
  Copy,
  Download,
  ExternalLink,
  File,
  FileCheck2,
  FilePlus2,
  FlaskConical,
  Folder,
  FolderKanban,
  FolderPlus,
  GitCompare,
  GraduationCap,
  ListTree,
  Map,
  Menu,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  Settings,
  Settings2,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Trash2,
  Undo2,
  Upload,
  Users,
  X,
} from "lucide-react";
import {
  type CurrentUserContext,
  type AgentMessageResponse,
  type AgentProviderInfo,
  type AssistanceLevelInfo,
  type Course,
  type CanonicalEvaluationReport,
  type TeachingCourse,
  type TeachingClass,
  type TeachingProject,
  type EnvironmentInfo,
  type EnvironmentSchema,
  type EvaluationComparison,
  type EvaluationReport,
  type MetricEvent,
  type LegacyRunRecord,
  type PipelineDocument,
  type PipelineValidation,
  type RunRecord,
  type RunArtifact,
  type RunComparisonResponse,
    type SandboxArtifactInfo,
    type SandboxJobInfo,
    type SandboxTemplateInfo,
  type SceneDocument,
  type SceneObject,
  type StrategyInfo,
  type WorkspaceNode,
  type WorkspaceProject,
  ApiError,
  api,
  apiBlob,
  apiWithFallback,
  getAccessToken,
  setAccessToken,
} from "./api";
import { ArmViewport } from "./ArmViewport";
import { FrozenSceneSummary } from "./FrozenSceneSummary";
import { SceneEditor } from "./SceneEditor";
import { RunComparisonView } from "./RunComparisonView";
import { RunArtifactsPage } from "./RunArtifactsPage";
import { clampS1Joint, s1JointLabel, S1_ARM_JOINTS, S1_INITIAL_ARM_POSITIONS } from "./s1Robot";
import { RunVideoPanel } from "./WhepPlayer";
import { ReplayPlayer, runVideoEnabled, runVideoMode, selectReplayArtifacts } from "./ReplayPlayer";
import { useStudio } from "./store";
import {
  isTerminalRunStatus,
  latestRunEventSequence,
  mergeRunEvents,
  normalizeRunEvent,
  normalizeRunEventsResponse,
  runConnectionLabel,
} from "./runEvents";
import { createCanonicalRunSubmission } from "./canonicalRunRequest";
const WorkbenchPage = lazy(() => import("./WorkbenchPage").then((module) => ({ default: module.WorkbenchPage })));
const LearningPage = lazy(() => import("./learning/LearningPage").then((module) => ({ default: module.LearningPage })));
import {
  SimulationTaskConfigurator,
  type SimulationTaskConfigurationState,
} from "./SimulationTaskConfigurator";
import { taskParameterPayload } from "./simulationCatalog";
import { runDisplayName } from "./runLabel";
import { buildAgentPageContext } from "./agentContext";
import {
  RUN_STATUS_PRESENTATIONS,
  formatElapsedDuration,
  parameterFields,
  runElapsedMilliseconds,
  runProgressPresentation,
  runStatusLabel,
  runStatusPresentation,
} from "./runPresentation";
import {
  PRODUCTION_EVALUATION_DEFINITIONS,
  RUN_METRIC_DEFINITIONS,
  buildRunMetricsCsv,
  evaluationMetric,
  metricSeriesData,
  missingSimulationMetricDefinitions,
  productionEvaluation,
  runMetricHistory,
  evaluationMetricText,
  runMetricText,
  visibleRunMetricDefinitions,
  type RunMetricPoint,
} from "./runMetrics";
import {
  CANONICAL_PLATFORM_RUNTIME_CONTEXT,
  LEGACY_QUICK_COURSE_RUNTIME_CONTEXT,
  runtimeModeLabels,
} from "./runtimeMode";
import { LanguageSwitcher, LocalizedContent, translateUiText, useI18n } from "./i18n";
import { TourEngine } from "./onboarding/TourEngine";
import { platformFirstTimeTour, tourRegistry, type TourDefinition } from "./onboarding/registry";
import { tourStorage } from "./onboarding/storage";
import { TasksSkillsPage } from "./TasksSkillsPage";
import { teachingProjectLabel } from "./projectPresentation";
import { GUIDE_PRINCIPLES, USER_GUIDE_BY_ID, USER_GUIDE_SECTIONS, USER_GUIDE_VERSION, guideForPath, guidePathForSection, type GuideSection } from "./userGuide";

type WorkspaceMode = "learning" | "experiment" | "analysis";

const PLATFORM_NAV_MIN_WIDTH = 204;
const PLATFORM_NAV_MAX_WIDTH = 360;
const PLATFORM_NAV_DEFAULT_WIDTH = 268;
const PLATFORM_NAV_WIDTH_STORAGE_KEY = "embodied-arm-platform-nav-width";

function clampPlatformNavWidth(width: number): number {
  return Math.round(Math.min(PLATFORM_NAV_MAX_WIDTH, Math.max(PLATFORM_NAV_MIN_WIDTH, width)));
}

const NAV_ITEMS = [
  { path: "/learn", label: "入门学习", icon: BookOpen, status: "ready", modes: ["learning"] },
  { path: "/roadmap", label: "学习路线与实验流程", icon: Map, status: "ready", modes: ["learning"] },
  { path: "/agent", label: "教学 Agent", icon: Users, status: "ready", modes: ["learning"] },
  { path: "/projects", label: "实验工作台", icon: FolderKanban, status: "ready", modes: ["experiment"] },
  { path: "/skills", label: "任务与技能", icon: Boxes, status: "ready", modes: ["experiment"] },
  { path: "/courses", label: "课程与高级任务", icon: GraduationCap, status: "ready", modes: ["experiment"] },
  { path: "/runs", label: "运行分析", icon: FlaskConical, status: "ready", modes: ["analysis"] },
  { path: "/run-comparison", label: "运行结果对比", icon: GitCompare, status: "ready", modes: ["analysis"] },
  { path: "/evaluation", label: "策略评估", icon: TestTube2, status: "ready", modes: ["analysis"] },
  { path: "/export", label: "运行制品", icon: FileCheck2, status: "ready", modes: ["analysis"] },
] as const;

const WORKSPACE_MODES: Array<{
  id: WorkspaceMode;
  label: string;
  shortLabel: string;
  description: string;
  landingPath: string;
}> = [
  { id: "learning", label: "学习", shortLabel: "学习", description: "认识平台并按步骤理解一次六轴 S1 仿真实验", landingPath: "/roadmap" },
  { id: "experiment", label: "实验", shortLabel: "实验", description: "编辑项目、S1 场景与策略，配置后启动正式实验", landingPath: "/projects" },
  { id: "analysis", label: "分析", shortLabel: "分析", description: "核对 Run、正式评估与持久化制品", landingPath: "/runs" },
];

const NAV_STATUS_LABELS = {
  ready: "可用",
  mock: "演示",
  planned: "规划中",
  blocked: "受限",
} as const;

function workspaceModeForPath(path: string, preferredMode?: WorkspaceMode): WorkspaceMode {
  if (path === "/learn" || path.startsWith("/learn/")) return "learning";
  if (path === "/guide" || path.startsWith("/guide/")) return preferredMode ?? "learning";
  const item = NAV_ITEMS.find(
    (entry) => entry.path === path || (entry.path === "/runs" && path.startsWith("/runs/")),
  );
  if (preferredMode && item?.modes.includes(preferredMode as never)) {
    return preferredMode;
  }
  return item?.modes[0] ?? "experiment";
}

function capabilityLabel(value: string): string {
  return {
    MOCK: "演示能力",
    mock: "演示能力",
    PLANNED: "规划中",
    planned: "规划中",
    READY: "可用",
    ready: "可用",
    BLOCKED: "受限",
    blocked: "受限",
  }[value] ?? value;
}

const ROADMAP_STAGES = [
  {
    id: "project",
    index: "01",
    title: "项目与代码结构",
    summary: "建立项目、工作区、文件树和可追踪的实验上下文。",
    evidence: "项目创建并生成初始工作区",
    capability: "MOCK",
  },
  {
    id: "theory",
    index: "02",
    title: "理论与安全",
    summary: "理解坐标系、关节、仿真边界、急停和真实部署风险。",
    evidence: "完成课程安全检查点",
    capability: "MOCK",
  },
  {
    id: "robot",
    index: "03",
    title: "机器人与场景",
    summary: "选择机械臂、场景版本、目标物和可交互对象。",
    evidence: "场景通过结构校验并生成修订版本",
    capability: "MOCK",
  },
  {
    id: "observation",
    index: "04",
    title: "Observation 设计",
    summary: "定义关节、目标、视觉和归一化观测输入。",
    evidence: "保存 observation schema 与设计说明",
    capability: "MOCK",
  },
  {
    id: "action",
    index: "05",
    title: "Action 设计",
    summary: "选择关节增量、关节目标或末端执行器控制方式。",
    evidence: "保存 action schema、范围和缩放",
    capability: "MOCK",
  },
  {
    id: "reward",
    index: "06",
    title: "Reward 设计",
    summary: "组合距离、姿态、成功、平滑度和惩罚项。",
    evidence: "保存 reward 组成与预期现象",
    capability: "MOCK",
  },
  {
    id: "termination",
    index: "07",
    title: "Termination / Reset",
    summary: "定义成功、超时、碰撞、无效状态和重置条件。",
    evidence: "环境逻辑通过契约验证",
    capability: "MOCK",
  },
  {
    id: "step",
    index: "08",
    title: "Environment Step",
    summary: "观察 action 到物理步进、奖励和终止的完整链路。",
    evidence: "Pipeline 节点产生运行状态",
    capability: "MOCK",
  },
  {
    id: "policy",
    index: "09",
    title: "Policy / Network",
    summary: "选择基线控制器、RL 策略或模仿学习策略契约。",
    evidence: "策略配置与环境版本绑定",
    capability: "MOCK",
  },
  {
    id: "data",
    index: "10",
    title: "数据采集",
    summary: "规划遥操作、episode、标注和数据质量检查。",
    evidence: "数据集协议已定义",
    capability: "PLANNED",
  },
  {
    id: "training",
    index: "11",
    title: "训练执行",
    summary: "绑定训练提供方、资源预算、seed 和 checkpoint。",
    evidence: "产生可追踪运行记录",
    capability: "MOCK",
  },
  {
    id: "simulation",
    index: "12",
    title: "实时仿真",
    summary: "观察机器人、目标、轨迹、action、observation 和 reward。",
    evidence: "Simulation Viewer 会话可复现",
    capability: "MOCK",
  },
  {
    id: "metrics",
    index: "13",
    title: "实时指标",
    summary: "查看任务指标、优化指标和基础设施状态。",
    evidence: "指标事件与运行 ID 关联",
    capability: "MOCK",
  },
  {
    id: "comparison",
    index: "14",
    title: "运行对比",
    summary: "比较代码、场景、配置、seed、策略和结果变化。",
    evidence: "生成多运行对比证据",
    capability: "MOCK",
  },
  {
    id: "evaluation",
    index: "15",
    title: "自动评估",
    summary: "结合静态检查、运行测试、多 seed、基线和可复现性。",
    evidence: "评估标准逐项关联 Evidence",
    capability: "MOCK",
  },
  {
    id: "export",
    index: "16",
    title: "Policy / Dataset 导出",
    summary: "封装策略、场景、映射、指标、报告和校验和。",
    evidence: "导出 manifest 并保留门禁结果",
    capability: "MOCK",
  },
  {
    id: "sim2real",
    index: "17",
    title: "Sim2Real Readiness",
    summary: "检查关节顺序、标定、延迟、限位、急停和映射。",
    evidence: "完成独立人工 readiness checklist",
    capability: "BLOCKED",
  },
  {
    id: "deployment",
    index: "18",
    title: "受控部署",
    summary: "在独立 Robot Gateway、lease、watchdog 和操作员授权下运行。",
    evidence: "真实机器人部署默认禁止",
    capability: "BLOCKED",
  },
] as const;

type RoadmapStageId = (typeof ROADMAP_STAGES)[number]["id"];

const S1_TASK_DRAFT_STORAGE_KEY = "s1-draft-simulation-task-v1";

interface S1TaskDraft {
  taskId: string;
  parameters: Record<string, unknown>;
}

function readS1TaskDraft(): S1TaskDraft | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(S1_TASK_DRAFT_STORAGE_KEY) ?? "null") as unknown;
    if (!value || typeof value !== "object") return null;
    const draft = value as Partial<S1TaskDraft>;
    if (typeof draft.taskId !== "string" || !draft.parameters || typeof draft.parameters !== "object") return null;
    return { taskId: draft.taskId, parameters: draft.parameters };
  } catch {
    return null;
  }
}

const LEGACY_ROUTE_REDIRECTS: Record<string, string> = {
  "/quick-course": "/roadmap",
  "/pipelines": "/roadmap",
  "/settings": "/agent",
  "/datasets": "/roadmap",
  "/robots": "/environments",
  "/deployment": "/export",
};

function currentPath() {
  if (location.pathname === "/") return "/roadmap";
  if (location.pathname === "/simulation") return "/training";
  return LEGACY_ROUTE_REDIRECTS[location.pathname] ?? location.pathname;
}

function navigate(path: string) {
  history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function LoginPage({ onLoggedIn, initialError = "" }: { onLoggedIn: (user: CurrentUserContext) => void; initialError?: string }) {
  const { locale, t } = useI18n();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(initialError);
  const [busy, setBusy] = useState(false);

  useEffect(() => setError(""), [locale]);

  async function login(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await api<{ access_token: string; user: { user_id: string; username: string; display_name: string; enabled: boolean } }>(
        "/api/v1/auth/login",
        { method: "POST", body: JSON.stringify({ username, password }) },
      );
      setAccessToken(response.access_token);
      const context = await api<CurrentUserContext>("/api/v1/me");
      onLoggedIn(context);
      navigate("/roadmap");
    } catch (loginError) {
      setError(loginError instanceof Error ? loginError.message : t("登录失败", "Sign-in failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login-page">
      <div className="login-language"><LanguageSwitcher /></div>
      <form className="login-form" onSubmit={login}>
        <span className="eyebrow">EMBODIED ARM STUDIO</span>
        <h1>登录研究平台</h1>
        <p>使用平台账号进入课程、项目和运行证据工作区。</p>
        <label>用户名<input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
        <label>密码<input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
        {error && <p className="inline-error">{error}</p>}
        <button className="primary" disabled={busy}>{busy ? "登录中..." : "登录"}</button>
      </form>
    </main>
  );
}

function runDetailPath(runId: string) {
  return `/runs/${encodeURIComponent(runId)}`;
}

async function createLegacyTrainingRun(payload: {
  project_id: string;
  parameters: Record<string, unknown>;
  project_config_snapshot: Record<string, unknown>;
  legacy: {
    environment_id: string;
    strategy_id: string;
    seed: number;
    max_steps: number;
    config: Record<string, unknown>;
  };
}) {
  return api<LegacyRunRecord>("/api/runs/train", {
      method: "POST",
      body: JSON.stringify(payload.legacy),
  });
}

export function LearningGuidePage() {
  const { t } = useI18n();
  const LEARNING_STEPS = [
    { icon: Map, path: "/projects", title: t("认识平台", "Understand the platform"), subtitle: t("浏览源码与受控奖励", "Browse source and controlled rewards"), action: t("打开实验工作台，切换 Push、Reach 或 Place，检查文件与编辑标记。", "Open the workbench, switch between Push, Reach and Place, and inspect source and edit markers."), files: t("运行源码 · reward.py · 评估脚本", "Runtime source \u00b7 reward.py \u00b7 evaluation scripts"), evidence: t("能区分平台只读源码、可执行奖励与 Run 制品。", "Distinguish read-only platform source, executable reward edits and Run artifacts."), caution: t("修改奖励不改变 PPO、动作观测和真实控制边界。", "Reward edits do not change PPO, action/observation contracts or real-control boundaries.") },
    { icon: Boxes, path: "/environments", title: t("准备六轴场景", "Prepare the six-axis scene"), subtitle: t("核对场景与预览边界", "Review the scene and preview boundaries"), action: t("检查 S1 场景中的机械臂、目标和物体，保存需要的预览修订。正式工作台使用任务基线场景。", "Inspect the S1 arm, goal and object. Save preview revisions when needed. The formal workbench uses the task baseline scene."), files: t("S1 Scene revision · URDF/STL", "S1 Scene revision \u00b7 URDF/STL"), evidence: t("知道预览场景与正式任务基线的区别。", "Distinguish preview scenes from the formal task baseline."), caution: t("任意障碍物未映射到 Worker 时不能用于正式训练；预览不是碰撞或 IK 证明。", "Untranslated obstacles cannot enter formal training. Preview is not collision or IK proof.") },
    { icon: Code2, path: "/projects", title: t("理解任务与策略", "Understand tasks and policies"), subtitle: t("选择任务并解读代码", "Choose a task and explain its code"), action: t("浏览任务实现、观测动作、奖励和 PPO 源码，选中代码向右侧 Agent 提问。", "Browse task, observation/action, reward and PPO source. Select code and ask the Agent on the right."), files: t("task_registry.py · PPO · reward.py", "task_registry.py \u00b7 PPO \u00b7 reward.py"), evidence: t("理解 Push、Reach 和 Place 的初始条件与成功定义。", "Understand entry conditions and success definitions for Push, Reach and Place."), caution: t("控制 Demo 不更新策略权重；Place 是搬运保持而非完整释放流程。", "Control demos do not train weights. Place is transport-and-hold, not a full release workflow.") },
    { icon: Activity, path: "/projects", title: t("配置训练", "Configure training"), subtitle: t("项目版本决定正式训练约束", "The Project version defines training constraints"), action: t("选择教学班和匹配 Project，展开实验参数，修改高亮奖励表达式并保存、预检。", "Select a class and matching Project. Expand Experiment parameters, edit the highlighted reward expression, save and preflight."), files: t("Project config · 参数 catalog · 奖励版本", "Project config \u00b7 parameter catalog \u00b7 reward revision"), evidence: t("奖励已保存、预检通过，参数和 Project 匹配。", "Reward is saved, preflight passes, and parameters match the Project."), caution: t("静态预检不保证物理安全或学习成功率。", "Static preflight does not prove physics safety or learning success.") },
    { icon: Play, path: "/projects", title: t("启动并观察 Run", "Start and observe the Run"), subtitle: t("冻结版本后创建正式运行", "Freeze a revision and create a formal Run"), action: t("在工作台启动训练，关联 Run 并检查底部后端日志与右侧诊断。", "Start training in the workbench. Associate the Run and inspect backend logs and diagnosis."), files: t("冻结代码与参数 · Run 事件", "Frozen code and parameters \u00b7 Run events"), evidence: t("拿到 run_id，并看到服务端状态与事件。", "Receive a run_id and inspect server-owned status and events."), caution: t("completed 表示执行结束，不自动表示任务成功。", "completed means execution ended; it does not establish task success.") },
    { icon: TestTube2, path: "/runs", title: t("评估和复盘", "Evaluate and review"), subtitle: t("以证据决定下一次修改", "Use evidence to decide the next change"), action: t("核对成功率、错误、指标与 checkpoint；在同任务、场景、预算和 seed 下做基线对比。", "Review success rates, errors, metrics and checkpoints. Compare baselines with matching task, scene, budget and seed."), files: t("metrics · manifest.json · checkpoints", "metrics \u00b7 manifest.json \u00b7 checkpoints"), evidence: t("区分训练完成、任务成功与证据不足。", "Distinguish training completion, task success and insufficient evidence."), caution: t("Agent 解释现有证据，不生成缺失指标或修改评分。", "The Agent explains existing evidence; it does not invent missing metrics or change scores.") },
  ];
  const [selectedStep, setSelectedStep] = useState(0);
  const step = LEARNING_STEPS[selectedStep];
  const Icon = step.icon;
  return <main className="learning-guide">
    <header className="learning-guide-header"><span className="eyebrow">S1 / LEARNING GUIDE</span><h1>{t("学习路线与实验流程", "Learning path and experiment flow")}</h1><p>{t("从认识项目到复盘证据，按六个阶段理解一次六轴 S1 仿真训练。这里是教学展示；真正的编辑、提交和评估在对应工作区进行。", "Follow six stages from project setup to evidence review. This is a guide; editing, training and evaluation happen in their workspaces.")}</p></header>
    <div className="learning-guide-flow" data-tour="learning-roadmap" aria-label={t("六步学习路线", "Six-step learning path")}>{LEARNING_STEPS.map((item, index) => <button key={item.title} className={index === selectedStep ? "active" : ""} aria-current={index === selectedStep ? "step" : undefined} onClick={() => setSelectedStep(index)}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item.title}</strong><small>{item.subtitle}</small><ChevronRight size={15} /></button>)}</div>
    <div className="learning-guide-body"><article className="learning-guide-detail" data-tour="learning-current-step"><span className="eyebrow">STEP {String(selectedStep + 1).padStart(2, "0")} / 06</span><div className="learning-guide-title"><Icon size={24} /><h2>{step.title}</h2></div><p>{step.subtitle}</p><dl><div><dt>{t("在网页中做什么", "What to do")}</dt><dd>{step.action}</dd></div><div><dt>{t("会接触哪些文件", "Related files")}</dt><dd><code>{step.files}</code></dd></div><div><dt>{t("怎样确认学会", "What to check")}</dt><dd>{step.evidence}</dd></div><div><dt>{t("别误解为", "Important boundary")}</dt><dd>{step.caution}</dd></div></dl><div className="learning-guide-actions" data-tour="workspace-entry"><button className="primary" onClick={() => navigate(step.path)}>{t("打开对应工作区", "Open workspace")} <ExternalLink size={15} /></button><button className="secondary" onClick={() => window.dispatchEvent(new CustomEvent("studio-agent-open", { detail: { prompt: t(`请解释六轴 S1 仿真教学第 ${selectedStep + 1} 步“${step.title}”：${step.action}。说明要看的文件、完成判断和常见误区，不编造训练结果。`, `Explain S1 learning stage ${selectedStep + 1}, "${step.title}": ${step.action}. Describe files, completion criteria and pitfalls using actual evidence.`) } }))}>{t("问教学 Agent", "Ask teaching Agent")} <Bot size={15} /></button></div></article><aside className="learning-guide-aside" data-tour="training-routes"><span className="eyebrow">THREE WORKBENCH TASKS</span><h3>{t("三条工作台训练路线", "Three workbench training paths")}</h3><div><strong>S1 Push</strong><span>{t("推动方块到目标；24 维观测，二维平面动作。", "Push a cube to a goal; 24D observation, 2D planar action.")}</span></div><div><strong>S1 Reach</strong><span>{t("让末端到达目标区域；18 维观测，二维平面动作。", "Reach a target region; 18D observation, 2D planar action.")}</span></div><div><strong>S1 Place</strong><span>{t("从已抓取抬起状态搬运并保持一秒。", "Transport from a validated lifted grasp and hold for one second.")}</span></div><p>{t("控制 Demo 位于任务与技能。真实机械臂控制始终禁用。", "Control demos are in Tasks and skills. Real robot control remains disabled.")}</p><button className="secondary" onClick={() => navigate("/agent")}>{t("教学 Agent 设置", "Teaching Agent settings")} <ChevronRight size={14} /></button></aside></div>
  </main>;
}

function RoadmapPage() {
  const [selected, setSelected] = useState<RoadmapStageId>(
    ROADMAP_STAGES[0].id,
  );
  const [evidence, setEvidence] = useState<Record<string, boolean>>(() => {
    try {
      return JSON.parse(
        localStorage.getItem("embodied-arm-roadmap-evidence") ?? "{}",
      );
    } catch {
      return {};
    }
  });
  const active =
    ROADMAP_STAGES.find((stage) => stage.id === selected) ?? ROADMAP_STAGES[0];
  const completed = ROADMAP_STAGES.filter((stage) => evidence[stage.id]).length;

  function recordEvidence() {
    const next = { ...evidence, [active.id]: !evidence[active.id] };
    setEvidence(next);
    localStorage.setItem("embodied-arm-roadmap-evidence", JSON.stringify(next));
  }

  return (
    <main className="research-page roadmap-page">
      <header className="research-header">
        <div>
          <span className="eyebrow">RESEARCH ROADMAP</span>
          <h1>机械臂智能，从可复现实验开始</h1>
          <p>
            沿着环境、策略、训练与证据评估推进。当前可执行能力均明确标记为
            Mock，真实机械臂控制保持禁用。
          </p>
        </div>
        <div className="roadmap-progress">
          <span>EVIDENCE PROGRESS</span>
          <strong>
            {String(completed).padStart(2, "0")}
            <span> / {ROADMAP_STAGES.length}</span>
          </strong>
          <small>Verified evidence · 已记录证据</small>
          <div>
            <span
              style={{ width: `${(completed / ROADMAP_STAGES.length) * 100}%` }}
            />
          </div>
        </div>
      </header>
      <section className="roadmap-workspace">
        <div className="roadmap-list" role="list" aria-label="科研阶段">
          {ROADMAP_STAGES.map((stage, stageIndex) => (
            <button
              key={stage.id}
              className={`${selected === stage.id ? "active" : ""} ${evidence[stage.id] ? "completed" : ""}`.trim()}
              aria-current={selected === stage.id ? "step" : undefined}
              onClick={() => setSelected(stage.id)}
            >
              <span className="roadmap-index">
                {evidence[stage.id] ? <Check size={16} /> : stage.index}
              </span>
              <span>
                <small>
                  {stageIndex
                    ? `依赖阶段 ${String(stageIndex).padStart(2, "0")}`
                    : "起点"}
                </small>
                <strong>{stage.title}</strong>
                <em>{stage.summary}</em>
              </span>
              <ChevronRight size={18} />
            </button>
          ))}
        </div>
        <article className="stage-detail">
          <div className="stage-title">
            <span className={`capability ${active.capability.toLowerCase()}`}>
              {active.capability}
            </span>
            <span>STAGE {active.index}</span>
          </div>
          <h2>{active.title}</h2>
          <p>{active.summary}</p>
          <div className="evidence-block">
            <span className={evidence[active.id] ? "verified" : ""}>
              <Check size={14} /> REQUIRED EVIDENCE · 完成证据
            </span>
            <strong>{active.evidence}</strong>
            <p>
              证据必须关联到版本化制品或运行记录。仅勾选状态不会使研究阶段自动通过。
            </p>
          </div>
          <div className="stage-actions">
            {active.id === "theory" && (
              <button
                className="primary"
                onClick={() => navigate("/quick-course")}
              >
                <Play size={16} /> 进入快速课程
              </button>
            )}
            <button className="secondary" onClick={recordEvidence}>
              {evidence[active.id] ? (
                <>
                  <X size={16} /> 移除证据标记
                </>
              ) : (
                <>
                  <Check size={16} /> 记录已有证据
                </>
              )}
            </button>
          </div>
        </article>
        <aside className="research-status">
          <span className="eyebrow">ACTIVE CONTEXT</span>
          <h3>Mock Reach Baseline</h3>
          <dl>
            <div>
              <dt>环境</dt>
              <dd><button onClick={() => navigate("/environments")}>mock-reach · rev 0 <ChevronRight size={13} /></button></dd>
            </div>
            <div>
              <dt>策略</dt>
              <dd><button onClick={() => navigate("/training-code")}>joint-p-control <ChevronRight size={13} /></button></dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd>42</dd>
            </div>
            <div>
              <dt>真实控制</dt>
              <dd className="blocked">BLOCKED</dd>
            </div>
          </dl>
          <div className="boundary-note">
            <strong>CAPABILITY BOUNDARY · 能力边界</strong>
            <p>
              当前动力学为教学近似，不是数字孪生或物理精确仿真。Pipeline、复现实验与完整评估仍按真实运行证据进行。
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}

function PlannedPage({ path }: { path: string }) {
  const item = NAV_ITEMS.find((entry) => entry.path === path) ?? NAV_ITEMS[0];
  return (
    <main className="planned-page">
      <span className="eyebrow">PLANNED CAPABILITY</span>
      <item.icon size={34} />
      <h1>{item.label}</h1>
      <p>
        该工作区已进入升级路线，但尚无可执行实现。这里不会用界面占位来冒充已接入能力。
      </p>
      <button className="primary" onClick={() => navigate("/roadmap")}>
        <Map size={16} /> 返回科研路线
      </button>
    </main>
  );
}

function UserGuidePage({ path }: { path: string }) {
  const { locale, t } = useI18n();
  const sectionFromPath = path.startsWith("/guide/") ? path.slice("/guide/".length).split(/[?#]/, 1)[0] : "overview";
  const [selectedId, setSelectedId] = useState(sectionFromPath in USER_GUIDE_BY_ID ? sectionFromPath : "overview");
  const [query, setQuery] = useState("");
  const selected = USER_GUIDE_BY_ID[selectedId] ?? USER_GUIDE_BY_ID.overview;
  const currentSection = guideForPath(currentPath());
  const textIndex = locale === "en" ? 1 : 0;
  const matches = (section: GuideSection) => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return true;
    const values = [...section.title, ...section.summary, ...section.steps.flatMap((item) => [...item.title, ...item.body]), ...section.checks.flat(), ...section.boundaries.flat()];
    const principles = GUIDE_PRINCIPLES[section.id] ?? [];
    return [...values, ...principles.flat()].some((value) => value.toLocaleLowerCase().includes(needle));
  };
  useEffect(() => {
    const nextId = path.startsWith("/guide/") ? path.slice("/guide/".length).split(/[?#]/, 1)[0] : "overview";
    if (USER_GUIDE_BY_ID[nextId]) setSelectedId(nextId);
  }, [path]);
  const goTo = (section: GuideSection) => { setSelectedId(section.id); navigate(guidePathForSection(section.id)); };
  return <main className="guide-page">
    <header className="guide-header"><div><span className="eyebrow">EMBODIED ARM · USER GUIDE</span><h1>{t("完整使用文档", "Complete user guide")}</h1><p>{t("按研究流程学习平台功能、前置条件、完成标准和能力边界。文档由当前前端版本维护，AI 只在此受控内容上提供个性化解释。", "Learn the platform by workflow, prerequisites, completion criteria and capability boundaries. This guide is maintained with the frontend version; AI provides personalized explanations over this controlled content.")}</p></div><div className="guide-version"><span>DOCUMENT VERSION</span><strong>{USER_GUIDE_VERSION}</strong><small>{t("随前端版本发布", "Published with the frontend")}</small></div></header>
    <section className="guide-workspace"><aside className="guide-index"><label className="guide-search"><span>{t("搜索文档", "Search guide")}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("搜索章节、参数或边界", "Search sections, parameters or boundaries")} /></label><div className="guide-index-list" aria-label={t("文档章节", "Guide sections")}>{USER_GUIDE_SECTIONS.filter(matches).map((section) => <button key={section.id} className={section.id === selected.id ? "active" : ""} aria-current={section.id === selected.id ? "page" : undefined} onClick={() => goTo(section)}><span>{String(USER_GUIDE_SECTIONS.indexOf(section) + 1).padStart(2, "0")}</span><strong>{section.title[textIndex]}</strong><small>{section.summary[textIndex]}</small></button>)}{!USER_GUIDE_SECTIONS.some(matches) && <p className="guide-empty">{t("没有匹配章节", "No matching sections")}</p>}</div></aside><article className="guide-article"><div className="guide-article-meta"><span>{selected.route ? `${selected.route} · ${selected.tourId ?? "GUIDE"}` : "REFERENCE"}</span>{selected.id === currentSection.id && <em>{t("当前页面相关", "Current page")}</em>}</div><h2>{selected.title[textIndex]}</h2><p className="guide-summary">{selected.summary[textIndex]}</p><div className="guide-audience"><strong>{t("适合谁", "For")}</strong><span>{selected.audience[textIndex]}</span></div>{GUIDE_PRINCIPLES[selected.id]?.length && <section className="guide-principles"><span className="eyebrow">CONCEPTS &amp; PRINCIPLES</span>{GUIDE_PRINCIPLES[selected.id].reduce<Array<{ title: string; body: string }>>((items, item, index) => index % 2 === 0 ? [...items, { title: item[textIndex], body: GUIDE_PRINCIPLES[selected.id][index + 1]?.[textIndex] ?? "" }] : items, []).map((item) => <div className="guide-principle" key={item.title}><h3>{item.title}</h3><p>{item.body}</p></div>)}</section>}<div className="guide-section-block"><span className="eyebrow">STEP BY STEP</span>{selected.steps.map((item, index) => <section key={item.title[0]} className="guide-step"><span>{String(index + 1).padStart(2, "0")}</span><div><h3>{item.title[textIndex]}</h3><p>{item.body[textIndex]}</p></div></section>)}</div><div className="guide-two-column"><section><span className="eyebrow">CHECK BEFORE CONTINUE</span><ul>{selected.checks.map((item) => <li key={item[0]}>{item[textIndex]}</li>)}</ul></section><section className="guide-boundaries"><span className="eyebrow">CAPABILITY BOUNDARIES</span><ul>{selected.boundaries.map((item) => <li key={item[0]}>{item[textIndex]}</li>)}</ul></section></div><footer className="guide-article-footer"><span>{t("文档版本", "Guide version")} {USER_GUIDE_VERSION}</span>{selected.route && <button className="secondary" onClick={() => navigate(selected.route!)}>{t("打开相关工作区", "Open related workspace")} <ExternalLink size={14} /></button>}</footer></article></section>
  </main>;
}

const PROVIDER_GUIDE_URLS: Record<string, string> = {
  "openai-compatible": "https://platform.openai.com/docs/quickstart",
  deepseek: "https://api-docs.deepseek.com/",
  anthropic: "https://platform.claude.com/docs/en/api/overview",
};

function AgentSettingsPage() {
  const { locale, t } = useI18n();
  const queryClient = useQueryClient();
  const providers = useQuery({
    queryKey: ["agent-providers"],
    queryFn: () => api<AgentProviderInfo[]>("/api/agent/providers"),
  });
  const levels = useQuery({
    queryKey: ["agent-assistance-levels"],
    queryFn: () => api<AssistanceLevelInfo[]>("/api/agent/assistance-levels"),
  });
  const [providerId, setProviderId] = useState(
    () => localStorage.getItem("embodied-arm-agent-provider") ?? "mock",
  );
  const [level, setLevel] = useState(() =>
    Number(localStorage.getItem("embodied-arm-agent-level") ?? "4"),
  );
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [configBusy, setConfigBusy] = useState(false);
  const [configMessage, setConfigMessage] = useState("");
  const [preferenceSaved, setPreferenceSaved] = useState(false);
  const preferenceFeedbackTimer = useRef<number | null>(null);
  const providerCatalog = providers.data ?? [];
  const assistanceLevels = levels.data ?? [];
  const selectedAssistanceLevel = assistanceLevels.find((item) => item.level === level);
  const selectedProvider =
    providerCatalog.find((provider) => provider.id === providerId) ??
    providerCatalog[0];
  const localizeProviderText = (value: string) =>
    locale === "en" ? translateUiText(value) : value;

  useEffect(() => setConfigMessage(""), [locale]);

  useEffect(() => () => {
    if (preferenceFeedbackTimer.current !== null) {
      window.clearTimeout(preferenceFeedbackTimer.current);
    }
  }, []);

  useEffect(() => {
    if (!selectedProvider || selectedProvider.id === "mock") {
      setModel("");
      setBaseUrl("");
      return;
    }
    setModel(selectedProvider.models[0] ?? (selectedProvider.id === "deepseek" ? "deepseek-chat" : ""));
    setBaseUrl(selectedProvider.id === "deepseek" ? "https://api.deepseek.com" : selectedProvider.id === "openai-compatible" ? "https://api.openai.com/v1" : "");
    setApiKey("");
    setConfigMessage("");
  }, [selectedProvider?.id]);

  function savePreference() {
    localStorage.setItem("embodied-arm-agent-provider", providerId);
    localStorage.setItem("embodied-arm-agent-level", String(level));
    window.dispatchEvent(new Event("studio-agent-preference-changed"));
    setPreferenceSaved(false);
    window.requestAnimationFrame(() => setPreferenceSaved(true));
    if (preferenceFeedbackTimer.current !== null) {
      window.clearTimeout(preferenceFeedbackTimer.current);
    }
    preferenceFeedbackTimer.current = window.setTimeout(() => {
      setPreferenceSaved(false);
      preferenceFeedbackTimer.current = null;
    }, 1800);
  }

  function syncProviderConfiguration(updatedProvider: AgentProviderInfo) {
    queryClient.setQueryData<AgentProviderInfo[]>(["agent-providers"], (current) =>
      current?.map((provider) =>
        provider.id === updatedProvider.id ? updatedProvider : provider,
      ) ?? [updatedProvider],
    );
    window.dispatchEvent(new CustomEvent("studio-agent-provider-configuration-changed", {
      detail: { providerId: updatedProvider.id },
    }));
  }

  async function checkSelectedProvider() {
    if (!selectedProvider || selectedProvider.id === "mock") return;
    setConfigBusy(true);
    setConfigMessage("");
    try {
      const checked = await api<AgentProviderInfo>(`/api/agent/providers/${selectedProvider.id}/check`, { method: "POST" });
      queryClient.setQueryData<AgentProviderInfo[]>(["agent-providers"], (current) => current?.map((item) => item.id === checked.id ? checked : item) ?? [checked]);
      setConfigMessage(checked.status === "healthy" ? t("连接验证通过，模型已返回有效回复。", "Connection verified: the model returned a valid reply.") : localizeProviderText(checked.connection?.message ?? t("连接验证失败", "Connection verification failed")));
    } catch (caught) {
      setConfigMessage(localizeProviderText(caught instanceof Error ? caught.message : t("连接验证失败", "Connection verification failed")));
    } finally { setConfigBusy(false); }
  }

  async function configureSelectedProvider() {
    if (!selectedProvider || selectedProvider.id === "mock") return;
    setConfigBusy(true);
    setConfigMessage("");
    try {
      const updatedProvider = await api<AgentProviderInfo>(`/api/agent/providers/${selectedProvider.id}/configuration`, {
        method: "PUT",
        body: JSON.stringify({ api_key: apiKey.trim(), model: model.trim(), base_url: baseUrl.trim() }),
      });
      syncProviderConfiguration(updatedProvider);
      setApiKey("");
      const checked = await api<AgentProviderInfo>(`/api/agent/providers/${selectedProvider.id}/check`, { method: "POST" });
      queryClient.setQueryData<AgentProviderInfo[]>(["agent-providers"], (current) => current?.map((item) => item.id === checked.id ? checked : item) ?? [checked]);
      setConfigMessage(checked.status === "healthy" ? t("临时配置已保存，连接验证通过。API 重启后需要重新配置。", "Temporary configuration saved and connection verified. Configure again after an API restart.") : t("配置已保存，但连接验证失败：", "Configuration saved, but connection verification failed: ") + localizeProviderText(checked.connection?.message ?? ""));
    } catch (caught) {
      setConfigMessage(localizeProviderText(caught instanceof Error ? caught.message : t("Provider 配置失败", "Provider configuration failed")));
    } finally { setConfigBusy(false); }
  }

  async function clearSelectedProvider() {
    if (!selectedProvider || selectedProvider.id === "mock") return;
    setConfigBusy(true);
    setConfigMessage("");
    try {
      const updatedProvider = await api<AgentProviderInfo>(`/api/agent/providers/${selectedProvider.id}/configuration`, { method: "DELETE" });
      syncProviderConfiguration(updatedProvider);
      setApiKey("");
      setConfigMessage(t("已清除当前 API 进程中的临时配置。", "Temporary configuration cleared from the current API process."));
    } catch (caught) {
      setConfigMessage(caught instanceof Error ? caught.message : t("Provider 配置清除失败", "Failed to clear Provider configuration"));
    } finally {
      setConfigBusy(false);
    }
  }

  if (providers.isLoading || levels.isLoading)
    return <div className="loading">正在读取服务端 Agent 能力...</div>;
  if (providers.isError || levels.isError || !selectedProvider)
    return (
      <div className="loading error">
        Agent Provider API 不可用，请检查后端服务。
      </div>
    );

  return (
    <main className="agent-settings-page">
      <header className="agent-settings-header">
        <div>
          <span className="eyebrow">TEACHING AGENT CONTROL</span>
          <h1>{t("模型来源与教学边界", "Model providers and teaching boundaries")}</h1>
          <p>
            {t("选择会话使用的模型来源和帮助强度。API Key 可在此临时配置到当前 API 进程，浏览器不会保存、读取或导出密钥。", "Choose a model provider and assistance level. API keys are temporary in the current API process and are never saved or exported by the browser.")}
          </p>
        </div>
        <div className="agent-boundary">
          <ShieldCheck size={19} />
          <span>
            <strong>GENERIC SHELL DISABLED</strong>
            <small>REAL ROBOT CONTROL DISABLED</small>
          </span>
        </div>
      </header>
      <section className="agent-settings-workspace">
        <aside className="provider-list" data-tour="agent-provider-list">
          <div className="section-label">AVAILABLE PROVIDERS</div>
          {providerCatalog.map((provider) => (
            <button
              key={provider.id}
              className={provider.id === selectedProvider.id ? "active" : ""}
              onClick={() => setProviderId(provider.id)}
              disabled={false}
            >
              <Bot size={18} />
              <span>
                <strong>{localizeProviderText(provider.name)}</strong>
                <small>{provider.id}</small>
              </span>
              <em className={provider.status}>{provider.id === "mock" ? "MOCK" : provider.status === "healthy" ? t("已验证", "Verified") : provider.status === "degraded" ? t("连接失败", "Connection failed") : provider.configured ? t("待验证", "Untested") : t("未配置", "Not configured")}</em>
            </button>
          ))}
        </aside>
        <section className="provider-detail" data-tour="agent-provider-profile">
          <span className="eyebrow">PROVIDER PROFILE</span>
          <h2>{localizeProviderText(selectedProvider.name)}</h2>
          <p>{localizeProviderText(selectedProvider.data_scope)}</p>
          {selectedProvider.id !== "mock" && (
            <div className="provider-config-form">
              <div className="provider-config-heading">
                <div className="section-label">RUNTIME PROVIDER CONFIG</div>
                {PROVIDER_GUIDE_URLS[selectedProvider.id] && (
                  <a href={PROVIDER_GUIDE_URLS[selectedProvider.id]} target="_blank" rel="noreferrer">
                    {t("官方 API 配置指南", "Official API setup guide")} <ExternalLink size={13} />
                  </a>
                )}
              </div>
              <label className="field">
                <span>API Key <small>{t("仅发送至当前 API，不会回显", "Sent only to the current API; never echoed")}</small></span>
                <input type="password" value={apiKey} autoComplete="new-password" onChange={(event) => setApiKey(event.target.value)} placeholder={t("输入后配置；不会保存到浏览器", "Enter a key; it is never saved in the browser")} />
              </label>
              <label className="field">
                <span>{t("模型 ID", "Model ID")}</span>
                <input value={model} onChange={(event) => setModel(event.target.value)} placeholder={selectedProvider.id === "deepseek" ? "deepseek-chat" : "e.g. gpt-4o-mini"} />
              </label>
              {selectedProvider.id !== "anthropic" && (
                <label className="field">
                  <span>{t("兼容 API 根地址", "Compatible API base URL")} <small>{t("必须 HTTPS，后端禁止内网地址", "HTTPS required; private network addresses are blocked")}</small></span>
                  <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} placeholder={selectedProvider.id === "deepseek" ? "https://api.deepseek.com" : "https://api.openai.com/v1"} />
                </label>
              )}
              <p className="provider-connection-note">{t("配置完整不代表连接成功。验证会发送一条固定测试消息，不包含页面或会话内容。", "Complete configuration does not prove connectivity. Verification sends a fixed test message without page or session content.")}</p><div className="provider-config-actions" data-tour="agent-connection-check">
                <button className="primary" onClick={() => void configureSelectedProvider()} disabled={configBusy || !apiKey.trim() || !model.trim()}>
                  <Check size={16} /> {configBusy ? t("验证中…", "Verifying…") : t("配置并验证", "Configure and verify")}
                </button>
                <button onClick={() => void checkSelectedProvider()} disabled={configBusy || !selectedProvider.configured}>{t("重新验证连接", "Verify connection again")}</button><button onClick={() => void clearSelectedProvider()} disabled={configBusy || !selectedProvider.configured}>{t("清除临时配置", "Clear temporary configuration")}</button>
              </div>
              {(configMessage || selectedProvider.connection?.message) && <p className="provider-config-message" role="status">{configMessage || localizeProviderText(selectedProvider.connection?.message ?? "")}</p>}
            </div>
          )}
          <dl>
            <div>
              <dt>{t("服务端配置", "Server configuration")}</dt>
              <dd>
                {selectedProvider.configured ? "READY" : "NOT CONFIGURED"}
              </dd>
            </div>
            <div>
              <dt>{t("对话", "Chat")}</dt>
              <dd>{selectedProvider.capabilities.chat ? "YES" : "NO"}</dd>
            </div>
            <div>
              <dt>{t("结构化输出", "Structured output")}</dt>
              <dd>
                {selectedProvider.capabilities.structured_output ? "YES" : "NO"}
              </dd>
            </div>
            <div>
              <dt>{t("Provider 工具调用", "Provider tool calls")}</dt>
              <dd>{selectedProvider.capabilities.tool_calls ? "YES" : "NO"}</dd>
            </div>
          </dl>
          <div className="privacy-note">
            <ShieldCheck size={18} />
            <span>
              <strong>{t("密钥隔离", "Credential isolation")}</strong>
              <p>{t("API 仅返回 configured 状态。密钥值、Authorization header 和服务端 secret reference 不进入此页面。", "The API returns only the configured status. Key values, Authorization headers, and server-side secret references never enter this page.")}</p>
            </span>
          </div>
        </section>
        <aside className="assistance-control" data-tour="assistance-level">
          <div className="section-label">ASSISTANCE LEVEL</div>
          <div className="level-selector">
            {assistanceLevels.map((item) => (
              <button
                key={item.level}
                className={level === item.level ? "active" : ""}
                onClick={() => setLevel(item.level)}
                disabled={false}
              >
                <span>{item.level}</span>
                <strong>{item.id.replace(/_/g, " ")}</strong>
                {item.requires_teacher_authorization && (
                  <small>TEACHER ONLY</small>
                )}
              </button>
            ))}
          </div>
          <div className="level-summary">
            <strong>Level {level}</strong>
            <p>
              {(selectedAssistanceLevel?.capabilities ?? []).map((capability) => ({
                explain: t("解释", "Explain"),
                socratic_hints: t("Socratic 提示", "Socratic hints"),
                review_student_proposal: t("审查学生方案", "Review student proposal"),
                read_page_context: t("读取当前页面", "Read current page"),
                read_workspace: t("读取受限 Workspace", "Read restricted workspace"),
                read_scene: t("读取 Scene", "Read scene"),
                read_run_evidence: t("读取 Run evidence", "Read run evidence"),
                suggest_configuration: t("配置建议", "Suggest configuration"),
                propose_patch: t("生成受审批 Patch 提案", "Propose an approval-gated patch"),
              } as Record<string, string>)[capability] ?? capability).join(" / ")}
            </p>
          </div>
          <button
            data-tour="save-agent-preference"
            className={`primary preference-save${preferenceSaved ? " saved" : ""}`}
            onClick={savePreference}
          >
            <Check size={16} />
            <span aria-live="polite">
              {preferenceSaved ? t("已保存", "Saved") : t("保存会话偏好", "Save session preferences")}
            </span>
          </button>
        </aside>
      </section>
    </main>
  );
}

function workspaceLanguage(path: string) {
  const extension = path.split(".").pop()?.toLowerCase();
  return extension === "py"
    ? "python"
    : extension === "json"
      ? "json"
      : extension === "yaml" || extension === "yml"
        ? "yaml"
        : extension === "cpp"
          ? "cpp"
          : extension === "md"
            ? "markdown"
            : "plaintext";
}

type BlueprintTask = "s1_push" | "s1_reach";

const TRAINING_BLUEPRINTS: Record<BlueprintTask, {
  name: string;
  goal: string;
  observation: string;
  action: string;
  taskSource: string;
  policySource: string;
  steps: Array<{ title: string; detail: string; files: string; route: string; status: string }>;
}> = {
  s1_push: {
    name: "S1 Push · 推物", goal: "推动桌面方块进入目标区域", observation: "24 维：关节、末端/物体/目标相对位置和速度", action: "二维末端 XY 位移；Z 和姿态固定",
    taskSource: "simulation/src/simulation_core/push_adapter.py", policySource: "simulation/src/simulation_core/ppo/policy.py",
    steps: [
      { title: "场景和机械臂", detail: "填写 scene.yaml 作为设计草稿；在 S1 环境编辑中保存实际场景修订。", files: "tasks/s1_push/scene.yaml · S1 Scene revision", route: "/environments", status: "场景可编辑" },
      { title: "任务定义", detail: "在 task.py 描述 reset、observe、step；正式逻辑仍由 Simulation 服务实现。", files: "tasks/s1_push/task.py · push_adapter.py", route: "/projects", status: "草稿可编辑" },
      { title: "观测 / 动作", detail: "在 policy.py 描述 24 维输入与二维 XY 输出；正式 PPO 网络不读取此草稿。", files: "tasks/s1_push/policy.py · ppo/policy.py", route: "/training-code", status: "草稿可编辑" },
      { title: "奖励 / 终止", detail: "在 reward.py 写下进度、成功与超时条件，再核对平台的正式 rubric。", files: "tasks/s1_push/reward.py · Project rubric", route: "/projects", status: "草稿可编辑" },
      { title: "PPO 训练", detail: "在 train.py 梳理训练流程；选 Push Project，冻结参数和 seed，再启动 GPU Worker。", files: "tasks/s1_push/train.py · Project config · training_config.json", route: "/training", status: "正式训练" },
      { title: "运行与评估", detail: "在 evaluate.py 记录评估方法；实际指标、checkpoint 与报告由 Run 生成。", files: "tasks/s1_push/evaluate.py · manifest.json · checkpoints/*.pt", route: "/runs", status: "运行后生成" },
    ],
  },
  s1_reach: {
    name: "S1 Reach · 平面到达", goal: "末端到达目标区域并保持", observation: "18 维：关节状态、末端与目标位置及运动状态", action: "二维末端 XY 位移；不是抓取",
    taskSource: "simulation/src/simulation_core/reach_adapter.py（在线运行副本）", policySource: "simulation/src/simulation_core/ppo/policy.py",
    steps: [
      { title: "场景和机械臂", detail: "填写 scene.yaml 作为设计草稿；在 S1 环境编辑中保存六轴场景修订。", files: "tasks/s1_reach/scene.yaml · S1 Scene revision", route: "/environments", status: "场景可编辑" },
      { title: "到达任务", detail: "在 task.py 定义目标中心、区域、距离阈值和保持步数；正式任务由 Reach adapter 实现。", files: "tasks/s1_reach/task.py · reach_adapter.py", route: "/projects", status: "草稿可编辑" },
      { title: "观测 / 动作", detail: "在 policy.py 描述 18 维输入与二维末端动作；平台 PPO 仍使用服务端实现。", files: "tasks/s1_reach/policy.py · ppo/policy.py", route: "/training-code", status: "草稿可编辑" },
      { title: "奖励 / 终止", detail: "在 reward.py 描述距离进度、持续到达和超时，并对照正式 rubric。", files: "tasks/s1_reach/reward.py · Project rubric", route: "/projects", status: "草稿可编辑" },
      { title: "PPO 训练", detail: "在 train.py 梳理训练流程；选 Reach Project，配置 seed 并提交 GPU 训练。", files: "tasks/s1_reach/train.py · Project config · training_config.json", route: "/training", status: "正式训练" },
      { title: "运行与评估", detail: "在 evaluate.py 记录评估方法；查看实际目标距离、成功率和 checkpoint。", files: "tasks/s1_reach/evaluate.py · manifest.json · checkpoints/*.pt", route: "/runs", status: "运行后生成" },
    ],
  },
};

const CONTROL_PARTS = [
  ["目标位姿", "目标 XY 偏移", "demo_s1_target_pose", "demos/target_pose.py"],
  ["直线跟踪", "方向 / 线长 / 周期", "demo_s1_line_tracking", "demos/line_tracking.py"],
  ["圆轨迹跟踪", "半径 / 周期", "demo_s1_circle_tracking", "demos/circle_tracking.py"],
  ["关节定位", "home / ready / side", "demo_s1_joint_position", "demos/joint_position.py"],
  ["关节扫描", "关节编号 / 速度", "demo_s1_joint_sweep", "demos/joint_sweep.py"],
] as const;

const BLUEPRINT_EXAMPLES: Record<BlueprintTask, string[]> = {
  s1_push: [
    "scene: S1 URDF/STL + table + movable cube + goal\noutput: saved Scene revision (visual preview only)",
    "task: reset robot, cube and goal\nstep: apply bounded XY action, advance physics\noutput: observation, reward, done, diagnostics",
    "observation: 24 values (arm_q[6], arm_dq[6], 6 XY pairs)\naction: [delta_x, delta_y] bounded to +/-0.005 m\npolicy: actor 24 -> 64 -> 64 -> 2; critic 24 -> 64 -> 64 -> 1",
    "reward: object-to-goal progress + contact/success terms\ntermination: goal reached or episode limit\nevidence: success_rate, timeout_rate, object_displacement",
    "input: matching Push Project + seed + S1 Scene revision\nserver: freeze configuration -> Redis/RQ -> Isaac Gym PPO\noutput: model_latest.pt and training metrics",
    "review: Run events + metric history + artifact manifest\ncompare: reward, success_rate, final_distance\nnext: generate evaluation from real evidence",
  ],
  s1_reach: [
    "scene: S1 URDF/STL + workspace + target region\noutput: saved Scene revision (visual preview only)",
    "task: sample target_center_xy_m within workspace\nstep: move end effector in XY and update observation\noutput: distance-to-target, reward, done",
    "observation: robot joints + end effector / target geometry\naction: bounded planar end-effector delta\npolicy: PPO actor and value networks (runtime-owned)",
    "reward: XY distance progress + held-success bonus\ntermination: within success_distance_m for success_hold_steps or timeout\nevidence: goal distance, success and timeout",
    "input: matching Reach Project + seed + S1 Scene revision\nserver: freeze configuration -> Redis/RQ -> Isaac Gym PPO\noutput: model_latest.pt and training metrics",
    "review: Run events + distance/success evidence\ninspect: logs, metrics and registered checkpoint\nnext: generate evaluation from real evidence",
  ],
};

const WORKSPACE_GUIDES: Record<string, string> = {
  "policy_overview.md": "# S1 training project\n\nChoose Push or Reach in the project blueprint. Describe observations, actions and success conditions here.\n\nThis document is not executed by the PPO trainer.\n",
  "network_architecture.json": '{\n  "algorithm": "PPO",\n  "actor": "MLP",\n  "critic": "MLP",\n  "notes": "Teaching reference; the server owns the executable policy."\n}\n',
  "training_config.yaml": "task: s1_push_or_s1_reach\ntraining_profile: frozen_by_project\nseed: supplied_at_run_creation\n# This file is a teaching reference, not a Run input.\n",
  "reward_notes.md": "# Reward and termination\n\nWrite down progress, success, timeout and safety assumptions. The formal task reward remains server-owned.\n",
};

const TASK_DRAFT_FILES = (taskId: BlueprintTask): Record<string, string> => {
  const task = TRAINING_BLUEPRINTS[taskId];
  const root = `tasks/${taskId}`;
  return {
    [`${root}/README.md`]: `# ${task.name}\n\nGoal: ${task.goal}\nObservation: ${task.observation}\nAction: ${task.action}\n\nEdit these files to describe a proposed task. The current formal Run uses the platform implementation, not these drafts.\n`,
    [`${root}/scene.yaml`]: `robot: s1_v2\narm_dof: 6\ngripper_dof: 2\ntask: ${taskId}\n# Scene geometry is saved separately in the S1 Environment editor.\n`,
    [`${root}/task.py`]: `"""Task interface sketch for ${taskId}; never executed by the platform."""\n\ndef reset(seed):\n    raise NotImplementedError("Define initial robot, object and target state")\n\ndef observe(state):\n    raise NotImplementedError("Define observation fields and shapes")\n\ndef step(state, action):\n    raise NotImplementedError("Apply bounded action in simulation")\n`,
    [`${root}/policy.py`]: `"""Policy interface sketch for ${taskId}; never executed by the platform."""\n\ndef actor(observation):\n    raise NotImplementedError("Map observations to bounded XY action")\n\ndef critic(observation):\n    raise NotImplementedError("Estimate state value for PPO")\n`,
    [`${root}/reward.py`]: `"""Reward and termination sketch for ${taskId}; never executed."""\n\ndef reward(previous_state, state, action):\n    raise NotImplementedError("Define progress, success and penalty terms")\n\ndef done(state, steps):\n    raise NotImplementedError("Define success and time-limit conditions")\n`,
    [`${root}/train.py`]: `"""Training pipeline sketch for ${taskId}; never executed."""\n\ndef train(project_snapshot, seed):\n    raise NotImplementedError("Connect environment, PPO and artifact writer")\n`,
    [`${root}/evaluate.py`]: `"""Evaluation sketch for ${taskId}; never executed."""\n\ndef evaluate(checkpoint, seed):\n    raise NotImplementedError("Measure success, timeout and task-specific distance")\n`,
  };
};

function workspaceContainsPath(nodes: WorkspaceNode[], path: string): boolean {
  return nodes.some((node) => node.path === path || workspaceContainsPath(node.children ?? [], path));
}

function ProjectWorkspacePage({ initialView = "blueprint" }: { initialView?: "blueprint" | "files" }) {
  const { t } = useI18n();
  const [blueprintTask, setBlueprintTask] = useState<BlueprintTask>("s1_push");
  const [workspaceView, setWorkspaceView] = useState<"blueprint" | "files">(initialView);
  const [activeBlueprintStep, setActiveBlueprintStep] = useState(0);
  const blueprint = TRAINING_BLUEPRINTS[blueprintTask];
  const [projectId, setProjectId] = useState("");
  const [tree, setTree] = useState<WorkspaceNode[]>([]);
  const [tabs, setTabs] = useState<
    Array<{ path: string; content: string; saved: string }>
  >([]);
  const [activePath, setActivePath] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<
    Array<{ path: string; line: number; preview: string }>
  >([]);
  const [message, setMessage] = useState(() => t("正在初始化受限工作区...", "Initializing restricted workspace..."));
  const [snapshotId, setSnapshotId] = useState("");
  const [history, setHistory] = useState<
    Array<{ version_id: string; created_at: string; content: string }>
  >([]);
  const [diff, setDiff] = useState<{
    baseline: string;
    current: string;
    status: string;
  } | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const activeTab = tabs.find((tab) => tab.path === activePath);

  async function refreshTree(targetProject = projectId) {
    if (!targetProject) return;
    const workspace = await api<{ tree: WorkspaceNode[] }>(
      `/api/v1/workspaces/${targetProject}`,
    );
    setTree(workspace.tree);
  }

  async function addGuides() {
    if (!projectId || !window.confirm(t("先创建快照，再补充 Push 和 Reach 两套教学草稿？已有文件不会被覆盖，草稿不会被执行。", "Create a snapshot and add Push/Reach teaching drafts? Existing files stay untouched; drafts are not executed."))) return;
    try {
      const snapshot = await api<{ snapshot_id: string }>(`/api/v1/workspaces/${projectId}/snapshots`, { method: "POST" });
      setSnapshotId(snapshot.snapshot_id);
      let added = 0;
      const guides = { ...WORKSPACE_GUIDES, ...TASK_DRAFT_FILES("s1_push"), ...TASK_DRAFT_FILES("s1_reach") };
      for (const [path, content] of Object.entries(guides)) {
        const current = await api<{ tree: WorkspaceNode[] }>(`/api/v1/workspaces/${projectId}`);
        if (workspaceContainsPath(current.tree, path)) continue;
        await api(`/api/v1/workspaces/${projectId}/files/${path}`, { method: "PUT", body: JSON.stringify({ content }) });
        added += 1;
      }
      await refreshTree();
      setWorkspaceView("files");
      setMessage(t(`已添加 ${added} 个教学文件；回滚快照 ${snapshot.snapshot_id}。`, `Added ${added} teaching files; rollback snapshot ${snapshot.snapshot_id}.`));
    } catch (error) {
      await refreshTree();
      setMessage(error instanceof Error ? error.message : t("创建教学文件失败", "Could not create teaching files"));
    }
  }

  useEffect(() => {
    api<WorkspaceProject[]>("/api/v1/projects")
      .then(async (projects) => {
        const selected =
          projects[0]?.project_id ??
          (
            await api<{ project_id: string }>("/api/v1/projects", {
              method: "POST",
              body: JSON.stringify({ project_id: "research-workspace" }),
            })
          ).project_id;
        setProjectId(selected);
        const workspace = await api<{ tree: WorkspaceNode[] }>(
          `/api/v1/workspaces/${selected}`,
        );
        setTree(workspace.tree);
        setMessage(t("工作区已连接；文件写入受类型、大小和路径策略限制。", "Workspace connected; file writes are limited by type, size, and path policy."));
      })
      .catch((error) =>
        setMessage(error instanceof Error ? error.message : t("工作区初始化失败", "Workspace initialization failed")),
      );
  }, []);

  async function openFile(path: string) {
    setWorkspaceView("files");
    const existing = tabs.find((tab) => tab.path === path);
    if (!existing) {
      const file = await api<{ content: string }>(
        `/api/v1/workspaces/${projectId}/files/${path}`,
      );
      setTabs((current) => [
        ...current,
        { path, content: file.content, saved: file.content },
      ]);
    }
    setActivePath(path);
    setShowDiff(false);
    setHistory(
      await api(`/api/v1/workspaces/${projectId}/history/${path}`),
    );
  }

  function openBlueprintStep() {
    const step = blueprint.steps[activeBlueprintStep];
    if (step.route !== "/projects" && step.route !== "/training-code") {
      navigate(step.route);
      return;
    }
    const filename = activeBlueprintStep === 1 ? "task.py" : activeBlueprintStep === 2 ? "policy.py" : "reward.py";
    const path = `tasks/${blueprintTask}/${filename}`;
    if (workspaceContainsPath(tree, path)) {
      void openFile(path);
    } else {
      setWorkspaceView("files");
      setMessage(t(`尚无 ${path}；请返回“搭建步骤”并使用“补齐两套任务草稿（不覆盖）”。`, `Missing ${path}. Return to Build steps and add the task drafts first.`));
    }
  }

  async function persistFile(path: string, content: string, automatic = false) {
    await api(`/api/v1/workspaces/${projectId}/files/${path}`, {
      method: "PUT",
      body: JSON.stringify({ content }),
    });
    setTabs((current) =>
      current.map((tab) =>
        tab.path === path && tab.content === content
          ? { ...tab, saved: content }
          : tab,
      ),
    );
    setMessage(`${automatic ? t("已自动保存", "Auto-saved") : t("已保存", "Saved")} ${path}`);
    await refreshTree();
    if (path === activePath) {
      setHistory(await api(`/api/v1/workspaces/${projectId}/history/${path}`));
    }
  }

  async function saveFile() {
    if (!activeTab) return;
    await persistFile(activeTab.path, activeTab.content);
  }

  useEffect(() => {
    if (!projectId || !activeTab || activeTab.content === activeTab.saved) return;
    const { path, content } = activeTab;
    setMessage(`${t("等待自动保存", "Waiting for auto-save")} ${path}`);
    const timer = window.setTimeout(() => {
      setMessage(`${t("正在自动保存", "Auto-saving")} ${path}`);
      persistFile(path, content, true).catch((error) =>
        setMessage(error instanceof Error ? error.message : `${t("自动保存失败", "Auto-save failed")} ${path}`),
      );
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [activeTab?.content, activeTab?.path, activeTab?.saved, projectId]);

  async function createEntry(kind: "file" | "directory") {
    const path = window.prompt(
      kind === "file"
        ? t("输入新文件路径（支持 py/json/yaml/md/cpp/txt）", "Enter a new file path (py/json/yaml/md/cpp/txt supported)")
        : t("输入新目录路径", "Enter a new directory path"),
    );
    if (!path) return;
    if (kind === "file")
      await api(`/api/v1/workspaces/${projectId}/files/${path}`, {
        method: "PUT",
        body: JSON.stringify({ content: "" }),
      });
    else
      await api(`/api/v1/workspaces/${projectId}/directories`, {
        method: "POST",
        body: JSON.stringify({ path }),
      });
    await refreshTree();
    if (kind === "file") await openFile(path);
  }

  async function renameActive() {
    if (!activeTab) return;
    const destination = window.prompt(t("输入新的工作区路径", "Enter the new workspace path"), activeTab.path);
    if (!destination || destination === activeTab.path) return;
    await api(`/api/v1/workspaces/${projectId}/move`, {
      method: "POST",
      body: JSON.stringify({ source: activeTab.path, destination }),
    });
    setTabs((current) =>
      current.map((tab) =>
        tab.path === activeTab.path ? { ...tab, path: destination } : tab,
      ),
    );
    setActivePath(destination);
    await refreshTree();
  }

  async function deleteActive() {
    if (
      !activeTab ||
      !window.confirm(t(`删除 ${activeTab.path}？该操作会写入审计记录。`, `Delete ${activeTab.path}? This action will be recorded in the audit log.`))
    )
      return;
    await api(`/api/v1/workspaces/${projectId}/files/${activeTab.path}`, {
      method: "DELETE",
    });
    setTabs((current) => current.filter((tab) => tab.path !== activeTab.path));
    setActivePath("");
    await refreshTree();
  }

  async function searchWorkspace() {
    if (!searchQuery.trim()) return;
    setSearchResults(
      await api(
        `/api/v1/workspaces/${projectId}/search?q=${encodeURIComponent(searchQuery)}`,
      ),
    );
  }

  async function createSnapshot() {
    const result = await api<{ snapshot_id: string }>(
      `/api/v1/workspaces/${projectId}/snapshots`,
      { method: "POST" },
    );
    setSnapshotId(result.snapshot_id);
    setMessage(`${t("已创建快照", "Snapshot created")} ${result.snapshot_id}`);
  }

  async function restoreSnapshot() {
    if (
      !snapshotId ||
      !window.confirm(t(`恢复 ${snapshotId}？当前未快照变更将丢失。`, `Restore ${snapshotId}? Changes made after the snapshot will be lost.`))
    )
      return;
    await api(
      `/api/v1/workspaces/${projectId}/snapshots/${snapshotId}/restore`,
      { method: "POST" },
    );
    setTabs([]);
    setActivePath("");
    await refreshTree();
    setMessage(`${t("已恢复快照", "Snapshot restored")} ${snapshotId}`);
  }

  async function loadDiff() {
    if (!activeTab) return;
    const result = await api<{
      baseline: string;
      current: string;
      status: string;
    }>(
      `/api/v1/workspaces/${projectId}/diff?path=${encodeURIComponent(activeTab.path)}`,
    );
    setDiff(result);
    setShowDiff(true);
  }

  async function restoreVersion(versionId: string) {
    if (!activeTab || !window.confirm(t(`恢复版本 ${versionId}？`, `Restore version ${versionId}?`))) return;
    await api(
      `/api/v1/workspaces/${projectId}/history/${activeTab.path}/versions/${versionId}/restore`,
      { method: "POST" },
    );
    const file = await api<{ content: string }>(
      `/api/v1/workspaces/${projectId}/files/${activeTab.path}`,
    );
    setTabs((current) =>
      current.map((tab) =>
        tab.path === activeTab.path
          ? { ...tab, content: file.content, saved: file.content }
          : tab,
      ),
    );
    setHistory(
      await api(`/api/v1/workspaces/${projectId}/history/${activeTab.path}`),
    );
    setMessage(`${t("已恢复历史版本", "Historical version restored")} ${activeTab.path}`);
  }

  async function restoreStarter() {
    if (!window.confirm(t("恢复 Starter 模板？系统会先创建可回滚快照。", "Restore the Starter template? A rollback snapshot will be created first."))) return;
    const result = await api<{ snapshot_id: string }>(
      `/api/v1/workspaces/${projectId}/starter/restore`,
      { method: "POST" },
    );
    setSnapshotId(result.snapshot_id);
    setTabs([]);
    setActivePath("");
    setHistory([]);
    setShowDiff(false);
    await refreshTree();
    setMessage(`${t("Starter 已恢复；回滚快照", "Starter restored; rollback snapshot")} ${result.snapshot_id}`);
  }

  function downloadActive() {
    if (!activeTab) return;
    const url = URL.createObjectURL(
      new Blob([activeTab.content], { type: "text/plain;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = activeTab.path.split("/").pop() ?? "workspace-file.txt";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function uploadFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const path = window.prompt(t("上传到工作区路径", "Upload to workspace path"), file.name);
    if (!path) return;
    await api(`/api/v1/workspaces/${projectId}/files/${path}`, {
      method: "PUT",
      body: JSON.stringify({ content: await file.text() }),
    });
    await refreshTree();
    await openFile(path);
    event.target.value = "";
  }

  function renderTree(nodes: WorkspaceNode[], depth = 0): React.ReactNode {
    return nodes.map((node) => (
      <div key={node.path}>
        {node.type === "directory" ? (
          <>
            <div
              className="workspace-tree-row directory"
              style={{ paddingLeft: 12 + depth * 14 }}
            >
              <Folder size={14} />
              <span>{node.name}</span>
            </div>
            {renderTree(node.children ?? [], depth + 1)}
          </>
        ) : (
          <button
            className={`workspace-tree-row ${activePath === node.path ? "active" : ""}`}
            style={{ paddingLeft: 12 + depth * 14 }}
            onClick={() => openFile(node.path)}
          >
            <File size={14} />
            <span>{node.name}</span>
            <small>{node.size ?? 0} B</small>
          </button>
        )}
      </div>
    ));
  }

  return (
    <main className="project-workspace-page">
      <header className="project-workspace-header">
        <div>
          <span className="eyebrow">PROJECT WORKSPACE · V1</span>
          <h1>{t("项目与文件工作区", "Project and file workspace")}</h1>
          <p>
            {projectId || "正在创建项目"} · {message}
          </p>
        </div>
        <div className="workspace-actions">
          <button title={t("新建文件", "New file")} onClick={() => createEntry("file")}>
            <FilePlus2 size={16} />
          </button>
          <button title={t("新建目录", "New directory")} onClick={() => createEntry("directory")}>
            <FolderPlus size={16} />
          </button>
          <label title={t("上传文本文件", "Upload text file")}>
            <Upload size={16} />
            <input type="file" onChange={uploadFile} />
          </label>
          <button
            title={t("下载当前文件", "Download current file")}
            onClick={downloadActive}
            disabled={!activeTab}
          >
            <Download size={16} />
          </button>
          <button
            title={t("重命名当前文件", "Rename current file")}
            onClick={renameActive}
            disabled={!activeTab}
          >
            <Pencil size={16} />
          </button>
          <button
            title={t("删除当前文件", "Delete current file")}
            onClick={deleteActive}
            disabled={!activeTab}
          >
            <Trash2 size={16} />
          </button>
          <button title={t("对比 Starter", "Compare with Starter")} onClick={loadDiff} disabled={!activeTab}>
            <GitCompare size={16} />
          </button>
          <button
            className="primary"
            onClick={saveFile}
            disabled={!activeTab || activeTab.content === activeTab.saved}
          >
            <Save size={15} /> {t("保存", "Save")}
          </button>
        </div>
      </header>
      <section className="project-blueprint" aria-label={t("仿真训练项目路线", "Simulation training project roadmap")}>
        <div className="blueprint-topline"><div><span className="eyebrow">S1 · SIX AXIS / PPO</span><h2>{t("从场景到策略", "From scene to policy")}</h2></div><div className="blueprint-switch" data-tour="project-tasks" role="group" aria-label={t("强化学习任务", "Reinforcement learning task")}>{(["s1_push", "s1_reach"] as const).map((id) => <button key={id} className={blueprintTask === id ? "active" : ""} aria-pressed={blueprintTask === id} onClick={() => setBlueprintTask(id)}>{TRAINING_BLUEPRINTS[id].name}</button>)}</div></div>
        <div className="blueprint-summary"><strong>{blueprint.goal}</strong><span>{t("观测", "Observation")}: {blueprint.observation}</span><span>{t("动作", "Action")}: {blueprint.action}</span><span>{t("任务实现", "Task implementation")}: {blueprint.taskSource}</span><span>{t("策略源码", "Policy source")}: {blueprint.policySource}</span></div>
        <div className="blueprint-view-switch" data-tour="project-files" role="tablist"><button role="tab" aria-selected={workspaceView === "blueprint"} className={workspaceView === "blueprint" ? "active" : ""} onClick={() => setWorkspaceView("blueprint")}>{t("搭建步骤", "Build steps")}</button><button role="tab" aria-selected={workspaceView === "files"} className={workspaceView === "files" ? "active" : ""} onClick={() => setWorkspaceView("files")}>{t("我的工作区文件", "My workspace files")}</button></div>
        {workspaceView === "blueprint" && <><div className="blueprint-steps" data-tour="recommended-steps">{blueprint.steps.map((step, index) => <article key={step.title} className={activeBlueprintStep === index ? "active" : ""}><small>{String(index + 1).padStart(2, "0")} / {step.status}</small><h3>{step.title}</h3><p>{step.detail}</p><code>{step.files}</code><button onClick={() => setActiveBlueprintStep(index)}>{t("查看输入与代码骨架", "Inspect inputs and code outline")} <ChevronRight size={14} /></button></article>)}</div>
          <section className="blueprint-step-detail" data-tour="project-current-step"><div><span className="eyebrow">STEP {String(activeBlueprintStep + 1).padStart(2, "0")} · {blueprint.steps[activeBlueprintStep].status}</span><h3>{blueprint.steps[activeBlueprintStep].title}</h3><p>{blueprint.steps[activeBlueprintStep].detail}</p><strong>{t("关联文件", "Related files")}</strong><code>{blueprint.steps[activeBlueprintStep].files}</code><button className="secondary" onClick={openBlueprintStep}>{blueprint.steps[activeBlueprintStep].route === "/projects" || blueprint.steps[activeBlueprintStep].route === "/training-code" ? t("打开任务草稿", "Open task draft") : t("前往该步骤", "Go to step")} <ChevronRight size={14} /></button></div><div><span className="eyebrow">REFERENCE OUTLINE · NOT EXECUTED</span>{activeBlueprintStep === 2 && <div className="blueprint-network"><div>{blueprintTask === "s1_push" ? "24D" : "18D"}<small>OBSERVATION</small></div><ChevronRight size={14} /><div><strong>ACTOR</strong><small>MLP · PPO</small><strong>CRITIC</strong><small>VALUE</small></div><ChevronRight size={14} /><div>2D<small>XY ACTION</small></div></div>}<pre>{BLUEPRINT_EXAMPLES[blueprintTask][activeBlueprintStep]}</pre><small>{t("以上为文件职责示意，不是正在运行的源码，也不会随工作区保存而进入 PPO。", "An outline of responsibilities, not live source or executable PPO code.")}</small></div></section>
          <section className="blueprint-file-map"><div><span className="eyebrow">FILE MAP</span><h3>{t("完成一次训练需要什么", "Files needed for a training run")}</h3></div><div><strong>{t("可编辑教学草稿", "Editable teaching drafts")}</strong><code>tasks/s1_push/ · tasks/s1_reach/ · README.md · scene.yaml · task.py · policy.py · reward.py · train.py · evaluate.py</code><small>{t("分别填写两种任务的场景、任务、网络、奖励、训练及评估接口。当前仅保存草稿，不作为 PPO 可执行输入。", "Fill in each task's scene, policy, reward, training and evaluation interfaces. Saved drafts are not executable PPO inputs.")}</small><button onClick={() => setWorkspaceView("files")}>{t("打开工作区文件", "Open workspace files")} <ChevronRight size={13} /></button><button onClick={() => void addGuides()} disabled={!projectId}>{t("补齐两套任务草稿（不覆盖）", "Add both task drafts without overwriting")} <FilePlus2 size={13} /></button></div><div><strong>{t("平台任务 / 策略源码", "Platform task / policy source")}</strong><code>{blueprint.taskSource} · {blueprint.policySource}</code><small>{t("只读参考。当前网页尚不能上传自定义训练环境和替换 PPO 实现。", "Read-only reference. The web app cannot yet upload a custom environment or replace the PPO implementation.")}</small></div><div><strong>{t("训练后生成", "Generated after training")}</strong><code>config/training_config.json · metrics/*.jsonl · checkpoints/*.pt · manifest.json</code><small>{t("由 Simulation Worker 创建，运行记录关联；并非创建项目前需要手写。", "Created by the Simulation Worker and linked to a Run; not files to hand-write before training.")}</small></div></section>
          <div className="blueprint-components"><div><span className="eyebrow">CONTROL PARTS</span><h3>{t("控制演示零件", "Control demo parts")}</h3><p>{t("这五个模块用于检查姿态、轨迹与关节运动；它们不进行 PPO 学习，不能作为训练任务提交。", "These modules inspect poses, trajectories and joints. They do not train PPO policies.")}</p></div><div>{CONTROL_PARTS.map(([name, detail, id, source]) => <div key={id}><strong>{name}</strong><span>{detail}</span><code>{id}</code><small>simulation/src/simulation_core/{source}</small></div>)}</div></div></>}
      </section>
      {workspaceView === "files" && <>
      <section className="project-ide">
        <aside className="workspace-explorer">
          <div className="workspace-pane-title">
            <span>EXPLORER</span>
            <small>{projectId}</small>
          </div>
          <div className="workspace-tree">
            {tree.length ? renderTree(tree) : <p>{t("工作区为空", "Workspace is empty")}</p>}
          </div>
        </aside>
        <section className="workspace-editor">
          <div className="workspace-tabs">
            {tabs.map((tab) => (
              <button
                key={tab.path}
                className={tab.path === activePath ? "active" : ""}
                onClick={() => setActivePath(tab.path)}
              >
                <span>{tab.path.split("/").pop()}</span>
                {tab.content !== tab.saved && <i />}
                <X
                  size={12}
                  onClick={(event) => {
                    event.stopPropagation();
                    setTabs((current) =>
                      current.filter((item) => item.path !== tab.path),
                    );
                    if (activePath === tab.path) setActivePath("");
                  }}
                />
              </button>
            ))}
          </div>
          {activeTab && showDiff && diff ? (
            <DiffEditor
              height="calc(100vh - 188px)"
              language={workspaceLanguage(activeTab.path)}
              theme="vs-light"
              original={diff.baseline}
              modified={diff.current}
              onMount={(editor) => {
                const container = editor.getContainerDomNode().parentElement;
                if (container) {
                  editor.layout({
                    width: container.clientWidth,
                    height: container.clientHeight,
                  });
                }
              }}
              options={{
                readOnly: true,
                renderSideBySide: true,
                minimap: { enabled: false },
                automaticLayout: true,
              }}
            />
          ) : activeTab ? (
            <Editor
              height="calc(100vh - 188px)"
              language={workspaceLanguage(activeTab.path)}
              theme="vs-light"
              value={activeTab.content}
              onMount={(editor) => {
                const container = editor.getContainerDomNode().parentElement;
                if (container) {
                  editor.layout({
                    width: container.clientWidth,
                    height: container.clientHeight,
                  });
                }
              }}
              onChange={(content) =>
                setTabs((current) =>
                  current.map((tab) =>
                    tab.path === activeTab.path
                      ? { ...tab, content: content ?? "" }
                      : tab,
                  ),
                )
              }
              options={{
                minimap: { enabled: false },
                fontSize: 13,
                scrollBeyondLastLine: false,
                automaticLayout: true,
              }}
            />
          ) : (
            <div className="workspace-empty">
              <FolderKanban size={34} />
              <strong>{t("选择或创建文件", "Select or create a file")}</strong>
              <span>{t("打开左侧文件查看策略、网络结构和训练配置；这些教学文件不会被执行。", "Open a file to inspect the policy, network and training setup. These teaching files are not executed.")}</span>
            </div>
          )}
        </section>
        <aside className="workspace-tools">
          <div className="workspace-pane-title">
            <span>SEARCH</span>
            <small>{t("最多 100 项", "Up to 100 items")}</small>
          </div>
          <div className="workspace-search">
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") searchWorkspace();
              }}
              placeholder={t("搜索工作区文本", "Search workspace text")}
            />
            <button title={t("搜索", "Search")} onClick={searchWorkspace}>
              <Search size={15} />
            </button>
          </div>
          <div className="workspace-search-results">
            {searchResults.map((result) => (
              <button
                key={`${result.path}:${result.line}`}
                onClick={() => openFile(result.path)}
              >
                <strong>
                  {result.path}:{result.line}
                </strong>
                <span>{result.preview}</span>
              </button>
            ))}
          </div>
          <div className="workspace-snapshot">
            <div className="workspace-pane-title">
              <span>SNAPSHOT</span>
              <small>{t("已审计", "AUDITED")}</small>
            </div>
            <button className="primary" onClick={createSnapshot}>
              <Save size={14} /> {t("创建快照", "Create snapshot")}
            </button>
            <button onClick={restoreSnapshot} disabled={!snapshotId}>
              <RotateCcw size={14} /> {t("恢复本次快照", "Restore snapshot")}
            </button>
            {snapshotId && <code>{snapshotId}</code>}
          </div>
          <div className="workspace-history">
            <div className="workspace-pane-title">
              <span>FILE HISTORY</span>
              <small>{history.length} VERSIONS</small>
            </div>
            <button onClick={loadDiff} disabled={!activeTab}>
              <GitCompare size={14} /> {t("Starter Diff", "Starter Diff")}
            </button>
            <button onClick={restoreStarter}>
              <RotateCcw size={14} /> {t("恢复 Starter", "Restore Starter")}
            </button>
            <div className="workspace-history-list">
              {history.map((version) => (
                <button
                  key={version.version_id}
                  onClick={() => restoreVersion(version.version_id)}
                >
                  <strong>{version.version_id}</strong>
                  <span>{new Date(version.created_at).toLocaleString()}</span>
                </button>
              ))}
            </div>
          </div>
        </aside>
      </section>
      </>}
    </main>
  );
}

function TrainingPage() {
  const { locale, t } = useI18n();
  const environments = useQuery({
    queryKey: ["training-environments"],
    queryFn: () => api<EnvironmentInfo[]>("/api/environments"),
  });
  const strategies = useQuery({
    queryKey: ["training-strategies"],
    queryFn: () => api<StrategyInfo[]>("/api/strategies"),
  });
  const scenes = useQuery({
    queryKey: ["training-scenes"],
    queryFn: () => api<SceneDocument[]>("/api/scenes"),
  });
  const sandboxTemplates = useQuery({
    queryKey: ["sandbox-templates"],
    queryFn: () => api<SandboxTemplateInfo[]>("/api/sandbox/templates"),
  });
  const [provider, setProvider] = useState<"mock" | "edge-gpu">("mock");
  const [environmentId, setEnvironmentId] = useState("mock-reach");
  const [sceneId, setSceneId] = useState("scene_mock_reach_lab");
  const [strategyId, setStrategyId] = useState("joint-p-control");
  const [seed, setSeed] = useState(42);
  const [maxSteps, setMaxSteps] = useState(120);
  const [actionScale, setActionScale] = useState(0.35);
  const [run, setRun] = useState<LegacyRunRecord | null>(null);
  const [events, setEvents] = useState<MetricEvent[]>([]);
  const [error, setError] = useState("");
  const [gpuJob, setGpuJob] = useState<SandboxJobInfo | null>(null);
  const [gpuMetrics, setGpuMetrics] = useState<Array<{ epoch: number; train_loss: number }>>([]);
  const [gpuArtifacts, setGpuArtifacts] = useState<SandboxArtifactInfo[]>([]);
  useEffect(() => setError(""), [locale]);
  const running = provider === "mock"
    ? run?.status === "queued" || run?.status === "running"
    : gpuJob?.status === "queued" || gpuJob?.status === "running";
  const metricEvents = events.filter(
    (event) => event.type === "metric" && event.payload.metrics,
  );
  const finalMetrics =
    metricEvents[metricEvents.length - 1]?.payload.metrics ?? {};
  const visibleMetricEvents = provider === "mock" ? metricEvents : gpuMetrics;

  async function launch() {
    setError("");
    setEvents([]);
    setGpuMetrics([]);
    setGpuArtifacts([]);
    try {
      if (provider === "edge-gpu") {
        const created = await api<SandboxJobInfo>("/api/sandbox/jobs", {
          method: "POST",
          body: JSON.stringify({ template_id: "gpu-pytorch-train" }),
        });
        setGpuJob(created);
        return;
      }
      const created = await createLegacyTrainingRun({
        project_id: "quick-start",
        parameters: {
          environment_id: environmentId,
          strategy_id: strategyId,
          seed,
          max_steps: maxSteps,
        },
        project_config_snapshot: {
          scene_id: sceneId,
          action_scale: actionScale,
        },
        legacy: {
          environment_id: environmentId,
          strategy_id: strategyId,
          seed,
          max_steps: maxSteps,
          config: { action_scale: actionScale, scene_id: sceneId },
        },
      });
      setRun(created);
      navigate(runDetailPath(created.run_id));
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const token = getAccessToken();
      const eventQuery = new URLSearchParams({ after_sequence: "0" });
      const socket = new WebSocket(
        `${protocol}://${location.host}/api/v1/runs/${created.run_id}/events?${eventQuery}`,
      );
      socket.addEventListener("open", () => {
        if (token) socket.send(JSON.stringify({ type: "AUTH", access_token: token }));
      }, { once: true });
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data) as MetricEvent;
        setEvents((current) => [...current, event]);
        if (["state", "status"].includes(event.type) && event.payload.status) {
          setRun((current) =>
            current
              ? {
                  ...current,
                  status: event.payload.status as LegacyRunRecord["status"],
                }
              : current,
          );
          if (["completed", "stopped", "failed"].includes(event.payload.status))
            socket.close();
        }
      };
      socket.onerror = () =>
        setError(t("运行已创建，但实时事件连接中断；可在运行记录中检查结果。", "The run was created, but the live event connection was interrupted. Check the result in Run History."));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("训练启动失败", "Failed to start training"));
    }
  }

  async function stop() {
    if (provider === "edge-gpu" && gpuJob) {
      const stopped = await api<SandboxJobInfo>(`/api/sandbox/jobs/${gpuJob.job_id}/stop`, { method: "POST" });
      setGpuJob(stopped);
      return;
    }
    if (!run) return;
    await apiWithFallback(
      `/api/v1/runs/${run.run_id}/stop`,
      `/api/runs/${run.run_id}/stop`,
      { method: "POST" },
    );
  }

  useEffect(() => {
    if (!gpuJob || !["queued", "running"].includes(gpuJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const current = await api<SandboxJobInfo>(`/api/sandbox/jobs/${gpuJob.job_id}`);
        setGpuJob(current);
        const logs = await api<{ stdout: string; stderr: string }>(`/api/sandbox/jobs/${gpuJob.job_id}/logs`);
        const metrics = logs.stdout.split("\n").flatMap((line) => {
          if (!line.startsWith("STUDIO_METRIC ")) return [];
          try { return [JSON.parse(line.slice(14)) as { epoch: number; train_loss: number }]; }
          catch { return []; }
        });
        setGpuMetrics(metrics);
        if (["completed", "failed", "stopped"].includes(current.status)) {
          window.clearInterval(timer);
          setGpuArtifacts(await api<SandboxArtifactInfo[]>(`/api/sandbox/jobs/${gpuJob.job_id}/artifacts`));
          if (current.error) setError(current.error);
        }
      } catch (caught) {
        window.clearInterval(timer);
        setError(caught instanceof Error ? caught.message : "GPU 训练状态读取失败");
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [gpuJob?.job_id, gpuJob?.status]);

  const chartOption = {
    animation: false,
    grid: { left: 48, right: 24, top: 28, bottom: 35 },
    tooltip: { trigger: "axis" },
    legend: { data: ["Reward", "Success"], right: 10, textStyle: { color: "#5f6966" } },
    xAxis: { type: "category", data: metricEvents.map((event) => event.payload.step), axisLabel: { color: "#73807b" } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#dde1dc" } } },
    series: [
      { name: "Reward", type: "line", symbol: "none", data: metricEvents.map((event) => event.payload.metrics?.mean_reward), lineStyle: { color: "#e05a3f", width: 2 } },
      { name: "Success", type: "line", symbol: "none", data: metricEvents.map((event) => event.payload.metrics?.success_rate), lineStyle: { color: "#277b69", width: 2 } },
    ],
  };

  const visibleStatus = provider === "mock" ? run?.status : gpuJob?.status;
  const visibleRunId = provider === "mock" ? run?.run_id : gpuJob?.job_id;
  const visibleChartOption = provider === "mock" ? chartOption : {
    yAxis: { type: "log", splitLine: { lineStyle: { color: "#dde1dc" } } },
    series: [{ name: "Train loss", type: "line", symbol: "circle", data: gpuMetrics.map((item) => item.train_loss), lineStyle: { color: "#277b69", width: 2 } }],
  };

  if (environments.isLoading || strategies.isLoading || scenes.isLoading || sandboxTemplates.isLoading)
    return <div className="loading">正在读取训练能力...</div>;
  if (environments.isError || strategies.isError || scenes.isError || sandboxTemplates.isError)
    return <div className="loading error">训练目录 API 不可用。</div>;

  return (
    <main className="operations-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">TRAINING LAB · {provider === "mock" ? "MOCK" : "EDGE GPU"}</span>
          <h1>{provider === "mock" ? "Mock 参数实验" : "GPU Worker 验证"}</h1>
          <p>
            {provider === "mock"
              ? "使用三关节教学近似验证 seed、参数与事件可复现性；这里不训练神经网络，也不创建课程正式 Run。"
              : "运行固定的 synthetic regression 模板，用于验证 GPU Worker、反向传播与 checkpoint 链路；它不是 S1 或课程 PPO 训练。"}
          </p>
        </div>
        <div className="operations-boundary">
          <ShieldCheck size={18} />
          <span>
            <strong>{provider === "mock" ? "MOCK ADAPTER" : "EDGE GPU WORKER"}</strong>
            <small>{provider === "mock" ? "REAL CONTROL DISABLED" : "GPU 0 · NO HOST MOUNTS"}</small>
          </span>
        </div>
      </header>
      <section className="training-workspace">
        <aside className="run-config">
          <div className="section-label">RUN CONFIGURATION</div>
          <div className="training-provider-switch" role="group" aria-label="训练提供方">
            <button className={provider === "mock" ? "active" : ""} onClick={() => setProvider("mock")} disabled={running}>Mock</button>
            <button className={provider === "edge-gpu" ? "active" : ""} onClick={() => setProvider("edge-gpu")} disabled={running || !sandboxTemplates.data?.some((item) => item.template_id === "gpu-pytorch-train")}>Edge GPU</button>
          </div>
          {provider === "edge-gpu" ? (
            <div className="gpu-template-facts">
              <span>PyTorch 2.4.1</span><span>CUDA 12.4</span><span>2 GiB RAM</span><span>180 s timeout</span>
            </div>
          ) : <>
          <label>
            环境
            <select
              value={environmentId}
              onChange={(event) => setEnvironmentId(event.target.value)}
              disabled={running}
            >
              {(environments.data ?? [])
                .filter((item) => item.available)
                .map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            策略
            <select
              value={strategyId}
              onChange={(event) => setStrategyId(event.target.value)}
              disabled={running}
            >
              {(strategies.data ?? [])
                .filter((item) =>
                  item.compatible_environments.includes(environmentId),
                )
                .map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Seed
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
              disabled={running}
            />
          </label>
          <label>
            最大步数
            <input
              type="number"
              min="1"
              max="2000"
              value={maxSteps}
              onChange={(event) => setMaxSteps(Number(event.target.value))}
              disabled={running}
            />
          </label>
          <label>
            动作缩放 <output>{actionScale.toFixed(2)}</output>
            <input
              type="range"
              min="0.05"
              max="1"
              step="0.05"
              value={actionScale}
              onChange={(event) => setActionScale(Number(event.target.value))}
              disabled={running}
            />
          </label>
          </>}
          <div className="run-command">
            <button className="primary" onClick={launch} disabled={running}>
              <Play size={16} /> {provider === "mock" ? "运行 Mock 实验" : "验证 GPU Worker"}
            </button>
            <button className="secondary" onClick={stop} disabled={!running}>
              <CircleStop size={16} /> 停止
            </button>
          </div>
          {error && <p className="inline-error">{error}</p>}
        </aside>
        <section className="training-monitor">
          <div className="monitor-title">
            <div>
              <span>LIVE METRICS</span>
              <strong>{visibleRunId ?? "等待启动"}</strong>
            </div>
            <span className={`run-state ${visibleStatus ?? "idle"}`}>
              {visibleStatus ?? "idle"}
            </span>
          </div>
          {visibleMetricEvents.length ? (
            <ReactECharts option={visibleChartOption} style={{ height: 360 }} />
          ) : (
            <div className="monitor-empty">
              <Activity size={30} />
              <strong>尚无运行事件</strong>
              <span>{provider === "mock" ? "启动后将实时显示 reward 与 success rate。" : "启动后将显示真实 GPU train loss。"}</span>
            </div>
          )}
        </section>
        <aside className="run-summary">
          <div className="section-label">RUN SUMMARY</div>
          <dl>
            <div>
              <dt>环境</dt>
              <dd>{provider === "mock" ? run?.environment_id ?? environmentId : "synthetic-regression"}</dd>
            </div>
            <div>
              <dt>策略</dt>
              <dd>{provider === "mock" ? run?.strategy_id ?? strategyId : "torch-mlp-adam"}</dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd>{provider === "mock" ? run?.seed ?? seed : 42}</dd>
            </div>
            <div>
              <dt>事件</dt>
              <dd>{provider === "mock" ? events.length : gpuMetrics.length}</dd>
            </div>
            <div>
              <dt>{provider === "mock" ? "Reward" : "Train loss"}</dt>
              <dd>
                {provider === "edge-gpu" && gpuMetrics.length
                  ? gpuMetrics[gpuMetrics.length - 1].train_loss.toFixed(6)
                  : typeof finalMetrics.mean_reward === "number"
                  ? finalMetrics.mean_reward.toFixed(3)
                  : "--"}
              </dd>
            </div>
            <div>
              <dt>{provider === "mock" ? "Success" : "Checkpoint"}</dt>
              <dd>
                {provider === "edge-gpu"
                  ? gpuArtifacts[0]?.sha256.slice(0, 12) ?? "--"
                  : typeof finalMetrics.success_rate === "number"
                  ? finalMetrics.success_rate.toFixed(3)
                  : "--"}
              </dd>
            </div>
          </dl>
          {provider === "edge-gpu" && gpuJob && gpuArtifacts.length ? (
            <a className="secondary artifact-download" href={`/api/sandbox/jobs/${gpuJob.job_id}/artifacts/checkpoint.json`} download>
              <Download size={16} /> 下载 Checkpoint
            </a>
          ) : (
            <button className="secondary" onClick={() => navigate("/runs")}>
              <FlaskConical size={16} /> 查看运行记录
            </button>
          )}
        </aside>
      </section>
    </main>
  );
}

function S1ParameterSandboxPage({ user }: { user: CurrentUserContext }) {
  const { locale, t } = useI18n();
  const [organizationId, setOrganizationId] = useState(
    () => sessionStorage.getItem("s1-draft-organization") ?? user.organizations[0]?.organization_id ?? "",
  );
  const [courseId, setCourseId] = useState(
    () => sessionStorage.getItem("s1-draft-course") ?? "",
  );
  const [projectId, setProjectId] = useState(
    () => sessionStorage.getItem("s1-draft-project") ?? "",
  );
  const [sceneId, setSceneId] = useState(
    () => sessionStorage.getItem("s1-draft-scene") ?? "",
  );
  const [taskDraft] = useState(readS1TaskDraft);
  const [taskConfiguration, setTaskConfiguration] = useState<SimulationTaskConfigurationState | null>(null);
  const courses = useQuery({
    queryKey: ["sandbox-courses", organizationId],
    queryFn: () => api<TeachingCourse[]>(`/api/v1/courses?organization_id=${encodeURIComponent(organizationId)}`),
    enabled: Boolean(organizationId),
  });
  const projects = useQuery({
    queryKey: ["sandbox-projects", courseId],
    queryFn: () => api<TeachingProject[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/projects`),
    enabled: Boolean(courseId),
  });
  const scenes = useQuery({
    queryKey: ["sandbox-s1-scenes"],
    queryFn: () => api<SceneDocument[]>("/api/scenes"),
  });
  const s1Scenes = (scenes.data ?? []).filter(
    (scene) => scene.capability !== "mock" && scene.robot.dof === S1_ARM_JOINTS.length,
  );
  const usableProjects = (projects.data ?? []).filter((project) => project.enabled && project.permissions.can_use && project.task_type === "training" && !project.environment_type.startsWith("demo_"));
  const selectedProject = usableProjects.find((project) => project.project_id === projectId);
  const selectedScene = s1Scenes.find((scene) => scene.scene_id === sceneId) ?? s1Scenes[0];

  useEffect(() => {
    if (!courseId && courses.data?.[0]) setCourseId(courses.data[0].course_id);
  }, [courseId, courses.data]);

  useEffect(() => {
    if (projects.data && !projects.data.some((project) => project.project_id === projectId && project.enabled && project.permissions.can_use && project.task_type === "training" && !project.environment_type.startsWith("demo_"))) {
      setTaskConfiguration(null);
      setProjectId(projects.data.find((project) => project.enabled && project.permissions.can_use && project.task_type === "training" && !project.environment_type.startsWith("demo_"))?.project_id ?? "");
    }
  }, [projectId, projects.data]);

  useEffect(() => {
    if (!sceneId && s1Scenes[0]) setSceneId(s1Scenes[0].scene_id);
  }, [sceneId, s1Scenes]);

  function continueToLaunch() {
    if (selectedProject?.task_type !== "training" || taskConfiguration?.task?.mode !== "training" || !taskConfiguration.validation?.valid || !taskConfiguration.task.supportsFormalRun ||
        taskConfiguration.task.id !== selectedProject?.environment_type) return;
    sessionStorage.setItem("s1-draft-organization", organizationId);
    sessionStorage.setItem("s1-draft-course", courseId);
    sessionStorage.setItem("s1-draft-project", projectId);
    sessionStorage.setItem("s1-draft-scene", selectedScene?.scene_id ?? "");
    sessionStorage.setItem(S1_TASK_DRAFT_STORAGE_KEY, JSON.stringify({
      taskId: taskConfiguration.task.id,
      parameters: taskConfiguration.validation.values,
    } satisfies S1TaskDraft));
    if (typeof taskConfiguration.parameters.seed === "number") {
      sessionStorage.setItem("s1-draft-seed", String(taskConfiguration.parameters.seed));
    }
    navigate("/courses");
  }

  if (courses.isError || projects.isError || scenes.isError) {
    return <div className="loading error">{t("正式课程或六轴 S1 场景目录不可用。", "The formal course or six-axis S1 scene catalog is unavailable.")}</div>;
  }

  return (
    <main className="operations-page s1-parameter-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">SIMULATION TASK · PARAMETER SANDBOX</span>
          <h1>{t("任务选择与参数配置", "Task and parameter configuration")}</h1>
          <p>{t("选择教学 Project 并配置对应的训练参数。控制 Demo 在任务与技能中启动，场景对象在 S1 环境编辑中修改。", "Choose a teaching Project and configure its training parameters. Launch control demos in Tasks and skills; edit scene objects in S1 Environment.")}</p>
        </div>
        <div className="operations-boundary"><ShieldCheck size={18} /><span><strong>6 ARM + 2 GRIPPER DOF</strong><small>SIMULATION ONLY · REAL CONTROL DISABLED</small></span></div>
      </header>
      <section className="s1-parameter-workspace">
        <aside className="run-config" data-tour="parameter-context">
          <div className="section-label">EXPERIMENT CONTEXT</div>
          <label>{t("组织", "Organization")}<select value={organizationId} onChange={(event) => { setOrganizationId(event.target.value); setCourseId(""); setProjectId(""); }}>
            <option value="">{t("选择组织", "Select organization")}</option>
            {user.organizations.map((organization) => <option key={organization.organization_id} value={organization.organization_id}>{organization.name || organization.slug}</option>)}
          </select></label>
          <label>{t("课程", "Course")}<select value={courseId} disabled={!organizationId || courses.isLoading} onChange={(event) => { setCourseId(event.target.value); setProjectId(""); }}>
            <option value="">{t("选择课程", "Select course")}</option>
            {(courses.data ?? []).map((course) => <option key={course.course_id} value={course.course_id}>{course.code} · {course.title}</option>)}
          </select></label>
          <label>{t("教学 Project", "Teaching Project")}<select value={projectId} disabled={!courseId || projects.isLoading} onChange={(event) => { setTaskConfiguration(null); setProjectId(event.target.value); }}>
            <option value="">{t("选择 Project", "Select Project")}</option>
            {usableProjects.map((project) => <option key={project.project_id} value={project.project_id}>{teachingProjectLabel(project, locale)}</option>)}
          </select></label>
          {selectedProject?.environment_type === "s1_push" && selectedProject.training_profile === "short" && <p className="field-note">{t(`${selectedProject.expected_training_steps} 步短程 PPO，用于运行链路检查，不代表策略充分训练。固定控制 Demo 请前往“任务与技能 → 控制演示”。`, `${selectedProject.expected_training_steps}-step short PPO run for pipeline checks, not sufficient policy training. Fixed demos are in Tasks and skills → Control demos.`)}</p>}
          {courseId && !projects.isLoading && !usableProjects.length && <p className="summary-empty">{t("此课程暂无可用的强化学习训练 Project。", "This course has no available RL training Project.")}</p>}
          <div className="s1-context-card">
            <span>{t("已保存的 S1 场景", "Saved S1 scene")}</span>
            <strong>{selectedScene ? `${selectedScene.name} · REV ${selectedScene.revision}` : t("尚未选择", "Not selected")}</strong>
            <small>{selectedScene ? `${selectedScene.robot.id} · ${selectedScene.objects.length} ${t("个场景对象", "scene objects")}` : t("请先在环境编辑中选择并保存六自由度场景。", "Select and save a six-DOF scene in the Environment page first.")}</small>
            <button className="secondary" onClick={() => navigate("/environments")}><Boxes size={14} /> {t("前往 S1 环境编辑", "Open S1 Environment")}</button>
          </div>
          <div className="section-label">{t("冻结训练配置", "Frozen training configuration")}</div>
          {selectedProject ? <dl className="s1-frozen-config" data-tour="parameter-scale"><div><dt>{t("算法", "Algorithm")}</dt><dd>{selectedProject.algorithm}</dd></div><div><dt>{t("训练档位", "Training profile")}</dt><dd>{selectedProject.training_profile}</dd></div><div><dt>{t("预计步数", "Expected steps")}</dt><dd>{selectedProject.expected_training_steps.toLocaleString(locale)}</dd></div><div><dt>{t("配置版本", "Config version")}</dt><dd>V{selectedProject.config_version}</dd></div></dl> : <p className="summary-empty">{t("选择 Project 后显示只读配置。", "Select a Project to view the read-only configuration.")}</p>}
        </aside>
        <section className="s1-task-configuration">
          {selectedProject ? <SimulationTaskConfigurator
            key={`${selectedProject.project_id}:${selectedProject.config_version}:${selectedProject.environment_type}`}
            initialTaskId={selectedProject.environment_type}
            initialParameters={taskDraft?.taskId === selectedProject.environment_type ? taskDraft.parameters : undefined}
            allowedTaskIds={[selectedProject.environment_type]}
            showTaskSelection={false}
            requiredTaskMode="training"
            groupParameters
            onStateChange={setTaskConfiguration}
          /> : <p className="summary-empty">{t("请选择教学 Project。", "Choose a teaching Project.")}</p>}
          {taskConfiguration?.task && !taskConfiguration.task.supportsFormalRun && (
            <div className="s1-task-launch-boundary"><CircleStop size={16} /><span><strong>{t("当前任务只能完成参数准备", "This task can only be configured")}</strong>{taskConfiguration.task.formalRunReason ?? t("平台尚未提供该任务的正式启动路径。", "The platform does not yet provide a formal launch path for this task.")}</span></div>
          )}
          {selectedProject && taskConfiguration?.task && taskConfiguration.task.supportsFormalRun && taskConfiguration.task.id !== selectedProject.environment_type && (
            <div className="s1-task-launch-boundary"><CircleStop size={16} /><span><strong>{t("任务与 Project 不匹配", "Task and Project do not match")}</strong>{t(`请选择 ${taskConfiguration.task.title} 对应的教学 Project。`, `Choose the teaching Project for ${taskConfiguration.task.title}.`)}</span></div>
          )}
          <button className="primary s1-continue" data-tour="launch-experiment" disabled={!selectedProject || !selectedScene || !taskConfiguration?.validation?.valid || !taskConfiguration.task?.supportsFormalRun || taskConfiguration.task.mode !== "training" || taskConfiguration.task.id !== selectedProject?.environment_type} onClick={continueToLaunch}><Play size={16} /> {t("进入启动实验", "Continue to launch experiment")}</button>
        </section>
      </section>
    </main>
  );
}

function RunsPage() {
  const { locale, t } = useI18n();
  const queryClient = useQueryClient();
  const runs = useQuery({
    queryKey: ["run-records"],
    queryFn: () => api<RunRecord[]>("/api/v1/runs"),
    refetchInterval: 1500,
  });
  const [selectedId, setSelectedId] = useState(
    () => new URLSearchParams(location.search).get("run") ?? "",
  );
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [deleteIds, setDeleteIds] = useState<string[]>([]);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [comparison, setComparison] = useState<RunComparisonResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [exporting, setExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState("");
  const [configExpanded, setConfigExpanded] = useState(false);
  const [configWrapped, setConfigWrapped] = useState(false);
  const [configCopyStatus, setConfigCopyStatus] = useState("");
  const records = runs.data ?? [];
  const terminalRecords = records.filter((run) => ["completed", "failed", "stopped"].includes(run.status));
  const selected =
    records.find((run) => run.run_id === selectedId) ?? records[0];
  const events = useQuery({
    queryKey: ["run-events", selected?.run_id],
    queryFn: () =>
      api<MetricEvent[]>(
        `/api/v1/runs/${selected?.run_id}/events?after_sequence=0`,
      ),
    enabled: Boolean(selected),
  });

  useEffect(() => {
    if (!selectedId && records[0]) setSelectedId(records[0].run_id);
  }, [records, selectedId]);

  function toggleCompare(runId: string) {
    setComparison(null);
    setCompareError("");
    setCompareIds((current) =>
      current.includes(runId)
        ? current.filter((id) => id !== runId)
        : current.length < 2
          ? [...current, runId]
          : [current[1], runId],
    );
  }

  async function deleteSelectedRuns() {
    if (deleting || !deleteIds.length) return;
    if (deleteIds.length > 10) {
      setDeleteError(t("每次最多删除 10 条运行记录。", "Delete up to 10 runs at a time."));
      return;
    }
    const selectedRuns = records.filter((run) => deleteIds.includes(run.run_id));
    if (selectedRuns.length !== deleteIds.length || selectedRuns.some((run) => !["completed", "failed", "stopped"].includes(run.status))) {
      setDeleteError(t("只能删除已完成、失败或已停止的运行；请先停止正在运行的任务。", "Only completed, failed or stopped runs can be deleted. Stop active runs first."));
      return;
    }
    if (!window.confirm(t(`确认删除 ${deleteIds.length} 条运行记录及其关联事件、评估和制品索引？此操作不可撤销。`, `Delete ${deleteIds.length} run records and their events, evaluations and artifact indexes? This cannot be undone.`))) return;
    setDeleting(true);
    setDeleteError("");
    try {
      await api<{ deleted_count: number }>("/api/v1/runs/delete", {
        method: "POST",
        body: JSON.stringify({ run_ids: deleteIds }),
      });
      if (deleteIds.includes(selectedId)) setSelectedId("");
      setCompareIds((current) => current.filter((id) => !deleteIds.includes(id)));
      setComparison(null);
      setCompareError("");
      setDeleteIds([]);
      queryClient.removeQueries({ predicate: (query) => deleteIds.some((id) => query.queryKey.includes(id)) });
      await queryClient.invalidateQueries({ queryKey: ["run-records"] });
      await queryClient.invalidateQueries({ queryKey: ["workbench-runs"] });
    } catch (caught) {
      setDeleteError(caught instanceof ApiError && [404, 405].includes(caught.status)
        ? t("删除接口尚未部署到当前后端，或记录不存在；记录未被删除。", "The deletion API is not deployed on this backend, or the run was not found. No records were deleted.")
        : caught instanceof Error ? caught.message : t("删除失败", "Deletion failed"));
    } finally {
      setDeleting(false);
    }
  }

  async function compare() {
    if (compareIds.length !== 2 || comparing) return;
    navigate(`/run-comparison?runs=${compareIds.map(encodeURIComponent).join(",")}`);
  }

  async function exportManifest() {
    if (!selected || exporting) return;
    setExporting(true);
    setExportStatus("");
    try {
      const response = await apiBlob("/api/v1/runs/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: [selected.run_id], format: "json" }),
      });
      const blob = await response.blob();
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(blob);
      const disposition = response.headers.get("content-disposition") ?? "";
      anchor.download =
        disposition.match(/filename="?([^";]+)"?/)?.[1] ??
        `${selected.run_id}-runs.json`;
      anchor.click();
      URL.revokeObjectURL(anchor.href);
      setExportStatus(`success:${t("运行 JSON 已生成，下载已开始", "Run JSON generated; download started.")}`);
    } catch (caught) {
      setExportStatus(`error:${caught instanceof Error ? caught.message : t("导出请求失败，请重试", "Export failed; try again.")}`);
    } finally {
      setExporting(false);
    }
  }

  async function copyFrozenConfiguration() {
    if (!selected) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(selected.project_config_snapshot, null, 2));
      setConfigCopyStatus(t("已复制", "Copied"));
    } catch {
      setConfigCopyStatus(t("复制失败", "Copy failed"));
    }
  }

  if (runs.isLoading) return <div className="loading">{t("正在读取运行记录...", "Reading run records...")}</div>;
  if (runs.isError)
    return <div className="loading error">{t("运行记录 API 不可用。", "The run records API is unavailable.")}</div>;
  if (!selected)
    return (
      <main className="evaluation-empty">
        <FlaskConical size={34} />
        <span className="eyebrow">NO RUN RECORDS</span>
        <h1>{t("尚无运行记录", "No run records yet")}</h1>
        <p>
          {t("从训练工作区启动一个正式教学 Run；只有真实服务产生的配置、状态和指标事件才会出现在这里。", "Start a formal teaching Run from the training workspace. Only configuration, status, and metric events produced by the real service appear here.")}
        </p>
        <button className="primary" onClick={() => navigate("/courses")}>
          <Play size={16} /> {t("选择课程", "Choose course")}
        </button>
      </main>
    );

  return (
    <main className="operations-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">RUN EVIDENCE</span>
          <h1>{t("运行分析", "Run analysis")}</h1>
          <p>
            {t("保留成功、失败和停止的运行。对比只展示实际记录的配置和末次指标，不从单次最佳结果推断可复现性。", "Completed, failed, and stopped runs are retained. Comparison shows only recorded configuration and final metrics; reproducibility is not inferred from a single best result.")}
          </p>
          <p>{t("已有运行可直接查看结果、制品和导出，无需重新训练。", "Existing runs can be inspected, exported, and reviewed without retraining.")}</p>
        </div>
        <button className="primary" onClick={() => navigate("/courses")}>
          <Play size={16} /> {t("从课程新建运行", "Create a Run from a course")}
        </button>
      </header>
      <section className="runs-workspace">
        <aside className="run-record-list" data-tour="run-list">
          <div className="section-label">{t("运行记录", "Run records")} · {records.length}</div>
          <div className="run-delete-controls" data-tour="run-delete-controls">
            <label><input type="checkbox" checked={terminalRecords.length > 0 && terminalRecords.slice(0, 10).every((run) => deleteIds.includes(run.run_id))} disabled={!terminalRecords.length || deleting} onChange={(event) => setDeleteIds(event.target.checked ? terminalRecords.slice(0, 10).map((run) => run.run_id) : [])} />{t("选择前 10 条已结束运行", "Select first 10 terminal runs")}</label>
            <button className="danger" disabled={!deleteIds.length || deleting} onClick={() => void deleteSelectedRuns()}><Trash2 size={14} /> {deleting ? t("删除中", "Deleting") : t(`删除选中 (${deleteIds.length})`, `Delete selected (${deleteIds.length})`)}</button>
            {deleteError && <p role="alert" className="action-feedback error">{deleteError}</p>}
          </div>
          {records.map((run) => (
            <div
              className={`run-record ${selected.run_id === run.run_id ? "active" : ""}`}
              key={run.run_id}
            >
              <button
                aria-pressed={selected.run_id === run.run_id}
                onClick={() => {
                  setSelectedId(run.run_id);
                  setExportStatus("");
                }}
              >
                <span>
                  <strong>{runDisplayName(run, locale)}</strong>
                  <small>
                    {t("项目", "Project")} {shortIdentifier(run.project_id)} · {t("随机种子", "Seed")} {String(run.parameters?.seed ?? t("未指定", "Not specified"))}
                  </small>
                </span>
                <em className={`run-state ${run.status}`}>{runStatusLabel(run.status, locale)}</em>
              </button>
              <label title={t("加入双运行对比", "Add to two-run comparison")}>
                <input
                  type="checkbox"
                  checked={compareIds.includes(run.run_id)}
                  onChange={() => toggleCompare(run.run_id)}
                />{" "}
                {t("对比", "Compare")}
              </label>
              <label className="run-delete-option" title={["completed", "failed", "stopped"].includes(run.status) ? t("选择删除", "Select for deletion") : t("先停止运行才可删除", "Stop the run before deleting")}>
                <input type="checkbox" aria-label={t(`选择删除运行 ${shortIdentifier(run.run_id)}`, `Select run ${shortIdentifier(run.run_id)} for deletion`)} disabled={deleting || !["completed", "failed", "stopped"].includes(run.status) || (deleteIds.length >= 10 && !deleteIds.includes(run.run_id))} checked={deleteIds.includes(run.run_id)} onChange={(event) => setDeleteIds((current) => event.target.checked ? [...current, run.run_id] : current.filter((id) => id !== run.run_id))} />
                {t("删除", "Delete")}
              </label>
              <a
                className="run-record-detail"
                href={runDetailPath(run.run_id)}
                aria-label={t(`查看运行 ${shortIdentifier(run.run_id)} 完整详情`, `View full details for run ${shortIdentifier(run.run_id)}`)}
                onClick={(event) => {
                  event.preventDefault();
                  navigate(runDetailPath(run.run_id));
                }}
              >
                <ExternalLink size={13} /> {t("查看详情", "View details")}
              </a>
            </div>
          ))}
        </aside>
        <section className="run-evidence" data-tour="selected-run">
          <div className="monitor-title">
            <div>
              <span>SELECTED RUN</span>
              <strong>{runDisplayName(selected, locale)} <span className={`run-state ${selected.status}`}>{selected.status}</span></strong>
            </div>
          </div>
          <div className="run-facts">
            <div>
              <span>PROJECT</span>
              <strong>{selected.project_id}</strong>
            </div>
            <div>
              <span>CLASS</span>
              <strong>{selected.class_id ?? "--"}</strong>
            </div>
            <div>
              <span>SEED</span>
              <strong>{String(selected.parameters?.seed ?? "--")}</strong>
            </div>
            <div>
              <span>UPDATES</span>
              <strong>{selected.update_steps}</strong>
            </div>
          </div>
          <section className="frozen-configuration" data-tour="run-config">
            <div className="run-section-heading">
              <div>
                <span>FROZEN CONFIGURATION</span>
                <h2>{t("服务端冻结配置", "Server-frozen configuration")}</h2>
              </div>
              <div className="json-tools" aria-label={t("配置显示控制", "Configuration display controls")}>
                <button title={t("复制完整 JSON", "Copy full JSON")} onClick={() => void copyFrozenConfiguration()}><Copy size={14} /> {configCopyStatus || t("复制", "Copy")}</button>
                <button aria-pressed={configExpanded} onClick={() => setConfigExpanded((value) => !value)}><ExternalLink size={14} /> {configExpanded ? t("收起", "Collapse") : t("展开", "Expand")}</button>
                <button aria-pressed={configWrapped} onClick={() => setConfigWrapped((value) => !value)}><Code2 size={14} /> {t("换行", "Wrap")}</button>
              </div>
            </div>
            <div className="configuration-summary-strip">
              <div><span>SCENE</span><strong>{selected.simulation_task_id ?? "--"}</strong></div>
              <div><span>ROBOT</span><strong>{String(selected.project_config_snapshot.robot_id ?? selected.project_config_snapshot.robot ?? "--")}</strong></div>
              <div><span>CONTROL</span><strong>{String(selected.project_config_snapshot.control_mode ?? selected.project_config_snapshot.strategy_id ?? "--")}</strong></div>
              <div><span>SEED</span><strong>{String(selected.parameters?.seed ?? "--")}</strong></div>
            </div>
            <pre className={`${configExpanded ? "expanded" : ""} ${configWrapped ? "wrapped" : ""}`.trim()}>{JSON.stringify(selected.project_config_snapshot, null, 2)}</pre>
          </section>
          <div className="run-section-heading event-evidence-heading" data-tour="event-evidence">
            <div><span>EVENT EVIDENCE</span><h2>{t("事件证据", "Event evidence")}</h2></div>
          </div>
          <div className="event-timeline" aria-label={t("运行状态轨迹", "Run status timeline")}>
            {["CREATED", "QUEUED", "RUNNING", ["completed", "failed", "stopped"].includes(selected.status) ? selected.status.toUpperCase() : "TERMINAL"].map((step, index) => {
              const reachedIndex = ["queued", "initializing"].includes(selected.status) ? 1 : ["running", "finalizing", "stopping"].includes(selected.status) ? 2 : 3;
              const terminalState = index === 3 && ["failed", "stopped"].includes(selected.status) ? selected.status : "";
              return <span key={`${index}-${step}`} className={`${index <= reachedIndex ? "reached" : ""} ${terminalState}`.trim()}>{step}</span>;
            })}
          </div>
          <div className="event-log">
            {(events.data ?? []).length === 0 && (
              <div className="event-empty"><strong>{t("暂无事件证据", "No event evidence yet")}</strong><span>{t("运行进入执行或终态后，证据会显示在这里。", "Evidence will appear here when the run enters running or a terminal state.")}</span></div>
            )}
            {(events.data ?? []).slice(-12).map((event) => (
              <div key={event.event_id}>
                <code>{event.type}</code>
                <span>
                  {event.payload.step !== undefined
                    ? `step ${event.payload.step}`
                    : String(event.payload.status ?? "event")}
                </span>
                <small>{event.event_id.slice(0, 8)}</small>
              </div>
            ))}
          </div>
        </section>
        <aside className="run-tools" data-tour="evidence-actions">
          <div className="section-label">EVIDENCE ACTIONS</div>
          <span className="action-group-label">PRIMARY</span>
          <button
            className="primary"
            onClick={() => navigate(runDetailPath(selected.run_id))}
          >
            <ExternalLink size={16} /> {t("查看完整运行详情", "View full run details")}
          </button>
          <span className="action-group-label">SECONDARY</span>
          <button
            className="secondary"
            onClick={() => void compare()}
            disabled={compareIds.length !== 2 || comparing}
            title={compareIds.length !== 2 ? t("需要选择两个 Run", "Requires two selected runs") : undefined}
          >
            <Settings2 size={16} /> {comparing ? t("正在对比", "Comparing") : compareError ? t("重试对比", "Retry comparison") : t("对比选中运行", "Compare selected runs")}
          </button>
          {compareIds.length !== 2 && <small className="disabled-reason">{t("需要选择两个 Run", "Requires two selected runs")}</small>}
          {compareError && <p className="action-feedback error" aria-live="polite">{compareError}</p>}
          <button
            className="secondary"
            onClick={() => navigate("/evaluation")}
            disabled={
              selected.status === "running" || selected.status === "queued"
            }
            title={selected.status === "running" || selected.status === "queued" ? t("Run 结束后可用", "Available after the run ends") : undefined}
          >
            <TestTube2 size={16} /> {t("进入证据评估", "Open evidence evaluation")}
          </button>
          {(selected.status === "running" || selected.status === "queued") && <small className="disabled-reason">{t("Run 结束后可用", "Available after the run ends")}</small>}
          <button
            className="secondary"
            onClick={() => void exportManifest()}
            disabled={selected.status !== "completed" || exporting}
            title={selected.status !== "completed" ? t("Run 完成后可下载", "Available after run completion") : undefined}
          >
            <Download size={16} /> {exporting ? t("正在生成", "Generating") : exportStatus.startsWith("error:") ? t("重试下载", "Retry download") : t("下载 JSON", "Download JSON")}
          </button>
          {selected.status !== "completed" && <small className="disabled-reason">{t("Run 完成后可下载", "Available after run completion")}</small>}
          {exportStatus && (
            <p
              className={`action-feedback ${exportStatus.startsWith("error:") ? "error" : "success"}`}
              aria-live="polite"
            >
              {exportStatus.replace(/^(success|error):/, "")}
            </p>
          )}
          <div className="sim2real-block" data-tour="real-control-boundary">
            <CircleStop size={17} />
            <span>
              <small>SIM2REAL</small>
              <strong>BLOCKED</strong>
              <small>MANIFEST DOES NOT ENABLE CONTROL</small>
              <p>{t("此运行配置不可执行真实机械臂控制。需要硬件清单、Robot Gateway 与操作员授权。", "Real robot execution is unavailable for this run. A hardware manifest, Robot Gateway, and operator approval are required.")}</p>
            </span>
          </div>
        </aside>
      </section>
    </main>
  );
}

function RunComparisonPage() {
  const { locale, t } = useI18n();
  const runs = useQuery({ queryKey: ["comparison-runs"], queryFn: () => api<RunRecord[]>("/api/v1/runs") });
  const [selectedIds, setSelectedIds] = useState<string[]>(() => (new URLSearchParams(location.search).get("runs") ?? "").split(",").filter(Boolean).slice(0, 2));
  const [comparison, setComparison] = useState<RunComparisonResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const terminalRuns = (runs.data ?? []).filter((run) => ["completed", "failed", "stopped"].includes(run.status));
  function toggle(runId: string) {
    setComparison(null); setError("");
    setSelectedIds((current) => current.includes(runId) ? current.filter((id) => id !== runId) : current.length < 2 ? [...current, runId] : [current[1], runId]);
  }
  async function compare() {
    if (selectedIds.length !== 2) return;
    setBusy(true); setError("");
    try {
      setComparison(await api<RunComparisonResponse>("/api/v1/runs/compare", { method: "POST", body: JSON.stringify({ run_ids: selectedIds }) }));
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("对比请求失败", "Comparison failed")); }
    finally { setBusy(false); }
  }
  if (runs.isLoading) return <div className="loading">{t("正在读取可对比运行…", "Reading comparable runs…")}</div>;
  if (runs.isError) return <div className="loading error">{t("运行记录 API 不可用。", "Run records API unavailable.")}</div>;
  return <main className="comparison-page">
    <header className="operations-header"><div><span className="eyebrow">RUN COMPARISON</span><h1>{t("运行结果对比", "Run result comparison")}</h1><p>{t("选择两次已结束运行，比较冻结配置、指标历史、失败证据和可用视频。对比不会把差异自动解释成因果关系。", "Select two terminal Runs to compare frozen configuration, metric history, failure evidence and available video. Comparison does not infer causality automatically.")}</p></div><button className="primary" disabled={selectedIds.length !== 2 || busy} onClick={() => void compare()}><GitCompare size={16} /> {busy ? t("对比中…", "Comparing…") : t("开始对比", "Compare runs")}</button></header>
    <section className="comparison-picker"><div className="comparison-picker-heading"><strong>{t("选择两次实验", "Choose two experiments")}</strong><span>{selectedIds.length}/2</span></div><div className="comparison-run-grid">{terminalRuns.map((run) => <button key={run.run_id} className={selectedIds.includes(run.run_id) ? "selected" : ""} onClick={() => toggle(run.run_id)} aria-pressed={selectedIds.includes(run.run_id)}><span><strong>{runDisplayName(run, locale)}</strong><small>{shortIdentifier(run.run_id)} · {run.status}</small></span><Check size={16} /></button>)}</div>{!terminalRuns.length && <p>{t("暂无已结束运行。", "No terminal Runs yet.")}</p>}{error && <p className="action-feedback error" role="alert">{error}</p>}</section>
    {comparison && <RunComparisonView comparison={comparison} runs={runs.data ?? []} />}
  </main>;
}

function EvaluationPage() {
  const { locale, t } = useI18n();
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const runs = useQuery({
    queryKey: ["evaluation-runs"],
    queryFn: () => api<RunRecord[]>("/api/v1/runs"),
  });
  const reports = useQuery({
    queryKey: ["evaluation-reports", selectedRunIds[0]],
    queryFn: () => api<EvaluationReport[]>(`/api/v1/evaluations?run_id=${encodeURIComponent(selectedRunIds[0])}`),
    enabled: Boolean(selectedRunIds[0]),
  });
  const completedRuns = (runs.data ?? []).filter(
    (run) => !["queued", "running"].includes(run.status),
  );
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [baselineId, setBaselineId] = useState("");
  const [comparison, setComparison] = useState<EvaluationComparison | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  useEffect(() => setEvaluationError(""), [locale]);
  const selectedRun =
    completedRuns.find((run) => run.run_id === selectedRunIds[0]) ??
    completedRuns[0];
  const cohortCandidates = completedRuns.filter(
    (run) =>
      run.project_id === selectedRun?.project_id &&
      run.class_id === selectedRun?.class_id,
  );
  const baselineReports = (reports.data ?? []).filter(
    (item) =>
      item.report_id !== report?.report_id &&
      item.run_ids.some((runId) =>
        cohortCandidates.some((run) => run.run_id === runId),
      ),
  );

  useEffect(() => {
    if (selectedRunIds.length === 0 && completedRuns[0])
      setSelectedRunIds([completedRuns[0].run_id]);
  }, [completedRuns, selectedRunIds]);

  function toggleCohortRun(runId: string) {
    setReport(null);
    setComparison(null);
    setEvaluationError("");
    setSelectedRunIds((current) =>
      current.includes(runId)
        ? current.filter((id) => id !== runId)
        : [...current, runId],
    );
  }

  async function evaluate() {
    if (!selectedRun || selectedRunIds.length === 0) return;
    setBusy(true);
    setEvaluationError("");
    try {
      const created = await api<EvaluationReport>("/api/v1/evaluations", {
        method: "POST",
        body: JSON.stringify({ run_id: selectedRun.run_id }),
      });
      setReport(created);
      setComparison(null);
      await reports.refetch();
    } catch (error) {
      setEvaluationError(
        error instanceof Error ? error.message : t("评估请求失败", "Evaluation request failed"),
      );
    } finally {
      setBusy(false);
    }
  }

  async function compareReports() {
    if (!report || !baselineId) return;
    setEvaluationError("");
    try {
      setComparison(
        await api<EvaluationComparison>("/api/evaluations/compare", {
          method: "POST",
          body: JSON.stringify({
            before_report_id: baselineId,
            after_report_id: report.report_id,
          }),
        }),
      );
    } catch (error) {
      setEvaluationError(
        error instanceof Error ? error.message : t("报告比较失败", "Report comparison failed"),
      );
    }
  }

  function openEvidence(reference: string) {
    const parts = reference.split(":");
    const runId = parts[0] === "run" ? parts[1] : report?.run_ids[0];
    if (runId) navigate(`/runs?run=${encodeURIComponent(runId)}`);
  }

  if (runs.isLoading)
    return <div className="loading">正在读取可评估运行...</div>;
  if (runs.isError)
    return (
      <div className="loading error">
        运行记录 API 不可用，无法建立证据评估。
      </div>
    );
  if (!selectedRun)
    return (
      <main className="evaluation-empty">
        <TestTube2 size={34} />
        <span className="eyebrow">NO EVALUATION EVIDENCE</span>
        <h1>尚无可评估运行</h1>
        <p>
          先完成一次正式教学短训练。评估中心只读取实际运行证据，不会生成推测分数。
        </p>
        <button className="primary" onClick={() => navigate("/quick-course")}>
          <Play size={16} /> 进入快速课程
        </button>
      </main>
    );

  return (
    <main className="evaluation-page">
      <header className="evaluation-header">
        <div>
          <span className="eyebrow">EVIDENCE-BOUND EVALUATION</span>
          <h1>环境与动作设计评估</h1>
          <p>
            每项结论都绑定到 Run 事件或制品。选择同环境、同策略的多个 seed
            可评估同一运行配置的复现性；这仍不能证明泛化或真实部署安全。
          </p>
        </div>
        <div className="evaluation-action">
          <span>
            {selectedRunIds.length >= 2
              ? `${selectedRunIds.length} RUN COHORT`
              : "SINGLE RUN REPORT"}
          </span>
          <button
            className="primary"
            onClick={evaluate}
            disabled={busy || selectedRunIds.length === 0}
          >
            <TestTube2 size={16} /> {busy ? "评估中" : "生成证据报告"}
          </button>
        </div>
      </header>
      <section className="evaluation-workspace">
        <aside className="evaluation-context">
          <div className="section-label">RUN COHORT</div>
          <div className="cohort-selector">
            {cohortCandidates.map((run) => (
              <label
                key={run.run_id}
                className={selectedRunIds.includes(run.run_id) ? "active" : ""}
              >
                <input
                  type="checkbox"
                  checked={selectedRunIds.includes(run.run_id)}
                  onChange={() => toggleCohortRun(run.run_id)}
                />
                <span>
                  <strong>{run.run_id.slice(-8)}</strong>
                  <small>
                    seed {String(run.parameters?.seed ?? "--")} · {run.status}
                  </small>
                </span>
              </label>
            ))}
          </div>
          <dl>
            <div>
              <dt>PROJECT</dt>
              <dd>{selectedRun.project_id}</dd>
            </div>
            <div>
              <dt>CLASS</dt>
              <dd>{selectedRun.class_id ?? "--"}</dd>
            </div>
            <div>
              <dt>SELECTED</dt>
              <dd>{selectedRunIds.length} RUNS</dd>
            </div>
          </dl>
          {evaluationError && <p className="inline-error">{evaluationError}</p>}
          <div className="evaluation-warning">
            <ShieldCheck size={17} />
            <span>
              <strong>MOCK ONLY</strong>
              <p>
                不同 seed 仅支持复现性证据，不构成 held-out 泛化或 Sim2Real
                证明。
              </p>
            </span>
          </div>
        </aside>
        <section className="rubric-panel">
          <div className="rubric-title">
            <span>RUBRIC CRITERIA</span>
            <small>
              {report ? `REPORT ${report.report_id.slice(-8)}` : "等待生成报告"}
            </small>
          </div>
          {!report && (
            <div className="rubric-idle">
              <TestTube2 size={28} />
              <h2>选择现有运行并生成报告</h2>
              <p>
                Evaluator
                将检查实际可用通道。缺失复现性、泛化或安全证据时不会补造分数。
              </p>
            </div>
          )}
          {report?.criteria.map((criterion) => (
            <article
              className={`criterion-row ${criterion.status}`}
              key={criterion.criterion_id}
            >
              <div className="criterion-score">
                {criterion.score === null ? "—" : criterion.score}
                <small>/4</small>
              </div>
              <div>
                <div className="criterion-heading">
                  <strong>{criterion.name}</strong>
                  <span>
                    {criterion.weight}% · {criterion.status.replace("_", " ")}
                  </span>
                </div>
                <p>
                  {criterion.status === "assessed"
                    ? criterion.finding
                    : `缺少：${criterion.missing_evidence}`}
                </p>
                {criterion.evidence_refs.length > 0 && (
                  <div className="evidence-refs">
                    {criterion.evidence_refs.map((reference) => (
                      <button
                        key={reference}
                        onClick={() => openEvidence(reference)}
                        title="在运行记录中查看证据"
                      >
                        <code>{reference}</code>
                        <ExternalLink size={11} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          {report && (
            <section className="report-comparison">
              <div className="rubric-title">
                <span>BEFORE / AFTER</span>
                <small>RUBRIC {report.rubric_version}</small>
              </div>
              <div className="comparison-controls">
                <select
                  aria-label="基线评估报告"
                  value={baselineId}
                  onChange={(event) => {
                    setBaselineId(event.target.value);
                    setComparison(null);
                  }}
                >
                  <option value="">选择同环境历史报告</option>
                  {baselineReports.map((item) => (
                    <option key={item.report_id} value={item.report_id}>
                      {item.report_id.slice(-8)} ·{" "}
                      {Math.round(item.evidence_completeness * 100)}%
                    </option>
                  ))}
                </select>
                <button
                  className="secondary"
                  onClick={compareReports}
                  disabled={!baselineId}
                >
                  <GitCompare size={15} /> 比较评估
                </button>
              </div>
              {comparison && (
                <>
                  <div className="comparison-overview">
                    <span>
                      完整度{" "}
                      <strong>
                        {comparison.completeness_delta >= 0 ? "+" : ""}
                        {Math.round(comparison.completeness_delta * 100)}%
                      </strong>
                    </span>
                    <span>
                      总分{" "}
                      <strong>
                        {comparison.overall.delta === null
                          ? "WITHHELD"
                          : `${comparison.overall.delta >= 0 ? "+" : ""}${comparison.overall.delta}`}
                      </strong>
                    </span>
                  </div>
                  <div className="criterion-changes">
                    {comparison.criteria
                      .filter((item) => item.change !== "unchanged")
                      .map((item) => (
                        <div className={item.change} key={item.criterion_id}>
                          <span>{item.name}</span>
                          <strong>
                            {item.before.score ?? "—"} →{" "}
                            {item.after.score ?? "—"}
                          </strong>
                          <small>
                            {item.change.replace("_", " ")}
                            {item.delta === null
                              ? ""
                              : ` ${item.delta > 0 ? "+" : ""}${item.delta}`}
                          </small>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </section>
          )}
        </section>
        <aside className="evaluation-summary">
          <div className="section-label">REPORT SUMMARY</div>
          {!report ? (
            <p className="summary-empty">
              报告生成后，这里会显示证据完整度、总分状态和下一次实验。
            </p>
          ) : (
            <>
              <div className="completeness">
                <strong>
                  {Math.round(report.evidence_completeness * 100)}%
                </strong>
                <span>证据完整度</span>
                <div>
                  <i
                    style={{ width: `${report.evidence_completeness * 100}%` }}
                  />
                </div>
              </div>
              <div className="overall-status">
                <span>OVERALL</span>
                <strong>{report.overall.score ?? "WITHHELD"}</strong>
                <small>{report.overall.status.replace("_", " ")}</small>
              </div>
              <h3>下一次需补证据</h3>
              <ul>
                {report.missing_evidence.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <div className="sim2real-block">
                <CircleStop size={17} />
                <span>
                  <strong>
                    SIM2REAL {report.sim2real.status.toUpperCase()}
                  </strong>
                  <small>REAL CONTROL DISABLED</small>
                </span>
              </div>
            </>
          )}
        </aside>
      </section>
    </main>
  );
}

function PolicyExportPage() {
  const { locale, t } = useI18n();
  const runs = useQuery({
    queryKey: ["export-runs"],
    queryFn: () => api<RunRecord[]>("/api/v1/runs"),
  });
  const reports = useQuery({
    queryKey: ["export-reports"],
    queryFn: () => api<EvaluationReport[]>("/api/evaluations"),
  });
  const completedRuns = (runs.data ?? []).filter(
    (run) => run.status === "completed",
  );
  const [runId, setRunId] = useState("");
  const [reportId, setReportId] = useState("");
  const [format, setFormat] = useState<"manifest" | "onnx" | "torchscript">(
    "manifest",
  );
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [exportError, setExportError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => setExportError(""), [locale]);
  const selectedRun =
    completedRuns.find((run) => run.run_id === runId) ?? completedRuns[0];
  const compatibleReports = (reports.data ?? []).filter(
    (report) => selectedRun && report.run_ids.includes(selectedRun.run_id),
  );
  const selectedReport = compatibleReports.find(
    (report) => report.report_id === reportId,
  );

  useEffect(() => {
    if (!runId && completedRuns[0]) setRunId(completedRuns[0].run_id);
  }, [completedRuns, runId]);

  useEffect(() => {
    setReportId(compatibleReports[0]?.report_id ?? "");
    setResult(null);
    setExportError("");
  }, [selectedRun?.run_id, reports.data]);

  async function exportPolicy() {
    if (!selectedRun) return;
    setBusy(true);
    setResult(null);
    setExportError("");
    try {
      setResult(
        await apiWithFallback<Record<string, unknown>>(
          "/api/v1/runs/export",
          `/api/runs/${selectedRun.run_id}/export`,
          {
            method: "POST",
            body: JSON.stringify({
              run_ids: [selectedRun.run_id],
              format: "json",
            }),
          },
        ),
      );
    } catch (error) {
      setExportError(error instanceof Error ? error.message : t("导出请求失败", "Export request failed"));
    } finally {
      setBusy(false);
    }
  }

  if (runs.isLoading || reports.isLoading)
    return <div className="loading">正在读取导出证据...</div>;
  if (runs.isError || reports.isError)
    return <div className="loading error">导出证据 API 不可用。</div>;
  if (!selectedRun)
    return (
      <main className="evaluation-empty">
        <FileCheck2 size={34} />
        <span className="eyebrow">NO COMPLETED RUN</span>
        <h1>尚无可导出运行</h1>
        <p>
          完成一个 Mock 训练运行后，才能生成教学 Manifest
          或提交模型格式门禁检查。
        </p>
        <button className="primary" onClick={() => navigate("/training")}>
          <Play size={16} /> 进入训练
        </button>
      </main>
    );

  return (
    <main className="export-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">POLICY EXPORT GATE</span>
          <h1>策略导出与复现清单</h1>
          <p>
            导出只封装当前 Mock
            运行的策略契约、关节顺序、限制与校验和。模型格式必须关联通过证据门禁的持久化报告，任何结果都不会启用真实机械臂控制。
          </p>
        </div>
        <div className="operations-boundary">
          <CircleStop size={18} />
          <span>
            <strong>SIM2REAL BLOCKED</strong>
            <small>OPERATOR REVIEW REQUIRED</small>
          </span>
        </div>
      </header>
      <section className="export-workspace">
        <aside className="export-config">
          <div className="section-label">EXPORT REQUEST</div>
          <label>
            完成运行
            <select
              value={selectedRun.run_id}
              onChange={(event) => {
                setRunId(event.target.value);
                setFormat("manifest");
              }}
            >
              {completedRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id.slice(-8)} · {run.project_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            评估报告
            <select
              value={reportId}
              onChange={(event) => {
                setReportId(event.target.value);
                setResult(null);
              }}
            >
              <option value="">未关联报告</option>
              {compatibleReports.map((report) => (
                <option key={report.report_id} value={report.report_id}>
                  {report.report_id.slice(-8)} ·{" "}
                  {Math.round(report.evidence_completeness * 100)}%
                </option>
              ))}
            </select>
          </label>
          <div className="format-selector" aria-label="导出格式">
            {(["manifest", "onnx", "torchscript"] as const).map((item) => (
              <button
                key={item}
                className={format === item ? "active" : ""}
                onClick={() => {
                  setFormat(item);
                  setResult(null);
                  setExportError("");
                }}
              >
                {item}
              </button>
            ))}
          </div>
          <button
            className="primary"
            onClick={exportPolicy}
            disabled={busy || (format !== "manifest" && !reportId)}
          >
            <Download size={16} /> {busy ? "检查中" : "执行门禁并导出"}
          </button>
          {exportError && <p className="inline-error">{exportError}</p>}
        </aside>
        <section className="export-evidence">
          <div className="monitor-title">
            <div>
              <span>SELECTED RUN</span>
              <strong>{selectedRun.run_id}</strong>
            </div>
            <span className="run-state completed">completed</span>
          </div>
          <div className="export-contract">
            <div>
              <span>PROJECT</span>
              <strong>{selectedRun.project_id}</strong>
            </div>
            <div>
              <span>CLASS</span>
              <strong>{selectedRun.class_id ?? "--"}</strong>
            </div>
            <div>
              <span>FORMAT</span>
              <strong>{format}</strong>
            </div>
            <div>
              <span>CONTROL</span>
              <strong>DISABLED</strong>
            </div>
          </div>
          {result ? (
            <>
              <div className="gate-result">
                <Check size={20} />
                <span>
                  <strong>导出响应已生成</strong>
                  <small>服务端门禁结果已写入制品</small>
                </span>
              </div>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </>
          ) : (
            <div className="monitor-empty">
              <FileCheck2 size={24} />
              <strong>等待导出请求</strong>
              <span>结果将包含 checksum、safety checks 与 export gate。</span>
            </div>
          )}
        </section>
        <aside className="gate-inspector">
          <div className="section-label">EVALUATION GATE</div>
          {selectedReport ? (
            <>
              <div className="completeness">
                <strong>
                  {Math.round(selectedReport.evidence_completeness * 100)}%
                </strong>
                <span>证据完整度</span>
                <div>
                  <i
                    style={{
                      width: `${selectedReport.evidence_completeness * 100}%`,
                    }}
                  />
                </div>
              </div>
              <dl>
                <div>
                  <dt>REPORT</dt>
                  <dd>{selectedReport.report_id.slice(-8)}</dd>
                </div>
                <div>
                  <dt>OVERALL</dt>
                  <dd>{selectedReport.overall.status}</dd>
                </div>
                <div>
                  <dt>SCORE</dt>
                  <dd>{selectedReport.overall.score ?? "WITHHELD"}</dd>
                </div>
                <div>
                  <dt>RUNS</dt>
                  <dd>{selectedReport.run_ids.length}</dd>
                </div>
              </dl>
            </>
          ) : (
            <p className="summary-empty">
              Manifest 可在无报告时生成。ONNX 和 TorchScript
              请求必须关联满足完整度、总评和 mandatory criteria 的报告。
            </p>
          )}
          <div className="sim2real-block">
            <CircleStop size={17} />
            <span>
              <strong>REAL CONTROL DISABLED</strong>
              <small>EXPORT IS NOT DEPLOYMENT</small>
            </span>
          </div>
        </aside>
      </section>
    </main>
  );
}

function EnvironmentPage() {
  const { locale, t } = useI18n();
  const scenes = useQuery({
    queryKey: ["scenes"],
    queryFn: () => api<SceneDocument[]>("/api/scenes"),
  });
  const s1Scenes = (scenes.data ?? []).filter((item) => item.capability !== "mock" && item.robot.dof === S1_ARM_JOINTS.length);
  const [scene, setScene] = useState<SceneDocument | null>(null);
  const [sceneHistory, setSceneHistory] = useState<SceneDocument[]>([]);
  const [selectedObjectId, setSelectedObjectId] = useState("target-01");
  const [jointPositions, setJointPositions] = useState(S1_INITIAL_ARM_POSITIONS);
  const [message, setMessage] = useState("");
  const [sceneError, setSceneError] = useState("");
  const [scenePreflight, setScenePreflight] = useState<{
    valid: boolean;
    trainable_push: boolean;
    trainable_place: boolean;
    errors: string[];
    warnings: string[];
  } | null>(null);
  const [boxDraft, setBoxDraft] = useState({
    label: "",
    x: 0.4,
    y: 0.25,
    z: 0.3,
    size: 0.5,
    color: "#2d6cdf",
    static: true,
  });
  const [showBoxComposer, setShowBoxComposer] = useState(false);
  const setActiveScene = useStudio((state) => state.setActiveScene);
  useEffect(() => {
    setMessage("");
    setSceneError("");
  }, [locale]);
  useEffect(() => {
    if (s1Scenes[0] && !scene) {
      const draftSceneId = sessionStorage.getItem("s1-draft-scene");
      const initialScene = s1Scenes.find((item) => item.scene_id === draftSceneId) ?? s1Scenes[0];
      setScene(initialScene);
      setActiveScene(initialScene.scene_id);
      sessionStorage.setItem("s1-draft-scene", initialScene.scene_id);
      void api<NonNullable<typeof scenePreflight>>(`/api/scenes/${encodeURIComponent(initialScene.scene_id)}/preflight`).then(setScenePreflight).catch(() => setScenePreflight(null));
    }
  }, [scenes.data, scene, setActiveScene]);

  const selectScene = (nextScene: SceneDocument) => {
    setScene(nextScene);
    setSceneHistory([]);
    setActiveScene(nextScene.scene_id);
    sessionStorage.setItem("s1-draft-scene", nextScene.scene_id);
    setSelectedObjectId(nextScene.objects[0]?.object_id ?? "");
    setJointPositions([...S1_INITIAL_ARM_POSITIONS]);
    setMessage("");
    setSceneError("");
    void api<NonNullable<typeof scenePreflight>>(`/api/scenes/${encodeURIComponent(nextScene.scene_id)}/preflight`).then(setScenePreflight).catch(() => setScenePreflight(null));
  };

  if (scenes.isLoading)
    return <div className="loading">{t("正在读取环境与场景契约...", "Reading environment and scene contracts...")}</div>;
  if (scenes.isError || !scene)
    return (
      <div className="loading error">
        {t("场景 API 不可用，请确认服务已更新并启动。", "The scene API is unavailable. Confirm that the service is updated and running.")}
      </div>
    );

  const selectedObject =
    scene.objects.find((item) => item.object_id === selectedObjectId) ??
    scene.objects[0];
  const commitSceneEdit = (nextScene: SceneDocument) => {
    setSceneHistory((history) => [
      ...history.slice(-49),
      {
        ...scene,
        robot: { ...scene.robot },
        objects: scene.objects.map((item) => ({
          ...item,
          position: [...item.position],
          rotation: [...item.rotation],
          scale: [...item.scale],
        })),
      },
    ]);
    setScene(nextScene);
    setMessage("");
    setSceneError("");
    setScenePreflight(null);
  };
  const undoSceneEdit = () => {
    const previousScene = sceneHistory[sceneHistory.length - 1];
    if (!previousScene) return;
    setScene(previousScene);
    setSelectedObjectId((currentId) =>
      previousScene.objects.some((item) => item.object_id === currentId)
        ? currentId
        : previousScene.objects[0]?.object_id ?? "",
    );
    setSceneHistory((history) => history.slice(0, -1));
    setMessage("");
    setSceneError("");
  };
  const updateJointPosition = (index: number, value: number) => {
    setJointPositions((current) => current.map((position, itemIndex) =>
      itemIndex === index ? clampS1Joint(index, value) : position,
    ));
  };
  const addObject = (type: SceneObject["type"]) => {
    if (type === "plane") return;
    const objectLabels: Partial<Record<SceneObject["type"], string>> = {
      box: t("新方块", "New box"),
      cylinder: t("新圆柱", "New cylinder"),
      sphere: t("新球体", "New sphere"),
      target: t("新目标区域", "New target area"),
      wall: t("新墙体", "New wall"),
    };
    const object: SceneObject = {
      object_id: `${type}-${Date.now().toString(36)}`,
      type,
      label: objectLabels[type] ?? t("新对象", "New object"),
      position: [0.4, type === "target" ? 0.02 : 0.25, 0.3],
      rotation: [0, 0, 0],
      scale: type === "wall" ? [1.4, 0.7, 1] : [0.5, 0.5, 0.5],
      color: type === "target" ? "#e05a3f" : "#2d6cdf",
      static: true,
    };
    commitSceneEdit({ ...scene, objects: [...scene.objects, object] });
    setSelectedObjectId(object.object_id);
  };
  const addConfiguredBox = () => {
    const size = Math.max(0.05, boxDraft.size);
    const object: SceneObject = {
      object_id: `box-${Date.now().toString(36)}`,
      type: "box",
      label: boxDraft.label.trim() || t("训练方块", "Training box"),
      position: [boxDraft.x, Math.max(size / 2, boxDraft.y), boxDraft.z],
      rotation: [0, 0, 0],
      scale: [size, size, size],
      color: boxDraft.color,
      static: boxDraft.static,
    };
    commitSceneEdit({ ...scene, objects: [...scene.objects, object] });
    setSelectedObjectId(object.object_id);
    setShowBoxComposer(false);
  };
  const updateObject = (
    key: "position" | "rotation" | "scale",
    index: number,
    value: number,
  ) => {
    if (!selectedObject) return;
    const objects = scene.objects.map((item) =>
      item.object_id === selectedObject.object_id
        ? {
            ...item,
            [key]: item[key].map((part, itemIndex) =>
              itemIndex === index ? value : part,
            ),
          }
        : item,
    );
    commitSceneEdit({ ...scene, objects });
  };
  const deleteSelectedObject = () => {
    if (!selectedObject || selectedObject.type === "plane") return;
    const nextObjects = scene.objects.filter(
      (item) => item.object_id !== selectedObject.object_id,
    );
    commitSceneEdit({ ...scene, objects: nextObjects });
    setSelectedObjectId(nextObjects[0]?.object_id ?? "");
  };
  async function saveScene() {
  setSceneHistory([]);
    setSceneError("");
    try {
      const saved = await api<SceneDocument>("/api/scenes", {
        method: "POST",
        body: JSON.stringify({ document: scene }),
      });
      setScene(saved);
      sessionStorage.setItem("s1-draft-scene", saved.scene_id);
      const preflight = await api<NonNullable<typeof scenePreflight>>(`/api/scenes/${encodeURIComponent(saved.scene_id)}/preflight`);
      setScenePreflight(preflight);
      setMessage(`${t("已保存", "Saved")} ${saved.scene_id} · REV ${saved.revision}`);
    } catch (caught) {
      setMessage("");
      setSceneError(caught instanceof Error ? caught.message : t("场景保存失败", "Scene save failed"));
    }
  }
  async function exportScene() {
    const currentScene = scene;
    if (!currentScene) return;
    setSceneError("");
    try {
      const result = await api<{ scene: SceneDocument }>(
        `/api/scenes/${currentScene.scene_id}/export`,
      );
      const blob = new Blob([JSON.stringify(result.scene, null, 2)], {
        type: "application/json",
      });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${currentScene.scene_id}-rev-${currentScene.revision}.json`;
      link.click();
      setMessage(t("场景 manifest 已导出", "Scene manifest exported"));
    } catch (caught) {
      setMessage("");
      setSceneError(caught instanceof Error ? caught.message : t("场景导出失败", "Scene export failed"));
    }
  }

  return (
    <main className="environment-page">
      <header className="environment-header">
        <div>
          <span className="eyebrow">S1 SCENE STUDIO</span>
          <h1>{t("六自由度场景设计", "Six-DOF scene design")}</h1>
          <p>{t("在 S1 场景中修改目标、障碍物和物体位置；关节姿态只供结构预览，不代替 Isaac Gym 碰撞验证。", "Edit targets, obstacles and objects in an S1 scene. Joint poses are structural previews, not Isaac Gym collision validation.")}</p>
        </div>
        <div className="environment-count">
            <strong>6 + 2</strong>
            <span>ARM + GRIPPER DOF</span>
          <small>REAL CONTROL DISABLED</small>
        </div>
      </header>
      <section className="environment-workspace">
        <aside className="environment-list" data-tour="scene-revision">
          <div className="section-label">{t("版本化场景", "Versioned scenes")}</div>
          {s1Scenes.map((availableScene) => (
            <button
              key={availableScene.scene_id}
              className={scene.scene_id === availableScene.scene_id ? "active" : ""}
              onClick={() => selectScene(availableScene)}
            >
              <span className="environment-list-icon">
                <Boxes size={17} />
              </span>
              <span>
                <strong>{availableScene.name}</strong>
                <small>
                  {availableScene.robot.id} · REV {availableScene.revision}
                </small>
              </span>
              <span className={`capability ${availableScene.capability}`}>
                {availableScene.capability}
              </span>
            </button>
          ))}
          <div className="fidelity-note">
            <ShieldCheck size={18} />
            <span>
              <strong>{t("S1 模型资产边界", "S1 model asset boundary")}</strong>
              {t("网页仅加载 S1 v2 URDF/STL 视觉资产；不解析碰撞几何，也不提供安全或可达性结论。", "The web preview loads S1 v2 URDF/STL visual assets only; it does not validate collision, safety or reachability.")}
            </span>
          </div>
        </aside>
        <section className="environment-contract scene-studio-main">
          <div className="scene-toolbar" data-tour="scene-toolbar">
            <div>
              <span className="eyebrow">
                S1 V2 · SCENE REV {scene.revision}
              </span>
              <strong>{scene.name}</strong>
            </div>
            <div>
              <button
                className="secondary"
                onClick={undoSceneEdit}
                disabled={!sceneHistory.length}
                title={t("撤销上一步环境编辑", "Undo previous environment edit")}
              >
                <Undo2 size={14} /> {t("回退", "Undo")}
              </button>
              <button
                className="secondary"
                onClick={() => setShowBoxComposer((value) => !value)}
              >
                <Plus size={14} /> {t("方块", "Box")}
              </button>
              <button className="secondary" onClick={() => addObject("sphere")}>
                <Plus size={14} /> {t("球体", "Sphere")}
              </button>
              <button
                className="secondary"
                onClick={() => addObject("cylinder")}
              >
                <Plus size={14} /> {t("圆柱", "Cylinder")}
              </button>
              <button className="primary" data-tour="save-revision" onClick={saveScene}>
                <Save size={14} /> {t("保存修订", "Save revision")}
              </button>
              <button className="secondary" onClick={exportScene}>
                <Download size={14} /> {t("导出", "Export")}
              </button>
            </div>
          </div>
          {showBoxComposer && (
            <div className="box-composer">
              <strong>{t("添加方块到场景", "Add a box to the scene")}</strong>
              <label>
                {t("名称", "Name")}
                <input
                  value={boxDraft.label}
                  placeholder={t("训练方块", "Training box")}
                  onChange={(event) =>
                    setBoxDraft({ ...boxDraft, label: event.target.value })
                  }
                />
              </label>
              <label>
                {t("边长 (m)", "Edge length (m)")}
                <input
                  type="number"
                  min="0.05"
                  step="0.05"
                  value={boxDraft.size}
                  onChange={(event) =>
                    setBoxDraft({
                      ...boxDraft,
                      size: Number(event.target.value),
                    })
                  }
                />
              </label>
              <div className="composer-vector">
                <label>
                  X
                  <input
                    type="number"
                    step="0.05"
                    value={boxDraft.x}
                    onChange={(event) =>
                      setBoxDraft({
                        ...boxDraft,
                        x: Number(event.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Y
                  <input
                    type="number"
                    step="0.05"
                    value={boxDraft.y}
                    onChange={(event) =>
                      setBoxDraft({
                        ...boxDraft,
                        y: Number(event.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Z
                  <input
                    type="number"
                    step="0.05"
                    value={boxDraft.z}
                    onChange={(event) =>
                      setBoxDraft({
                        ...boxDraft,
                        z: Number(event.target.value),
                      })
                    }
                  />
                </label>
              </div>
              <label>
                {t("颜色", "Color")}
                <input
                  type="color"
                  value={boxDraft.color}
                  onChange={(event) =>
                    setBoxDraft({ ...boxDraft, color: event.target.value })
                  }
                />
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={boxDraft.static}
                  onChange={(event) =>
                    setBoxDraft({ ...boxDraft, static: event.target.checked })
                  }
                />{" "}
                {t("固定物体（取消后可推动）", "Static object (can be pushed when disabled)")}
              </label>
              <button className="primary" onClick={addConfiguredBox}>
                <Check size={14} /> {t("加入 3D 场景", "Add to 3D scene")}
              </button>
            </div>
          )}
          <div data-tour="scene-editor"><SceneEditor
            scene={scene}
            selectedObjectId={selectedObjectId}
            onSelect={setSelectedObjectId}
            jointPositions={jointPositions}
          /></div>
          <div className="scene-support-strip">
            <span><Check size={14} /><strong>{t("当前可保存", "Can be saved")}</strong> {t("目标与障碍物的位置、旋转、尺寸和静态属性", "target and obstacle position, rotation, size, and static properties")}</span>
            <span><CircleStop size={14} /><strong>{t("当前仅预览", "Preview only")}</strong> {t("S1 视觉网格与关节姿态；未计算工作空间或碰撞", "S1 visual meshes and joint poses; workspace and collisions are not calculated")}</span>
          </div>
          <div className={`scene-preflight ${scenePreflight?.trainable_push ? "pass" : "blocked"}`} role="status">
            {scenePreflight?.trainable_push ? <Check size={15} /> : <CircleStop size={15} />}
            <span>
              <strong>{scenePreflight?.trainable_push ? t("场景预检通过", "Scene preflight passed") : t("当前仅为视觉预览", "Visual preview only")}</strong>
              <small>{scenePreflight?.trainable_push ? t(`可用于 Push；Place 门禁：${scenePreflight.trainable_place ? "通过" : "未通过"}`, `Push-ready; Place gate: ${scenePreflight.trainable_place ? "passed" : "blocked"}`) : t("保存后仍需通过结构、工作区和 Worker 几何门禁才能训练。", "Saving is not enough; structure, workspace and Worker geometry gates must pass before training.")}</small>
            </span>
          </div>
          <div className="environment-task-compatibility">
            <div>
              <span className="eyebrow">NEXT STEP · TASK PARAMETERS</span>
              <strong>{t("场景保存后进入参数沙箱", "Continue to the parameter sandbox after saving")}</strong>
              <p>{t("这里编辑 S1 场景和姿态预览；下一步仅配置 Push 或 Reach 强化学习训练。控制演示零件可在项目工作区查看。", "Edit the S1 scene and pose preview here; next configure Push or Reach training. Control demo parts are listed in the project workspace.")}</p>
            </div>
            <button className="primary" data-tour="continue-parameters" onClick={() => navigate("/training")}><ChevronRight size={15} /> {t("进入参数沙箱", "Open parameter sandbox")}</button>
          </div>
          {message && (
            <div className="scene-message">
              <Check size={14} />
              {message}
            </div>
          )}
          {sceneError && <div className="scene-message error"><CircleAlert size={14} />{sceneError}</div>}
          <details className="advanced-contract">
            <summary>{t("查看环境输入、动作、奖励与终止契约", "View environment input, action, reward, and termination contracts")}</summary>
          <div className="fidelity-note">
            <CircleStop size={18} />
            <span><strong>{t("S1 运行合同未由 Scene API 提供", "The Scene API does not provide an S1 runtime contract")}</strong> {t("此处只保存六轴 S1 场景对象与资产引用；观测、动作、奖励和终止条件由所选 Push 或 Reach 训练任务定义。", "This editor stores six-axis S1 scene objects and assets; observations, actions, rewards and termination come from the selected Push or Reach task.")}</span>
          </div>
          </details>
        </section>
        <aside className="evidence-inspector scene-inspector">
          <div className="section-label">{t("S1 六轴姿态预览", "S1 six-axis pose preview")}</div>
          {<div className="s1-joint-controls" data-tour="joint-controls">
            <div className="joint-control-header">
              <span>URDF: yhrg_s1_v2 · EE: ee_link</span>
              <button className="secondary" onClick={() => setJointPositions([...S1_INITIAL_ARM_POSITIONS])}>
                <RotateCcw size={13} /> {t("标准姿态", "Reset pose")}
              </button>
            </div>
            {S1_ARM_JOINTS.map((joint, index) => (
              <label key={joint.name}>
                <span><strong>J{index + 1} · {s1JointLabel(joint, locale)}</strong><small>{joint.name}</small></span>
                <output>{jointPositions[index].toFixed(3)} rad</output>
                <input
                  type="range"
                  data-tour={index === 1 ? "joint-j2" : undefined}
                  min={joint.lower}
                  max={joint.upper}
                  step="0.01"
                  value={jointPositions[index]}
                  onChange={(event) => updateJointPosition(index, Number(event.target.value))}
                />
                <small>{joint.lower.toFixed(2)} / {joint.upper.toFixed(2)}</small>
              </label>
            ))}
          </div>}
          <div className="section-label">{t("场景对象", "Scene objects")}</div>
          <div className="object-tree" data-tour="scene-objects">
            {scene.objects.map((item) => (
              <button
                key={item.object_id}
                className={
                  selectedObject.object_id === item.object_id ? "active" : ""
                }
                onClick={() => setSelectedObjectId(item.object_id)}
              >
                <span>{item.type.toUpperCase()}</span>
                <strong>{item.label}</strong>
                <small>{item.object_id}</small>
              </button>
            ))}
          </div>
          {selectedObject && (
            <div className="transform-panel">
              <h3>TRANSFORM · {selectedObject.type}</h3>
              <label className="inspector-field">
                {t("对象名称", "Object name")}
                <input
                  value={selectedObject.label}
                  onChange={(event) =>
                    commitSceneEdit({
                      ...scene,
                      objects: scene.objects.map((item) =>
                        item.object_id === selectedObject.object_id
                          ? { ...item, label: event.target.value }
                          : item,
                      ),
                    })
                  }
                />
              </label>
              {(["position", "rotation", "scale"] as const).map((key) => (
                <div className="vector-row" key={key}>
                  <span>{key}</span>
                  {selectedObject[key].map((value, index) => (
                    <input
                      key={index}
                      type="number"
                      step="0.05"
                      value={value}
                      aria-label={`${key}-${index}`}
                      onChange={(event) =>
                        updateObject(key, index, Number(event.target.value))
                      }
                    />
                  ))}
                </div>
              ))}
              {selectedObject.type === "box" && (
                <label className="inspector-field">
                  {t("边长 (m)", "Edge length (m)")}
                  <input
                    type="number"
                    min="0.05"
                    step="0.05"
                    value={selectedObject.scale[0]}
                    onChange={(event) => {
                      const size = Math.max(0.05, Number(event.target.value));
                      commitSceneEdit({
                        ...scene,
                        objects: scene.objects.map((item) =>
                          item.object_id === selectedObject.object_id
                            ? { ...item, scale: [size, size, size] }
                            : item,
                        ),
                      });
                    }}
                  />
                </label>
              )}
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={selectedObject.static}
                  onChange={(event) =>
                    commitSceneEdit({
                      ...scene,
                      objects: scene.objects.map((item) =>
                        item.object_id === selectedObject.object_id
                          ? { ...item, static: event.target.checked }
                          : item,
                      ),
                    })
                  }
                />{" "}
                {selectedObject.static ? t("固定物体", "Static object") : t("可推动物体", "Pushable object")}
              </label>
              <button
                className="danger object-delete-button"
                disabled={selectedObject.type === "plane"}
                onClick={deleteSelectedObject}
                title={
                  selectedObject.type === "plane"
                    ? t("场景平面不能删除", "The scene plane cannot be deleted")
                    : t("删除当前场景对象", "Delete the selected scene object")
                }
              >
                <Trash2 size={14} />
                {selectedObject.type === "plane"
                  ? t("场景平面不可删除", "Scene plane cannot be deleted")
                  : t("删除当前对象", "Delete selected object")}
              </button>
              <div className="safety-mini">
                <ShieldCheck size={14} />
                <span>
                  S1 visual asset + scene objects
                  <br />
                    <strong>{t("保存对象环境，不保存关节预览", "Saves object environment, not joint preview")}</strong>
                </span>
              </div>
            </div>
          )}
          {(
            <div className="fidelity-note">
              <CircleStop size={18} />
              <span><strong>{t("运行证据不可用", "Run evidence unavailable")}</strong> {t("此 Scene 只声明 runtime-aligned 视觉资产；参数 Schema、碰撞、IK、轨迹与训练指标必须由真实运行服务另行提供。", "This Scene declares only runtime-aligned visual assets; parameter schema, collision, IK, trajectories, and training metrics must come from the real run service.")}</span>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

type FloatingAgentEntry = {
  role: "user" | "assistant";
  content: string;
  toolsUsed?: string[];
};

function FloatingAgent({ path }: { path: string }) {
  const floating = useFloatingAgentPosition();
  const { locale, t } = useI18n();
  const studio = useStudio();
  const providers = useQuery({
    queryKey: ["agent-providers"],
    queryFn: () => api<AgentProviderInfo[]>("/api/agent/providers"),
  });
  const [open, setOpen] = useState(false);
  const [providerId, setProviderId] = useState(
    () => localStorage.getItem("embodied-arm-agent-provider") ?? "mock",
  );
  const [level, setLevel] = useState(() =>
    Number(localStorage.getItem("embodied-arm-agent-level") ?? "4"),
  );
  const [session, setSession] = useState<{ id: string; key: string } | null>(null);
  const [entries, setEntries] = useState<FloatingAgentEntry[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const messagesEnd = useRef<HTMLDivElement>(null);
  const selectedProvider = providers.data?.find((item) => item.id === providerId);
  const providerReady = selectedProvider?.configured === true;
  const currentGuide = guideForPath(path);
  const guideTextIndex = locale === "en" ? 1 : 0;

  useEffect(() => {
    const refreshPreference = () => {
      setProviderId(localStorage.getItem("embodied-arm-agent-provider") ?? "mock");
      setLevel(Number(localStorage.getItem("embodied-arm-agent-level") ?? "4"));
      setSession(null);
      setEntries([]);
      setError("");
    };
    window.addEventListener("studio-agent-preference-changed", refreshPreference);
    return () => window.removeEventListener("studio-agent-preference-changed", refreshPreference);
  }, []);

  useEffect(() => {
    const resetRevokedSession = (event: Event) => {
      const changedProviderId = (event as CustomEvent<{ providerId?: string }>).detail?.providerId;
      if (changedProviderId !== providerId) return;
      setSession(null);
      setEntries([]);
      setError("");
    };
    window.addEventListener("studio-agent-provider-configuration-changed", resetRevokedSession);
    return () => window.removeEventListener("studio-agent-provider-configuration-changed", resetRevokedSession);
  }, [providerId]);

  useEffect(() => {
    const showAgent = (event: Event) => {
      setOpen(true);
      const prompt = (event as CustomEvent<{ prompt?: string }>).detail?.prompt;
      if (prompt) setInput(prompt);
    };
    window.addEventListener("studio-agent-open", showAgent);
    return () => window.removeEventListener("studio-agent-open", showAgent);
  }, []);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ block: "nearest" });
  }, [entries, busy]);

  async function sendMessage() {
    const content = input.trim();
    if (!content || busy || !providerReady) return;
    setBusy(true);
    setError("");
    setInput("");
    setEntries((current) => [...current, { role: "user", content }]);
    try {
      let projectId = studio.projectId;
      if (!projectId) {
        const initialized = await initializeProject();
        projectId = initialized.projectId;
        studio.setProject(projectId);
      }
      const sessionKey = `${projectId}:${providerId}:${level}:${locale}`;
      let sessionId = session?.key === sessionKey ? session.id : "";
      if (!sessionId) {
        const created = await api<{ session_id: string }>("/api/agent/sessions", {
          method: "POST",
          body: JSON.stringify({
            project_id: projectId,
            provider: providerId,
            assistance_level: level,
            locale,
          }),
        });
        sessionId = created.session_id;
        setSession({ id: sessionId, key: sessionKey });
      }
      const reply = await api<AgentMessageResponse>(
        `/api/agent/sessions/${sessionId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            content,
            context: buildAgentPageContext(
              path,
              projectId,
              studio.currentRun?.run_id,
              studio.activeSceneId,
              undefined,
              {
                version: USER_GUIDE_VERSION,
                id: currentGuide.id,
                title: currentGuide.title[guideTextIndex],
                summary: currentGuide.summary[guideTextIndex],
                boundaries: currentGuide.boundaries.map((item) => item[guideTextIndex]),
                steps: currentGuide.steps.map((item) => `${item.title[guideTextIndex]}: ${item.body[guideTextIndex]}`),
                principles: (GUIDE_PRINCIPLES[currentGuide.id] ?? []).map((item) => item[guideTextIndex]),
              },
            ),
          }),
        },
      );
      setEntries((current) => [
        ...current,
        { role: "assistant", content: reply.content, toolsUsed: reply.tools_used },
      ]);
    } catch (caught) {
      setInput(content);
      setError(caught instanceof Error ? (locale === "en" ? translateUiText(caught.message) : caught.message) : t("Agent 请求失败", "Agent request failed"));
      void providers.refetch();
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className={`floating-agent ${open ? "open" : ""}`} style={{ left: floating.position.x, top: floating.position.y, right: "auto", bottom: "auto" }}>
      {open && (
        <section className="floating-agent-panel" style={floating.panelStyle} aria-label={t("教学 Agent 对话", "Teaching Agent conversation")}>
          <header>
            <span className="floating-agent-mark"><Bot size={18} /></span>
            <div>
              <strong>{t("教学 Agent", "Teaching Agent")}</strong>
              <small>{selectedProvider ? (locale === "en" ? translateUiText(selectedProvider.name) : selectedProvider.name) : t("Provider 不可用", "Provider unavailable")}</small>
            </div>
            <em className={providerReady && selectedProvider?.status === "healthy" ? "ready" : "unavailable"}>
              {providerReady ? (providerId === "mock" ? "MOCK" : selectedProvider?.status === "healthy" ? "VERIFIED" : selectedProvider?.status === "degraded" ? "FAILED" : "UNTESTED") : "UNAVAILABLE"}
            </em>
            <button title={t("Agent 设置", "Agent settings")} onClick={() => navigate("/agent")}>
              <Settings size={16} />
            </button>
            <button title={t("收起 Agent", "Collapse Agent")} onClick={() => setOpen(false)}>
              <X size={17} />
            </button>
          </header>
          <div className="floating-agent-context">
            <span>{path}</span>
            <span>LEVEL {level}</span>
            <span>{providerId === "mock" ? t("不发送外部请求", "No external request is sent") : t("上下文将发送到外部模型", "Context will be sent to an external model")}</span>
          </div>
          <div className="floating-agent-guide-tools">
            <button onClick={() => navigate(guidePathForSection(currentGuide.id))}><BookOpen size={14} /> {t("查看完整文档", "Open full guide")}</button>
            <button onClick={() => setInput(t(`请根据《${currentGuide.title[0]}》说明当前页面的使用方法、完成标准和能力边界，并告诉我下一步应该检查什么。`, `Using the "${currentGuide.title[1]}" guide, explain this page, its completion criteria and capability boundaries, then tell me what I should check next.`))}><Sparkles size={14} /> {t("解释当前章节", "Explain this chapter")}</button>
          </div>
          <div className="floating-agent-messages" aria-live="polite">
            {!entries.length && (
              <div className="floating-agent-empty">
                <Sparkles size={20} />
                <strong>{t("询问当前页面", "Ask about this page")}</strong>
                <p>{t("我会携带已知的页面、Project、Run 和 Scene 标识，不会执行 Shell 或直接控制机器人。", "I will include known page, Project, Run, and Scene identifiers, but will not execute Shell commands or directly control a robot.")}</p>
              </div>
            )}
            {entries.map((entry, index) => (
              <div key={`${entry.role}-${index}`} className={`floating-agent-message ${entry.role}`}>
                <span>{entry.role === "user" ? "YOU" : "AGENT"}</span>
                <p>{entry.content}</p>
                {!!entry.toolsUsed?.length && (
                  <small className="agent-tools-used">
                    READ: {entry.toolsUsed.join(" / ")}
                  </small>
                )}
              </div>
            ))}
            {busy && <div className="floating-agent-thinking">{t("正在生成回复...", "Generating a reply...")}</div>}
            {error && <p className="floating-agent-error" role="alert">{error}</p>}
            <div ref={messagesEnd} />
          </div>
          <form onSubmit={(event) => { event.preventDefault(); void sendMessage(); }}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder={providerReady ? t("询问原因、证据或下一步...", "Ask about causes, evidence, or next steps...") : t("请先在 Agent 设置中选择可用 Provider", "Choose an available Provider in Agent settings first")}
              disabled={!providerReady || busy}
              rows={2}
            />
            <button type="submit" title={t("发送消息", "Send message")} disabled={!input.trim() || busy || !providerReady}>
              <ChevronRight size={18} />
            </button>
          </form>
        </section>
      )}
      <button
        className="floating-agent-trigger"
        data-tour="floating-agent"
        {...floating.handlers}
        aria-label={t("教学 Agent：点击打开，拖动调整位置", "Teaching Agent: click to open, drag to reposition")}
        title={open ? t("收起教学 Agent", "Collapse Teaching Agent") : t("打开教学 Agent", "Open Teaching Agent")}
        aria-expanded={open}
        onClick={() => { if (!floating.consumeDragClick()) setOpen((current) => !current); }}
      >
        {open ? <X size={20} /> : <Bot size={21} />}
        {!open && <span className={providerReady && selectedProvider?.status === "healthy" ? "ready" : "unavailable"} />}
      </button>
    </aside>
  );
}

function PipelinePage() {
  const { locale, t } = useI18n();
  const pipeline = useQuery({
    queryKey: ["pipeline", "mock-reach"],
    queryFn: () =>
      api<PipelineDocument>("/api/pipelines/pipeline_mock_reach_baseline"),
  });
  const validation = useQuery({
    queryKey: ["pipeline-validation", pipeline.data?.revision],
    queryFn: () =>
      api<PipelineValidation>("/api/pipelines/validate", {
        method: "POST",
        body: JSON.stringify({ document: pipeline.data }),
      }),
    enabled: Boolean(pipeline.data),
  });
  const [selectedStepNumber, setSelectedStepNumber] = useState("01");
  const [showAdvancedDetails, setShowAdvancedDetails] = useState(false);
  if (pipeline.isLoading)
    return <div className="loading">{t("正在解析 Pipeline 修订...", "Parsing Pipeline revision...")}</div>;
  if (pipeline.isError || !pipeline.data)
    return (
      <div className="loading error">
        {t("Pipeline API 不可用，请确认服务已更新并启动。", "The Pipeline API is unavailable. Confirm that the service is updated and running.")}
      </div>
    );
  const experimentSteps = [
    { number: "01", title: t("定义环境", "Define environment"), description: t("选择 S1 场景、目标和可观察量", "Select the S1 scene, target, and observables"), detail: t("确认六自由度 S1 场景版本、任务目标、观测空间和动作边界。场景预览只用于配置核对，不代表碰撞或真实机器人安全验证。", "Confirm the six-degree-of-freedom S1 scene revision, task target, observation space, and action limits. The preview is for configuration checks and does not represent collision or real-robot safety validation."), completion: t("保存场景修订，并确认 Run 将冻结对应 scene revision。", "Save the scene revision and confirm that the Run will freeze the corresponding scene revision."), path: "/environments", nodeKind: "environment" },
    { number: "02", title: t("编写策略", "Write policy"), description: t("确认输入、动作和策略入口", "Confirm inputs, actions, and the policy entrypoint"), detail: t("阅读 Project 冻结的 PPO 训练约束，编写或选择与六轴动作空间匹配的策略代码，并先通过静态检查。", "Read the PPO training constraints frozen by the Project, write or select policy code that matches the six-axis action space, and pass static checks first."), completion: t("策略代码已保存，输入输出与所选 S1 场景一致。", "Policy code is saved, and its inputs and outputs match the selected S1 scene."), path: "/training-code", nodeKind: "policy" },
    { number: "03", title: t("配置训练", "Configure training"), description: t("选择可复现 seed 并核对冻结配置", "Choose a reproducible seed and verify frozen configuration"), detail: t("正式 S1 Project 只允许提交 schema 声明的参数。当前可调参数是随机种子；算法、训练步数和安全边界由服务端 Project 版本冻结。", "The formal S1 Project accepts only schema-declared parameters. The current adjustable parameter is the random seed; the algorithm, training steps, and safety boundaries are frozen by the server-side Project revision."), completion: t("seed 在合法范围内，并理解它用于复现实验随机过程。", "The seed is within the valid range and its role in reproducing the experiment's randomness is understood."), path: "/training", nodeKind: "trainer" },
    { number: "04", title: t("启动实验", "Start experiment"), description: t("选择课程 Project 并提交正式 Run", "Select the course Project and submit a formal Run"), detail: t("在组织、课程和教学班中选择可用 Project，绑定场景修订与 seed，创建一条可追溯的 simulation Run。", "Select an available Project from the organization, course, and teaching class, bind the scene revision and seed, and create a traceable simulation Run."), completion: t("Run 创建成功并获得唯一 run_id。", "The Run is created successfully and receives a unique run_id."), path: "/courses", nodeKind: "runner" },
    { number: "05", title: t("检查指标", "Inspect metrics"), description: t("查看奖励、损失和状态事件", "Review rewards, losses, and status events"), detail: t("沿时间顺序检查 Run 状态、训练指标和事件证据。不要用单条曲线代替多 seed 的复现性判断。", "Review Run status, training metrics, and event evidence in time order. Do not use a single curve as a substitute for reproducibility across multiple seeds."), completion: t("确认终态、关键指标与异常事件，并记录证据缺口。", "Confirm the terminal state, key metrics, and anomalous events, and record evidence gaps."), path: "/runs", nodeKind: "metrics" },
    { number: "06", title: t("评估证据", "Evaluate evidence"), description: t("按 rubric 检查完成度与限制", "Check completion and limitations against the rubric"), detail: t("从不可变 Run 证据生成正式评估，再让 AI 对报告作个性化解读。AI 点评不会覆盖正式分数或补造缺失证据。", "Generate a formal evaluation from immutable Run evidence, then ask AI for a personalized interpretation. AI commentary does not override the formal score or invent missing evidence."), completion: t("正式报告已生成，限制和下一轮改进项已阅读。", "The formal report is generated, and the limitations and next improvements have been reviewed."), path: "/evaluation", nodeKind: "evaluation" },
  ];
  const nodeForStep = (step: (typeof experimentSteps)[number]) =>
    pipeline.data.nodes.find((node) =>
      node.node_id.toLowerCase().includes(step.nodeKind) ||
      node.kind.toLowerCase().includes(step.nodeKind),
    );
  const selectedStep = experimentSteps.find((step) => step.number === selectedStepNumber) ?? experimentSteps[0];
  const selectedStepNode = nodeForStep(selectedStep);

  function selectStep(step: (typeof experimentSteps)[number]) {
    setSelectedStepNumber(step.number);
  }

  function askAgent(step: (typeof experimentSteps)[number]) {
    window.dispatchEvent(new CustomEvent("studio-agent-open", {
      detail: {
        prompt: t(`请结合当前页面，详细介绍实验流程第 ${step.number} 步“${step.title}”：解释学习目标、需要检查的配置、完成标准、常见错误，以及我下一步应该在对应界面做什么。`, `Using the current page, explain experiment step ${step.number}, "${step.title}": describe the learning objective, configuration checks, completion criteria, common mistakes, and what I should do next in the corresponding interface.`,),
      },
    }));
  }

  return (
    <main className="pipeline-page">
      <header className="pipeline-header">
        <div>
          <span className="eyebrow">
            {t("实验流程", "Experiment flow")} · SCHEMA {pipeline.data.schema_version}
          </span>
          <h1>{t("从环境到证据的六步实验", "A six-step experiment from environment to evidence")}</h1>
          <p>
            {t("按六个步骤完成一次可复现的教学实验。每一步都连接到已有工作区，运行结果保留配置、事件和限制说明；当前流程绑定 Mock Reach 教学近似。", "Complete a reproducible teaching experiment in six steps. Each step connects to an existing workspace, while results retain configuration, events, and limitations; this flow is bound to the Mock Reach teaching approximation.")}
          </p>
        </div>
        <div className="pipeline-meta">
          <span className="capability">{t("教学流程", "Teaching flow")}</span>
          <dl>
            <div>
              <dt>REVISION</dt>
              <dd>{pipeline.data.revision}</dd>
            </div>
            <div>
              <dt>ENVIRONMENT</dt>
              <dd>{pipeline.data.environment_ref.id}</dd>
            </div>
            <div>
              <dt>DETERMINISTIC</dt>
              <dd>
                {pipeline.data.reproducibility.deterministic ? "YES" : "NO"}
              </dd>
            </div>
          </dl>
        </div>
      </header>
      <section className="pipeline-workspace">
        <div className="pipeline-canvas">
          <div className="canvas-toolbar">
            <span>{t("六步实验主线", "Six-step experiment path")}</span>
            <small>
              {experimentSteps.length} {t("步骤", "STEPS")} · {pipeline.data.revision} {t("修订", "REVISION")}
            </small>
          </div>
          <div className="experiment-step-list">
            {experimentSteps.map((step, index) => {
              const node = nodeForStep(step);
              const active = selectedStep.number === step.number;
              return (
                <div className={`experiment-step ${active ? "expanded" : ""}`} key={step.number}>
                  <button
                    className={active ? "active" : ""}
                    aria-expanded={active}
                    onClick={() => selectStep(step)}
                  >
                    <span className="experiment-step-number">{step.number}</span>
                    <span><strong>{step.title}</strong><small>{step.description}</small></span>
                    <em>{node?.capability ?? t("未绑定", "Unbound")}</em>
                    <ChevronDown size={16} className="experiment-step-toggle" />
                  </button>
                  {active && (
                    <div className="experiment-step-detail" key={step.number}>
                      <p>{step.detail}</p>
                      <div><span>{t("完成标准", "Completion criteria")}</span><strong>{step.completion}</strong></div>
                      <div className="experiment-step-actions">
                        <button className="secondary" onClick={() => navigate(step.path)}>
                          {t("打开对应界面", "Open interface")} <ExternalLink size={14} />
                        </button>
                        <button className="secondary" onClick={() => askAgent(step)}>
                          {t("问教学 Agent", "Ask teaching Agent")} <Bot size={14} />
                        </button>
                      </div>
                    </div>
                  )}
                  {index < experimentSteps.length - 1 && <ChevronRight size={18} className="experiment-step-arrow" />}
                </div>
              );
            })}
          </div>
          <div className="pipeline-warning">
            <FlaskConical size={17} />
            <span>
              <strong>{t("当前运行绑定：Mock Adapter", "Current runtime: Mock Adapter")}</strong>
              {t("流程可验证、可复现，但动力学是近似教学行为；外部 worker 和真实控制尚未接入。", "The flow is verifiable and reproducible, but the dynamics are an approximate teaching behavior; external workers and real control are not connected.")}
            </span>
          </div>
        </div>
        <aside className="node-inspector">
          <div className="inspector-title">
            <span>{t("高级流程数据", "Advanced flow data")}</span>
            <small>{showAdvancedDetails ? selectedStepNode?.node_id ?? t("当前步骤未绑定", "Current step is unbound") : t("按需查看", "View on demand")}</small>
          </div>
          {!showAdvancedDetails ? (
            <div className="workspace-empty">
              <ListTree size={26} />
              <strong>{t("六步实验主线已就绪", "Six-step experiment path is ready")}</strong>
              <span>{t("节点绑定、输入输出和配置快照属于高级流程信息。", "Node bindings, inputs, outputs, and configuration snapshots are advanced flow information.")}</span>
              <button className="secondary" onClick={() => setShowAdvancedDetails(true)}>
                {t("查看高级流程数据", "View advanced flow data")} <ChevronRight size={14} />
              </button>
            </div>
          ) : (
            <>
              {selectedStepNode ? (
                <>
                  <h2>{selectedStepNode.kind === "environment" ? t("环境", "Environment") : selectedStepNode.kind}</h2>
                  <code>{selectedStepNode.implementation}</code>
                  <section>
                    <h3>INPUTS</h3>
                    {selectedStepNode.inputs.length ? selectedStepNode.inputs.map((port) => (
                      <div className="port-row" key={port.name}><span>{port.name}</span><code>{port.type}</code></div>
                    )) : <p>{t("无输入端口", "No input ports")}</p>}
                  </section>
                  <section>
                    <h3>OUTPUTS</h3>
                    {selectedStepNode.outputs.map((port) => (
                      <div className="port-row" key={port.name}><span>{port.name}</span><code>{port.type}</code></div>
                    ))}
                  </section>
                  <section><h3>{t("配置快照", "Configuration snapshot")}</h3><pre>{JSON.stringify(selectedStepNode.config, null, 2)}</pre></section>
                </>
              ) : (
                <div className="workspace-empty"><ListTree size={24} /><strong>{t("当前教学步骤未绑定底层节点", "The current teaching step has no underlying node")}</strong><span>{t("这一步由对应业务页面完成，Pipeline 不会伪造 trainer、runner 或 evaluation 节点。", "This step is completed in its corresponding business page; Pipeline will not fabricate trainer, runner, or evaluation nodes.")}</span></div>
              )}
              <button className="secondary pipeline-open-step" onClick={() => navigate(selectedStep.path)}>
                {t(`打开“${selectedStep.title}”界面`, `Open the "${selectedStep.title}" interface`)} <ChevronRight size={14} />
              </button>
            </>
          )}
        </aside>
      </section>
      <section
        className={`validation-strip ${validation.data?.valid ? "valid" : "pending"}`}
      >
        <span>
          {validation.data?.valid ? (
            <Check size={17} />
          ) : (
            <Activity size={17} />
          )}
        </span>
        <div>
          <strong>
            {validation.data?.valid ? t("结构验证通过", "Structure validation passed") : t("正在验证结构", "Validating structure")}
          </strong>
          <small>
            {validation.data
              ? `${validation.data.errors.length} errors · ${validation.data.warnings.length} warning`
              : t("等待验证器响应", "Waiting for validator response")}
          </small>
        </div>
        {validation.data?.warnings.map((warning) => (
          <p key={warning.path}>{locale === "en" ? translateUiText(warning.message) : warning.message}</p>
        ))}
      </section>
    </main>
  );
}

const DEFAULT_TRAINING_CODE = `def train(env, seed):
    """Mock-safe neural policy training entrypoint."""
    policy = {
        "algorithm": "mlp-policy",
        "observation": ["joint_positions", "target"],
        "action": "joint_delta[3]",
        "seed": seed,
    }
    return policy
`;

function TrainingCodePage() {
  const { locale, t } = useI18n();
  const scenes = useQuery({
    queryKey: ["scenes"],
    queryFn: () => api<SceneDocument[]>("/api/scenes"),
  });
  const codes = useQuery({
    queryKey: ["training-code"],
    queryFn: () =>
      api<
        Array<{
          code_id: string;
          revision: number;
          name: string;
          content: string;
          scene_ref?: { scene_id: string; revision: number };
        }>
      >("/api/training-code"),
  });
  const [code, setCode] = useState(DEFAULT_TRAINING_CODE);
  const [codeId, setCodeId] = useState("training_policy_baseline");
  const [sceneId, setSceneId] = useState("scene_mock_reach_lab");
  const [result, setResult] = useState<{
    valid: boolean;
    errors: string[];
    warnings: string[];
    execution: string;
  } | null>(null);
  const [message, setMessage] = useState("");
  useEffect(() => setMessage(""), [locale]);
  useEffect(() => {
    const saved = codes.data?.find((item) => item.code_id === codeId);
    if (saved) {
      setCode(saved.content);
      if (saved.scene_ref) setSceneId(saved.scene_ref.scene_id);
    }
  }, [codes.data, codeId]);
  async function validate() {
    setResult(
      await api("/api/training-code/validate", {
        method: "POST",
        body: JSON.stringify({ content: code }),
      }),
    );
  }
  async function save() {
    const saved = await api<{ code_id: string; revision: number }>(
      "/api/training-code",
      {
        method: "POST",
        body: JSON.stringify({
          document: {
            code_id: codeId,
            name: "Student neural policy",
            language: "python",
            content: code,
            scene_ref: {
              scene_id: sceneId,
              revision:
                scenes.data?.find((item) => item.scene_id === sceneId)
                  ?.revision ?? 1,
            },
          },
        }),
      },
    );
    setMessage(`${t("已保存", "Saved")} ${saved.code_id} · REV ${saved.revision}`);
  }
  return (
    <main className="code-page">
      <header className="code-header">
        <div>
          <span className="eyebrow">POLICY WORKSPACE · PYTHON</span>
          <h1>{t("策略草稿与网络参考", "Policy drafts and network reference")}</h1>
          <p>
            面向可复现实验的训练入口。编辑器支持 Python
            语法内容保存和安全静态检查，服务端不会执行提交代码。
          </p>
        </div>
        <div className="code-boundary">
          <ShieldCheck size={18} />
          <span>
            <strong>STATIC VALIDATION ONLY</strong>
            <small>STORED DRAFT · NOT EXECUTED</small>
          </span>
        </div>
      </header>
      <section className="code-workspace">
        <aside className="code-files">
          <div className="section-label">TRAINING ARTIFACTS</div>
          {(codes.data ?? []).map((item) => (
            <button
              key={item.code_id}
              className={item.code_id === codeId ? "active" : ""}
              onClick={() => setCodeId(item.code_id)}
            >
              <Code2 size={15} />
              <span>
                <strong>{item.name}</strong>
                <small>
                  {item.code_id} · REV {item.revision}
                </small>
              </span>
            </button>
          ))}
          <label>
            绑定场景
            <select
              value={sceneId}
              onChange={(event) => setSceneId(event.target.value)}
            >
              {(scenes.data ?? []).map((sceneItem) => (
                <option key={sceneItem.scene_id} value={sceneItem.scene_id}>
                  {sceneItem.name} · REV {sceneItem.revision}
                </option>
              ))}
            </select>
          </label>
        </aside>
        <section className="code-editor-pane">
          <div className="editor-head">
            <span>train_policy.py</span>
            <div>
              <button className="secondary" onClick={validate}>
                <Check size={14} /> 静态检查
              </button>
              <button className="primary" onClick={save}>
                <Save size={14} /> 保存 revision
              </button>
            </div>
          </div>
          <div className="code-editor-shell">
            <Editor
              height="560px"
              language="python"
              theme="vs-light"
              value={code}
              onChange={(value) => setCode(value ?? "")}
              options={{
                minimap: { enabled: false },
                fontSize: 13,
                lineNumbersMinChars: 3,
                scrollBeyondLastLine: false,
                padding: { top: 16 },
              }}
            />
          </div>
          {message && (
            <div className="scene-message">
              <Check size={14} />
              {message}
            </div>
          )}
        </section>
        <aside className="code-inspector">
          <figure className="policy-network" aria-label={t("Push PPO 默认网络参考", "Push PPO default network reference")}>
            <figcaption>{t("Push PPO 默认网络参考", "Push PPO default network reference")}</figcaption>
            <div className="policy-network-input">{t("24 维观测", "24D observation")}<small>q / dq / XY / velocity</small></div>
            <div className="policy-network-branches">
              <div><strong>ACTOR</strong><span>Linear 24 → 64</span><span>Tanh → 64 → Tanh</span><span>Gaussian → tanh</span><strong>Δx, Δy · ±0.005 m</strong></div>
              <div><strong>CRITIC</strong><span>Linear 24 → 64</span><span>Tanh → 64 → Tanh</span><span>Linear → 1</span><strong>V(s)</strong></div>
            </div>
            <small>{t("源码默认结构参考，不代表当前草稿或已加载的 checkpoint；修改草稿不会替换正式训练网络。", "Source default reference, not the current draft or a loaded checkpoint. Editing a draft does not replace the formal training network.")}</small>
          </figure>
          <div className="section-label">VALIDATION REPORT</div>
          {result ? (
            <>
              <div
                className={`code-result ${result.valid ? "valid" : "invalid"}`}
              >
                <strong>{result.valid ? "代码结构通过" : "代码被阻止"}</strong>
                <small>{result.execution}</small>
              </div>
              {result.errors.map((error) => (
                <p className="code-finding error" key={error}>
                  {error}
                </p>
              ))}
              {result.warnings.map((warning) => (
                <p className="code-finding" key={warning}>
                  {warning}
                </p>
              ))}
            </>
          ) : (
            <div className="summary-empty">
              运行静态检查后显示 AST 安全结果。
            </div>
          )}
          <div className="code-contract">
            <span>ENTRYPOINT</span>
            <strong>train(env, seed)</strong>
            <span>SCENE REF</span>
            <strong>{sceneId}</strong>
            <span>EXECUTION</span>
            <strong>STATIC ONLY</strong>
          </div>
        </aside>
      </section>
    </main>
  );
}

const RUN_METRIC_COLORS = [
  "#e05a3f",
  "#277b69",
  "#b07a16",
  "#476f9b",
  "#9a493e",
  "#4f6f64",
] as const;

const RUN_METRIC_LABELS: Record<string, { "zh-CN": string; en: string }> = {
  training_steps: { "zh-CN": "训练步数", en: "Training steps" },
  update_index: { "zh-CN": "更新次数", en: "Updates" },
  average_reward: { "zh-CN": "平均奖励", en: "Average reward" },
  episode_length: { "zh-CN": "回合长度", en: "Episode length" },
  episodes: { "zh-CN": "完成回合数", en: "Completed episodes" },
  policy_loss: { "zh-CN": "策略损失", en: "Policy loss" },
  value_loss: { "zh-CN": "价值损失", en: "Value loss" },
  entropy: { "zh-CN": "策略熵", en: "Policy entropy" },
};

const RUN_EVENT_LABELS: Record<string, { "zh-CN": string; en: string }> = {
  state: { "zh-CN": "状态变化", en: "Status change" },
  status: { "zh-CN": "状态变化", en: "Status change" },
  metric: { "zh-CN": "训练指标", en: "Training metrics" },
  progress: { "zh-CN": "进度更新", en: "Progress update" },
  artifact: { "zh-CN": "制品生成", en: "Artifact generated" },
  error: { "zh-CN": "错误", en: "Error" },
  heartbeat: { "zh-CN": "运行心跳", en: "Run heartbeat" },
};

const ARTIFACT_LABELS: Record<string, { "zh-CN": string; en: string }> = {
  checkpoint: { "zh-CN": "模型检查点", en: "Model checkpoint" },
  checkpoint_json: { "zh-CN": "模型检查点", en: "Model checkpoint" },
  training_video: { "zh-CN": "训练录像", en: "Training video" },
  metrics: { "zh-CN": "训练指标", en: "Training metrics" },
  metrics_json: { "zh-CN": "训练指标", en: "Training metrics" },
  config: { "zh-CN": "训练配置", en: "Training configuration" },
  manifest: { "zh-CN": "运行清单", en: "Run manifest" },
  evaluation: { "zh-CN": "评估报告", en: "Evaluation report" },
};

function shortIdentifier(value: string) {
  return value.length > 18 ? `${value.slice(0, 8)}...${value.slice(-6)}` : value;
}

export function isDemoRunRecord(record: RunRecord): boolean {
  return record.run_kind === "play";
}

export function runEnvironmentType(record: RunRecord): string {
  const snapshot = record.project_config_snapshot;
  const catalogTask = snapshot.catalog_task as { id?: unknown } | undefined;
  const taskConfig = snapshot.task_config as { environment_type?: unknown } | undefined;
  if (typeof catalogTask?.id === "string") return catalogTask.id;
  if (typeof snapshot.environment_type === "string") return snapshot.environment_type;
  if (typeof taskConfig?.environment_type === "string") return taskConfig.environment_type;
  return record.simulation_task_id ?? "--";
}

export function demoMetricEntries(record: RunRecord): Array<[string, unknown]> {
  return Object.entries(record.metrics_summary).filter(([key]) => key !== "evaluation");
}

export function runExportEligible(record: RunRecord): boolean {
  return record.status === "completed" || (isDemoRunRecord(record) && record.status === "stopped");
}

function runEventLabel(type: string, locale: "zh-CN" | "en") {
  return RUN_EVENT_LABELS[type]?.[locale] ?? (locale === "en" ? "Run event" : "运行事件");
}

function artifactLabel(type: string, locale: "zh-CN" | "en") {
  return ARTIFACT_LABELS[type]?.[locale] ?? (locale === "en" ? "Training artifact" : "训练制品");
}

function formatEvaluationMetric(
  value: number | null,
  format: "percent" | "number" | "distance",
  locale: "zh-CN" | "en",
) {
  if (value === null) return locale === "en" ? "Not recorded" : "未记录";
  if (format === "percent") return `${(value * 100).toFixed(2)}%`;
  if (format === "distance") return `${Number(value.toFixed(4))} m`;
  return String(Number(value.toFixed(4)));
}

export function RunEvaluationCard({ record, locale = "zh-CN" }: { record: RunRecord; locale?: "zh-CN" | "en" }) {
  const demo = isDemoRunRecord(record);
  const evaluation = productionEvaluation(record.metrics_summary);
  const runStatus = record.status;
  const policyStatus = evaluation?.status ?? "unavailable";
  const policyStatusLabel = locale === "en"
    ? ({ passed: "Passed", failed: "Failed", unavailable: "Unavailable" } as const)[policyStatus]
    : ({ passed: "通过", failed: "未通过", unavailable: "不可用" } as const)[policyStatus];
  const interval = evaluation
    && typeof evaluation.wilson_95_lower_bound === "number"
    && typeof evaluation.wilson_95_upper_bound === "number"
      ? `${(evaluation.wilson_95_lower_bound * 100).toFixed(2)}% – ${(evaluation.wilson_95_upper_bound * 100).toFixed(2)}%`
      : locale === "en" ? "Not recorded" : "未记录";

  return (
    <section className="run-evaluation-card" aria-labelledby="run-evaluation-title">
      <header>
        <div>
          <span className="eyebrow">{demo ? (locale === "en" ? "DEMO RESULT" : "Demo 结果") : (locale === "en" ? "PRODUCTION EVALUATION" : "正式策略评估")}</span>
          <h2 id="run-evaluation-title">{demo ? (locale === "en" ? "Demo execution result" : "Demo 执行结果") : (locale === "en" ? "Policy evaluation evidence" : "策略评估证据")}</h2>
        </div>
        <div className="run-evaluation-statuses">
          <span><small>Run</small><strong>{runStatusLabel(runStatus, locale)}</strong></span>
          {!demo && <span><small>{locale === "en" ? "Policy" : "策略"}</small><strong>{policyStatusLabel}</strong></span>}
        </div>
      </header>
      {demo ? (
        <p className="run-empty-note">{locale === "en" ? "A Demo does not train or evaluate a policy; missing policy evidence does not mean the Demo failed." : "Demo 不训练或评估策略；没有 Policy 证据不代表失败。"}</p>
      ) : !evaluation ? (
        <p className="run-empty-note">{locale === "en" ? "This run has no formal evaluation. Run completed does not mean Policy passed and is not shown as 0%." : "该运行未生成正式评估。Run completed 不代表 Policy passed，也不会显示为 0%。"}</p>
      ) : (
        <>
          <div className="run-facts">
            <div><span title={locale === "en" ? "Explicitly successful episodes / completed evaluation episodes for fixed evaluation seeds" : "固定评估 seed 中明确成功的回合数 / 已完成评估回合数"}>{locale === "en" ? "Successful episodes" : "成功回合"}</span><strong>{evaluation.success_count ?? (locale === "en" ? "Not recorded" : "未记录")} / {evaluation.episode_count ?? (locale === "en" ? "Not recorded" : "未记录")}</strong></div>
            <div><span title={locale === "en" ? "Wilson score 95% confidence interval for the binomial success rate" : "二项成功率的 Wilson score 95% 置信区间"}>Wilson 95% {locale === "en" ? "interval" : "区间"}</span><strong>{interval}</strong></div>
            {PRODUCTION_EVALUATION_DEFINITIONS.map((definition) => (
              <div key={definition.key}>
                <span title={`${evaluationMetricText(definition, "description", locale)}; ${locale === "en" ? "unit" : "单位"}: ${evaluationMetricText(definition, "unit", locale)}`}>{evaluationMetricText(definition, "label", locale)}</span>
                <strong>{formatEvaluationMetric(evaluationMetric(evaluation, definition.key), definition.format, locale)}</strong>
              </div>
            ))}
          </div>
          <small className="evaluation-source-note">{locale === "en" ? "Uses only persisted deterministic evidence from metrics_summary.evaluation." : "仅使用 metrics_summary.evaluation 中的持久化确定性证据。"}</small>
        </>
      )}
    </section>
  );
}

function eventStatusLabel(value: unknown) {
  if (typeof value !== "string") return null;
  return RUN_STATUS_PRESENTATIONS.some((item) => item.status === value)
    ? runStatusLabel(value as RunRecord["status"])
    : value;
}

export function CopyableIdentifier({ value, label }: { value: string; label: string }) {
  const { locale, t } = useI18n();
  const valueRef = useRef<HTMLSpanElement>(null);
  const [feedback, setFeedback] = useState<"" | "copied" | "manual">("");

  function selectCompleteValue() {
    const node = valueRef.current;
    const selection = window.getSelection();
    if (!node || !selection) return false;
    const range = document.createRange();
    range.selectNodeContents(node);
    selection.removeAllRanges();
    selection.addRange(range);
    return selection.toString() === value;
  }

  async function copy() {
    if (window.isSecureContext && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(value);
        setFeedback("copied");
        return;
      } catch {
        // Continue into the selectable-text fallback.
      }
    }
    const selected = selectCompleteValue();
    if (selected) {
      try {
        if (document.execCommand("copy")) {
          setFeedback("copied");
          return;
        }
      } catch {
        // Keep the complete value selected for manual copying.
      }
    }
    setFeedback("manual");
  }

  return (
    <span className="copyable-identifier">
      <span
        ref={valueRef}
        className="copyable-identifier-value"
        title={value}
        tabIndex={0}
      >
        {value}
      </span>
      <button
        type="button"
        title={t(`复制完整${label}`, `Copy full ${label}`)}
        aria-label={t(`复制完整${label}`, `Copy full ${label}`)}
        onClick={() => void copy()}
      >
        {feedback === "copied" ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <small className={feedback === "manual" ? "error" : ""} aria-live="polite">
        {feedback === "copied"
          ? t("已复制", "Copied")
          : feedback === "manual"
            ? t("请手动复制已选中的完整值", "Please manually copy the selected full value")
            : ""}
      </small>
    </span>
  );
}

function formatRunMetricValue(
  value: number | null | undefined,
  locale: "zh-CN" | "en" = "zh-CN",
) {
  if (value == null) return "--";
  const magnitude = Math.abs(value);
  return magnitude !== 0 && (magnitude >= 10000 || magnitude < 0.001)
    ? value.toExponential(3)
    : value.toLocaleString(locale, { maximumFractionDigits: 4 });
}

function RunMetricChart({ option }: { option: echarts.EChartsOption }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.EChartsType | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = echarts.init(container, undefined, { renderer: "svg" });
    chartRef.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true, lazyUpdate: false });
  }, [option]);

  return <div ref={containerRef} style={{ width: "100%", height: 190 }} />;
}

function RunMetricCharts({ points }: { points: RunMetricPoint[] }) {
  const { locale, t } = useI18n();
  if (!points.length) {
    return <p className="run-empty-note">{t("等待 Worker 产生首个完整 PPO update。", "Waiting for the Worker to produce the first complete PPO update.")}</p>;
  }

  const visibleDefinitions = visibleRunMetricDefinitions(points);
  const missingDefinitions = missingSimulationMetricDefinitions(points);
  return (
    <div className="run-metric-history">
      <p>{t("横轴为 Worker 上报的 update_index；悬停可核对对应 training_steps。缺失值保留为空，不补零。", "The x-axis uses the update_index reported by the Worker; hover to check the corresponding training_steps. Missing values remain empty and are not filled with zero.")}</p>
      <div className="run-metric-grid">
        {visibleDefinitions.map((definition, index) => {
          const label = runMetricText(definition, "label", locale);
          const description = runMetricText(definition, "description", locale);
          const unit = runMetricText(definition, "unit", locale);
          const data = metricSeriesData(points, definition.key);
          const hasValues = data.some(([, value]) => value !== null);
          const color = RUN_METRIC_COLORS[index];
          const option: echarts.EChartsOption = {
            animation: false,
            grid: { left: 12, right: 18, top: 14, bottom: 34, containLabel: true },
            tooltip: {
              trigger: "axis",
              formatter: (raw: unknown) => {
                const first = (Array.isArray(raw) ? raw[0] : raw) as
                  | { value?: unknown }
                  | undefined;
                const coordinates = Array.isArray(first?.value) ? first.value : [];
                const updateIndex = typeof coordinates[0] === "number" ? coordinates[0] : null;
                const metricValue = typeof coordinates[1] === "number" ? coordinates[1] : null;
                const point = points.find((candidate) => candidate.updateIndex === updateIndex);
                return [
                  `${t("更新", "Update")} ${updateIndex ?? "--"}`,
                  `${t("训练步数", "Training steps")} ${point?.trainingSteps ?? "--"}`,
                  `${label} ${formatRunMetricValue(metricValue, locale)}`,
                ].join("<br/>");
              },
            },
            xAxis: {
              type: "value",
              minInterval: 1,
              name: t("更新", "Update"),
              nameLocation: "middle",
              nameGap: 23,
              axisLabel: { color: "#6c7773", fontSize: 9 },
              axisLine: { lineStyle: { color: "#c9cec8" } },
              splitLine: { show: false },
            },
            yAxis: {
              type: "value",
              scale: true,
              axisLabel: { color: "#6c7773", fontSize: 9 },
              splitLine: { lineStyle: { color: "#e5e7e2" } },
            },
            series: [{
              name: label,
              type: "line",
              data,
              connectNulls: false,
              symbol: "circle",
              symbolSize: 5,
              itemStyle: { color },
              lineStyle: { color, width: 2 },
            }],
          };

          return (
            <article className="run-metric-chart" key={definition.key}>
              <header>
                <div>
                  <span>{definition.key}</span>
                  <h3 title={`${description}; ${t("单位", "unit")}: ${unit}`}>{label}</h3>
                </div>
                <strong>{data.filter(([, value]) => value !== null).length} {t("点", "points")}</strong>
              </header>
              <p>{description}</p>
              {hasValues ? (
                <div role="img" aria-label={t(`${definition.label}随 PPO 更新变化的折线图`, `${label} line chart over PPO updates`)}>
                  <RunMetricChart option={option} />
                </div>
              ) : (
                <div className="run-metric-empty">{t("本次运行尚未上报该指标", "This run has not reported this metric")}</div>
              )}
            </article>
          );
        })}
      </div>
      {missingDefinitions.length > 0 && (
        <aside className="simulation-metric-gaps" aria-label={t("Simulation 指标接口缺口", "Simulation metric interface gaps")}>
          <strong>{t("Simulation 接口缺口", "Simulation interface gaps")}</strong>
          <p>{missingDefinitions.map((definition) => runMetricText(definition, "label", locale)).join(locale === "en" ? ", " : "、")}{t("尚未由本次 Run 的 Event 输出；页面不推算、不补 0。", " were not emitted by this Run's Events; the page does not infer or fill them with zero.")}</p>
        </aside>
      )}
    </div>
  );
}

function RunDetailPage({ runId }: { runId: string }) {
  const { locale, t } = useI18n();
  const queryClient = useQueryClient();
  const previousRunStatusRef = useRef<RunRecord["status"] | null>(null);
  const run = useQuery({
    queryKey: ["run-detail", runId],
    queryFn: () =>
      api<RunRecord>(`/api/v1/runs/${encodeURIComponent(runId)}`),
    refetchInterval: (query) =>
      query.state.data && !["completed", "failed", "stopped"].includes(query.state.data.status)
        ? 2000
        : false,
  });
  const artifacts = useQuery({
    queryKey: ["run-artifacts", runId],
    queryFn: () => api<RunArtifact[]>(
      `/api/v1/runs/${encodeURIComponent(runId)}/artifacts`,
    ),
    enabled: Boolean(run.data),
  });
  const [events, setEvents] = useState<MetricEvent[]>([]);
  const [lastSequence, setLastSequence] = useState(0);
  const lastSequenceRef = useRef(0);
  const [connection, setConnection] = useState<"connecting" | "live" | "reconnecting" | RunRecord["status"]>("connecting");
  const [replayError, setReplayError] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState("");
  const [exporting, setExporting] = useState(false);
  const [exportFeedback, setExportFeedback] = useState<{
    status: "success" | "error";
    message: string;
  } | null>(null);
  const [artifactDownloads, setArtifactDownloads] = useState<Record<
    string,
    { status: "downloading" | "success" | "error"; message: string }
  >>({});
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const nextStatus = run.data?.status;
    const previousStatus = previousRunStatusRef.current;
    previousRunStatusRef.current = nextStatus ?? null;
    if (
      nextStatus &&
      previousStatus &&
      !isTerminalRunStatus(previousStatus) &&
      isTerminalRunStatus(nextStatus)
    ) {
      void queryClient.invalidateQueries({
        queryKey: ["run-artifacts", runId],
        exact: true,
      });
    }
  }, [queryClient, run.data?.status, runId]);

  useEffect(() => {
    if (!run.data || isTerminalRunStatus(run.data.status)) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [run.data?.run_id, run.data?.status]);

  useEffect(() => {
    if (!run.data) return;
    setReplayError(false);
    if (isTerminalRunStatus(run.data.status)) {
      setConnection(run.data.status);
    }
    let socket: WebSocket | null = null;
    let reconnectTimer = 0;
    let disposed = false;
    let observedTerminal = false;
    const terminal = () =>
      observedTerminal || (run.data ? isTerminalRunStatus(run.data.status) : false);
    const appendEvents = (incoming: MetricEvent[]) => {
      setEvents((current) => mergeRunEvents(current, incoming));
      const newest = latestRunEventSequence(incoming, lastSequenceRef.current);
      if (newest > lastSequenceRef.current) {
        lastSequenceRef.current = newest;
        setLastSequence(newest);
      }
    };
    const replayEvents = async () => {
      try {
        const response = await api<unknown>(
          `/api/v1/runs/${encodeURIComponent(runId)}/events?after_sequence=${lastSequenceRef.current}`,
        );
        if (!disposed) {
          appendEvents(normalizeRunEventsResponse(response));
          setReplayError(false);
        }
      } catch {
        if (!disposed) {
          setReplayError(true);
        }
      }
    };
    const connect = async () => {
      if (disposed) return;
      await replayEvents();
      if (disposed || terminal()) return;
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const token = getAccessToken();
      const query = new URLSearchParams({ after_sequence: String(lastSequenceRef.current) });
      socket = new WebSocket(
        `${protocol}://${location.host}/api/v1/runs/${encodeURIComponent(runId)}/events?${query}`,
      );
      socket.onopen = () => {
        if (token) socket?.send(JSON.stringify({ type: "AUTH", access_token: token }));
      };
      socket.onmessage = (message) => {
        const raw = JSON.parse(message.data) as { type?: string };
        if (raw.type === "AUTH_ACK") {
          setConnection("live");
          return;
        }
        const event = normalizeRunEvent(raw);
        if (!event) return;
        appendEvents([event]);
        if (["state", "status"].includes(event.type) && event.payload.status) {
          if (isTerminalRunStatus(event.payload.status)) {
            observedTerminal = true;
            setConnection(event.payload.status);
            socket?.close();
          }
          void run.refetch();
        }
      };
      socket.onerror = () => {
        if (!terminal()) setConnection("reconnecting");
      };
      socket.onclose = () => {
        if (!disposed && !terminal()) {
          setConnection("reconnecting");
          reconnectTimer = window.setTimeout(connect, 1500);
        }
      };
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [locale, run.data?.run_id, run.data?.status, runId]);

  if (run.isLoading) return <div className="loading">{t("正在恢复运行记录...", "Restoring run record...")}</div>;
  if (run.isError || !run.data) {
    return (
      <main className="operations-page">
        <section className="run-results-panel query-error">
          <header>
            <div>
              <span className="eyebrow">{t("运行结果", "Run results")}</span>
              <h1>{t("训练结果与导出", "Training results and export")}</h1>
            </div>
            <button className="secondary" disabled>
              <Download size={16} /> {t("导出结果", "Export results")}
            </button>
          </header>
          <p className="action-feedback error" role="alert">
            {t("运行记录加载失败，当前无法确认结果或导出资格。", "The run record could not be loaded, so the result and export eligibility cannot be confirmed.")}
          </p>
          <button className="secondary" onClick={() => void run.refetch()}>
            <RotateCcw size={14} /> {t("重试查询", "Retry query")}
          </button>
        </section>
      </main>
    );
  }
  const record = run.data;
  const metricHistory = runMetricHistory(events);
  const latestMetricPoint = metricHistory[metricHistory.length - 1];
  const summary: Record<string, unknown> = {
    ...record.metrics_summary,
    ...(latestMetricPoint?.values ?? {}),
    ...(latestMetricPoint ? { update_index: latestMetricPoint.updateIndex } : {}),
  };
  const statusPresentation = runStatusPresentation(record.status, locale);
  const connectionLabel = connection === "connecting"
    ? t("连接中", "Connecting")
    : connection === "live"
      ? t("实时连接", "Live connection")
      : connection === "reconnecting"
        ? t("实时连接中断，正在恢复", "Live connection interrupted; reconnecting")
        : runConnectionLabel(connection, locale) ?? runStatusPresentation(connection, locale).label;
  const progressPresentation = runProgressPresentation(
    record,
    latestMetricPoint?.trainingSteps ?? undefined,
    typeof summary.update_index === "number" ? summary.update_index : undefined,
  );
  const elapsed = formatElapsedDuration(runElapsedMilliseconds(record, now), locale);
  const runParameters = parameterFields(
    record.project_config_snapshot.parameter_schema,
    record.parameters,
  );
  const canStop = !["completed", "failed", "stopped", "stopping"].includes(record.status);
  const canExport = runExportEligible(record);
  const isDemoRun = isDemoRunRecord(record);
  const taskConfig = record.project_config_snapshot.task_config as { video_enabled?: unknown } | undefined;
  const videoEnabled = runVideoEnabled(record.project_config_snapshot)
    || taskConfig?.video_enabled === true
    || record.parameters?.video_enabled === true;
  const { trainingVideo, demoVideo, evaluationVideo } = selectReplayArtifacts(
    artifacts.data ?? [],
  );
  const primaryVideo = isDemoRun ? demoVideo : trainingVideo;
  const videoMode = runVideoMode(
    videoEnabled,
    record.status,
    Boolean(primaryVideo),
    artifacts.isLoading || artifacts.isFetching,
    artifacts.isError,
  );
  const emptyArtifactMessage =
    record.status === "finalizing"
      ? isDemoRun
        ? t("Demo 正在结束；该任务不生成训练 checkpoint。", "The Demo is finishing; this task does not produce a training checkpoint.")
        : t("正在回传训练结果", "Returning training results")
      : record.status === "completed"
        ? isDemoRun
          ? t("Demo 已正常完成；未生成训练 checkpoint，符合该任务规格。", "The Demo completed normally; no training checkpoint was produced, as specified for this task.")
          : t("训练已完成，暂未收到制品；可重试读取。", "Training is complete, but artifacts have not arrived yet; retry reading them.")
        : record.status === "running"
          ? isDemoRun
            ? t("Demo 进行中；不会生成训练 checkpoint。", "The Demo is running; it will not produce a training checkpoint.")
            : t("训练进行中，结果将在结束后回传。", "Training is in progress; results will be returned when it finishes.")
          : record.status === "initializing"
            ? isDemoRun
              ? t("正在初始化 Demo。", "Initializing the Demo.")
              : t("正在初始化训练。", "Initializing training.")
            : record.status === "queued"
              ? isDemoRun
                ? t("Demo 正在排队；不会生成训练 checkpoint。", "The Demo is queued; it will not produce a training checkpoint.")
                : t("训练尚未产生结果。", "Training has not produced results yet.")
              : t("当前没有已回传的制品。", "No returned artifacts are available.");

  async function exportRunResults() {
    if (!canExport || exporting) return;
    setExporting(true);
    setExportFeedback(null);
    try {
      const response = await apiBlob("/api/v1/runs/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: [record.run_id], format: "json" }),
      });
      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") ?? "";
      const filename =
        disposition.match(/filename="?([^";]+)"?/)?.[1] ??
        `${record.run_id}-runs.json`;
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(blob);
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(anchor.href);
      setExportFeedback({
        status: "success",
        message: `${t("下载已开始：", "Download started: ")}${filename}`,
      });
    } catch (caught) {
      setExportFeedback({
        status: "error",
        message: caught instanceof Error ? caught.message : t("导出失败，请重试", "Export failed; try again."),
      });
    } finally {
      setExporting(false);
    }
  }
  function exportMetricsCsv() {
    const contents = buildRunMetricsCsv(record, metricHistory);
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(
      new Blob([contents], { type: "text/csv;charset=utf-8" }),
    );
    anchor.download = `${record.run_id}-metrics.csv`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }
  async function stopRun() {
    if (!window.confirm(t("确认停止这次训练？已保存的事件和制品会保留，停止后不能继续本次 Run。", "Stop this training run? Saved events and artifacts will be kept, but this Run cannot continue after it stops."))) return;
    setStopping(true);
    setStopError("");
    try {
      await api<RunRecord>(`/api/v1/runs/${encodeURIComponent(runId)}/stop`, {
        method: "POST",
      });
      await run.refetch();
    } catch (caught) {
      setStopError(caught instanceof Error ? caught.message : t("停止请求失败，请重试", "The stop request failed; try again."));
    } finally {
      setStopping(false);
    }
  }
  async function downloadArtifact(artifactId: string, path: string) {
    setArtifactDownloads((current) => ({
      ...current,
      [artifactId]: { status: "downloading", message: t("正在准备下载", "Preparing download") },
    }));
    try {
      const response = await apiBlob(
        `/api/v1/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}/download`,
      );
      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") ?? "";
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(blob);
      anchor.download = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? path.split("/").pop() ?? "artifact";
      anchor.click();
      URL.revokeObjectURL(anchor.href);
      setArtifactDownloads((current) => ({
        ...current,
        [artifactId]: { status: "success", message: t("下载已开始", "Download started") },
      }));
    } catch (caught) {
      setArtifactDownloads((current) => ({
        ...current,
        [artifactId]: {
          status: "error",
          message: caught instanceof Error ? caught.message : t("下载失败，请重试", "Download failed; try again."),
        },
      }));
    }
  }
  return (
    <main className="operations-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">{t("运行详情", "Run details")}</span>
          <h1 className="run-identifier-heading">{runDisplayName(record, locale)}</h1>
          <CopyableIdentifier value={record.run_id} label="Run ID" />
          <p>{t("状态以服务端 Run 为准；浏览器刷新或实时连接中断不会停止训练。", "The server-side Run is authoritative; refreshing the browser or losing the live connection does not stop training.")}</p>
        </div>
        <div className={`run-status-card ${record.status}`} data-tour="detail-status">
          <span>{t("当前状态", "Current status")}</span>
          <strong>{statusPresentation.label}</strong>
          <small>{statusPresentation.description}</small>
        </div>
      </header>
      <section className="run-evidence" style={{ margin: "0 24px 24px", padding: 24 }}>
        <RunEvaluationCard record={record} locale={locale} />
        <div className="run-status-map" aria-label={t("Run 的八种服务端状态", "The eight server-side Run states")}>
          {RUN_STATUS_PRESENTATIONS.map((item) => (
            <div
              className={item.status === record.status ? `active ${item.status}` : ""}
              key={item.status}
            >
              <span>{runStatusLabel(item.status, locale)}</span>
              <strong>{runStatusPresentation(item.status, locale).description}</strong>
            </div>
          ))}
        </div>
        <div className="run-progress-panel" data-tour="detail-progress">
          <div className="run-progress-heading">
            <span>{t("服务端估算进度", "Server-estimated progress")}</span>
            <strong>{progressPresentation.percentage}%</strong>
          </div>
          <div
            className="run-progress-track"
            role="progressbar"
            aria-label={t("服务端估算进度", "Server-estimated progress")}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progressPresentation.percentage}
          >
            <span style={{ width: `${progressPresentation.percentage}%` }} />
          </div>
          <small>{t("该进度直接来自 Run 快照，不按浏览器停留时间推算。", "This progress comes directly from the Run snapshot and is not inferred from browser time.")}</small>
        </div>
        <div className="run-facts">
          <div><span>{t("项目", "Project")}</span><strong>{record.project_id ? <CopyableIdentifier value={record.project_id} label={t("项目编号", "Project ID")} /> : t("未指定", "Not specified")}</strong></div>
          <div><span>{t("训练步数", "Training steps")}</span><strong>{progressPresentation.trainingSteps} / {progressPresentation.expectedSteps ?? "--"}</strong></div>
          <div><span>{t("更新次数", "Updates")}</span><strong>{progressPresentation.updateIndex || t("等待更新", "Waiting for updates")}</strong></div>
          <div><span>{t("已用时", "Elapsed")}</span><strong>{elapsed}</strong></div>
          <div><span>{t("事件数量", "Events")}</span><strong>{events.length} · {t("序号", "sequence")} {lastSequence}</strong></div>
          <div><span>{t("实时连接", "Live connection")}</span><strong>{connectionLabel}</strong></div>
        </div>
        {replayError && (
          <p className="action-feedback error" role="alert">
            {isTerminalRunStatus(record.status)
              ? t("运行已结束，但历史事件读取失败；当前结果可能缺少部分指标。", "The run has ended, but historical events could not be read; some metrics may be missing.")
              : t("历史事件读取失败，正在继续尝试实时连接。", "Historical events could not be read; continuing to try the live connection.")}
          </p>
        )}
        <FrozenSceneSummary snapshot={record.project_config_snapshot} />
        <h2>{t("运行参数", "Run parameters")}</h2>
        {runParameters.length ? (
          <div className="run-parameter-list" data-tour="detail-parameters">
            {runParameters.map((field) => (
              <div key={field.name}>
                <span>{field.label}</span>
                <strong>{field.value == null ? "--" : String(field.value)}</strong>
                <small>
                  {field.type}
                  {field.minimum !== null ? ` · ${t("最小", "min")} ${field.minimum}` : ""}
                  {field.maximum !== null ? ` · ${t("最大", "max")} ${field.maximum}` : ""}
                  {field.required ? ` · ${t("必填", "required")}` : ""}
                </small>
              </div>
            ))}
          </div>
        ) : <p className="run-empty-note">{t("冻结快照未声明可展示参数。", "The frozen snapshot declares no displayable parameters.")}</p>}
        <h2>{t("关键指标", "Key metrics")}</h2>
        <div className="run-facts">
          {([
            ["training_steps", latestMetricPoint?.trainingSteps ?? record.training_steps],
            ["update_index", summary.update_index ?? record.update_steps],
            ["episodes", summary.episodes],
            ...visibleRunMetricDefinitions(metricHistory).map(
              (definition) => [definition.key, summary[definition.key]],
            ),
          ] as Array<[string, unknown]>).map(([key, value]) => (
            <div key={key}>
              <span title={RUN_METRIC_DEFINITIONS.find((item) => item.key === key) ? runMetricText(RUN_METRIC_DEFINITIONS.find((item) => item.key === key)!, "description", locale) : undefined}>
                {RUN_METRIC_DEFINITIONS.find((item) => item.key === key) ? runMetricText(RUN_METRIC_DEFINITIONS.find((item) => item.key === key)!, "label", locale) : RUN_METRIC_LABELS[key]?.[locale] ?? key}
              </span>
              <strong>{value == null ? t("等待指标", "Waiting for metrics") : formatRunMetricValue(typeof value === "number" ? value : null, locale)}</strong>
            </div>
          ))}
        </div>
        <h2>{t("指标历史", "Metric history")}</h2>
        <RunMetricCharts points={metricHistory} />
        {record.error && (
          <p className="inline-error">
            {typeof record.error === "string"
              ? record.error
              : record.error.message ?? t("运行失败", "Run failed")}
          </p>
        )}
        <h2>{t("视频流", "Video stream")}</h2>
        {videoMode === "disabled" ? (
          <section className="run-video-panel replay-state" data-video-presentation="disabled">
            {t("本次运行未启用视频", "Video was not enabled for this run")}
          </section>
        ) : videoMode === "live" ? (
          <RunVideoPanel runId={record.run_id} runStatus={record.status} />
        ) : videoMode === "finalizing" ? (
          <section className="run-video-panel replay-state" data-video-presentation="finalizing">
            {t("实时训练已结束，正在准备录像", "Live training has ended; preparing the recording")}
          </section>
        ) : videoMode === "replay" && primaryVideo ? (
            <ReplayPlayer runId={record.run_id} artifact={primaryVideo} />
        ) : videoMode === "error" ? (
            <section className="run-video-panel replay-state replay-error" data-video-presentation="error">
              <p role="alert">{t("训练录像信息读取失败", "Could not read training recording information")}</p>
              <button className="secondary" onClick={() => void artifacts.refetch()}>
                <RotateCcw size={14} /> {t("重试", "Retry")}
              </button>
            </section>
        ) : videoMode === "preparing" ? (
            <section className="run-video-panel replay-state" data-video-presentation="preparing">
              {t("正在准备训练录像", "Preparing training recording")}
            </section>
        ) : videoMode === "unavailable" ? (
            <section className="run-video-panel replay-state" data-video-presentation="unavailable">
              {t("本次运行没有可用的训练录像", "No training recording is available for this run")}
            </section>
        ) : (
          <RunVideoPanel runId={record.run_id} runStatus={record.status} />
        )}
        <h2>{t("成功策略评估回放", "Successful-policy evaluation replay")}</h2>
        {evaluationVideo ? (
          <ReplayPlayer runId={record.run_id} artifact={evaluationVideo} />
        ) : artifacts.isLoading || artifacts.isFetching ? (
          <section className="run-video-panel replay-state">{t("正在读取评估回放信息", "Reading evaluation replay information")}</section>
        ) : artifacts.isError ? (
          <section className="run-video-panel replay-state replay-error">{t("评估回放信息读取失败", "Could not read evaluation replay information")}</section>
        ) : (
          <section className="run-video-panel replay-state">{t("本次运行没有可用的成功策略评估回放；这不改变 Run 或 Policy 结论。", "No successful-policy evaluation replay is available for this run; this does not change the Run or Policy conclusion.")}</section>
        )}
        <section className="run-results-panel" data-tour="detail-results" aria-labelledby="run-results-title">
          <header>
            <div>
              <span className="eyebrow">{t("运行结果", "Run results")}</span>
              <h2 id="run-results-title">{t("训练结果与导出", "Training results and export")}</h2>
            </div>
            <div className="button-row">
              <button className="secondary" onClick={exportMetricsCsv}>
                <Download size={16} /> {t("导出指标 CSV", "Export metrics CSV")}
              </button>
              <button
                className={canExport ? "primary run-export-ready" : "secondary"}
                disabled={!canExport || exporting}
                onClick={() => void exportRunResults()}
              >
                <Download size={16} />
                {exporting
                  ? t("正在导出", "Exporting")
                  : exportFeedback?.status === "error"
                    ? t("重试导出", "Retry export")
                    : isDemoRun
                      ? t("导出记录", "Export record")
                      : t("导出结果", "Export results")}
              </button>
            </div>
          </header>
          {!canExport && (
            <p className="run-result-state">
              {t("Run 完成且 Backend 允许导出后，此按钮将自动启用。", "This button is enabled when the Run is complete and the Backend allows export.")}
            </p>
          )}
          {exportFeedback && (
            <p
              className={`action-feedback ${exportFeedback.status}`}
              role={exportFeedback.status === "error" ? "alert" : undefined}
              aria-live="polite"
            >
              {exportFeedback.message}
            </p>
          )}
          <h3>{isDemoRun ? t("Demo 输出", "Demo outputs") : t("制品", "Artifacts")}</h3>
          {artifacts.isLoading || artifacts.isFetching ? (
            <p>{isDemoRun ? t("正在读取 Demo 输出", "Reading Demo outputs") : t("正在回传训练结果", "Returning training results")}</p>
          ) : artifacts.isError ? (
            <div className="artifact-query-error">
              <p className="action-feedback error" role="alert">
                {t("制品目录读取失败。", "Could not read the artifact catalog.")}
              </p>
              <button className="secondary" onClick={() => void artifacts.refetch()}>
                <RotateCcw size={14} /> {t("重试读取", "Retry reading")}
              </button>
            </div>
          ) : artifacts.data?.length ? (
            <div className="artifact-list">
              {artifacts.data.map((artifact) => {
                const feedback = artifactDownloads[artifact.artifact_id];
                return (
                  <div className="artifact-row" key={artifact.artifact_id}>
                    <code>{artifactLabel(artifact.artifact_type, locale)}</code>
                    <span title={artifact.path}>{shortIdentifier(artifact.path)}</span>
                    <small>{artifact.sha256?.slice(0, 12) ?? t("无校验值", "No checksum")}</small>
                    <button
                      className="secondary"
                      disabled={feedback?.status === "downloading"}
                      onClick={() =>
                        void downloadArtifact(artifact.artifact_id, artifact.path)
                      }
                    >
                      <Download size={14} />
                      {feedback?.status === "downloading"
                        ? t("下载中", "Downloading")
                        : feedback?.status === "error"
                          ? t("重试", "Retry")
                          : t("下载", "Download")}
                    </button>
                    {feedback && (
                      <p
                        className={`artifact-feedback ${feedback.status}`}
                        aria-live="polite"
                      >
                        {feedback.message}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="run-result-empty">
              <p>{emptyArtifactMessage}</p>
              {isTerminalRunStatus(record.status) && !isDemoRun && (
                <button className="secondary" onClick={() => void artifacts.refetch()}>
                  <RotateCcw size={14} /> {t("重试读取", "Retry reading")}
                </button>
              )}
            </div>
          )}
        </section>
        <h2>{t("事件", "Events")}</h2>
        <div className="event-log" data-tour="detail-events">{events.slice(-12).map((event) => (
          <div key={event.event_id}><code>{event.sequence ?? "-"}</code><span>{runEventLabel(event.type, locale)}</span><small title={event.event_id}>{shortIdentifier(event.event_id)}</small></div>
        ))}</div>
        <div className="button-row">
          {canStop && <button className="danger" disabled={stopping} onClick={() => void stopRun()}><CircleStop size={16} /> {stopping ? t("正在停止", "Stopping") : t("停止运行", "Stop run")}</button>}
          <button className="secondary" onClick={() => navigate("/runs")}><FlaskConical size={16} /> {t("返回运行记录", "Back to runs")}</button>
        </div>
        {stopError && <p className="inline-error">{stopError}</p>}
      </section>
    </main>
  );
}

const S1_DEMO_TASK_IDS = [
  "demo_s1_scripted_pick_place",
  "demo_s1_target_pose",
  "demo_s1_line_tracking",
  "demo_s1_circle_tracking",
  "demo_s1_joint_position",
  "demo_s1_joint_sweep",
] as const;
const SHORT_PROJECT_TASK_IDS = ["s1_push", ...S1_DEMO_TASK_IDS] as const;
const STANDARD_PROJECT_TASK_IDS = ["s1_push"] as const;
const REACH_PROJECT_TASK_IDS = ["s1_reach"] as const;
const MANIPULATION_PROJECT_TASK_IDS = ["s1_grasp", "s1_lift", "s1_lower_release", "s1_place"] as const;

function TeachingFlowPage({ user }: { user: CurrentUserContext }) {
  const { locale, t } = useI18n();
  const [organizationId, setOrganizationId] = useState(() => sessionStorage.getItem("s1-draft-organization") ?? "");
  const [courseId, setCourseId] = useState(() => sessionStorage.getItem("s1-draft-course") ?? "");
  const [classId, setClassId] = useState(() => sessionStorage.getItem("s1-draft-class") ?? "");
  const [projectId, setProjectId] = useState(() => sessionStorage.getItem("s1-draft-project") ?? "");
  const [sceneId, setSceneId] = useState(() => sessionStorage.getItem("s1-draft-scene") ?? "");
  const [taskDraft] = useState(readS1TaskDraft);
  const [seed, setSeed] = useState(() => Number(taskDraft?.parameters.seed ?? sessionStorage.getItem("s1-draft-seed") ?? "42"));
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [taskConfiguration, setTaskConfiguration] = useState<SimulationTaskConfigurationState | null>(null);
  const runSubmission = useRef(createCanonicalRunSubmission());
  const userOrganizations = user.organizations;
  const courseQuery = useQuery({
    queryKey: ["teaching-courses", organizationId],
    queryFn: () => api<TeachingCourse[]>(`/api/v1/courses?organization_id=${encodeURIComponent(organizationId)}`),
    enabled: Boolean(organizationId),
  });
  const classQuery = useQuery({
    queryKey: ["teaching-classes", courseId],
    queryFn: () => api<TeachingClass[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/classes`),
    enabled: Boolean(courseId),
  });
  const projectQuery = useQuery({
    queryKey: ["teaching-projects", courseId],
    queryFn: () => api<TeachingProject[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/projects`),
    enabled: Boolean(courseId),
  });
  const scenesQuery = useQuery({
    queryKey: ["teaching-scenes"],
    queryFn: () => api<SceneDocument[]>("/api/scenes"),
  });
  const selectedProject = projectQuery.data?.find((project) => project.project_id === projectId);
  const selectedCourse = courseQuery.data?.find((course) => course.course_id === courseId);
  const selectedClass = classQuery.data?.find((item) => item.class_id === classId);
  const formalScenes = (scenesQuery.data ?? []).filter((item) => item.capability !== "mock" && item.robot.dof === S1_ARM_JOINTS.length);
  const selectedScene = formalScenes.find((item) => item.scene_id === sceneId) ?? formalScenes[0];
  const allowedTaskIds = MANIPULATION_PROJECT_TASK_IDS.includes(
    selectedProject?.environment_type as typeof MANIPULATION_PROJECT_TASK_IDS[number],
  )
    ? [selectedProject!.environment_type]
    : selectedProject?.environment_type === "s1_reach"
    ? REACH_PROJECT_TASK_IDS
    : selectedProject?.training_profile === "short"
      ? SHORT_PROJECT_TASK_IDS
      : STANDARD_PROJECT_TASK_IDS;
  const seedField = selectedProject
    ? parameterFields(selectedProject.parameter_schema, { seed }).find(
        (field) => field.name === "seed" && field.type === "integer",
      )
    : undefined;
  const minimumSeed = seedField?.minimum ?? 0;
  const maximumSeed = seedField?.maximum ?? 2147483647;
  const updateTaskConfiguration = (next: SimulationTaskConfigurationState) => {
    setTaskConfiguration(next);
    if (typeof next.parameters.seed === "number") setSeed(next.parameters.seed);
  };

  async function createRun() {
    if (!selectedProject || !selectedClass || runSubmission.current.isPending()) return;
    if (!taskConfiguration?.task || !taskConfiguration.validation?.valid) {
      setError(t("请先完成有效的 Simulation task 配置。", "Complete a valid Simulation task configuration first."));
      return;
    }
    const isDemo = taskConfiguration.task.mode === "demo";
    if (!allowedTaskIds.includes(taskConfiguration.task.id as never)) {
      setError(t("这个任务与所选 Project 不兼容。", "This task is not compatible with the selected Project."));
      return;
    }
    setCreating(true);
    setError("");
    try {
      const run = await runSubmission.current.submit({
        project_id: selectedProject.project_id,
        class_id: selectedClass.class_id,
        project_config_version: selectedProject.config_version,
        run_kind: isDemo ? "play" : "train",
        environment_type: taskConfiguration.task.id,
        parameters: taskParameterPayload(taskConfiguration.task, taskConfiguration.parameters),
        code: "",
        ...(!isDemo && taskConfiguration.task.id === "s1_push" && selectedScene
          ? { scene_ref: { scene_id: selectedScene.scene_id, revision: selectedScene.revision } }
          : {}),
      });
      navigate(runDetailPath(run.run_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("正式 Run 创建失败", "Could not create formal Run"));
    } finally {
      setCreating(false);
    }
  }

  return (
    <main className="research-page">
      <header className="research-header">
        <div>
          <span className="eyebrow">S1 PUSH / REACH / MANIPULATION · TEACHING FLOW</span>
          <h1>{t("课程与实验", "Courses and experiments")}</h1>
          <p>{t("从组织、课程和教学班进入 Project；训练配置由服务端版本冻结。", "Enter a Project through an organization, course, and class; training configuration is frozen by the server revision.")}</p>
        </div>
        <div className="roadmap-progress"><strong>FORMAL</strong><small>SIMULATION ONLY</small></div>
      </header>
      <section className="roadmap-workspace">
        <aside className="roadmap-list" data-tour="teaching-context">
          <label className="field"><span>{t("组织", "Organization")}</span><select value={organizationId} onChange={(event) => { setOrganizationId(event.target.value); setCourseId(""); setClassId(""); setProjectId(""); }}>
            <option value="">{t("选择组织", "Select organization")}</option>
            {userOrganizations.map((organization) => <option key={organization.organization_id} value={organization.organization_id}>{organization.name || organization.slug || shortIdentifier(organization.organization_id)}</option>)}
          </select></label>
          <label className="field"><span>{t("课程", "Course")}</span><select value={courseId} disabled={!organizationId} onChange={(event) => { setCourseId(event.target.value); setClassId(""); setProjectId(""); }}>
            <option value="">{t("选择课程", "Select course")}</option>
            {(courseQuery.data ?? []).map((course) => <option key={course.course_id} value={course.course_id}>{course.code} · {course.title}</option>)}
          </select></label>
          <label className="field"><span>{t("教学班", "Class")}</span><select value={classId} disabled={!courseId} onChange={(event) => setClassId(event.target.value)}>
            <option value="">{t("选择教学班", "Select class")}</option>
            {(classQuery.data ?? []).map((item) => <option key={item.class_id} value={item.class_id}>{item.code} · {item.name} · {item.term}</option>)}
          </select></label>
          <div className="section-label" data-tour="teaching-projects">{t("TEACHING PROJECTS", "TEACHING PROJECTS")}</div>
          {(projectQuery.data ?? []).map((project) => <button key={project.project_id} className={project.project_id === projectId ? "active" : ""} onClick={() => setProjectId(project.project_id)} disabled={!project.enabled}>
            <FolderKanban size={15} /><span><strong>{teachingProjectLabel(project, locale)}</strong><small>{t(`项目配置第 ${project.config_version} 版`, `Project config v${project.config_version}`)} · {project.video_enabled ? t("包含训练视频", "Training video included") : t("不含训练视频", "No training video")}</small></span><ChevronRight size={15} />
          </button>)}
        </aside>
        <section className="code-editor-pane" data-tour="project-detail" style={{ padding: 28 }}>
          {!selectedProject ? <div className="workspace-empty"><GraduationCap size={34} /><strong>{t("选择一个教学 Project", "Select a teaching Project")}</strong><span>{t("Project 配置、指标和评估 rubric 由服务端提供。", "Project configuration, metrics, and the evaluation rubric are provided by the server.")}</span></div> : <>
            <span className="eyebrow">PROJECT DETAIL</span><h2>{teachingProjectLabel(selectedProject, locale)}</h2><p>{selectedProject.description || t("暂无描述", "No description")}</p>
            <div className="run-facts"><div><span>{t("课程", "Course")}</span><strong>{selectedCourse?.title ?? selectedProject.course_id}</strong></div><div><span>{t("教学班", "Class")}</span><strong>{selectedClass?.name ?? t("未选择", "Not selected")}</strong></div><div><span>{t("训练方法", "Training method")}</span><strong>{selectedProject.algorithm}</strong></div><div><span>{t("预计步数", "Expected steps")}</span><strong>{selectedProject.expected_training_steps.toLocaleString(locale)}</strong></div><div><span>{t("训练视频", "Training video")}</span><strong>{selectedProject.video_enabled ? t("已开启", "Enabled") : t("未开启", "Disabled")}</strong></div><div><span>{t("使用权限", "Usage permission")}</span><strong>{selectedProject.permissions.can_use ? t("可以开始", "Ready to start") : t("暂不可用", "Unavailable")}</strong></div></div>
            <h3>{t("教学说明", "Teaching notes")}</h3><p>{selectedProject.instructions || t("暂无教学说明", "No teaching notes")}</p>
            <h3>{t("只读训练配置", "Read-only training configuration")}</h3><pre>{JSON.stringify({ task_type: selectedProject.task_type, environment_type: selectedProject.environment_type, training_profile: selectedProject.training_profile, success_condition: selectedProject.success_condition, parameter_schema: selectedProject.parameter_schema, evaluation_rubric: selectedProject.evaluation_rubric, config_version: selectedProject.config_version }, null, 2)}</pre>
            <div data-tour="training-task"><SimulationTaskConfigurator
              key={`${selectedProject.project_id}:${selectedProject.config_version}`}
              initialTaskId={selectedProject.environment_type}
              initialParameters={taskDraft?.taskId === selectedProject.environment_type ? taskDraft.parameters : undefined}
              allowedTaskIds={allowedTaskIds}
              compact
              onStateChange={updateTaskConfiguration}
              disabled={creating}
            /></div>
            <div className="field" data-tour="scene-selection">
              <label htmlFor="formal-scene">{t("实验场景", "Experiment scene")}</label>
              <select
                id="formal-scene"
                value={selectedScene?.scene_id ?? ""}
                onChange={(event) => setSceneId(event.target.value)}
                disabled={scenesQuery.isLoading || scenesQuery.isError}
              >
                <option value="" disabled>{t("选择六轴 S1 场景", "Select a six-axis S1 scene")}</option>
                {formalScenes.map((scene) => (
                  <option key={scene.scene_id} value={scene.scene_id}>
                    {scene.name} · REV {scene.revision} · S1
                  </option>
                ))}
              </select>
              <small>
                {scenesQuery.isError
                  ? t("场景服务暂不可用，本次 Run 不会伪造场景证据。", "The scene service is unavailable; this Run will not fabricate scene evidence.")
                  : selectedScene
                    ? selectedProject?.environment_type === "s1_push"
                      ? t(`服务端将在创建时冻结 ${selectedScene.scene_id} · REV ${selectedScene.revision}。`, `The server will freeze ${selectedScene.scene_id} · REV ${selectedScene.revision} when creating the Run.`)
                      : t("Reach Run 使用 Project 的任务配置；当前 canonical contract 不发送 scene_ref。", "Reach Runs use the Project task configuration; the current canonical contract does not send scene_ref.")
                    : t("请先在环境编辑中保存六轴 S1 场景。", "Save a six-axis S1 scene in the Environment editor first.")}
              </small>
            </div>
            {!seedField && <p className="inline-error">{t("当前 Project 快照未声明可提交的整数 seed。", "The current Project snapshot does not declare a submit-ready integer seed.")}</p>}
            {seedField && taskConfiguration?.task && (
              <p className="field-note">
                {t("seed 已由 Simulation catalog 参数编辑器管理；Project schema 仍用于提交前范围校验。", "Seed is managed by the Simulation catalog editor; the Project schema is still used for pre-submit range validation.")}
              </p>
            )}
            {error && <p className="inline-error">{error}</p>}
            <button className="primary" data-tour="create-run" disabled={creating || !classId || !selectedProject.permissions.can_use || !taskConfiguration?.task || !taskConfiguration.validation?.valid || !allowedTaskIds.includes(taskConfiguration.task.id as never) || (taskConfiguration.task.mode === "training" && taskConfiguration.task.id === "s1_push" && !selectedScene)} onClick={createRun}><Play size={16} fill="currentColor" /> {creating ? t("正在启动实验...", "Starting experiment...") : taskConfiguration?.task?.mode === "demo" ? t("启动 Demo", "Start Demo") : t("开始训练", "Start training")}</button>
          </>}
        </section>
      </section>
    </main>
  );
}

function evaluationSummary(report: CanonicalEvaluationReport, locale: "zh-CN" | "en"): string {
  if (report.status === "insufficient_evidence") {
    return locale === "en"
      ? "An evaluation record was generated, but the available evidence is insufficient for a trustworthy score. Complete the evidence listed in the report before using the result for experiment comparison."
      : "这次运行已经生成评估记录，但现有证据不足以给出可信分数。请先补齐报告列出的证据，再把结果用于实验比较。";
  }
  const score = report.score === null
    ? locale === "en" ? "no score was assigned" : "未给出分数"
    : locale === "en" ? `the score is ${report.score}` : `分数为 ${report.score}`;
  return locale === "en"
    ? `The evidence evaluation for this Run is complete; ${score}, with grade ${report.grade}. This conclusion applies only to this record and does not represent real-robot deployment safety or cross-task generalization.`
    : `这次运行的证据评估已完成，${score}，等级为 ${report.grade}。该结论只适用于本次记录，不代表真实机器人部署安全或跨任务泛化。`;
}

function CanonicalEvaluationPage() {
  const { locale, t } = useI18n();
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedReport, setSelectedReport] = useState<CanonicalEvaluationReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const [agentProviderId, setAgentProviderId] = useState(
    () => localStorage.getItem("embodied-arm-agent-provider") ?? "mock",
  );
  const [agentLevel, setAgentLevel] = useState(
    () => Number(localStorage.getItem("embodied-arm-agent-level") ?? "4"),
  );
  const [agentFocus, setAgentFocus] = useState("next_experiment");
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentError, setAgentError] = useState("");
  const [agentResult, setAgentResult] = useState<AgentMessageResponse | null>(null);
  const runs = useQuery({
    queryKey: ["canonical-evaluation-runs"],
    queryFn: () => api<RunRecord[]>("/api/v1/runs"),
  });
  const terminalRuns = (runs.data ?? []).filter((run) =>
    ["completed", "failed", "stopped"].includes(run.status),
  );
  const activeRun = terminalRuns.find((run) => run.run_id === selectedRunId) ?? terminalRuns[0];
  const history = useQuery({
    queryKey: ["canonical-evaluation-history", activeRun?.run_id],
    queryFn: () => api<CanonicalEvaluationReport[]>(
      `/api/v1/evaluations?run_id=${encodeURIComponent(activeRun!.run_id)}`,
    ),
    enabled: Boolean(activeRun),
  });
  const agentProviders = useQuery({
    queryKey: ["agent-providers"],
    queryFn: () => api<AgentProviderInfo[]>("/api/agent/providers"),
  });
  const selectedAgentProvider = agentProviders.data?.find(
    (provider) => provider.id === agentProviderId,
  );
  const canReadEvaluationEvidence =
    agentLevel === 5 && selectedAgentProvider?.configured === true;

  useEffect(() => {
    if (!selectedRunId && terminalRuns[0]) setSelectedRunId(terminalRuns[0].run_id);
  }, [selectedRunId, terminalRuns]);

  useEffect(() => {
    setSelectedReport(history.data?.[0] ?? null);
  }, [activeRun?.run_id, history.data]);

  useEffect(() => {
    setAgentResult(null);
    setAgentError("");
  }, [activeRun?.run_id, selectedReport?.evaluation_id]);

  useEffect(() => {
    const refreshAgentPreference = () => {
      setAgentProviderId(localStorage.getItem("embodied-arm-agent-provider") ?? "mock");
      setAgentLevel(Number(localStorage.getItem("embodied-arm-agent-level") ?? "4"));
      setAgentResult(null);
      setAgentError("");
    };
    window.addEventListener("studio-agent-preference-changed", refreshAgentPreference);
    return () => window.removeEventListener("studio-agent-preference-changed", refreshAgentPreference);
  }, []);

  async function createEvaluation() {
    if (!activeRun) return;
    setBusy(true);
    setError("");
    setFeedback("");
    try {
      const report = await api<CanonicalEvaluationReport>("/api/v1/evaluations", {
        method: "POST",
        body: JSON.stringify({ run_id: activeRun.run_id }),
      });
      setSelectedReport(report);
      await history.refetch();
      setFeedback(t(`评估已生成并保存：${report.evaluation_id}`, `Evaluation generated and saved: ${report.evaluation_id}`));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("评估请求失败", "Evaluation request failed"));
    } finally {
      setBusy(false);
    }
  }

  async function requestAgentEvaluation() {
    if (!activeRun || !selectedReport || !canReadEvaluationEvidence) return;
    setAgentBusy(true);
    setAgentError("");
    setAgentResult(null);
    const focusLabels: Record<string, string> = {
      next_experiment: t("下一轮实验设计与参数优先级", "next-experiment design and parameter priorities"),
      failure_analysis: t("失败模式、证据缺口与排查顺序", "failure modes, evidence gaps, and troubleshooting order"),
      learning_review: t("面向学习者的策略理解与反思问题", "learner-facing policy understanding and reflection questions"),
    };
    try {
      const result = await api<AgentMessageResponse>(
        `/api/v1/evaluations/${encodeURIComponent(selectedReport.evaluation_id)}/commentary`,
        {
          method: "POST",
          body: JSON.stringify({
            provider: agentProviderId,
            assistance_level: agentLevel,
            focus: agentFocus,
            locale,
          }),
        },
      );
      setAgentResult(result);
    } catch (caught) {
      setAgentError(caught instanceof Error ? caught.message : t("AI 点评请求失败", "AI commentary request failed"));
    } finally {
      setAgentBusy(false);
    }
  }

  if (runs.isLoading) return <div className="loading">{t("正在读取可评估运行...", "Reading runs available for evaluation...")}</div>;
  if (runs.isError) return <div className="loading error">{t("正式 Run API 不可用。", "The formal Run API is unavailable.")}</div>;
  if (!activeRun) return <main className="evaluation-empty"><TestTube2 size={34} /><h1>{t("尚无终态 Run", "No terminal Run yet")}</h1><p>{t("完成或停止一次正式训练后才能生成证据评估。", "Complete or stop a formal training run before generating an evidence evaluation.")}</p><button className="primary" onClick={() => navigate("/courses")}><BookOpen size={16} /> {t("返回课程", "Back to courses")}</button></main>;

  return <main className="evaluation-page">
    <header className="evaluation-header"><div><span className="eyebrow">CANONICAL EVALUATION</span><h1>{t("运行证据评估", "Run evidence evaluation")}</h1><p>{t("每份不可变报告只绑定一个正式 Run；证据不足时不生成推测分数。", "Each immutable report binds to one formal Run; insufficient evidence never produces a speculative score.")}</p></div><button className="primary" data-tour="generate-evaluation" disabled={busy} onClick={() => void createEvaluation()}><TestTube2 size={16} /> {busy ? t("评估中", "Evaluating") : error ? t("重试评估", "Retry evaluation") : t("生成评估", "Generate evaluation")}</button></header>
    <section className="evaluation-workspace" data-tour="evaluation-run">
      <aside className="evaluation-context"><div className="section-label">TERMINAL RUN</div><select value={activeRun.run_id} onChange={(event) => { setSelectedRunId(event.target.value); setSelectedReport(null); setError(""); setFeedback(""); }}>{terminalRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id.slice(-8)} · {run.status}</option>)}</select><dl><div><dt>PROJECT</dt><dd>{activeRun.project_id}</dd></div><div><dt>CLASS</dt><dd>{activeRun.class_id ?? "--"}</dd></div><div><dt>SEED</dt><dd>{String(activeRun.parameters?.seed ?? "--")}</dd></div></dl>{error && <p className="action-feedback error" role="alert">{error}</p>}{feedback && <p className="action-feedback success" aria-live="polite">{feedback}</p>}</aside>
      <div data-tour="evaluation-report">
      <section className="rubric-panel"><div className="rubric-title"><span>IMMUTABLE REPORT</span><small>{selectedReport ? selectedReport.evaluation_id : t("等待报告", "Waiting for report")}</small></div>{selectedReport ? <><div className={`code-result ${selectedReport.status === "assessed" ? "valid" : "invalid"}`}><strong>{selectedReport.status === "assessed" ? t("已完成评估", "Evaluation complete") : t("证据不足", "Insufficient evidence")}</strong><small>{selectedReport.grade} · {selectedReport.score ?? t("暂不评分", "Not scored")}</small></div><div className="evaluation-natural-summary"><h2>{t("这份结果说明什么", "What this result means")}</h2><p>{evaluationSummary(selectedReport, locale)}</p></div><section className="agent-evaluation" aria-labelledby="agent-evaluation-title"><div className="agent-evaluation-heading"><span><Sparkles size={17} /><strong id="agent-evaluation-title">{t("AI 个性化策略点评", "AI personalized policy commentary")}</strong></span><small>{selectedAgentProvider?.name ?? agentProviderId} · LEVEL {agentLevel}</small></div><p>{t("AI 只解释所选不可变报告及 Run 证据，不参与正式评分，也不会改变报告。", "AI only interprets the selected immutable report and Run evidence; it does not participate in formal scoring or change the report.")}</p><div className="agent-evaluation-controls"><select aria-label={t("AI 点评重点", "AI commentary focus")} value={agentFocus} onChange={(event) => setAgentFocus(event.target.value)} disabled={agentBusy}><option value="next_experiment">{t("下一轮实验设计", "Next experiment design")}</option><option value="failure_analysis">{t("失败模式与证据缺口", "Failure modes and evidence gaps")}</option><option value="learning_review">{t("策略理解与学习反思", "Policy understanding and learning review")}</option></select><button className="secondary" onClick={() => void requestAgentEvaluation()} disabled={agentBusy || !canReadEvaluationEvidence}><Sparkles size={15} /> {agentBusy ? t("分析中", "Analyzing") : t("生成个性化点评", "Generate commentary")}</button></div>{agentProviders.isLoading ? <p className="agent-evaluation-status">{t("正在读取 Agent 配置...", "Reading Agent configuration...")}</p> : !selectedAgentProvider?.configured ? <div className="agent-evaluation-gate"><CircleAlert size={16} /><span>{t("当前 Provider 尚未在服务端配置。", "The current Provider is not configured on the server.")}<button onClick={() => navigate("/agent-settings")}>{t("配置 Agent", "Configure Agent")}</button></span></div> : agentLevel !== 5 ? <div className="agent-evaluation-gate"><ShieldCheck size={16} /><span>{t(`当前 Level ${agentLevel} 不允许读取 Run/Evaluation 证据。请由使用者在 Agent 设置中选择 Level 5，本页不会自动提升权限。`, `Level ${agentLevel} cannot read Run/Evaluation evidence. Select Level 5 in Agent settings; this page will not elevate permissions automatically.`)}<button onClick={() => navigate("/agent-settings")}>{t("查看五档权限", "View five permission levels")}</button></span></div> : null}{agentError && <p className="action-feedback error" role="alert">{agentError}</p>}{agentResult && <div className="agent-evaluation-result"><div><Bot size={17} /><strong>{t("证据解读", "Evidence interpretation")}</strong></div><p>{agentResult.content}</p><footer><span>{t("只读工具", "Read-only tools")}</span>{agentResult.tools_used.map((tool) => <code key={tool}>{tool}</code>)}</footer></div>}</section><details className="advanced-contract"><summary>{t("查看原始评估数据", "View raw evaluation data")}</summary><h3>{t("摘要", "Summary")}</h3><pre>{JSON.stringify(selectedReport.summary, null, 2)}</pre><h3>{t("证据", "Evidence")}</h3><pre>{JSON.stringify(selectedReport.evidence, null, 2)}</pre></details></> : <div className="rubric-idle"><TestTube2 size={28} /><h2>{t("尚无评估报告", "No evaluation report yet")}</h2><p>{t("生成评估后显示服务端持久化的证据与结论。", "Generate an evaluation to view server-persisted evidence and conclusions.")}</p></div>}</section>
      </div>
      <div data-tour="evaluation-boundary">
      <aside className="evaluation-summary"><div className="section-label">HISTORY</div>{history.isLoading ? <p>{t("正在读取历史...", "Loading history...")}</p> : <div className="event-log">{(history.data ?? []).map((report) => <button key={report.evaluation_id} onClick={() => setSelectedReport(report)}><code>{report.evaluation_id.slice(-8)}</code><span>{report.status}</span><small>{new Date(report.created_at).toLocaleString()}</small></button>)}</div>}{selectedReport && <><h3>{t("限制", "Limitations")}</h3><ul>{selectedReport.limitations.map((item) => <li key={item}>{item}</li>)}</ul></>}<div className="sim2real-block"><CircleStop size={17} /><span><strong>SIM2REAL BLOCKED</strong><small>REAL CONTROL DISABLED</small></span></div></aside>
      </div>
    </section>
  </main>;
}

function CanonicalExportPage() {
  const { t } = useI18n();
  const [runId, setRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const runs = useQuery({
    queryKey: ["canonical-export-runs"],
    queryFn: () => api<RunRecord[]>("/api/v1/runs"),
  });
  const completedRuns = (runs.data ?? []).filter((run) => run.status === "completed");
  const selectedRun = completedRuns.find((run) => run.run_id === runId) ?? completedRuns[0];

  useEffect(() => {
    if (!runId && completedRuns[0]) setRunId(completedRuns[0].run_id);
  }, [completedRuns, runId]);

  async function downloadExport() {
    if (!selectedRun) return;
    setBusy(true);
    setError("");
    setFeedback("");
    try {
      const response = await apiBlob("/api/v1/runs/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: [selectedRun.run_id], format: "json" }),
      });
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(await response.blob());
      const disposition = response.headers.get("content-disposition") ?? "";
      anchor.download = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? `${selectedRun.run_id}-runs.json`;
      anchor.click();
      URL.revokeObjectURL(anchor.href);
      setFeedback(t(`服务端 JSON 已生成，下载已开始：${anchor.download}`, `Server JSON generated; download started: ${anchor.download}`));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("导出请求失败", "Export request failed"));
    } finally {
      setBusy(false);
    }
  }

  if (runs.isLoading) return <div className="loading">{t("正在读取可导出运行...", "Reading runs available for export...")}</div>;
  if (runs.isError) return <div className="loading error">{t("正式 Run API 不可用。", "The formal Run API is unavailable.")}</div>;
  if (!selectedRun) return <main className="evaluation-empty"><FileCheck2 size={34} /><h1>{t("尚无已完成 Run", "No completed Run yet")}</h1><p>{t("正式 Run 完成后才能导出服务端 JSON。", "Complete a formal Run before exporting server-generated JSON.")}</p><button className="primary" onClick={() => navigate("/courses")}><BookOpen size={16} /> {t("返回课程", "Back to courses")}</button></main>;

  return <main className="export-page"><header className="operations-header"><div><span className="eyebrow">CANONICAL JSON EXPORT</span><h1>{t("运行证据导出", "Run evidence export")}</h1><p>{t("下载服务端生成的原始 JSON bytes 与权威响应头；导出不代表真实控制许可。", "Download server-generated raw JSON bytes and authoritative response headers; export does not grant real-control permission.")}</p></div></header><section className="export-workspace"><aside className="export-config"><div className="section-label">EXPORT REQUEST</div><label data-tour="artifact-run-select">{t("已完成 Run", "Completed Run")}<select value={selectedRun.run_id} onChange={(event) => { setRunId(event.target.value); setError(""); setFeedback(""); }}>{completedRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id.slice(-8)} · {run.project_id}</option>)}</select></label><button className="primary" data-tour="json-export" disabled={busy} onClick={() => void downloadExport()}><Download size={16} /> {busy ? t("生成中", "Generating") : error ? t("重试下载", "Retry download") : t("下载 JSON", "Download JSON")}</button>{error && <p className="action-feedback error" role="alert">{error}</p>}{feedback && <p className="action-feedback success" aria-live="polite">{feedback}</p>}</aside><section className="export-evidence" data-tour="canonical-evidence"><div className="run-facts"><div><span>PROJECT</span><strong>{selectedRun.project_id}</strong></div><div><span>CLASS</span><strong>{selectedRun.class_id ?? "--"}</strong></div><div><span>FORMAT</span><strong>json</strong></div><div><span>CONTROL</span><strong>DISABLED</strong></div></div><pre>{JSON.stringify({ run_id: selectedRun.run_id, status: selectedRun.status, parameters: selectedRun.parameters, metrics_summary: selectedRun.metrics_summary }, null, 2)}</pre></section></section></main>;
}

export function App() {
  const { locale, t } = useI18n();
  const [path, setPath] = useState(currentPath);
  const [navOpen, setNavOpen] = useState(false);
  const [navWidth, setNavWidth] = useState(() => {
    const storedWidth = Number(localStorage.getItem(PLATFORM_NAV_WIDTH_STORAGE_KEY));
    return Number.isFinite(storedWidth) && storedWidth > 0
      ? clampPlatformNavWidth(storedWidth)
      : PLATFORM_NAV_DEFAULT_WIDTH;
  });
  const [navResizing, setNavResizing] = useState(false);
  const navResizeRef = useRef<{ pointerId: number; startX: number; startWidth: number } | null>(null);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>(() => {
    const stored = localStorage.getItem("embodied-arm-workspace-mode");
    return stored === "learning" || stored === "experiment" || stored === "analysis"
      ? stored : stored === "engineering" ? "analysis" : "learning";
  });
  const [user, setUser] = useState<CurrentUserContext | null>(null);
  const [activeTour, setActiveTour] = useState<{ definition: TourDefinition; automatic: boolean } | null>(null);
  const autoTourAttempt = useRef("");
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState("");
  useEffect(() => setAuthError(""), [locale]);
  useEffect(() => {
    localStorage.setItem(PLATFORM_NAV_WIDTH_STORAGE_KEY, String(navWidth));
  }, [navWidth]);
  useEffect(() => {
    localStorage.setItem("embodied-arm-workspace-mode", workspaceMode);
  }, [workspaceMode]);

  function startNavResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (window.matchMedia("(max-width: 820px)").matches) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    navResizeRef.current = { pointerId: event.pointerId, startX: event.clientX, startWidth: navWidth };
    setNavResizing(true);
  }

  function resizeNav(event: ReactPointerEvent<HTMLDivElement>) {
    const resize = navResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    setNavWidth(clampPlatformNavWidth(resize.startWidth + event.clientX - resize.startX));
  }

  function stopNavResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (navResizeRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    navResizeRef.current = null;
    setNavResizing(false);
  }

  function resizeNavWithKeyboard(event: ReactKeyboardEvent<HTMLDivElement>) {
    const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (direction) {
      event.preventDefault();
      setNavWidth((width) => clampPlatformNavWidth(width + direction * 16));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setNavWidth(event.key === "Home" ? PLATFORM_NAV_MIN_WIDTH : PLATFORM_NAV_MAX_WIDTH);
    }
  }
  const permissions = new Set(
    user?.organizations.flatMap((organization) => organization.permissions) ?? [],
  );
  const hasPermission = (permission?: string) =>
    !permission || !user?.organizations.length || permissions.has(permission);
  const visibleNavItems = NAV_ITEMS.filter((item) =>
    item.modes.includes(workspaceMode as never) &&
    hasPermission(item.path === "/projects" ? "own_run:create" : undefined),
  );
  const activeWorkspaceMode = WORKSPACE_MODES.find((mode) => mode.id === workspaceMode)!;
  function changeWorkspaceMode(mode: (typeof WORKSPACE_MODES)[number]) {
    setWorkspaceMode(mode.id);
    navigate(mode.landingPath);
    setNavOpen(false);
  }
  const runtimeLabels = runtimeModeLabels(
    path === "/quick-course"
      ? LEGACY_QUICK_COURSE_RUNTIME_CONTEXT
      : CANONICAL_PLATFORM_RUNTIME_CONTEXT,
    locale,
  );
  useEffect(() => {
    const onPopState = () => {
      const redirectedPath = currentPath();
      if (redirectedPath !== location.pathname) {
        history.replaceState({}, "", redirectedPath);
      }
      setPath(redirectedPath);
      setNavOpen(false);
    };
    window.addEventListener("popstate", onPopState);
    const redirectedPath = currentPath();
    if (redirectedPath !== location.pathname) history.replaceState({}, "", redirectedPath);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    const routeMode = workspaceModeForPath(path, workspaceMode);
    if (routeMode !== workspaceMode) setWorkspaceMode(routeMode);
  }, [path, workspaceMode]);

  useEffect(() => {
    setActiveTour((current) => current?.automatic ? current : null);
  }, [path]);

  useEffect(() => {
    if (!user) autoTourAttempt.current = "";
  }, [user]);

  useEffect(() => {
    if (activeTour?.definition.id !== platformFirstTimeTour.id || !window.matchMedia("(max-width: 820px)").matches) return;
    const wasOpen = navOpen;
    setNavOpen(true);
    return () => setNavOpen(wasOpen);
  }, [activeTour?.definition.id]);

  useEffect(() => {
    if (!user || activeTour || autoTourAttempt.current === user.user_id || tourStorage.hasSeenFirstTimeOnboarding(user.user_id)) return;
    let pending: number | null = null;
    const ready = () => {
      if (pending !== null) return;
      pending = window.setTimeout(() => {
        pending = null;
        if (document.querySelector(".platform-content .loading, .platform-content [role='dialog']")) return;
        if (!["brand", "primary-navigation", "secondary-navigation", "system-status", "guide-actions"].every((id) => document.querySelector(`[data-tour="${id}"]`))) return;
        observer.disconnect();
        autoTourAttempt.current = user.user_id;
        setActiveTour({ definition: platformFirstTimeTour, automatic: true });
      }, 0);
    };
    const observer = new MutationObserver(ready);
    observer.observe(document.getElementById("root") ?? document.body, { childList: true, subtree: true });
    ready();
    return () => { observer.disconnect(); if (pending !== null) window.clearTimeout(pending); };
  }, [user, path, activeTour]);

  useEffect(() => {
    if ((import.meta as ImportMeta & { env: { DEV: boolean } }).env.DEV) {
      (window as Window & { resetOnboardingForDebug?: () => void }).resetOnboardingForDebug = () => {
        tourStorage.resetOnboardingForDebug(user?.user_id);
        autoTourAttempt.current = "";
      };
      return () => { delete (window as Window & { resetOnboardingForDebug?: () => void }).resetOnboardingForDebug; };
    }
  }, [user?.user_id]);

  useEffect(() => {
    const expire = () => {
      setAccessToken(null);
      setUser(null);
      setActiveTour(null);
      setAuthError(t("登录状态已失效，请重新登录。", "Your session has expired. Please sign in again."));
    };
    window.addEventListener("studio-auth-expired", expire);
    const token = getAccessToken();
    if (!token) {
      setAuthLoading(false);
      return () => window.removeEventListener("studio-auth-expired", expire);
    }
    api<CurrentUserContext>("/api/v1/me")
      .then((context) => {
        if (!context.enabled) throw new Error("当前用户已被禁用");
        setUser(context);
      })
      .catch((error) => {
        setAccessToken(null);
        setAuthError(error instanceof Error ? error.message : t("身份恢复失败", "Failed to restore the session"));
      })
      .finally(() => setAuthLoading(false));
    return () => window.removeEventListener("studio-auth-expired", expire);
  }, []);

  if (authLoading) return <div className="loading">正在恢复登录状态...</div>;
  if (!user) return <LoginPage initialError={authError} onLoggedIn={(context) => { setAuthError(""); setUser(context); }} />;

  return (
    <LocalizedContent>
    <div
      className={`platform-shell ${navResizing ? "nav-resizing" : ""}`}
      style={{ "--platform-nav-width": `${navWidth}px` } as React.CSSProperties}
    >
      <button
        className="mobile-nav-toggle"
        title="打开导航"
        onClick={() => setNavOpen(true)}
      >
        <Menu size={19} />
      </button>
      <aside className={`platform-nav ${navOpen ? "open" : ""}`}>
        <div className="nav-brand" data-tour="brand">
          <span>EA</span>
          <div>
            <strong>Embodied Arm</strong>
            <small>RESEARCH STUDIO</small>
          </div>
          <button title="关闭导航" onClick={() => setNavOpen(false)}>
            <X size={18} />
          </button>
        </div>
        <LanguageSwitcher compact />
        <nav aria-label="主导航">
          <div className="workspace-mode-switcher" aria-label="工作模式" data-tour="primary-navigation">
            {WORKSPACE_MODES.map((mode) => (
              <button
                key={mode.id}
                className={workspaceMode === mode.id ? "active" : ""}
                aria-pressed={workspaceMode === mode.id}
                onClick={() => changeWorkspaceMode(mode)}
              >
                {mode.shortLabel}
              </button>
            ))}
          </div>
          <div className="nav-mode-context">
            <strong>{activeWorkspaceMode.label}</strong>
            <small>{activeWorkspaceMode.description}</small>
          </div>
          <div className="nav-group" data-tour="secondary-navigation">
            {visibleNavItems.map((item) => (
              <button
                key={item.path}
                className={path === item.path || (item.path === "/learn" && path.startsWith("/learn/")) || (item.path === "/runs" && path.startsWith("/runs/")) ? "active" : ""}
                onClick={() => navigate(item.path)}
              >
                <item.icon size={17} />
                <span>{item.label}</span>
                <small>{NAV_STATUS_LABELS[item.status]}</small>
              </button>
            ))}
          </div>
          <div className="nav-guide-block">
            <span className="nav-guide-label">{t("参考资料", "REFERENCE")}</span>
            <button className={path === "/guide" || path.startsWith("/guide/") ? "active" : ""} aria-current={path === "/guide" || path.startsWith("/guide/") ? "page" : undefined} onClick={() => { navigate("/guide/overview"); setNavOpen(false); }}>
              <BookOpen size={17} />
              <span>{t("完整使用教程", "Complete user guide")}</span>
              <small>{USER_GUIDE_VERSION}</small>
            </button>
          </div>
        </nav>
        <div className="nav-boundary" data-tour="system-status">
          <span className="nav-boundary-label">SYSTEM STATUS</span>
          <span />
          <strong>{runtimeLabels.mode}</strong>
          <small>{runtimeLabels.capability}</small>
          <small>{runtimeLabels.safety}</small>
          <button className="nav-logout" onClick={() => { setAccessToken(null); setUser(null); setActiveTour(null); }}>退出登录</button>
        </div>
      </aside>
      <div
        className="platform-nav-resizer"
        role="separator"
        aria-label={t("调整侧栏宽度", "Resize sidebar")}
        aria-orientation="vertical"
        aria-valuemin={PLATFORM_NAV_MIN_WIDTH}
        aria-valuemax={PLATFORM_NAV_MAX_WIDTH}
        aria-valuenow={navWidth}
        tabIndex={0}
        title={t("拖动以调整侧栏宽度", "Drag to resize the sidebar")}
        onDoubleClick={() => setNavWidth(PLATFORM_NAV_DEFAULT_WIDTH)}
        onKeyDown={resizeNavWithKeyboard}
        onPointerDown={startNavResize}
        onPointerMove={resizeNav}
        onPointerUp={stopNavResize}
        onPointerCancel={stopNavResize}
      />
      <section className="platform-content">
        <div className="page-tour-actions" data-tour="guide-actions" role="group" aria-label={t("引导", "Guides")}>
          <button disabled={!tourRegistry.resolve(path)} title={tourRegistry.resolve(path) ? t("当前页面引导", "Tour this page") : t("当前页面暂未提供引导", "No tour for this page yet")} onClick={() => {
            const definition = tourRegistry.resolve(path);
            if (definition) setActiveTour({ definition, automatic: false });
          }}><Map size={16} /> {t("页面引导", "Page Tour")}</button>
          <button onClick={() => setActiveTour({ definition: platformFirstTimeTour, automatic: false })}><RotateCcw size={16} /> {t("重新查看平台介绍", "Replay Platform Introduction")}</button>
        </div>
        {path === "/guide" || path.startsWith("/guide/") ? (
          <UserGuidePage path={path} />
        ) : path === "/learn" || path.startsWith("/learn/") ? (
          <Suspense fallback={<div className="loading">{t("正在读取入门学习内容…", "Loading learning materials…")}</div>}><LearningPage path={path} navigate={navigate} /></Suspense>
        ) : path === "/roadmap" ? (
          <LearningGuidePage />
        ) : path === "/courses" ? (
          <TeachingFlowPage user={user} />
        ) : path === "/skills" ? (
          <TasksSkillsPage user={user} navigate={navigate} />
        ) : path === "/projects" || path === "/training-code" ? (
          path === "/projects" ? <Suspense fallback={<div className="loading">正在读取实验工作台…</div>}><WorkbenchPage user={user} navigate={navigate} /></Suspense> : <ProjectWorkspacePage key={path} initialView="files" />
        ) : path === "/environments" ? (
          <EnvironmentPage />
        ) : path === "/training" ? (
          <Suspense fallback={<div className="loading">正在读取实验工作台…</div>}><WorkbenchPage user={user} navigate={navigate} /></Suspense>
        ) : path === "/evaluation" ? (
          <CanonicalEvaluationPage />
        ) : path === "/runs" ? (
          <RunsPage />
        ) : path === "/run-comparison" ? (
          <RunComparisonPage />
        ) : path.startsWith("/runs/") ? (
          <RunDetailPage
            key={path}
            runId={decodeURIComponent(path.slice("/runs/".length))}
          />
        ) : path === "/export" ? (
          <RunArtifactsPage />
        ) : path === "/agent" ? (
          <AgentSettingsPage />
        ) : (
          <PlannedPage path={path} />
        )}
      </section>
      <FloatingAgent path={path} />
      {activeTour && <TourEngine key={`${activeTour.definition.id}-${user.user_id}`} tour={activeTour.definition} confirmSkip={activeTour.definition.id === platformFirstTimeTour.id} onComplete={() => {
        if (activeTour.automatic) tourStorage.markFirstTimeOnboardingCompleted(user.user_id);
        setActiveTour(null);
      }} onSkip={() => {
        if (activeTour.automatic) tourStorage.markFirstTimeOnboardingSkipped(user.user_id);
        setActiveTour(null);
      }} onClose={() => setActiveTour(null)} />}
    </div>
    </LocalizedContent>
  );
}

const DEFAULT_CODE = `{
  "action_scale": 0.35,
  "reward_distance_weight": 1.0,
  "learning_rate": 0.0003
}`;

type InitializedProject = { projectId: string; content: string };

let projectInitialization: Promise<InitializedProject> | null = null;

function initializeProject() {
  if (!projectInitialization) {
    projectInitialization = api<{ project_id: string }>("/api/projects", {
      method: "POST",
      body: JSON.stringify({}),
    })
      .then(({ project_id }) =>
        api<{ path: string; content: string }>(
          `/api/projects/${project_id}/files/controller.json`,
        ).then((file) => ({ projectId: project_id, content: file.content })),
      )
      .catch((error) => {
        projectInitialization = null;
        throw error;
      });
  }
  return projectInitialization;
}

function QuickCoursePage({ navigate }: { navigate: (path: string) => void }) {
  const { locale, t } = useI18n();
  const [tab, setTab] = useState<"metrics" | "agent" | "compare">("metrics");
  const course = useQuery({
    queryKey: ["course"],
    queryFn: () => api<Course>("/api/courses/quick-start"),
  });
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<LegacyRunRecord[]>("/api/runs"),
    enabled: tab === "compare",
    refetchInterval: tab === "compare" ? 1500 : false,
  });
  const [code, setCode] = useState(DEFAULT_CODE);
  const [baseCode, setBaseCode] = useState(DEFAULT_CODE);
  const [agentReply, setAgentReply] = useState(
    t("生成 Patch 前需提交学生设计；应用前必须人工审批。", "Submit the student design before generating a Patch; manual approval is required before applying it."),
  );
  const [agentSession, setAgentSession] = useState("");
  const [pendingPatch, setPendingPatch] = useState<{
    patch_id: string;
    risk: string;
    diff: string;
  } | null>(null);
  const [lastSnapshot, setLastSnapshot] = useState("");
  const [busy, setBusy] = useState(false);
  const [runPhase, setRunPhase] = useState<LegacyRunRecord["status"] | "idle">("idle");
  const studio = useStudio();

  useEffect(() => {
    setAgentReply(t("生成 Patch 前需提交学生设计；应用前必须人工审批。", "Submit the student design before generating a Patch; manual approval is required before applying it."));
  }, [locale]);

  useEffect(() => {
    if (studio.projectId) return;
    initializeProject()
      .then(({ projectId, content }) => {
        studio.setProject(projectId);
        setCode(content);
        setBaseCode(content);
      })
      .catch((error) => setAgentReply(error.message));
  }, [studio.projectId]);

  const metricEvents = studio.metrics.filter((event) => event.payload.metrics);
  const latestEvent = metricEvents[metricEvents.length - 1];
  const latestArm = latestEvent?.payload.arm ?? {
    joint_positions: [-0.9, 0.45, -0.55],
    target: [0.72, -0.35, 0.62],
  };
  const activeStep = course.data?.steps[studio.activeStep];

  async function startMockExperiment() {
    setBusy(true);
    setRunPhase("queued");
    setAgentReply(t("正在创建 Mock Run，并准备启动三关节教学仿真器。", "Creating a Mock Run and preparing the three-joint teaching simulator."));
    try {
      const parsed = JSON.parse(code) as Record<string, number>;
      studio.setConfig(parsed);
      const run = await createLegacyTrainingRun({
        project_id: studio.projectId || "quick-start",
        parameters: {
          environment_id: "mock-reach",
          strategy_id: studio.selectedStrategy,
          seed: 42,
          max_steps: 80,
        },
        project_config_snapshot: parsed,
        legacy: {
          environment_id: "mock-reach",
          strategy_id: studio.selectedStrategy,
          seed: 42,
          max_steps: 80,
          config: parsed,
        },
      });
      studio.beginRun(run);
      setRunPhase("running");
      setAgentReply(t("仿真器已启动：正在执行关节控制、采集轨迹并计算 reward。", "The simulator has started: executing joint control, collecting trajectories, and computing rewards."));

      // Legacy teaching Runs expose HTTP events. The canonical WebSocket is
      // reserved for persisted /api/v1 Runs and cannot stream this demo.
      let afterEventId = "";
      for (let attempt = 0; attempt < 240; attempt += 1) {
        const [current, events] = await Promise.all([
          api<LegacyRunRecord>(`/api/runs/${run.run_id}`),
          api<MetricEvent[]>(
            `/api/runs/${run.run_id}/events?after_event_id=${encodeURIComponent(afterEventId)}`,
          ),
        ]);
        events.forEach((runEvent) => {
          studio.appendEvent(runEvent);
          afterEventId = runEvent.event_id;
        });
        studio.setRunStatus(current.status, current.error);
        setRunPhase(current.status);
        if (["completed", "stopped", "failed"].includes(current.status)) {
          setBusy(false);
          setAgentReply(
            current.status === "completed"
              ? t("训练演示完成：可以查看 reward、success rate、运行对比并导出演示清单。", "Training demo complete: review reward, success rate, run comparisons, and export the demo manifest.")
              : current.error ?? (current.status === "stopped"
                ? t("演示已停止；当前 Run 的事件和指标已保留用于复盘。", "The demo was stopped; this Run's events and metrics have been retained for review.")
                : t("演示失败。", "The demo failed.")),
          );
          await runs.refetch();
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
      throw new Error(t("演示运行超时，请到运行分析查看该 Run 的状态。", "The demo run timed out. Check this Run's status in Run Analysis."));
    } catch (error) {
      setBusy(false);
      setRunPhase("failed");
      setAgentReply(error instanceof Error ? error.message : t("无法启动 Mock 演示", "Unable to start the Mock demo"));
    }
  }

  async function stopMockExperiment() {
    const currentRun = useStudio.getState().currentRun;
    if (!currentRun) return;
    const stopped = await apiWithFallback<LegacyRunRecord>(
      `/api/v1/runs/${currentRun.run_id}/stop`,
      `/api/runs/${currentRun.run_id}/stop`,
      { method: "POST" },
    );
    studio.setRunStatus(stopped.status, stopped.error);
    setRunPhase(stopped.status);
    setBusy(false);
    setAgentReply(t("演示已停止；当前 Run 的事件和指标已保留用于复盘。", "The demo was stopped; this Run's events and metrics have been retained for review."));
  }

  async function completeStep() {
    if (!activeStep) return;
    studio.completeStep(activeStep.id);
    if (studio.activeStep < (course.data?.steps.length ?? 1) - 1)
      studio.setStep(studio.activeStep + 1);
  }

  async function proposeConfigPatch() {
    if (!studio.projectId) return;
    let sessionId = agentSession;
    if (!sessionId) {
      const session = await api<{ session_id: string }>("/api/agent/sessions", {
        method: "POST",
        body: JSON.stringify({
          project_id: studio.projectId,
          provider: "mock",
          assistance_level: 5,
        }),
      });
      sessionId = session.session_id;
      setAgentSession(sessionId);
    }
    await api(`/api/agent/sessions/${sessionId}/student-design`, {
      method: "PUT",
      body: JSON.stringify({
        observation: "读取三个关节位置与末端到目标的距离。",
        action: `学生配置的关节动作缩放为 ${studio.config.action_scale ?? 0.35}。`,
        reward: "使用目标距离下降作为主要奖励证据。",
        termination: "到达目标或达到课程最大步数时终止。",
        expected_result: "使用相同 seed 比较修改前后的 distance 曲线。",
      }),
    });
    const oldLines = baseCode.trim().split("\n");
    const newLines = code.trim().split("\n");
    const diff = [
      "--- a/controller.json",
      "+++ b/controller.json",
      `@@ -1,${oldLines.length} +1,${newLines.length} @@`,
      ...oldLines.map((line) => `-${line}`),
      ...newLines.map((line) => `+${line}`),
    ].join("\n");
    const patch = await api<{ patch_id: string; risk: string; diff: string }>(
      "/api/agent/patches",
      {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          project_id: studio.projectId,
          path: "controller.json",
          diff,
        }),
      },
    );
    setPendingPatch(patch);
    setTab("agent");
  }

  async function approvePatch() {
    if (!pendingPatch) return;
    const applied = await api<{ snapshot_id: string }>(
      `/api/agent/patches/${pendingPatch.patch_id}/approve`,
      {
        method: "POST",
        body: JSON.stringify({ confirm_high_risk: true }),
      },
    );
    setLastSnapshot(applied.snapshot_id);
    setBaseCode(code.trim());
    setPendingPatch(null);
    setAgentReply(t("Patch 已审批应用，并已创建可回滚快照。", "The Patch was approved and applied, and a rollback snapshot was created."));
  }

  async function rollback() {
    if (!lastSnapshot) return;
    await api(`/api/projects/${studio.projectId}/rollback`, {
      method: "POST",
      body: JSON.stringify({ snapshot_id: lastSnapshot }),
    });
    const file = await api<{ content: string }>(
      `/api/projects/${studio.projectId}/files/controller.json`,
    );
    setCode(file.content);
    setBaseCode(file.content);
    setAgentReply(t("已回滚到审批前快照。", "Rolled back to the snapshot created before approval."));
  }

  async function exportMockManifest() {
    if (!studio.currentRun || studio.currentRun.status !== "completed") return;
    const manifest = await apiWithFallback<Record<string, unknown>>(
      "/api/v1/runs/export",
      `/api/runs/${studio.currentRun.run_id}/export`,
      {
        method: "POST",
        body: JSON.stringify({
          run_ids: [studio.currentRun.run_id],
          format: "json",
        }),
      },
    );
    const blob = new Blob([JSON.stringify(manifest, null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = "mock-experiment-manifest.json";
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  const chartOption = {
    animation: false,
    grid: { left: 48, right: 24, top: 28, bottom: 35 },
    tooltip: { trigger: "axis" },
    legend: {
      data: ["Reward", "Success"],
      right: 10,
      textStyle: { color: "#5f6966" },
    },
    xAxis: {
      type: "category",
      data: metricEvents.map((event) => event.payload.step),
      axisLabel: { color: "#73807b" },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#73807b" },
      splitLine: { lineStyle: { color: "#dde1dc" } },
    },
    series: [
      {
        name: "Reward",
        type: "line",
        symbol: "none",
        data: metricEvents.map((event) => event.payload.metrics?.mean_reward),
        lineStyle: { color: "#e05a3f", width: 2 },
      },
      {
        name: "Success",
        type: "line",
        symbol: "none",
        data: metricEvents.map((event) => event.payload.metrics?.success_rate),
        lineStyle: { color: "#278c78", width: 2 },
      },
    ],
  };

  if (course.isLoading)
    return <div className="loading">{t("正在加载课程工作台...", "Loading the course workspace...")}</div>;
  if (course.isError || !course.data)
    return (
      <div className="loading error">{t("课程加载失败，请确认 API 已启动。", "The course could not be loaded. Confirm that the API is running.")}</div>
    );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">EA</span>
          <div>
            <strong>Embodied Arm Studio</strong>
            <small>{t("具身机械臂教学平台", "Embodied Robot Arm Teaching Platform")}</small>
          </div>
        </div>
        <div className="mode">
          <span /> MOCK MODE
        </div>
        <div className="run-actions">
            <button className="secondary" onClick={() => navigate("/pipelines")}>
            <ListTree size={16} /> {t("S1 实验流程", "S1 Experiment Flow")}
          </button>
          <select
            aria-label={t("策略", "Policy")}
            value={studio.selectedStrategy}
            onChange={(event) => studio.setStrategy(event.target.value)}
          >
            <option value="joint-p-control">{t("关节 P 控制", "Joint P Control")}</option>
            <option value="policy-gradient-demo">{t("策略梯度入门", "Policy Gradient Introduction")}</option>
          </select>
          <button
            className="icon-button"
            title="回滚"
            onClick={rollback}
            disabled={!lastSnapshot}
          >
            <RotateCcw size={17} />
          </button>
          <button className="secondary" onClick={stopMockExperiment} disabled={!busy}>
            <CircleStop size={16} /> {t("停止", "Stop")}
          </button>
          <button className="primary" onClick={startMockExperiment} disabled={busy}>
            <Play size={16} fill="currentColor" /> {busy ? t("演示运行中", "Demo running") : t("运行演示", "Run demo")}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="lesson-panel">
          <div className="lesson-heading">
            <span>QUICK COURSE · {course.data.estimated_minutes} MIN</span>
            <h1>{course.data.title}</h1>
            <p>{course.data.summary}</p>
          </div>
          <nav className="steps" aria-label={t("课程步骤", "Course steps")}>
            {course.data.steps.map((step, index) => (
              <button
                key={step.id}
                className={studio.activeStep === index ? "active" : ""}
                onClick={() => studio.setStep(index)}
              >
                <span className="step-number">
                  {studio.completedSteps.includes(step.id) ? (
                    <Check size={15} />
                  ) : (
                    index + 1
                  )}
                </span>
                <span>
                  <strong>{step.title}</strong>
                  <small>{step.objective}</small>
                </span>
                <ChevronRight size={16} />
              </button>
            ))}
          </nav>
          {activeStep && (
            <article className="lesson-content">
              <h2>{activeStep.objective}</h2>
              <p>{activeStep.theory}</p>
              <div className="prediction">
                <FlaskConical size={17} />
                <span>
                  <strong>{t("运行前预测", "Pre-run prediction")}</strong>
                  {activeStep.prediction}
                </span>
              </div>
              <div className="success-rule">
                <Check size={16} />
                <span>{activeStep.success}</span>
              </div>
              <button className="complete" onClick={completeStep}>
                {t("完成此检查点", "Complete this checkpoint")}
              </button>
            </article>
          )}
        </aside>

        <section className="main-stage">
          <div className="sim-header">
            <div>
              <span>SIMULATION</span>
              <strong>{t("三关节 Mock 教学臂", "Three-joint Mock teaching arm")}</strong>
            </div>
            <div className="telemetry">
              <span>
                STEP <b>{latestEvent?.payload.step ?? 0}</b>
              </span>
              <span>
                DIST{" "}
                <b>
                  {latestEvent?.payload.metrics?.distance?.toFixed(3) ?? "--"}
                </b>
              </span>
              <span>
                FPS <b>20</b>
              </span>
            </div>
          </div>
          <div className="quick-run-flow" aria-live="polite" aria-label={t("Mock 训练流程", "Mock training flow")}>
            {([
              [t("创建 Run", "Create Run"), runPhase !== "idle"],
              [t("关节控制", "Joint control"), runPhase === "running" || runPhase === "completed"],
              [t("采集指标", "Collect metrics"), metricEvents.length > 0],
              [t("完成评估", "Complete evaluation"), runPhase === "completed"],
              [t("导出清单", "Export manifest"), runPhase === "completed"],
            ] as [string, boolean][]).map(([label, complete], index) => (
              <span key={label} className={complete ? "complete" : index === 0 && runPhase !== "idle" ? "active" : ""}>
                <b>{index + 1}</b>{label}
              </span>
            ))}
            <strong className={`run-phase ${runPhase}`}>
              {runPhase === "idle" ? t("等待启动", "Waiting to start") : runPhase === "queued" ? t("已排队", "Queued") : runPhase === "running" ? t("运行中", "Running") : runPhase === "completed" ? t("已完成", "Completed") : runPhase === "stopped" ? t("已停止", "Stopped") : t("失败", "Failed")}
            </strong>
          </div>
          <ArmViewport
            joints={latestArm.joint_positions}
            target={latestArm.target}
            status={studio.currentRun?.status ?? "ready"}
          />
          <div className="bottom-panel">
            <div className="tabs">
              <button
                className={tab === "metrics" ? "active" : ""}
                onClick={() => setTab("metrics")}
              >
                <FlaskConical size={15} /> {t("指标", "Metrics")}
              </button>
              <button
                className={tab === "compare" ? "active" : ""}
                onClick={() => setTab("compare")}
              >
                <Settings2 size={15} /> 运行对比
              </button>
              <button
                className={tab === "agent" ? "active" : ""}
                onClick={() => setTab("agent")}
              >
                <ShieldCheck size={15} /> {t("Patch 审批", "Patch approval")}
              </button>
              <button
                className="export-tab"
                onClick={exportMockManifest}
                disabled={studio.currentRun?.status !== "completed"}
              >
                <Download size={15} /> {t("导出演示清单", "Export demo manifest")}
              </button>
            </div>
            {tab === "metrics" && (
              <div className="chart-wrap">
                {metricEvents.length ? (
                  <ReactECharts option={chartOption} style={{ height: 230 }} />
                ) : (
                  <div className="empty-chart">
                    <FlaskConical size={28} />
                    <span>{t("运行 deterministic Mock 后，reward 与 success rate 会实时出现", "Reward and success rate will appear in real time after running the deterministic Mock.")}</span>
                  </div>
                )}
              </div>
            )}
            {tab === "compare" && (
              <div className="run-table">
                <div className="table-head">
                  <span>RUN</span>
                  <span>策略</span>
                  <span>SEED</span>
                  <span>STATUS</span>
                </div>
                {(runs.data ?? []).slice(0, 4).map((run) => (
                  <div className="table-row" key={run.run_id}>
                    <code>{run.run_id.slice(-6)}</code>
                    <span>{run.strategy_id}</span>
                    <span>{run.seed}</span>
                    <span className={`run-state ${run.status}`}>
                      {run.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {tab === "agent" && (
              <div className="agent-panel">
                <div className="agent-response">
                  <div className="agent-avatar">
                    <ShieldCheck size={18} />
                  </div>
                  <p>{agentReply}</p>
                </div>
                {pendingPatch && (
                  <div className="diff-box">
                    <div>
                      <strong>{t("待审批 Patch", "Patch awaiting approval")}</strong>
                      <span className="risk">{pendingPatch.risk} risk</span>
                    </div>
                    <pre>{pendingPatch.diff}</pre>
                    <div className="diff-actions">
                      <button onClick={() => setPendingPatch(null)}>
                        {t("拒绝", "Reject")}
                      </button>
                      <button className="primary" onClick={approvePatch}>
                        {t("确认并应用", "Confirm and apply")}
                      </button>
                    </div>
                  </div>
                )}
                <button
                  className="secondary"
                  onClick={() => window.dispatchEvent(new Event("studio-agent-open"))}
                >
                  <Bot size={15} /> {t("打开全局教学 Agent", "Open global Teaching Agent")}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside className="config-panel">
          <div className="panel-title">
            <span>
              <Settings2 size={17} /> {t("Mock 参数", "Mock parameters")}
            </span>
            <small>
              {studio.projectId ? studio.projectId.slice(-8) : "initializing"}
            </small>
          </div>
          <div className="field">
            <label htmlFor="action-scale">
              {t("动作缩放", "Action scale")} <span>{studio.config.action_scale?.toFixed(2)}</span>
            </label>
            <input
              id="action-scale"
              type="range"
              min="0.05"
              max="1"
              step="0.05"
              value={studio.config.action_scale}
              onChange={(event) => {
                const next = {
                  ...studio.config,
                  action_scale: Number(event.target.value),
                };
                studio.setConfig(next);
                setCode(JSON.stringify(next, null, 2));
              }}
            />
            <p>{t("控制每步关节位移。过高会增加震荡风险。", "Controls the joint displacement per step. Excessive values may increase oscillation risk.")}</p>
          </div>
          <div className="field">
            <label htmlFor="seed">{t("随机种子", "Random seed")}</label>
            <input id="seed" type="number" value="42" readOnly />
            <p>{t("对比实验时保持一致，才能区分策略与随机性。", "Keep this consistent when comparing experiments so policy changes can be distinguished from randomness.")}</p>
          </div>
          <div className="editor-head">
            <span>controller.json</span>
            <button onClick={proposeConfigPatch}>{t("生成 Patch", "Generate Patch")}</button>
          </div>
          <div className="editor-shell">
            <Editor
              height="285px"
              language="json"
              theme="vs-light"
              value={code}
              onChange={(value) => setCode(value ?? "")}
              options={{
                minimap: { enabled: false },
                fontSize: 13,
                lineNumbersMinChars: 3,
                scrollBeyondLastLine: false,
                padding: { top: 12 },
              }}
            />
          </div>
          <div className="safety-note">
            <span>{t("安全边界", "Safety boundary")}</span>
            <p>
              {t("仅 Mock 仿真。无宿主机 Shell，无真实机械臂写控制。所有 Agent 修改需审批并可回滚。", "Mock simulation only. No host Shell and no real robot write control. All Agent changes require approval and can be rolled back.")}
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
