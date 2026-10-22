import { simulationTaskTitle } from "./simulationTaskPresentation";
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, BookOpen, ChevronRight, CircleAlert, Film, Play, RotateCcw, ShieldCheck } from "lucide-react";
import { api, getAccessToken, type CurrentUserContext, type RunRecord, type TeachingClass, type TeachingCourse, type TeachingProject } from "./api";
import { createCanonicalRunSubmission } from "./canonicalRunRequest";
import { fetchSimulationCatalog, taskParameterPayload, type SimulationTaskDefinition } from "./simulationCatalog";
import { SimulationTaskConfigurator, type SimulationTaskConfigurationState } from "./SimulationTaskConfigurator";
import { useI18n } from "./i18n";
import { teachingProjectLabel } from "./projectPresentation";

type View = "training" | "demo";
interface Props { user: CurrentUserContext; navigate: (path: string) => void }

const TASK_DETAILS = [
  { id: "s1_push", title: ["推动", "Push"], input: ["可推动物体和目标位于任务工作区，使用匹配的 Push Project 和场景版本。", "Movable object and goal in the task workspace, with a matching Push Project and scene revision."], output: ["物体向目标移动；核对成功率、最终目标距离和物体位移，而不是只看末端动作。", "Object moves toward the goal; inspect success rate, final goal distance and object displacement, not just end-effector motion."] },
  { id: "s1_reach", title: ["平面到达", "Planar reach"], input: ["目标区域位于允许的 XY 工作区，观测和动作契约匹配 Reach Project。", "Target region in the allowed XY workspace, with observation and action contracts matching the Reach Project."], output: ["末端进入目标区域并保持；核对目标距离、成功条件和超时，不要求抓取物体。", "End effector enters and holds the target region; inspect distance, success conditions and timeout. No object grasping is required."] },
  { id: "s1_grasp", title: ["抓取", "Grasp"], input: ["末端接近物体；物体位置、夹爪和观测契约匹配。", "End effector near the object; compatible object, gripper and observation contracts."], output: ["对齐、下降、闭合与抓取验证。候选抓取不等于验证成功。", "Alignment, descent, closure and grasp verification. A candidate is not a verified grasp."] },
  { id: "s1_lift", title: ["抬升与保持", "Lift and hold"], input: ["稳定抓持的初始状态；不能假定 Lift 自带完整抓取。", "A stably grasped initial state; Lift must not be assumed to include grasping."], output: ["目标高度、保持时间、物体漂移和抓持稳定性。", "Target height, hold duration, object drift and grasp stability."] },
  { id: "s1_lower_release", title: ["下降与释放", "Lower and release"], input: ["物体已被抓持并位于目标上方；不是从自由物体开始。", "Object held above the target; this does not start with a free object."], output: ["下降高度、脱离状态、目标误差与放置稳定性。", "Descent height, detachment, target error and placement stability."] },
  { id: "s1_place", title: ["搬运与放置", "Transport and place"], input: ["从平台验证过的抓取抬起状态开始，目标由场景候选检查确定。", "Starts from a platform-validated grasped-and-lifted state; the target comes from the scene candidate check."], output: ["方块进入 15 mm 目标区域并保持 1 秒，同时满足抓持、姿态、力和工作空间安全条件。", "The cube enters the 15 mm target zone and holds for one second while grasp, pose, force and workspace safety remain valid."] },
] as const;

export function TasksSkillsPage({ user, navigate }: Props) {
  const { locale, t } = useI18n();
  const textIndex = locale === "en" ? 1 : 0;
  const [view, setView] = useState<View>("training");
  const [demoId, setDemoId] = useState<string>("demo_s1_circle_tracking");
  const [configuration, setConfiguration] = useState<SimulationTaskConfigurationState | null>(null);
  const [organizationId, setOrganizationId] = useState(user.organizations[0]?.organization_id ?? "");
  const [courseId, setCourseId] = useState("");
  const [classId, setClassId] = useState("");
  const [projectId, setProjectId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const submission = useRef(createCanonicalRunSubmission());
  const catalog = useQuery({
    queryKey: ["simulation-task-catalog"],
    queryFn: () => fetchSimulationCatalog((input, init) => {
      const headers = new Headers(init?.headers);
      const token = getAccessToken();
      if (token) headers.set("Authorization", `Bearer ${token}`);
      return fetch(input, { ...init, headers });
    }),
  });
  const courses = useQuery({ queryKey: ["skills-courses", organizationId], queryFn: () => api<TeachingCourse[]>(`/api/v1/courses?organization_id=${encodeURIComponent(organizationId)}`), enabled: Boolean(organizationId) });
  const classes = useQuery({ queryKey: ["skills-classes", courseId], queryFn: () => api<TeachingClass[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/classes`), enabled: Boolean(courseId) });
  const projects = useQuery({ queryKey: ["skills-projects", courseId], queryFn: () => api<TeachingProject[]>(`/api/v1/courses/${encodeURIComponent(courseId)}/projects`), enabled: Boolean(courseId) });
  const runs = useQuery({ queryKey: ["skills-runs"], queryFn: () => api<RunRecord[]>("/api/v1/runs"), enabled: view === "training" });
  const tasks = catalog.data?.tasks ?? [];
  const demoTasks = tasks.filter((task) => task.mode === "demo");
  const demoIds = demoTasks.map((task) => task.id);
  const serverReady = catalog.data?.source === "server" && !catalog.isError;
  const availableProjects = (projects.data ?? []).filter((project) => project.enabled && project.permissions.can_use && project.training_profile === "short" && project.environment_type === "s1_push");
  const project = availableProjects.find((item) => item.project_id === projectId);
  const selectedClass = (classes.data ?? []).find((item) => item.class_id === classId && item.enabled);
  const demo = tasks.find((item) => item.id === demoId && item.mode === "demo");
  const permitted = user.organizations.find((item) => item.organization_id === organizationId)?.permissions.includes("own_run:create") === true;
  const canStart = serverReady && Boolean(demo && project && selectedClass && permitted && configuration?.validation?.valid && configuration.task?.id === demoId && configuration.task.mode === "demo");
  const taskStatus = (task?: SimulationTaskDefinition) => !serverReady ? t("待服务端确认", "Awaiting server confirmation") : task ? t("已注册", "Registered") : t("未注册", "Not registered");
  const unregisteredTasks = serverReady ? TASK_DETAILS.filter((item) => !tasks.some((task) => task.id === item.id) && item.id !== "s1_push" && item.id !== "s1_reach") : [];

  useEffect(() => {
    if (busy || !catalog.data) return;
    const available = catalog.data.tasks.filter((task) => task.mode === "demo");
    if (!available.some((task) => task.id === demoId)) {
      setConfiguration(null);
      setDemoId(available[0]?.id ?? "");
    }
  }, [catalog.data, demoId, busy]);

  function trainingProjects(taskId: string) {
    return (projects.data ?? []).filter((item) => item.task_type === "training" && item.environment_type === taskId && item.enabled && item.permissions.can_use);
  }

  function trainingBlocker(task: SimulationTaskDefinition): string {
    if (!serverReady) return t("需要在线服务目录确认。", "Online service catalog required.");
    if (!task.supportsFormalRun || !task.producesCheckpoint) return t("目录未声明完整训练支持。", "Complete training support not declared.");
    if (!organizationId || !courseId) return t("先选择组织和课程，检查可用教学 Project。", "Select organization and course to check teaching Projects.");
    if (courses.isError || projects.isError || classes.isError) return t("教学上下文读取失败。", "Teaching context failed to load.");
    if (!courses.data?.some((item) => item.course_id === courseId && item.enabled)) return t("所选课程尚未就绪或已停用。", "Selected course is not ready or disabled.");
    if (projects.isLoading || classes.isLoading) return t("正在检查 Project 和教学班…", "Checking Projects and classes…");
    if (!trainingProjects(task.id).length) return t("此课程没有匹配且可使用的训练 Project，需要教师配置或授权。", "No matching permitted training Project in this course; instructor configuration or authorization required.");
    if (!selectedClass) return t("选择可用教学班后进入训练配置。", "Select an available class before training configuration.");
    if (!permitted) return t("当前组织缺少创建 Run 权限。", "Run creation permission missing in this organization.");
    return "";
  }

  function openTrainingProject(item: TeachingProject) {
    const task = tasks.find((entry) => entry.id === item.environment_type && entry.mode === "training");
    if (!task || trainingBlocker(task)) return;
    sessionStorage.setItem("s1-draft-organization", organizationId);
    sessionStorage.setItem("s1-draft-course", courseId);
    sessionStorage.setItem("s1-draft-class", classId);
    sessionStorage.setItem("s1-draft-project", item.project_id);
    sessionStorage.removeItem("s1-draft-simulation-task-v1");
    navigate(item.environment_type === "s1_push" || item.environment_type === "s1_reach" ? "/projects" : "/courses");
  }

  function renderTeachingContext() {
    return <div className="skills-training-context">
      <label>{t("组织", "Organization")}<select value={organizationId} onChange={(event) => { setOrganizationId(event.target.value); setCourseId(""); setClassId(""); setProjectId(""); }}><option value="">{t("选择组织", "Select organization")}</option>{user.organizations.map((item) => <option key={item.organization_id} value={item.organization_id}>{item.name}</option>)}</select></label>
      <label>{t("课程", "Course")}<select value={courseId} disabled={!organizationId || courses.isLoading} onChange={(event) => { setCourseId(event.target.value); setClassId(""); setProjectId(""); }}><option value="">{t("选择课程", "Select course")}</option>{(courses.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.course_id} value={item.course_id}>{item.title}</option>)}</select></label>
      <label>{t("教学班", "Class")}<select value={classId} disabled={!courseId || classes.isLoading} onChange={(event) => setClassId(event.target.value)}><option value="">{t("选择教学班", "Select class")}</option>{(classes.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.class_id} value={item.class_id}>{item.name}</option>)}</select></label>
    </div>;
  }

  function renderTaskDetails(taskId: string) {
    const detail = TASK_DETAILS.find((item) => item.id === taskId);
    const matchingRuns = (runs.data ?? []).filter((run) => run.run_kind === "train" && run.project_config_snapshot.environment_type === taskId);
    return <details className="skill-task-details">
      <summary>{t("任务说明与运行证据", "Task details and Run evidence")}</summary>
      <dl>
        <div><dt>{t("起始条件", "Entry conditions")}</dt><dd>{detail?.input[textIndex] ?? t("以当前教学 Project 的任务说明和冻结配置为准。", "Use the current teaching Project instructions and frozen configuration.")}</dd></div>
        <div><dt>{t("学习目标与成功证据", "Learning objective and success evidence")}</dt><dd>{detail?.output[textIndex] ?? t("按该任务的服务端成功条件和评估 rubric 检查，不从 Run completed 推断成功。", "Check server success conditions and evaluation rubric; do not infer success from Run completed.")}</dd></div>
        <div><dt>{t("参数与输出", "Parameters and outputs")}</dt><dd>{t("在左侧教学 Project 中选择匹配任务，参数沙箱随 Project 切换；Run 详情提供实际指标、Checkpoint 和可用的视频制品。", "Choose a matching task through the teaching Project; sandbox parameters follow the Project. Run details provide actual metrics, checkpoints and available video artifacts.")}</dd></div>
        {taskId === "s1_place" && <div><dt>{t("正式任务边界", "Formal task boundaries")}</dt><dd>{t("正式 Place 在实验工作台提供受控奖励修改、参数与运行入口，从验证过的抓取抬起状态进行 XY 搬运并保持一秒。完整的抓取、下降、释放与撤离仍是另一套实验流程；任务接入不等于所有策略都通过 GPU 成功率认证。", "Formal Place in the workbench supports controlled reward edits, parameters and Runs. It transports in XY from a validated lifted grasp and holds for one second. The full grasp, lower, release and retreat sequence is a separate experimental workflow. Integration does not certify every policy's GPU success rate.")}</dd></div>}
      </dl>
      <div className="skills-evidence-links">{matchingRuns.slice(0, 3).map((run) => <button className="secondary" key={run.run_id} onClick={() => navigate(`/runs/${encodeURIComponent(run.run_id)}`)}><Film size={14} /> {run.run_id.slice(-8)} · {run.status}</button>)}</div>
      {!matchingRuns.length && <p>{runs.isLoading ? t("读取已授权 Run 证据…", "Loading authorized Run evidence…") : runs.isError ? t("Run 证据读取失败。", "Failed to load Run evidence.") : t("当前账号暂无关联训练 Run。", "No associated training Run for this account.")}</p>}
    </details>;
  }

  async function startDemo() {
    if (!canStart || busy || !project || !selectedClass || !configuration?.task || submission.current.isPending()) return;
    setBusy(true);
    setError("");
    try {
      const run = await submission.current.submit({ project_id: project.project_id, class_id: selectedClass.class_id, project_config_version: project.config_version, run_kind: "play", environment_type: demoId, parameters: taskParameterPayload(configuration.task, configuration.parameters), code: "" });
      navigate(`/runs/${encodeURIComponent(run.run_id)}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("Demo 启动失败", "Demo launch failed"));
    } finally { setBusy(false); }
  }

  return <main className="skills-page">
    <header className="skills-header"><div><span className="eyebrow">{t("S1 · 任务与技能", "S1 · TASKS & SKILLS")}</span><h1>{t("任务与技能", "Tasks and skills")}</h1><p>{t("选择强化学习任务，或配置并运行固定控制演示。", "Choose a reinforcement learning task, or configure and run a fixed control demonstration.")}</p></div><span className="skills-safety"><ShieldCheck size={18} /> {t("仅限仿真", "SIMULATION ONLY")}</span></header>
    <div className="skills-tabs" data-tour="skills-categories" role="tablist" aria-label={t("任务类型", "Task categories")}>
      {([["training", "强化学习任务", "RL tasks"], ["demo", "控制演示", "Control demos"]] as const).map(([id, zh, en]) => <button key={id} id={`skills-tab-${id}`} role="tab" aria-selected={view === id} aria-controls="skills-panel" onClick={() => { setView(id); setError(""); }}>{t(zh, en)}</button>)}
    </div>
    <div className="skills-source" role="status">{catalog.isLoading ? t("读取任务目录…", "Reading catalog…") : catalog.isError ? t("服务端目录不可用，不能启动任务。", "Server catalog unavailable; launch is blocked.") : serverReady ? t("SERVER CATALOG · 能力以当前后端响应为准", "SERVER CATALOG · Current backend response is authoritative") : t("LOCAL FALLBACK · 仅供浏览，不能提交任务", "LOCAL FALLBACK · Browse only; submission is blocked")}{catalog.isError && <button className="secondary" onClick={() => void catalog.refetch()}><RotateCcw size={14} /> {t("重试", "Retry")}</button>}</div>
    <div className="skills-note" data-tour="skills-catalog">{t("服务目录确认能力，Run 详情确认实际直播、录像和证据；真实控制保持禁用。", "The service catalog confirms capability; Run details confirm streams, recordings and evidence. Real control remains disabled.")}</div>
    <section id="skills-panel" role="tabpanel" aria-labelledby={`skills-tab-${view}`}>
      {view === "training" && <div className="skills-list" data-tour="skills-training">
        {renderTeachingContext()}
        {tasks.filter((item) => item.mode === "training").map((task) => <article className="skill-row" key={task.id}><Activity size={20} /><div><h2>{simulationTaskTitle(task, locale)}</h2><code>{task.id}</code><p>{task.description || TASK_DETAILS.find((item) => item.id === task.id)?.output[textIndex] || t("策略通过观测、动作和奖励更新；正式配置由 Project 冻结。", "The policy learns from observations, actions and rewards; Projects freeze formal configuration.")}</p>{renderTaskDetails(task.id)}</div><div className="skill-meta"><span>{taskStatus(task)}</span><small>{task.supportsVideo ? t("声明视频支持", "Video declared") : t("未声明视频", "No video declared")}</small><small>{task.producesCheckpoint ? "CHECKPOINT" : t("无权重声明", "No checkpoint declared")}</small>{trainingBlocker(task) && <p role="status">{trainingBlocker(task)}</p>}{trainingProjects(task.id).map((item) => <button key={item.project_id} className="secondary" disabled={Boolean(trainingBlocker(task))} onClick={() => openTrainingProject(item)}><Play size={14} /> {teachingProjectLabel(item, locale)} · V{item.config_version}</button>)}</div></article>)}
        {!catalog.isLoading && !tasks.some((task) => task.mode === "training") && <p>{t("当前目录没有训练任务。", "No training task in the current catalog.")}</p>}
        <p className="skills-note">{t("训练必须匹配 Project、权限和任务类型。视频声明不保证每次直播或录像都成功；可在 Run 详情中检查实际结果。", "Training requires a matching Project, permissions and task type. Video support does not guarantee a stream or recording; check actual results in Run details.")}</p>
        {unregisteredTasks.length > 0 && <details className="skills-experimental-reference"><summary>{t("未开放的实验任务", "Experimental tasks not yet available")} ({unregisteredTasks.length})</summary><p className="skills-note">{t("这些条目仅说明已有实验方向，当前服务目录未注册，不提供训练入口。", "These entries describe experimental directions only. They are not registered in the current service catalog and offer no training action.")}</p>{unregisteredTasks.map((item) => <article className="skill-row" key={item.id}><Activity size={20} /><div><h2>{item.title[textIndex]}</h2><code>{item.id}</code><p>{item.output[textIndex]}</p>{renderTaskDetails(item.id)}</div><div className="skill-meta"><span>{t("未开放 · 实验任务", "Unavailable · experimental")}</span></div></article>)}</details>}
      </div>}
      {view === "demo" && <div className="skills-demo-layout">
        <aside className="skills-demo-context"><h2>{t("演示上下文", "Demo context")}</h2>
          <label>{t("组织", "Organization")}<select value={organizationId} disabled={busy} onChange={(event) => { setOrganizationId(event.target.value); setCourseId(""); setClassId(""); setProjectId(""); }}><option value="">{t("选择组织", "Select organization")}</option>{user.organizations.map((item) => <option key={item.organization_id} value={item.organization_id}>{item.name}</option>)}</select></label>
          <label>{t("课程", "Course")}<select value={courseId} disabled={busy || courses.isLoading || !organizationId} onChange={(event) => { setCourseId(event.target.value); setClassId(""); setProjectId(""); }}><option value="">{t("选择课程", "Select course")}</option>{(courses.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.course_id} value={item.course_id}>{item.title}</option>)}</select></label>
          <label>{t("教学班", "Class")}<select value={classId} disabled={busy || classes.isLoading || !courseId} onChange={(event) => setClassId(event.target.value)}><option value="">{t("选择教学班", "Select class")}</option>{(classes.data ?? []).filter((item) => item.enabled).map((item) => <option key={item.class_id} value={item.class_id}>{item.name}</option>)}</select></label>
          <label>Short Push Project<select value={projectId} disabled={busy || projects.isLoading || !courseId} onChange={(event) => setProjectId(event.target.value)}><option value="">{t("选择可用 Project", "Select an available Project")}</option>{availableProjects.map((item) => <option key={item.project_id} value={item.project_id}>{teachingProjectLabel(item, locale)} · V{item.config_version}</option>)}</select></label>
          {courseId && !projects.isLoading && !availableProjects.length && <p>{t("此课程暂无可用 Short Push Project。", "This course has no available Short Push Project.")}</p>}
          {(courses.isError || classes.isError || projects.isError) && <p role="alert">{t("教学上下文读取失败，请重新选择或刷新。", "Teaching context failed to load. Reselect or refresh.")}</p>}
          {!permitted && <p>{t("当前组织未授予创建 Run 权限。", "Run creation permission is not granted in this organization.")}</p>}
          <p className="skills-note">{t("固定控制器，不进行 PPO 学习。Demo 不提交 scene_ref，不生成训练权重。", "Fixed controller, no PPO learning. Demos submit no scene_ref and produce no trained weights.")}</p>
        </aside>
        <div className="skills-demo-editor" data-tour="skills-controls">
          <div className="skills-demo-selector">{demoTasks.map((task) => <button key={task.id} aria-pressed={demoId === task.id} disabled={busy} onClick={() => { setDemoId(task.id); setConfiguration(null); setError(""); }}>{simulationTaskTitle(task, locale)}</button>)}</div>
          {demo && <SimulationTaskConfigurator key={demoId} initialTaskId={demoId} allowedTaskIds={demoIds} requiredTaskMode="demo" showTaskSelection={false} disabled={busy} onStateChange={(next) => { if (next.taskId !== demoId) setDemoId(next.taskId); setConfiguration(next); }} />}
          {!demo && <p>{t("当前目录未提供此 Demo。", "This Demo is not in the current catalog.")}</p>}
          <div className="skills-launch"><button className="primary" disabled={!canStart || busy} onClick={() => void startDemo()}><Play size={16} /> {busy ? t("提交中…", "Submitting…") : t("启动控制演示", "Start control demo")}</button><span><Film size={16} /> {t("进入 Run 详情查看直播、录像与控制指标", "Live video, recording and control metrics are in Run details")}</span></div>
          {!canStart && <p className="skills-note">{t("需要服务端 Demo 目录、可用 Short Push Project、教学班、创建权限和有效参数。", "Requires server Demo catalog, an available Short Push Project, class, creation permission and valid parameters.")}</p>}
          {error && <p role="alert" className="skills-error"><CircleAlert size={16} /> {error}</p>}
        </div>
      </div>}
    </section>
    <footer className="skills-footer"><button className="secondary" onClick={() => navigate("/guide/task-skills")}><BookOpen size={15} /> {t("阅读任务与技能教程", "Read the tasks and skills guide")}</button><button className="secondary" onClick={() => navigate("/runs")}><ChevronRight size={15} /> {t("查看运行记录", "Open Run records")}</button></footer>
  </main>;
}
