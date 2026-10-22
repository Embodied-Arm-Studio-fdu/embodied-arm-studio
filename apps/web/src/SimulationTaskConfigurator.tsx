import { simulationTaskTitle } from "./simulationTaskPresentation";
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, CircleAlert, Plus, RotateCcw, ShieldCheck, X } from "lucide-react";
import { getAccessToken } from "./api";
import {
  fetchSimulationCatalog,
  getTaskDefaults,
  isFrontendTaskParameter,
  type SimulationTaskCatalog,
  type SimulationTaskDefinition,
  type SimulationTaskParameter,
  type SimulationTaskMode,
  type TaskParameterValidation,
  validateTaskParameters,
} from "./simulationCatalog";
import { translateUiText, useI18n } from "./i18n";

export interface SimulationTaskConfigurationState {
  catalog?: SimulationTaskCatalog;
  task?: SimulationTaskDefinition;
  taskId: string;
  parameters: Record<string, unknown>;
  validation?: TaskParameterValidation;
}

export interface SimulationTaskConfiguratorProps {
  initialTaskId?: string;
  initialParameters?: Record<string, unknown>;
  allowedTaskIds?: readonly string[];
  onStateChange?: (state: SimulationTaskConfigurationState) => void;
  compact?: boolean;
  disabled?: boolean;
  summaryOnly?: boolean;
  showTaskSelection?: boolean;
  requiredTaskMode?: SimulationTaskMode;
  groupParameters?: boolean;
  workbenchSection?: "scene" | "settings";
}

export const isSceneParameter = (name: string) => !/^(seed$|video_|camera_|fps$|width$|height$|record_)/.test(name);
const workbenchParameterLessons: Record<string, [string, string, string, string]> = {
  video_enabled: ["记录实验视频", "Record experiment video", "开启后记录实际运行画面，便于检查行为；视频设置不会直接改变奖励或成功标准。", "Record actual execution to inspect behavior. Video settings do not directly change rewards or success criteria."],
  video_fps: ["视频帧率", "Video frame rate", "每秒保存的画面数量；较高帧率增加记录开销。", "Frames saved per second; higher rates increase recording overhead."],
  video_width: ["视频宽度（像素）", "Video width (pixels)", "影响录像清晰度与渲染开销。", "Affects recording clarity and rendering overhead."],
  video_height: ["视频高度（像素）", "Video height (pixels)", "影响录像清晰度与渲染开销。", "Affects recording clarity and rendering overhead."],
  seed: ["随机种子", "Random seed", "保持同一种子方便比较；换种子可检查结果是否依赖一次随机过程。", "Keep the same seed for comparisons; change it to check sensitivity to randomness."],
  action_delta_m: ["单步最大位移（米）", "Maximum motion per step (m)", "0.005 米 = 5 毫米。小步便于精细调整，大步可移动更远；效果需要实验验证。", "0.005 m = 5 mm. Smaller steps allow finer adjustments; larger steps cover more distance. Test the outcome."],
  target_center_xy_m: ["目标区域中心 XY（米）", "Target region center XY (m)", "依次填写 X、Y，改变目标所在位置；服务端会检查允许范围。", "Enter X then Y to move the target region. The server checks the permitted range."],
  target_half_extent_xy_m: ["目标采样半范围 XY（米）", "Target sampling half-extents XY (m)", "训练目标在中心两侧的范围内采样；这不是成功距离。", "Training targets are sampled within these extents on either side of the center. This is not the success tolerance."],
  success_distance_m: ["成功距离（课程固定）", "Success tolerance (course controlled)", "本工作台保持成功标准一致，避免通过放宽标准改变评分。", "This workbench keeps the success criterion consistent across experiments."],
  success_hold_steps: ["成功保持步数（课程固定）", "Success hold steps (course controlled)", "到达目标后需要持续保持的步数。", "Number of steps that must remain within the target after reaching it."],
  hold_seconds: ["稳定保持时间（秒）", "Stable hold duration (s)", "Place 需要在目标区域稳定保持一秒。", "Place requires a stable one-second hold in the target region."],
  randomize_xy_m: ["初始 XY 随机偏移（米）", "Initial XY randomization (m)", "课程固定值；不额外开放未经验证的初始状态。", "Course-controlled value; additional initial states require validation."],
  release_trigger_threshold: ["松爪阈值", "Release threshold", "当前搬运任务锁定夹爪，不增加释放动作。", "The transport task keeps the gripper locked."],
};

const hasValue = (values: Record<string, unknown>, name: string) =>
  Object.prototype.hasOwnProperty.call(values, name);

const taskPurposes: Record<string, [string, string]> = {
  demo_s1_scripted_pick_place: ["规则抓取放下：固定控制器依次执行对齐、抓取、抬起、放下、松爪和稳定性检查。查看阶段、失败原因与实际视频；不进行 PPO 学习，不生成策略权重。", "Scripted pick and place: a fixed controller aligns, grasps, lifts, lowers, releases and checks settling. Inspect stages, failure reasons and actual video. No PPO learning or policy weights."],
  s1_push: ["推物强化学习：末端在桌面 XY 平面推动方块接近目标。关注成功率、最终距离、物体位移；训练输出策略权重。", "Push RL: move a cube toward the goal using planar end-effector actions. Evidence: success rate, final distance, displacement and policy weights."],
  s1_reach: ["平面到达强化学习：末端移动到目标区域并保持，不是抓取任务。可配置目标中心、区域范围、成功距离和保持步数。", "Planar reach RL: move the end effector into a target region and hold, not grasping. Configure target center, extent, success distance and hold steps."],
  s1_grasp: ["抓取强化学习：对齐方块、下降并闭合夹爪，以验证器确认稳定抓取。重点观察候选抓取率、验证成功率、接触阻塞与安全失败。", "Grasp RL: align with the cube, descend and close the gripper until the verifier confirms a stable grasp. Inspect candidate rate, verified success, blockage and safety failures."],
  s1_lift: ["抓取抬起强化学习：从稳定抓取状态抬升到 60 mm 目标带并持续保持。重点观察方块升高、保持时长、相对漂移和稳定抓取率。", "Grasp-and-lift RL: raise a stably grasped cube into the 60 mm target band and hold. Inspect cube rise, hold duration, relative drift and grasp stability."],
  s1_lower_release: ["抓取放下强化学习：把已抓取方块下降到目标区域，合法触发松爪并等待稳定。重点观察下降率、脱离率、目标误差和稳定放置率。", "Lower-and-release RL: descend a grasped cube into the target, trigger a legal release and wait for settling. Inspect descent, detachment, target error and stable placement."],
  s1_place: ["搬运放置强化学习：从验证过的抓取抬起状态水平搬运方块到经过候选检查的目标，并保持一秒。重点观察目标到达、保持、掉落、抓持丢失和安全失败。", "Transport-and-place RL: move a validated grasped-and-lifted cube to a candidate-checked target and hold for one second. Inspect target reached, hold, drop, grasp loss and safety failures."],
  demo_s1_target_pose: ["目标位姿控制：按指定 XY 偏移移动末端，检查到达过程；使用固定控制器，不进行 PPO 学习。", "Target pose: move the end effector by an XY offset using a fixed controller, without PPO learning."],
  demo_s1_line_tracking: ["直线跟踪：末端沿 X 或 Y 方向往返，调整长度与周期观察轨迹。无训练权重。", "Line tracking: move along X or Y; configure length and period. No trained weights."],
  demo_s1_circle_tracking: ["圆轨迹跟踪：末端按半径和周期执行圆轨迹，观察控制稳定性。无训练权重。", "Circle tracking: configure radius and period to inspect controller stability. No trained weights."],
  demo_s1_joint_position: ["关节位置控制：选择 home、ready 或 side 预设姿态，检查关节和机械臂姿态。不是强化学习。", "Joint position: select home, ready or side to inspect joint poses. Not reinforcement learning."],
  demo_s1_joint_sweep: ["关节范围扫描：选择单关节或全部机械臂关节，以指定速度扫描范围。不是碰撞检测或抓取。", "Joint sweep: scan one or all arm joints at a chosen speed. Not collision detection or grasping."],
};

const formatValue = (value: unknown): string => {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === undefined) return "-";
  return String(value);
};

const emptyArrayItem = (parameter: SimulationTaskParameter): unknown => {
  if (parameter.items?.type === "integer" || parameter.items?.type === "number") return 0;
  if (parameter.items?.type === "boolean") return false;
  return "";
};

function authenticatedCatalogFetch(input: RequestInfo | URL, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  const token = getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}

export function SimulationTaskConfigurator({
  initialTaskId,
  initialParameters,
  allowedTaskIds,
  onStateChange,
  compact = false,
  disabled = false,
  summaryOnly = false,
  showTaskSelection = true,
  requiredTaskMode,
  groupParameters = false,
  workbenchSection,
}: SimulationTaskConfiguratorProps) {
  const { locale, t } = useI18n();
  const catalogQuery = useQuery({
    queryKey: ["simulation-task-catalog"],
    queryFn: () => fetchSimulationCatalog(authenticatedCatalogFetch),
  });
  const [selectedTaskId, setSelectedTaskId] = useState(initialTaskId ?? "");
  const [parameters, setParameters] = useState<Record<string, unknown>>(initialParameters ?? {});
  const [recordingAdvanced, setRecordingAdvanced] = useState(false);
  const initializedTaskRef = useRef("");
  const onStateChangeRef = useRef(onStateChange);
  onStateChangeRef.current = onStateChange;

  const catalog = catalogQuery.data;
  const availableTasks = catalog?.tasks.filter(
    (item) => (!requiredTaskMode || item.mode === requiredTaskMode) && (allowedTaskIds === undefined ? item.mode === "training" : allowedTaskIds.includes(item.id)),
  ) ?? [];
  const task = availableTasks.find((item) => item.id === selectedTaskId);
  const validation = task ? validateTaskParameters(task, parameters) : undefined;
  const configurableParameters = task?.parameters.filter(
    (parameter) => isFrontendTaskParameter(parameter.name),
  ) ?? [];
  const visibleParameters = workbenchSection ? configurableParameters.filter((parameter) => isSceneParameter(parameter.name) === (workbenchSection === "scene")) : configurableParameters;

  useEffect(() => {
    if (!catalog) return;
    const compatibleTasks = catalog.tasks.filter(
      (item) => (!requiredTaskMode || item.mode === requiredTaskMode) && (allowedTaskIds === undefined ? item.mode === "training" : allowedTaskIds.includes(item.id)),
    );
    const currentTask = compatibleTasks.find((item) => item.id === selectedTaskId);
    if (currentTask) {
      if (initializedTaskRef.current !== currentTask.id) {
        setParameters({
          ...getTaskDefaults(currentTask),
          ...(currentTask.id === initialTaskId ? initialParameters : {}),
        });
        initializedTaskRef.current = currentTask.id;
      }
      return;
    }
    const preferredTask =
      compatibleTasks.find((item) => item.id === initialTaskId) ??
      compatibleTasks.find((item) => item.mode === "training" && item.supportsFormalRun) ??
      compatibleTasks[0];
    if (!preferredTask) return;
    setSelectedTaskId(preferredTask.id);
  }, [allowedTaskIds, catalog, initialParameters, initialTaskId, selectedTaskId, requiredTaskMode]);

  useEffect(() => {
    if (!catalog || !task) return;
    const currentValidation = validateTaskParameters(task, parameters);
    onStateChangeRef.current?.({
      catalog,
      task,
      taskId: task.id,
      parameters: currentValidation.values,
      validation: currentValidation,
    });
  }, [catalog, task, parameters]);

  const updateParameter = (name: string, value: unknown) => {
    setParameters((current) => ({ ...current, [name]: value }));
  };

  const selectTask = (taskId: string) => {
    const nextTask = availableTasks.find((item) => item.id === taskId);
    if (!nextTask) return;
    initializedTaskRef.current = taskId;
    setSelectedTaskId(taskId);
    setParameters(getTaskDefaults(nextTask));
  };

  const resetParameters = () => {
    if (task) {
      const defaults = getTaskDefaults(task);
      setParameters((current) => workbenchSection ? { ...current, ...Object.fromEntries(visibleParameters.map((parameter) => [parameter.name, defaults[parameter.name]])) } : defaults);
    }
  };

  const localizedParameter = (parameter: SimulationTaskParameter): SimulationTaskParameter => {
    const lesson = workbenchParameterLessons[parameter.name];
    if (lesson) {
      return {
        ...parameter,
        title: t(lesson[0], lesson[1]),
        description: t(lesson[2], lesson[3]),
      };
    }
    if (locale !== "en") return parameter;
    return {
      ...parameter,
      title: parameter.title ? translateUiText(parameter.title) : parameter.title,
      description: parameter.description ? translateUiText(parameter.description) : parameter.description,
    };
  };

  if (catalogQuery.isLoading) {
    return (
      <section className={`simulation-task-configurator${compact ? " compact" : ""}`}>
        <div className="simulation-task-state">
          {t("正在读取 Simulation task catalog...", "Reading the Simulation task catalog...")}
        </div>
      </section>
    );
  }

  if (catalogQuery.isError || !catalog) {
    return (
      <section className={`simulation-task-configurator${compact ? " compact" : ""}`}>
        <div className="simulation-task-state error">
          <CircleAlert size={17} />
          <span>
            {catalogQuery.error instanceof Error
              ? catalogQuery.error.message
              : t("Simulation task catalog 不可用。", "The Simulation task catalog is unavailable.")}
          </span>
          <button className="secondary" onClick={() => void catalogQuery.refetch()}>
            <RotateCcw size={14} /> {t("重试", "Retry")}
          </button>
        </div>
      </section>
    );
  }

  if (!task || !validation) {
    return (
      <section className={`simulation-task-configurator${compact ? " compact" : ""}`}>
        <div className="simulation-task-state error">
          <CircleAlert size={17} />
          {t("catalog 没有可用任务。", "The catalog has no usable task.")}
        </div>
      </section>
    );
  }

  return (
    <section className={`simulation-task-configurator${compact ? " compact" : ""}`} data-tour={summaryOnly ? undefined : "parameter-form"}>
      <div className={`simulation-catalog-source ${catalog.source}`}>
        <span>
          {catalog.source === "server"
            ? t("服务端目录", "SERVER CATALOG")
            : t("本地备用目录", "LOCAL FALLBACK CATALOG")}
        </span>
        <small>
          {catalog.source === "server"
            ? t("任务和参数来自服务端响应。", "Tasks and parameters come from the server response.")
            : t("服务端未提供 catalog 路由；以下规格来自本地 fallback，未宣称真实执行能力。", "The server did not expose the catalog route; these local fallback specs do not claim real execution capability.")}
        </small>
      </div>
      {!summaryOnly && showTaskSelection && <div className="simulation-task-selector">
        <div>
          <span className="section-label">{t("仿真任务", "SIMULATION TASK")}</span>
          <strong>{t("任务选择", "Task selection")}</strong>
        </div>
        <select
          value={task.id}
          disabled={disabled}
          onChange={(event) => selectTask(event.target.value)}
          aria-label={t("Simulation 任务", "Simulation task")}
        >
          <option value="" disabled>{t("选择任务", "Select a task")}</option>
          {availableTasks.map((item) => <option key={item.id} value={item.id}>{simulationTaskTitle(item, locale)} · {item.id}</option>)}
        </select>
      </div>}
      {!summaryOnly && showTaskSelection && <div className="simulation-task-overview" aria-label={t("可用仿真任务", "Available simulation tasks")}>
        {availableTasks.map((item) => (
          <button
            key={item.id}
            type="button"
            className={item.id === task.id ? "active" : ""}
            aria-pressed={item.id === task.id}
            disabled={disabled}
            onClick={() => selectTask(item.id)}
          >
            <span className="simulation-task-overview-title"><strong>{simulationTaskTitle(item, locale)}</strong><code>{item.id}</code></span>
            <span>{taskPurposes[item.id] ? t(...taskPurposes[item.id]) : item.description || t("服务端未提供描述", "No description from server")}</span>
            <small className={item.supportsFormalRun ? "available" : "limited"}>{item.supportsFormalRun
              ? item.mode === "demo" ? t("控制演示 · 不进行策略学习", "Control demo · no policy learning") : t("Simulation 支持训练 · 网页启动需匹配课程 Project", "Simulation training · requires matching course Project")
              : item.mode === "demo"
                ? t("Simulation 控制演示 · 可从 Short Push Project 启动", "Simulation control demo · launch from the Short Push Project")
                : t("仅可配置 · 此页不能启动训练", "Configuration only · training unavailable here")}</small>
          </button>
        ))}
      </div>}
      <div className="simulation-task-heading">
        <div>
          <span className="eyebrow">{task.mode === "demo" ? t("Demo 任务", "DEMO TASK") : t("训练任务", "TRAINING TASK")}</span>
          <h3>{simulationTaskTitle(task, locale)}</h3>
          <p>{taskPurposes[task.id] ? t(...taskPurposes[task.id]) : task.description || t("catalog 未提供任务描述。", "The catalog did not provide a task description.")}</p>
        </div>
        <div className={`simulation-task-mode ${task.mode}`}>
          <strong>{task.mode === "demo" ? "Demo" : t("训练", "TRAINING")}</strong>
          <small>{task.supportsVideo ? t("支持视频", "Video supported") : t("未声明视频", "Video not declared")}</small>
        </div>
      </div>
      <div className="simulation-capability-grid">
        <div>
          <span>{task.mode === "demo" ? t("控制演示 Run", "Control demo Run") : t("正式 Run", "Formal Run")}</span>
          <strong className={task.supportsFormalRun ? "available" : "blocked"}>
            {task.mode === "demo" ? t("需服务目录和 Short Push Project", "Requires service catalog and Short Push Project") : task.supportsFormalRun ? t("catalog 已声明支持", "Declared by catalog") : t("当前不可提交", "Unavailable")}
          </strong>
        </div>
        <div>
          <span>{t("Checkpoint", "Checkpoint")}</span>
          <strong className={task.producesCheckpoint ? "available" : "blocked"}>
            {task.producesCheckpoint ? t("训练产物", "Training artifact") : t("不生成", "Not produced")}
          </strong>
        </div>
      </div>
      <div className={`simulation-boundary-note ${task.supportsFormalRun ? "available" : "blocked"}`}>
        {task.supportsFormalRun ? <Check size={16} /> : <ShieldCheck size={16} />}
        <span>
          <strong>
            {task.mode === "demo" ? t("控制演示边界", "Control demo boundary") : task.supportsFormalRun
              ? t("Formal Run 边界", "Formal Run boundary")
              : t("不能伪造 Formal Run 能力", "Formal Run is not claimed")}
          </strong>
          <small>
            {task.mode === "demo" ? t("固定控制器使用 play Run，不进行策略训练。服务端目录、Short Push Project 和权限校验通过后才可启动。", "Fixed controllers use play Runs, not policy training. Launch requires server catalog, Short Push Project and permission validation.") : task.formalRunReason ?? t("该任务可进入正式 Simulation Run。", "This task can enter a formal Simulation Run.")}
          </small>
        </span>
      </div>
      {summaryOnly ? (
        <div className="simulation-parameter-summary">
          {configurableParameters.map((parameter) => (
            <div key={parameter.name}>
              <span>{parameter.name}</span>
              <strong>{formatValue(validation.values[parameter.name])}</strong>
            </div>
          ))}
        </div>
      ) : <>
      <div className="simulation-parameters-header">
        <div>
          <span className="section-label">{t("参数", "PARAMETERS")}</span>
          <strong>{t("任务参数", "Task parameters")}</strong>
        </div>
        <button className="secondary" onClick={resetParameters} disabled={disabled} title={t("恢复 catalog 默认值", "Reset to catalog defaults")}>
          <RotateCcw size={14} /> {t("恢复默认", "Reset")}
        </button>
      </div>
      {workbenchSection && !visibleParameters.length && <p className="workbench-fixed-note">{t("此任务的场景由课程 Project 固定，当前没有独立场景参数。请通过奖励设计进行实验。", "The course Project fixes this task's scene; no independent scene parameters are available. Use Reward design for experiments.")}</p>}
      {(workbenchSection === "settings" ? [
        { id: "primary", title: t("基础设置", "Basic settings"), parameters: visibleParameters.filter((parameter) => parameter.name === "seed" || parameter.type === "boolean") },
        { id: "advanced", title: t("高级录像设置", "Advanced recording settings"), parameters: visibleParameters.filter((parameter) => parameter.name !== "seed" && parameter.type !== "boolean") },
      ] : groupParameters ? [
        { id: "training", title: t("训练参数", "Training parameters"), parameters: visibleParameters.filter((parameter) => !/^(video_|camera_|fps$|width$|height$|record_)/.test(parameter.name)) },
        { id: "recording", title: t("视频记录参数", "Video recording parameters"), parameters: visibleParameters.filter((parameter) => /^(video_|camera_|fps$|width$|height$|record_)/.test(parameter.name)) },
      ] : [{ id: "all", title: "", parameters: visibleParameters }]).filter((group) => group.parameters.length > 0).map((group) => <section className="simulation-parameter-group" key={group.id}>
      {group.id === "advanced" ? <details className="workbench-recording-advanced" open={recordingAdvanced}><summary onClick={(event) => { event.preventDefault(); setRecordingAdvanced((current) => !current); }}>{group.title}{visibleParameters.some((parameter) => group.parameters.includes(parameter) && validation.errors[parameter.name]) && t(" · 请修正参数", " · Correct parameters")}</summary><p>{t("固定 overview 视角；这些选项控制记录清晰度与开销，不改变任务目标。", "Fixed overview camera; these options control recording clarity and overhead, without changing the task goal.")}</p><div className="simulation-parameter-list">{group.parameters.map((parameter) => <SimulationParameterField key={parameter.name} parameter={localizedParameter(parameter)} value={hasValue(parameters, parameter.name) ? parameters[parameter.name] : parameter.default} error={validation.errors[parameter.name]} disabled={disabled || parameters.video_enabled === false || parameter.enum?.length === 1} onChange={(value) => updateParameter(parameter.name, value)} t={t} />)}</div></details> : <>
      {group.title && <h4>{group.title}</h4>}
      <div className="simulation-parameter-list">
        {group.parameters.map((parameter) => {
          const value = hasValue(parameters, parameter.name) ? parameters[parameter.name] : parameter.default;
          const error = validation.errors[parameter.name];
          return (
            <SimulationParameterField
              key={parameter.name}
              parameter={localizedParameter(parameter)}
              value={value}
              error={error}
              disabled={disabled || parameter.enum?.length === 1 || Boolean(workbenchSection && /^(success_distance_m|success_hold_steps)$/.test(parameter.name))}
              onChange={(nextValue) => updateParameter(parameter.name, nextValue)}
              t={t}
            />
          );
        })}
      </div>
      </>}
      </section>)}
      </>}
      {!validation.valid && (
        <div className="simulation-validation-summary error">
          <CircleAlert size={16} />
          <span>{t("请修正上面的参数后再提交或启动。", "Correct the parameter errors before submitting or starting.")}</span>
        </div>
      )}
    </section>
  );
}

function SimulationParameterField({
  parameter,
  value,
  error,
  disabled,
  onChange,
  t,
}: {
  parameter: SimulationTaskParameter;
  value: unknown;
  error?: string;
  disabled: boolean;
  onChange: (value: unknown) => void;
  t: (chinese: string, english: string) => string;
}) {
  const itemType = parameter.items?.type ?? "number";
  const numeric = parameter.type === "integer" || parameter.type === "number";
  const inputBounds = {
    min: parameter.minimum,
    max: parameter.maximum,
    step: parameter.step ?? (parameter.type === "integer" ? 1 : "any"),
  };

  const renderScalar = (scalarValue: unknown, update: (nextValue: unknown) => void) => {
    if (parameter.enum && parameter.type !== "boolean") {
      return (
        <select
          aria-label={parameter.title ?? parameter.name}
          value={String(parameter.enum.findIndex((option) => Object.is(option, scalarValue)))}
          disabled={disabled}
          onChange={(event) => update(parameter.enum?.[Number(event.target.value)])}
        >
          {parameter.enum.map((option, index) => <option key={`${index}-${formatValue(option)}`} value={index}>{formatValue(option)}</option>)}
        </select>
      );
    }
    if (parameter.type === "boolean") {
      return (
        <label className="simulation-checkbox">
          <input type="checkbox" aria-label={parameter.title ?? parameter.name} checked={scalarValue === true} disabled={disabled} onChange={(event) => update(event.target.checked)} />
          <span>{scalarValue === true ? t("启用", "Enabled") : t("停用", "Disabled")}</span>
        </label>
      );
    }
    if (numeric) {
      return (
        <input
          type="number"
          aria-label={parameter.title ?? parameter.name}
          value={typeof scalarValue === "number" && Number.isFinite(scalarValue) ? scalarValue : ""}
          disabled={disabled}
          min={inputBounds.min}
          max={inputBounds.max}
          step={inputBounds.step}
          onChange={(event) => update(event.target.value === "" ? undefined : Number(event.target.value))}
        />
      );
    }
    return <input type="text" aria-label={parameter.title ?? parameter.name} value={typeof scalarValue === "string" ? scalarValue : formatValue(scalarValue)} disabled={disabled} onChange={(event) => update(event.target.value)} />;
  };

  return (
    <div className={`simulation-parameter-field${error ? " has-error" : ""}`} data-tour={parameter.name === "video_enabled" || parameter.name === "record_video" ? "parameter-recording" : undefined}>
      <div className="simulation-parameter-label">
        <div>
          <strong>{parameter.title ?? parameter.name}</strong>
          <code>{parameter.name}</code>
        </div>
        <small>{parameter.type}{parameter.items ? `[${parameter.items.type}]` : ""}</small>
      </div>
      {parameter.description && <p>{parameter.description}</p>}
      {parameter.type === "array" ? (
        <div className="simulation-array-editor">
          {(Array.isArray(value) ? value : []).map((item, index) => (
            <div className="simulation-array-item" key={`${parameter.name}-${index}`}>
              {parameter.items?.type === "boolean" ? (
                <label className="simulation-checkbox">
                  <input type="checkbox" checked={item === true} disabled={disabled} onChange={(event) => {
                    const next = Array.isArray(value) ? [...value] : [];
                    next[index] = event.target.checked;
                    onChange(next);
                  }} />
                  <span>{index + 1}</span>
                </label>
              ) : (
                <input
                  type={itemType === "integer" || itemType === "number" ? "number" : "text"}
                  value={typeof item === "string" || typeof item === "number" ? item : formatValue(item)}
                  disabled={disabled}
                  min={inputBounds.min}
                  max={inputBounds.max}
                  step={inputBounds.step}
                  aria-label={`${parameter.name}[${index}]`}
                  onChange={(event) => {
                    const next = Array.isArray(value) ? [...value] : [];
                    next[index] = itemType === "integer" || itemType === "number"
                      ? event.target.value === "" ? undefined : Number(event.target.value)
                      : event.target.value;
                    onChange(next);
                  }}
                />
              )}
              {!parameter.fixedLength && <button className="icon-button" type="button" disabled={disabled} title={t("删除数组项", "Remove array item")} onClick={() => {
                const next = Array.isArray(value) ? value.filter((_, itemIndex) => itemIndex !== index) : [];
                onChange(next);
              }}>
                <X size={13} />
              </button>}
            </div>
          ))}
          {!parameter.fixedLength && <button className="secondary array-add" type="button" disabled={disabled} onClick={() => onChange([...(Array.isArray(value) ? value : []), emptyArrayItem(parameter)])}>
            <Plus size={13} /> {t("添加数组项", "Add array item")}
          </button>}
        </div>
      ) : (
        <div className="simulation-scalar-editor">{renderScalar(value, onChange)}</div>
      )}
      <div className="simulation-parameter-meta">
        <span>DEFAULT {formatValue(parameter.default)}</span>
        {parameter.minimum !== undefined && <span>MIN {parameter.minimum}{parameter.exclusiveMinimum !== undefined ? " (exclusive)" : ""}</span>}
        {parameter.maximum !== undefined && <span>MAX {parameter.maximum}</span>}
        {parameter.step !== undefined && <span>STEP {parameter.step}</span>}
      </div>
      {error && <small className="simulation-parameter-error"><CircleAlert size={13} /> {error}</small>}
    </div>
  );
}
