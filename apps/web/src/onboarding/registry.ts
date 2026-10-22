export interface TourStep {
  id: string;
  target: string;
  title: [string, string];
  description: [string, string];
  waitFor?: "valueChange";
}

export interface TourDefinition {
  id: string;
  version: number;
  steps: TourStep[];
}

const step = (id: string, zhTitle: string, enTitle: string, zh: string, en: string, waitFor?: "valueChange"): TourStep => ({
  id, target: `[data-tour="${id}"]`, title: [zhTitle, enTitle], description: [zh, en], waitFor,
});

export const platformFirstTimeTour: TourDefinition = {
  id: "platform-introduction", version: 1, steps: [
    step("brand", "欢迎来到 Embodied Arm", "Welcome to Embodied Arm", "这里是一套面向具身智能机械臂实验的研究与教学工作台。你可以从学习、场景设计和参数配置开始，一直完成训练、运行分析与证据评估。", "This research and teaching workspace takes you from learning, scene design and parameters through training, run analysis and evidence evaluation."),
    step("primary-navigation", "完整实验流程", "The full experiment flow", "平台按研究流程分为学习、实验和分析三个阶段。先理解实验，再配置场景与训练，最后检查 Run、Evidence 和实验制品。", "Learn the experiment, configure a scene and training, then inspect runs, evidence and artifacts."),
    step("secondary-navigation", "进入具体工作区", "Open a workspace", "实验阶段在统一工作台浏览源码、修改奖励、配置参数并启动；分析阶段查看运行和证据。", "The available workspaces change with the selected research stage."),
    step("system-status", "确认当前能力边界", "Check capability boundaries", "这里显示 Simulation、GPU 和真实机械臂控制状态，避免把仿真能力误认为真实控制能力。", "Check simulation, GPU and real-robot control status. Simulation never grants real-control permission."),
    step("guide-actions", "需要时重新打开引导", "Reopen guidance when needed", "以后系统不会再自动打扰你。需要帮助时，可以从页面右上角打开「页面引导」，或重新查看平台介绍。", "The introduction will not open automatically again. Use the controls at the top right for a Page Tour or to replay the platform introduction."),
  ],
};

const tours: Record<string, TourDefinition> = {
  "/skills": { id: "tasks-skills", version: 1, steps: [
    step("skills-categories", "选择任务类别", "Choose a category", "强化学习任务更新策略权重，控制演示运行固定控制器。起始条件、成功证据和相关 Run 集中在各训练任务详情中。", "RL tasks update policy weights; control demos run fixed controllers. Entry conditions, success evidence and related Runs are in each task's details."),
    step("skills-catalog", "能力与证据分开核对", "Check capability and evidence separately", "服务目录决定可执行任务。本地 fallback 不获得提交权限。Demo 使用 play 创建 Run，实际视频和指标在详情中检查。", "Catalog determines executable tasks. Local fallback cannot submit. Demos create play Runs; verify actual video and metrics in Run details."),
  ] },
  "/roadmap": { id: "learning-guide", version: 1, steps: [
    step("learning-roadmap", "完整实验路线", "Experiment roadmap", "顶部 01–06 展示从认识平台到证据复盘的六个阶段。", "Stages 01–06 take you from platform orientation through evidence review."),
    step("learning-current-step", "当前学习阶段", "Current learning stage", "这里展示当前阶段的行动、相关文件、完成证据与能力边界。", "Review the action, related files, evidence and limits for this stage."),
    step("training-routes", "三条工作台训练路线", "Three workbench training paths", "Push、Reach 与 Place 是不同任务；Place 仅搬运保持，控制 Demo 不更新策略。", "Push, Reach and Place are separate tasks. Place is transport-and-hold; control demos do not train policies."),
    step("workspace-entry", "从教学进入真实工作区", "Open the real workspace", "使用当前阶段的入口进入真正的编辑、提交或评估页面。", "Use this action to enter the actual editing, submission or evaluation workspace."),
  ] },
  "/agent": { id: "teaching-agent", version: 2, steps: [
    step("agent-provider-list", "选择教学 Agent", "Choose a teaching Agent", "从 Available Providers 选择模型来源。", "Choose a model source from Available Providers."),
    step("agent-provider-profile", "确认模型能力边界", "Check provider boundaries", "核对 Provider 能力和数据范围；引导不会读取 API Key。", "Review provider capabilities and data scope. The tour never reads API keys."),
    step("agent-connection-check", "验证真实连接", "Verify the connection", "配置并验证会发送固定测试消息；只有返回有效回复才标记已验证。连接失败时查看密钥、余额、模型或服务器网络原因。", "Configure and verify sends a fixed test message. Only a valid response marks the connection verified. Failures distinguish key, balance, model and server network problems."),
    step("assistance-level", "设置帮助强度", "Set assistance level", "选择当前会话的帮助等级。", "Choose how much assistance to receive."),
    step("save-agent-preference", "保存当前教学设置", "Save teaching preferences", "保存 Provider 和帮助等级偏好。", "Save provider and assistance preferences."),
  ] },
  "/projects": { id: "experiment-workbench", version: 6, steps: [
    step("workbench-tasks", "选择实验任务", "Choose an experiment task", "选择推物、平面到达或搬运放置。先保持默认场景，每次只改一项。", "Choose Push, Reach or Place. Keep the default scene initially and change one thing at a time."),
    step("workbench-context", "确认课程与教学班", "Confirm course and class", "只有一个合法选项时自动选择；点击摘要可以切换。训练方案在训练设置中选择。", "A single eligible option is selected automatically. Open the summary to switch. Choose the training plan under Training settings."),
    step("workbench-modules", "三个实验编辑入口", "Three experiment controls", "顶部切换任务场景、奖励设计、训练设置。先从奖励设计开始，其他条件可以沿用默认值。", "Switch between Task scene, Reward design and Training settings above. Start with Reward design and keep other conditions at their defaults."),
    step("workbench-editor", "先只修改一个权重", "Change one weight first", "建议将动作惩罚从 1 改为 2。卡片说明作用与默认值，计算示例和代码按需展开；示例并非实测结果。", "Try changing the action penalty from 1 to 2. Cards explain behavior and defaults; expand calculations and code when needed. Examples are not measured outcomes."),
    step("workbench-scene", "理解任务场景", "Understand the task scene", "Push 的起点、目标和尺寸进入实际训练，尺寸以厘米输入。Reach 可调目标采样和步长；Place 的场景固定。", "Push start, goal and dimensions reach actual training; dimensions are entered in centimeters. Reach supports target sampling and motion steps. The Place scene is fixed."),
    step("workbench-parameters", "选择训练预算与种子", "Choose the budget and seed", "入门方案只检查流程，正式训练才用于观察学习。比较时保持预算与种子一致；录像高级设置可展开。", "The introductory plan checks the workflow; formal training is for observing learning. Match budget and seed for comparisons. Advanced recording settings can be expanded."),
    step("workbench-changes", "核对本次改动", "Review this draft's changes", "这里汇总相对默认奖励、场景和参数的变化。奖励修改记录可核对保存时间，再确认恢复为新版本。", "Review differences from default rewards, scene and parameters here. Reward history shows save times; explicitly confirm restoration as a new revision."),
    step("workbench-files", "需要时查看实现", "Inspect source when needed", "参考源码按钮打开只读文件浏览器。保留完整目录、搜索、长文件滚动；修改奖励请回到奖励设计。", "Open Reference source for the read-only file browser, full inventory, search and long-file scrolling. Return to Reward design to edit rewards."),
    step("workbench-agent", "随时询问实验助手", "Ask the experiment assistant", "卡片提问与悬浮球打开同一个助手，收起后保留当前上下文。它可解释选区或历史运行证据；代码建议先审查再保存。", "Card questions and the floating button open the same assistant, retaining context when collapsed. It explains selections and recorded execution evidence. Review code suggestions before saving."),
    step("workbench-actions", "保存并检查，再启动", "Save and check, then start", "固定操作栏显示训练预算、种子与下一步。保存并检查会依次保存场景、奖励并校验；任何一步失败都不会启动。检查详情可定位缺少的条件。", "The persistent bar shows budget, seed and next action. Save and check saves the scene and reward and validates them in sequence; failure never launches training. Check details locate unmet conditions."),
    step("workbench-evidence", "查看结果与对照实验", "Inspect results and a reference", "选择已有实验不会覆盖编辑方案。训练完成不代表任务成功；查看正式评估、视频与运行过程，再选择对照实验核对指标及比较条件。缺失指标显示未记录。", "Selecting a recorded experiment leaves the editing draft unchanged. Training completion does not prove task success. Inspect formal evaluation, video and timeline, then choose a reference to compare metrics and conditions. Missing metrics remain Not recorded."),
  ] },
  "/training-code": { id: "project-files", version: 1, steps: [
    step("project-tasks", "选择任务路线", "Choose a task", "切换 Push 和 Reach 的项目文件上下文。", "Switch between Push and Reach project contexts."),
    step("project-files", "项目文件", "Project files", "这里是可编辑教学草稿；保存不会替换服务端 PPO 代码。", "These editable drafts do not replace server-side PPO code."),
  ] },
  "/environments": { id: "scene-studio", version: 1, steps: [
    step("scene-revision", "场景 Revision", "Scene revision", "版本化场景使 Run 可以追溯到具体的 Scene Revision。", "Versioned scenes make each Run traceable to its Scene revision."),
    step("scene-editor", "实验场景", "Experiment scene", "3D Scene Studio 用于预览环境，不等于真实机械臂控制或碰撞验证。", "The 3D preview is not real-robot control or collision validation."),
    step("scene-toolbar", "编辑场景对象", "Edit scene objects", "可添加方块、球体或圆柱；修改后保存 Revision。", "Add boxes, spheres or cylinders, then save a revision."),
    step("joint-controls", "六自由度机械臂", "Six-degree-of-freedom arm", "J1–J6 对应机械臂各关节；当前姿态仅用于预览。", "J1–J6 are the arm joints. Their poses are for preview only."),
    step("joint-j2", "试着调整 J2", "Adjust J2", "请直接拖动真实 J2 滑块。关节数值变化后会自动进入下一步。", "Move the real J2 slider. The tour advances when its value changes.", "valueChange"),
    step("scene-objects", "场景对象", "Scene objects", "选择场景中的对象，查看并编辑其属性。", "Select a scene object to inspect and edit its properties."),
    step("save-revision", "保存实验场景", "Save the scene", "保存后产生新的 Scene Revision；引导不会代替你提交。", "Saving creates a new Scene revision. The tour never submits on your behalf."),
    step("continue-parameters", "继续配置实验", "Configure the experiment", "保存后进入参数沙箱配置 Push 或 Reach 训练。", "Continue to the parameter sandbox for Push or Reach training."),
  ] },
  "/courses": { id: "launch-experiment", version: 1, steps: [
    step("teaching-context", "教学上下文", "Teaching context", "先选择组织、课程和教学班。", "Select organization, course and class."),
    step("teaching-projects", "选择正式训练 Project", "Choose a formal Project", "选择可用的教学 Project。", "Choose an available teaching Project."),
    step("project-detail", "确认服务端配置", "Confirm server configuration", "检查只读的训练配置和 rubric。", "Review read-only training configuration and rubric."),
    step("training-task", "确认训练任务", "Confirm training task", "核对 Simulation 任务是否与 Project 匹配。", "Verify the Simulation task matches the Project."),
    step("scene-selection", "绑定 Scene Revision", "Bind Scene revision", "场景版本与正式 Run 绑定；Reach 使用其 Project 任务配置。", "The scene revision is bound to the Run; Reach uses its Project task configuration."),
    step("create-run", "创建正式 Run", "Create formal Run", "开始训练会提交正式 Simulation Run，不会授予真实机械臂控制权限。", "This submits a Simulation Run, not real-robot control permission."),
  ] },
  "/runs": { id: "run-evidence", version: 2, steps: [
    step("run-list", "运行记录", "Run records", "选择要分析的 Run。", "Choose a Run to inspect."),
    step("run-delete-controls", "清理已结束运行", "Delete terminal runs", "每次最多选择 10 条自己的已结束运行，确认后删除记录及关联索引。运行中记录不可删除；先导出需要保留的证据。", "Select up to 10 of your terminal runs. Confirm deletion of records and related indexes. Active runs cannot be deleted; export evidence you need to retain first."),
    step("selected-run", "当前 Run", "Selected Run", "核对运行状态、Project、班级和 seed。", "Check status, Project, class and seed."),
    step("run-config", "冻结配置", "Frozen configuration", "这里是 Run 创建时冻结的服务端配置。", "This is the server configuration frozen at Run creation."),
    step("event-evidence", "事件证据", "Event evidence", "沿时间线查看状态变化和事件。", "Inspect state transitions and events."),
    step("evidence-actions", "进一步分析", "Further analysis", "使用现有证据入口进行评估或导出。", "Use the existing actions for evaluation or export."),
    step("real-control-boundary", "真实控制边界", "Real-control boundary", "Simulation Run 不代表获得真实机械臂控制权限。", "A Simulation Run does not grant real-robot control permission."),
  ] },
  "/runs/:id": { id: "run-detail", version: 1, steps: [
    step("detail-status", "当前运行状态", "Current run status", "状态以服务端 Run 为准，刷新页面不会停止训练。", "The server Run is authoritative. Refreshing does not stop training."),
    step("detail-progress", "训练进度", "Training progress", "进度来自 Run 快照，而不是浏览器停留时间。", "Progress comes from the Run snapshot, not time spent in the browser."),
    step("detail-parameters", "冻结参数", "Frozen parameters", "核对本次 Run 的训练参数与配置边界。", "Review the Run parameters and configuration boundaries."),
    step("detail-results", "结果与制品", "Results and artifacts", "按现有入口下载指标、可用制品和正式记录。", "Use the existing controls to download metrics, available artifacts and the formal record."),
    step("detail-events", "运行事件", "Run events", "依据事件证据核对运行状态，Simulation 不代表真实控制授权。", "Review event evidence. Simulation does not grant real-control permission."),
  ] },
  "/evaluation": { id: "evaluation", version: 1, steps: [
    step("evaluation-run", "评估对象", "Run to evaluate", "选择终态 Run。", "Select a terminal Run."),
    step("evaluation-report", "不可变评估报告", "Immutable report", "报告只绑定一条正式 Run；证据不足不会生成推测分数。", "Reports bind to one Run and never invent missing evidence."),
    step("generate-evaluation", "生成证据评估", "Generate evaluation", "手动生成正式评估；引导不会代替你操作。", "Generate a formal evaluation yourself."),
    step("evaluation-boundary", "评估不等于控制授权", "Evaluation is not control authorization", "SIM2REAL BLOCKED，真实控制始终禁用。", "Sim2Real is blocked; real control remains disabled."),
  ] },
  "/export": { id: "artifacts", version: 1, steps: [
    step("artifact-run-select", "选择实验制品", "Choose artifacts", "选择已完成的 Run。", "Select a completed Run."),
    step("canonical-evidence", "Canonical Evidence", "Canonical evidence", "预览关联的 JSON 证据，导出由服务端生成。", "Preview the associated JSON evidence; export is generated by the server."),
    step("json-export", "导出实验记录", "Export the record", "下载正式的 JSON 实验记录。", "Download the canonical JSON record."),
  ] },
};

export const tourRegistry = {
  resolve(path: string): TourDefinition | undefined {
    const pathname = path.split(/[?#]/, 1)[0];
    return pathname.startsWith("/runs/") ? tours["/runs/:id"] : tours[pathname === "/training" ? "/projects" : pathname];
  },
};
