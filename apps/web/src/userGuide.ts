export const USER_GUIDE_VERSION = "2026.10.12";

export interface GuideStep { title: [string, string]; body: [string, string]; }
export interface GuideSection {
  id: string;
  route?: string;
  tourId?: string;
  title: [string, string];
  summary: [string, string];
  audience: [string, string];
  steps: GuideStep[];
  checks: [string, string][];
  boundaries: [string, string][];
}

export const GUIDE_PRINCIPLES: Record<string, [string, string][]> = {
  "task-skills": [
    ["训练、执行与组合的区别", "Training, execution and composition"],
    ["训练更新策略权重；Demo 按固定规则产生动作；已训练技能执行固定策略。组合包含学习策略，不代表组合本身正在训练。", "Training updates weights; demos use fixed rules; trained skills execute frozen policies. A workflow containing learned skills is not necessarily being trained."],
    ["零件如何形成完整动作", "How skills form a complete action"],
    ["典型流程：准备姿态 → 接近物体 → 验证抓取 → 抬升并保持 → 搬运到目标上方 → 下降释放 → 稳定放置 → 撤离回位。所有阶段需要共享同一仿真环境，保留物体和夹持状态；上一阶段的输出必须满足下一阶段的初始契约，失败不能偷偷 reset。当前仅提供教学说明，不提供流程编排或连续执行入口。", "Typical flow: ready pose → approach → verified grasp → lift and hold → transport above goal → descend and release → settle → retreat and return. All stages need one shared simulator preserving object and grasp state. Outputs must satisfy the next entry contract, without hidden resets on failure. This is instructional content only, not an orchestration or execution entry."],
    ["视频是独立验收项", "Video requires separate verification"],
    ["Run completed 不证明直播或录像成功。检查流状态、MP4 时长和动作覆盖；Demo 的跟踪误差不能当作 PPO 学习效果。", "Run completed does not prove video success. Inspect stream state, MP4 duration and action coverage; Demo tracking error is not PPO learning performance."],
  ],
  overview: [
    ["研究工作台的核心", "The core idea of the research workspace"],
    ["平台把学习、配置、运行和证据连接成一条链。每一次实验都应该能够回答三个问题：使用了什么配置、运行产生了什么证据、结论的边界在哪里。", "The platform connects learning, configuration, execution and evidence. Every experiment should answer three questions: which configuration was used, what evidence was produced, and where the conclusion stops."],
    ["为什么要区分 Simulation 和真实控制", "Why simulation is separated from real control"],
    ["仿真可以帮助学习任务结构和比较策略，但它不包含真实硬件的磨损、延迟、标定、碰撞风险和操作员授权。因此页面中的 Simulation、Evaluation 和 Sim2Real 状态必须分开理解。", "Simulation helps learn task structure and compare policies, but it does not include hardware wear, latency, calibration, collision risk or operator authorization. Simulation, Evaluation and Sim2Real states must remain separate."],
  ],
  workflow: [
    ["从状态到证据", "From state to evidence"],
    ["实验流程不是一组孤立页面，而是一条状态转换：场景和任务定义形成配置，配置创建 Run，Run 产生事件和指标，证据再支持评估。前一步没有稳定记录，后一步就不能声称可复现。", "The workflow is a state transition, not a set of isolated pages: scene and task definitions form configuration, configuration creates a Run, the Run produces events and metrics, and evidence supports evaluation. Without a stable record at one step, reproducibility cannot be claimed at the next."],
    ["为什么要保留冻结快照", "Why frozen snapshots matter"],
    ["训练过程中页面草稿可能继续变化，但 Run 的冻结快照必须保持不变。比较实验时应优先读取 Run 创建时的快照，而不是当前页面的临时选择。", "Page drafts may change while training runs, but the Run snapshot must remain immutable. Compare experiments using the snapshot at Run creation, not a current temporary selection."],
  ],
  "project-workspace": [
    ["任务、策略和奖励的关系", "How task, policy and reward relate"],
    ["任务定义环境如何重置、提供什么观测、接受什么动作以及何时结束；策略根据观测产生动作；奖励把任务目标转成训练信号。改变其中一项，可能改变整个实验问题。", "The task defines reset, observations, actions and termination; the policy produces actions from observations; the reward turns the objective into a training signal. Changing one can change the entire experiment problem."],
    ["受控奖励如何进入正式运行", "How controlled rewards enter a formal Run"],
    ["只有 reward.py 的受控组合表达式会冻结进正式 Run；其他源码用于完整浏览。保存、预检、版本和 SHA-256 共同确定运行内容。", "Only the controlled reward.py combiner is frozen into a formal Run. Other source is browsable. Saving, preflight, revision history and SHA-256 identify exactly what runs."],
    ["S1 Push 和 S1 Reach 的差异", "How S1 Push and S1 Reach differ"],
    ["Push 的核心对象是可推动方块：策略要让物体接近目标；Reach 的核心对象是末端和目标区域：策略要让末端进入区域并保持。两者的观测维度、成功条件、奖励和评估指标都不能直接混用。", "Push centers on a movable object: the policy brings it closer to a goal. Reach centers on the end effector and a target region: the policy enters and holds the region. Their observations, success conditions, rewards and metrics must not be mixed."],
    ["Observation、Action 和 Reward 的关系", "How observation, action and reward relate"],
    ["Observation 是策略看到的状态摘要；Action 是策略可以施加的受限输出；Reward 是训练过程用来判断动作好坏的信号。它们共同定义了学习问题，但 Reward 不是最终评估报告本身。", "Observation is the state summary seen by the policy. Action is its bounded output. Reward is the training signal for judging actions. Together they define the learning problem, but reward is not the final evaluation report."],
  ],
  "scene-studio": [
    ["Scene Revision 的意义", "What a Scene Revision means"],
    ["场景不是一张截图，而是一组带版本号的对象、姿态和资产引用。Revision 让之后的 Run 能指出“当时使用的是哪一个环境定义”。", "A scene is not a screenshot. It is a versioned set of objects, poses and asset references. A Revision lets a later Run identify exactly which environment definition was used."],
    ["六自由度与姿态预览", "Six degrees of freedom and pose preview"],
    ["J1–J6 描述机械臂六个关节的姿态自由度。Slider 改变的是前端预览状态，不等于求解碰撞、IK 可达性或真实执行安全性。", "J1–J6 describe the six joint degrees of freedom. A slider changes preview state; it does not solve collision, IK reachability or real execution safety."],
  ],
  "parameter-sandbox": [
    ["参数为什么需要 Schema", "Why parameters use a schema"],
    ["Schema 把参数名称、类型、默认值、范围和是否必填变成机器可校验的契约。它防止页面提交服务端无法解释的值，也让不同 Run 的配置可以比较。", "A schema turns parameter names, types, defaults, ranges and requiredness into a machine-checkable contract. It prevents values the server cannot interpret and makes Run configurations comparable."],
    ["训练规模和记录参数的区别", "Training scale versus recording parameters"],
    ["Rollout、Updates 和 Episode 影响训练数据量与优化过程；Video、Camera、FPS 和 Resolution 影响记录方式。记录更详细不等于策略训练得更好。", "Rollout, Updates and Episode affect training data and optimization. Video, Camera, FPS and Resolution affect recording. More detailed recording does not mean a better-trained policy."],
    ["PPO 中 Actor 和 Critic 做什么", "What Actor and Critic do in PPO"],
    ["Actor 根据 observation 产生 action 分布；Critic 估计当前状态的价值，帮助计算优势并稳定更新。页面中的蓝图用于理解职责，真正的网络和优化器由服务端训练实现负责。", "The Actor produces an action distribution from observations. The Critic estimates state value to help compute advantages and stabilize updates. The blueprint explains responsibilities; the server owns the executable network and optimizer."],
    ["Rollout、Update 和 Epoch 的关系", "How rollout, update and epoch relate"],
    ["Rollout 收集一批轨迹；Update 使用这批数据更新策略；Update Epoch 表示同一批数据被重复用于优化的次数。增加规模会改变训练成本和证据解释，不能简单当作“效果一定更好”。", "A rollout collects trajectories. An update optimizes the policy with that batch. Update epochs control how many optimization passes use the same batch. Larger values change cost and interpretation; they do not guarantee better results."],
  ],
  "launch-experiment": [
    ["为什么正式 Run 需要教学上下文", "Why a formal Run needs teaching context"],
    ["组织、课程和教学班决定谁可以使用哪个 Project，以及一次运行如何归档。Project 冻结配置则决定算法、任务和评估规则。", "Organization, course and class determine which Project is available and how a Run is archived. The frozen Project configuration determines algorithm, task and evaluation rules."],
    ["Run ID 是审计入口", "Why the Run ID matters"],
    ["run_id 连接状态、事件、指标、评估和制品。没有唯一 Run ID，就无法把后续结论可靠地归因到一次具体执行。", "The run_id connects status, events, metrics, evaluation and artifacts. Without a unique Run ID, later conclusions cannot be reliably attributed to one execution."],
  ],
  "run-analysis": [
    ["事件和指标的区别", "Events versus metrics"],
    ["事件描述发生了什么以及发生顺序；指标描述训练或任务表现的数值。事件可以证明状态经过了某个阶段，不能自动替代成功率或距离等指标。", "Events describe what happened and in which order. Metrics describe numeric training or task behavior. An event can prove a state transition but cannot replace success rate or distance metrics."],
    ["为什么前端不补零", "Why the UI does not fill missing values with zero"],
    ["缺失指标可能表示服务没有产生、没有回传或当前任务不适用。把缺失当作 0 会把“未知”错误地变成“失败”。", "A missing metric may mean it was not produced, not returned or not applicable. Treating missing as zero incorrectly turns unknown into failure."],
  ],
  evaluation: [
    ["评估不是重新训练", "Evaluation is not retraining"],
    ["评估读取已完成 Run 的持久化证据，按照 rubric 生成报告。它不修改策略、不补造证据，也不因为 AI 的自然语言解释而改变正式结果。", "Evaluation reads persisted evidence from a completed Run and applies the rubric. It does not modify the policy, invent evidence or change the formal result because of AI commentary."],
    ["证据不足的科学含义", "What insufficient evidence means"],
    ["insufficient_evidence 是一个有效结论：当前数据不足以支持可信分数。它提示下一次实验需要补充事件、指标、重复运行或评估条件。", "insufficient_evidence is a valid conclusion: the current data cannot support a trustworthy score. It indicates that future work needs more events, metrics, repetitions or evaluation conditions."],
  ],
  artifacts: [
    ["Canonical Evidence 的作用", "What Canonical Evidence is for"],
    ["Canonical JSON 把 Run 的关键状态、参数和指标以可保存格式固定下来，便于审计、提交、复现和跨时间比较。", "Canonical JSON fixes key Run state, parameters and metrics in a storable format for audit, submission, reproduction and comparison over time."],
    ["导出与控制授权的区别", "Export versus control authorization"],
    ["导出的文件证明某次实验记录可以被保存，不证明硬件存在、策略安全或操作者已经获得真实控制授权。", "An export proves that an experiment record can be saved. It does not prove hardware exists, policy safety or operator authorization for real control."],
  ],
  "teaching-agent": [
    ["AI 在系统中的角色", "The role of AI in the system"],
    ["文档和服务端状态是受控事实来源，AI 负责把这些事实翻译成适合当前学习者的问题解释。AI 不应替代 Project 配置、正式评分或权限系统。", "The guide and server state are controlled sources of truth. AI translates them into explanations for the learner. It must not replace Project configuration, formal scoring or permissions."],
    ["为什么要限制上下文", "Why context is restricted"],
    ["只传递当前页面、必要标识和当前章节，可以减少隐私泄露、避免把 API Key 带入对话，也能让回答围绕当前任务而不是泛泛扫描源码。", "Passing only the current page, necessary identifiers and current guide chapter reduces privacy exposure, keeps API keys out of chat and keeps answers focused instead of scanning source indiscriminately."],
  ],
  troubleshooting: [
    ["先分类，再处理", "Classify before fixing"],
    ["Loading、权限拒绝、参数校验失败、Run 失败和证据不足是不同问题。先看错误位置和服务端状态，再决定是等待、修改配置、重新提交还是补证据。", "Loading, permission denial, parameter validation, Run failure and insufficient evidence are different problems. Check the error location and server state before deciding whether to wait, change configuration, resubmit or collect evidence."],
    ["为什么不能只看按钮颜色", "Why button color is not enough"],
    ["按钮 disabled 通常是业务契约的可视化结果。真正原因在附近的权限、匹配、状态或边界说明中，而不是颜色本身。", "A disabled button usually visualizes a business contract. The actual reason is in nearby permission, match, status or boundary information, not the color itself."],
  ],
  glossary: [
    ["这些术语为什么必须稳定", "Why these terms must stay stable"],
    ["相同词汇对应相同对象，才能让学习记录、Run 报告和 Agent 对话互相引用。术语变化应随文档版本发布，而不是由 AI 临时改写。", "Stable terms let learning records, Run reports and Agent conversations refer to the same objects. Terminology changes should be versioned with the guide, not rewritten by AI at runtime."],
  ],
};

const section = (id: string, title: [string, string], summary: [string, string], audience: [string, string], steps: GuideStep[], checks: [string, string][], boundaries: [string, string][], route?: string, tourId?: string): GuideSection => ({ id, title, summary, audience, steps, checks, boundaries, route, tourId });

export const USER_GUIDE_SECTIONS: GuideSection[] = [
  section("task-skills", ["任务与技能", "Tasks and skills"], ["选择强化学习任务或控制演示，阅读任务条件并检查运行证据。", "Choose RL tasks or control demos, read task conditions and inspect Run evidence."], ["希望训练策略或观察固定控制动作的使用者。", "Users training policies or observing fixed control actions."], [
    { title: ["查看训练任务", "Browse training tasks"], body: ["只有服务端目录声明可训练的任务才能进入课程选择匹配 Project，不能凭历史实验推断开放状态。在任务页选择组织、课程和教学班后，页面会检查匹配且可使用的 Project、配置版本及创建权限；缺少条件时显示原因。Grasp、Lift、Lower Release 的 Standard 配置由服务端解析，前端不覆盖 run_mode 或训练规模。", "Only server-declared training tasks proceed to matching teaching Projects. Select organization, course and class to check permitted Projects, versions and creation permission. Missing conditions are explained. The backend resolves Standard Grasp, Lift and Lower Release configuration; the frontend does not override run mode or training scale."] },
    { title: ["启动控制技能", "Launch control skills"], body: ["选择技能、组织、课程、教学班和可用 Short Push Project。校验参数后创建 play Run；不发送 scene_ref，不生成训练权重。", "Choose a skill, organization, course, class and available Short Push Project. Validate parameters and create a play Run, without scene_ref or training weights."] },
    { title: ["规则抓取放下", "Scripted pick and place"], body: ["demo_s1_scripted_pick_place 已作为固定控制 Demo 注册。控制器按规则执行对齐、抓取、抬起、放下、松爪和稳定性检查；使用 play Run，不学习策略。视频是否开启、时长和合法参数范围来自当前目录，实际阶段、失败原因和录像在 Run 详情检查。它独立于正式 s1_place 搬运保持训练。", "demo_s1_scripted_pick_place is a registered fixed-control Demo. It aligns, grasps, lifts, lowers, releases and checks settling in a play Run, without learning. Video options, duration and valid ranges come from the catalog; inspect stages, failures and recordings in Run details. It is separate from formal s1_place transport-and-hold training."] },
    { title: ["正式 Place 与完整动作流程", "Formal Place and the full workflow"], body: ["正式 s1_place 已接入目录与工作台：从已抓取抬起状态进行 XY 搬运，并在目标保持一秒。它不是完整的抓取、下降、松爪与撤离流程。开始训练仍需匹配的教学 Project、权限和 GPU；正式接线与静态测试通过不等于成功率认证。", "Formal s1_place is wired into the catalog and workbench: XY transport from a validated lifted grasp followed by a one-second target hold. It is not the complete grasp, lower, release and retreat workflow. Training still requires a matching Project, permission and GPU. Formal wiring and static tests do not certify success rates."] },
    { title: ["检查任务说明和证据", "Inspect task details and evidence"], body: ["展开训练任务详情，阅读起始条件、学习目标、成功证据与参数输出，并打开账号已授权的关联 Run。提交后在 Run 详情查看直播、录像、指标和停止操作。未注册实验任务仅供了解，不开放训练。", "Expand task details for entry conditions, objectives, success evidence, parameters and outputs, and open authorized related Runs. Run details provide video, metrics and stop controls. Unregistered tasks offer no training action."] },
    { title: ["理解完整操作流程", "Understand complete manipulation workflows"], body: ["组合原理见本章说明。后端需要具备流程保存、阶段兼容校验、共享环境连续执行和逐阶段证据记录，才适合开放任务编排。", "See composition principles in this chapter. Orchestration requires backend workflow saving, stage compatibility validation, continuous shared-world execution and per-stage evidence."] },
  ], [["服务目录可用", "Service catalog available"], ["Project、班级和权限匹配", "Project, class and permissions match"]], [["Demo 不是策略训练。", "Demo is not policy training."], ["流程说明不代表已提供组合执行。", "Workflow explanations do not imply executable orchestration."], ["真实控制始终禁用。", "Real control stays disabled."]], "/skills", "tasks-skills"),
  section("overview", ["平台总览", "Platform overview"], ["理解 Embodied Arm 的研究流程、能力边界和每个工作区的职责。", "Understand the research flow, capability boundaries and responsibilities of each workspace."], ["第一次使用平台的学习者、教师和研究人员。", "New learners, instructors and researchers."], [
    { title: ["平台解决什么问题", "What the platform does"], body: ["Embodied Arm 是面向具身智能机械臂实验的研究与教学工作台。它把学习路线、项目文件、六自由度场景、训练参数、正式 Run、证据评估和运行制品放在同一条可追溯流程中。", "Embodied Arm is a research and teaching workspace for embodied robotic-arm experiments. It connects learning, project files, six-DoF scenes, training parameters, formal Runs, evidence evaluation and artifacts in one traceable flow."] },
    { title: ["推荐使用顺序", "Recommended order"], body: ["先进入学习路线理解 01–06 阶段；然后在项目工作区确认任务路线；在 S1 环境编辑中保存 Scene Revision；在参数沙箱校验任务和参数；最后从课程创建正式 Run，再进入运行分析、策略评估和运行制品。", "Start with the learning roadmap and its stages 01–06. Then choose a task in Project Workspace, save a Scene Revision in S1 Environment, validate the task and parameters, create a formal Run from Courses, and finish with Run Analysis, Evaluation and Artifacts."] },
    { title: ["页面引导和完整文档的分工", "Tour versus full guide"], body: ["页面引导解释当前页面的真实控件，完整文档解释概念、前置条件、完成标准、限制和跨页面流程。", "Page Tour points to real controls. The full guide explains concepts, prerequisites, completion criteria, limits and cross-page workflows."] },
  ], [["身份已加载", "Your identity is loaded"], ["知道当前实验属于哪个 Project、Scene 和 Run", "You know which Project, Scene and Run your experiment belongs to"]], [["仿真能力不等于真实机械臂控制权限。", "Simulation capability never grants real-robot control permission."], ["教学草稿不等于服务端正在执行的训练代码。", "Teaching drafts are not the training code executed by the server."]], "/roadmap", "platform-introduction"),
  section("workflow", ["完整实验流程", "Complete experiment workflow"], ["从学习、实验到分析，了解一次可复现实验如何产生证据。", "Understand how a reproducible experiment produces evidence from learning through analysis."], ["需要完成一次完整 S1 教学实验的使用者。", "Users completing a full S1 teaching experiment."], [
    { title: ["01–02：理解项目和环境", "01–02: Understand the project and environment"], body: ["认识 Project、任务、机械臂、场景和安全边界，不要把控制 Demo 当作 PPO 训练任务。", "Understand Projects, tasks, the arm, scenes and safety boundaries. Do not treat control demos as PPO training tasks."] },
    { title: ["03–04：配置任务和训练", "03–04: Configure the task and training"], body: ["在实验工作台选择 Push、Reach 或 Place，确认教学班和匹配 Project，保存奖励版本并通过预检。", "Choose Push, Reach or Place in the workbench, select a class and matching Project, then save and preflight the reward revision."] },
    { title: ["05–06：创建、分析和评估 Run", "05–06: Create, analyze and evaluate the Run"], body: ["从实验工作台创建正式 Run，再通过后端日志、运行分析、策略评估和运行制品检查证据。", "Create the formal Run in the workbench, then inspect backend logs, Run Analysis, Evaluation and Artifacts."] },
  ], [["任务和 Project 的 environment_type 匹配", "The task matches the Project environment_type"], ["Run 有唯一 run_id 且状态来自服务端", "The Run has a unique run_id and a server-owned status"], ["评估结论引用实际证据", "The evaluation cites actual evidence"]], [["completed 不自动代表任务成功。", "completed does not automatically mean task success."], ["证据不足时不应推测分数。", "Insufficient evidence must not become a speculative score."]], "/roadmap", "learning-guide"),
  section("project-workspace", ["实验工作台", "Experiment workbench"], ["每次只改一项：配置实验、保存检查、启动训练，再查看结果。", "Change one thing at a time: configure, save and check, start training, then inspect results."], ["首次进行 Push、Reach 或 Place 实验的使用者。", "Users starting a Push, Reach or Place experiment."], [
    { title: ["确认任务、课程与训练方案", "Confirm task, class and training plan"], body: ["顶部选择推物、平面到达或搬运放置。唯一合法的课程、教学班或方案自动选中；有多个训练方案时到训练设置明确选择。入门方案只验证流程，正式方案用于观察学习。预算来自课程版本。", "Choose Push, Reach or Place above. Single eligible courses, classes and plans are selected automatically; explicitly choose among multiple plans under Training settings. The introductory plan checks the workflow; the formal plan is for learning. The course revision controls the budget."] },
    { title: ["第一次只改一个奖励权重", "Change one reward weight first"], body: ["建议先保持默认场景、预算与种子，将动作惩罚权重从 1 调到 2。卡片显示行为含义和默认值，本次改动汇总差异。展开计算说明后，同一步教学示例小计从 0.03 变为 0.02；示例不是实际训练结果。", "Keep the default scene, budget and seed, and change the action penalty weight from 1 to 2. Cards explain behavior and defaults; Changes in this draft summarizes differences. Expand the calculation to see the illustrative same-step subtotal change from 0.03 to 0.02. This is not measured training evidence."] },
    { title: ["需要时修改场景和训练设置", "Edit scene and settings when needed"], body: ["Push 可修改方块初始 XY、目标 XY 和各边 6–10 厘米的尺寸，示意图即时更新，训练使用已保存并校验的场景。Reach 可改目标采样和动作步长；Place 从已抓取抬起状态搬运，场景固定。随机种子与录像在训练设置中，高级录像默认收起。", "Push supports initial XY, goal XY and dimensions of 6–10 cm per edge. The diagram updates immediately; training uses the saved and validated scene. Reach supports target sampling and motion steps. Place transports from a lifted grasp with a fixed scene. Seed and video are under Training settings; advanced recording is initially collapsed."] },
    { title: ["保存并检查，然后启动训练", "Save and check, then start training"], body: ["底部固定操作栏始终可见。保存并检查依次保存必要的场景、奖励并验证表达式；失败时停止，不会自动训练。检查详情提供缺少条件的入口。启动前核对预算、种子和奖励版本。静态检查不执行物理仿真，不保证学习成功率。", "The persistent bottom bar stays visible. Save and check saves the required scene and reward and validates the expression in sequence. Failure stops preparation; training never starts automatically. Check details link to unmet conditions. Review budget, seed and reward revision before launch. Static checks do not run physics or guarantee learning success."] },
    { title: ["查看结果，与历史实验对照", "Inspect results and compare experiments"], body: ["启动后自动显示进度与结果摘要。已有实验与结果里可以选择历史训练，它不会覆盖正在编辑的方案。已完成只代表流程完成；正式成功率和最终距离来自评估，缺失时显示未记录。展开对比，选择对照实验，核对场景、预算、种子、任务参数与奖励版本。平均训练奖励在奖励定义不同时不能直接排名；单次比较不证明因果。完整曲线、视频、制品与停止操作在实验详情和分析页。", "After launch, progress and outcome summaries appear automatically. Choose a recorded training run under Recorded experiments and results; this leaves the editing draft unchanged. Completed describes process completion. Success rate and final distance come from formal evaluation; missing values remain Not recorded. Expand comparison and choose a reference, checking scene, budget, seed, task parameters and reward version. Different reward definitions prevent directly ranking training rewards. One comparison does not establish causality. Full charts, video, artifacts and stop controls are in details and analysis."] },
    { title: ["按需查看源码、助手和修改记录", "Use source, assistant and history when needed"], body: ["参考源码打开只读文件浏览器，保留搜索与完整文件滚动。查看同步代码可展开奖励预览，进阶代码编辑保留高亮 return 表达式标记。卡片和可拖动悬浮球打开同一个实验助手，收起后保留当前上下文。奖励修改记录显示保存时间；先核对记录，再确认恢复为新版本。历史接口未记录完整改动摘要时不会编造内容。", "Reference source opens the read-only browser with search and complete file scrolling. View synchronized code expands the reward preview; Advanced code editing retains the highlighted return expression. Cards and the draggable floating button open the same experiment assistant, retaining context when collapsed. Reward history shows save times; review a record before restoring it as a new revision. Full change summaries absent from the history API are not invented."] },
  ], [["能说清本次改动与训练预算", "Explain the change and training budget"], ["能检查并启动实验", "Check and launch an experiment"], ["能区分教学示例、训练完成与正式评估", "Distinguish illustrations, process completion and formal evaluation"]], [["PPO、成功标准、安全终止与真实控制边界保持锁定。", "PPO, success criteria, safety termination and real-control boundaries stay locked."]], "/projects", "experiment-workbench"),
  section("scene-studio", ["S1 六自由度场景设计", "S1 six-DoF scene design"], ["创建可追溯的 Scene Revision，并用真实 J1–J6 控件预览姿态。", "Create a traceable Scene Revision and preview poses with the real J1–J6 controls."], ["需要准备 Push 或 Reach 场景的使用者。", "Users preparing a Push or Reach scene."], [
    { title: ["选择版本化场景", "Choose a versioned scene"], body: ["从左侧选择六轴 S1 场景，确认 REV 号。", "Choose a six-axis S1 scene and confirm its REV number."] },
    { title: ["预览 3D Scene Studio", "Preview the 3D Scene Studio"], body: ["3D 画面用于查看视觉资产和对象位置，不是碰撞、IK 或真实控制验证。", "The 3D view previews visual assets and object positions; it is not collision, IK or real-control validation."] },
    { title: ["编辑对象并保存 Revision", "Edit objects and save a Revision"], body: ["添加对象仅用于预览，修改后保存修订。任意障碍物尚未映射为正式 Worker 几何，将被场景预检阻止提交。", "Added objects are preview-only. Save a revision after edits. Arbitrary obstacles without Worker geometry translation are blocked by scene preflight."] },
    { title: ["调整 J1–J6", "Adjust J1–J6"], body: ["J1–J6 是真实业务 Slider。J2 引导监听真实 value change，不创建假控件。", "J1–J6 are real business sliders. The J2 tour listens for the real value change and does not create a duplicate control."] },
    { title: ["进入参数沙箱", "Continue to parameters"], body: ["保存 Scene Revision 后进入参数沙箱。", "After saving the Scene Revision, continue to the Parameter Sandbox."] },
  ], [["Scene Revision 已保存", "The Scene Revision is saved"], ["对象、关节姿态和控制边界已检查", "Objects, joint pose and boundaries are checked"]], [["视觉资产不代表碰撞安全或可达性结论。", "Visual assets do not prove collision safety or reachability."], ["真实控制始终禁用。", "Real control remains disabled."]], "/environments", "scene-studio"),
  section("parameter-sandbox", ["实验参数与预检", "Experiment parameters and preflight"], ["参数配置已整合到实验工作台；旧 /training 入口打开同一工作台。", "Parameters are integrated into the experiment workbench. The legacy /training route opens the same workbench."], ["正在配置实验的使用者。", "Users configuring an experiment."], [{"title": ["打开实验修改入口", "Open experiment controls"], "body": ["在工作台上方打开任务场景或训练设置，检查服务端目录允许的参数、范围和校验。", "Open Task scene or Training settings at the top to review server catalog parameters, ranges and validation."]}, {"title": ["检查 Project 和奖励版本", "Check the Project and reward revision"], "body": ["确认教学班，在训练设置中选择匹配方案，再点击保存并检查。启动按钮上方说明缺少的条件。", "Confirm your class, choose a matching training plan under Training settings, then use Save and check. The note above Start training explains missing conditions."]}], [["服务端目录校验通过", "Server catalog validation passes"], ["奖励版本已保存且预检通过", "Reward revision is saved and passes preflight"]], [["Project 快照决定训练预算。", "The Project snapshot determines the training budget."]], "/training", "experiment-workbench"),
  section("launch-experiment", ["启动正式实验", "Launch a formal experiment"], ["在组织、课程、教学班和 Project 上下文中创建可追溯 Simulation Run。", "Create a traceable Simulation Run inside the organization, course, class and Project context."], ["参数已校验、准备提交训练的使用者。", "Users with validated parameters ready to submit training."], [
    { title: ["选择教学上下文", "Choose the teaching context"], body: ["选择组织、课程和教学班，确认权限和课程状态。", "Choose organization, course and class, then confirm permissions and course state."] },
    { title: ["选择训练方案", "Choose a training plan"], body: ["在训练设置中选择课程授权的方案，核对预算。入门方案验证流程，正式训练方案观察学习；比较时保持预算一致。", "Choose a course-authorized plan under Training settings and check its budget. Introductory plans verify the workflow; formal training observes learning. Match budgets for comparisons."] },
    { title: ["保存并检查", "Save and check"], body: ["核对任务、场景、奖励和种子，再点击保存并检查。操作栏解释缺少的条件；任何一步失败都不会自动启动。", "Check the task, scene, reward and seed, then use Save and check. The action bar explains unmet conditions; any failed step prevents an automatic launch."] },
    { title: ["创建 Run", "Create the Run"], body: ["开始训练后服务端返回 run_id；刷新浏览器不会停止运行。", "Starting training returns a run_id; refreshing does not stop the Run."] },
  ], [["拿到 run_id", "You received a run_id"], ["状态来自服务端", "Status is server-owned"]], [["completed 不自动表示任务成功。", "completed does not mean task success."], ["Simulation Run 不获得真实控制权限。", "A Simulation Run does not grant real-control permission."]], "/projects", "experiment-workbench"),
  section("run-analysis", ["运行分析与事件证据", "Run Analysis and event evidence"], ["通过 Run List、冻结配置和 Event Evidence 判断发生了什么。", "Use the Run List, frozen configuration and Event Evidence to understand what happened."], ["需要解释训练状态、指标和异常的使用者。", "Users interpreting training status, metrics and anomalies."], [
    { title: ["选择 Run", "Select a Run"], body: ["选择 Run 后查看状态、Project、seed 和更新时间。", "Select a Run to inspect status, Project, seed and timestamps."] },
    { title: ["删除已结束的记录", "Delete terminal records"], body: ["勾选删除列或选择前 10 条已结束运行，再点击删除选中并确认。只能删除自己的 completed、failed、stopped 运行；每批最多 10 条。整批成功后列表更新，记录、事件、评估和制品索引不可恢复；服务器上的制品文件不在此次清理范围。", "Use the deletion checkboxes or select the first 10 terminal runs, then confirm Delete selected. Only your completed, failed or stopped runs can be deleted, with up to 10 per batch. After the entire batch succeeds, the list updates. Records, events, evaluations and artifact indexes cannot be restored; server artifact files are retained."] },
    { title: ["阅读冻结配置", "Read frozen configuration"], body: ["冻结配置是创建 Run 时的服务端快照，适合复现和审计。", "Frozen configuration is the server snapshot for reproduction and audit."] },
    { title: ["阅读事件证据", "Read event evidence"], body: ["检查 CREATED、QUEUED、RUNNING、终态和异常事件。", "Review CREATED, QUEUED, RUNNING, terminal and anomalous events."] },
    { title: ["进入评估或导出", "Continue to evaluation or export"], body: ["终态 Run 可评估，完成 Run 可导出，两者只使用服务端证据。", "Terminal Runs can be evaluated and completed Runs exported using server evidence only."] },
  ], [["状态和事件顺序合理", "Status and event order are coherent"], ["知道哪些指标真实存在", "You know which metrics exist"]], [["单次最佳结果不能替代复现性判断。", "One best result cannot replace reproducibility."], ["Simulation evidence 不能证明真实机械臂安全。", "Simulation evidence does not prove real-robot safety."]], "/runs", "run-evidence"),
  section("evaluation", ["策略与证据评估", "Policy and evidence evaluation"], ["从终态 Run 生成不可变评估报告，并理解证据不足。", "Generate an immutable report from a terminal Run and understand insufficient evidence."], ["需要比较策略质量或记录教学结论的使用者。", "Users comparing policy quality or recording a teaching conclusion."], [
    { title: ["选择评估对象", "Choose the evaluation target"], body: ["只选择终态 Run，并核对 Project、班级和 seed。", "Choose a terminal Run and check Project, class and seed."] },
    { title: ["生成不可变报告", "Generate the immutable report"], body: ["服务端依据持久化 Run 证据生成报告，AI 点评不会改变报告。", "The server generates the report from persisted evidence; AI commentary does not change it."] },
    { title: ["阅读限制", "Read limitations"], body: ["insufficient_evidence 表示不能给出可信分数，不是前端故障。", "insufficient_evidence means a trustworthy score cannot be given; it is not a UI failure."] },
    { title: ["使用 AI 点评", "Use AI commentary"], body: ["AI 只解释选中的报告和 Run 证据，不参与正式评分。", "AI explains the selected report and evidence; it does not score."] },
  ], [["报告绑定唯一 Run", "The report binds to one Run"], ["证据和限制已阅读", "Evidence and limitations are reviewed"]], [["评估不等于控制授权。", "Evaluation is not control authorization."], ["Sim2Real remains blocked.", "Sim2Real remains blocked."]], "/evaluation", "evaluation"),
  section("artifacts", ["运行制品与 JSON 记录", "Run artifacts and JSON records"], ["选择完成 Run，查看 Canonical Evidence 并导出正式记录。", "Choose a completed Run, inspect Canonical Evidence and export the formal record."], ["需要保存、审计或提交实验结果的使用者。", "Users saving, auditing or submitting experiment results."], [
    { title: ["选择已完成 Run", "Choose a completed Run"], body: ["选择服务端确认完成的 Run，核对 Project、class 和 status。", "Choose a server-confirmed completed Run and check Project, class and status."] },
    { title: ["阅读 Canonical Evidence", "Read Canonical Evidence"], body: ["JSON Viewer 展示服务端 Run、parameters 和 metrics_summary。", "JSON Viewer shows the server Run, parameters and metrics_summary."] },
    { title: ["下载 JSON", "Download JSON"], body: ["下载服务端生成的原始记录，并保留 Run ID 和文档版本。", "Download the server-generated record and keep the Run ID and guide version."] },
  ], [["导出对象是正确 Run", "The exported object is the correct Run"], ["JSON 下载成功", "The JSON download succeeds"]], [["导出不代表真实机械臂控制许可。", "Export does not grant real-robot control permission."], ["不要把 Simulation 结果描述成真实部署结果。", "Do not describe Simulation results as real deployment results."]], "/export", "artifacts"),
  section("teaching-agent", ["Teaching Agent 使用说明", "Using the Teaching Agent"], ["了解 Provider、帮助等级、页面上下文和安全边界。", "Understand providers, assistance levels, page context and safety boundaries."], ["需要获得页面解释、证据解读或下一步建议的使用者。", "Users who need page explanations, evidence interpretation or next steps."], [
    { title: ["选择 Provider", "Choose a Provider"], body: ["Mock 不发送外部请求。外部 Provider 配置后用固定测试消息验证，只有有效回复才标记已验证；失败显示可排查原因。网页不保存或回显 API Key。", "Mock sends no external request. External providers are verified with a fixed test message; only a valid reply marks them verified. Failures show actionable reasons. The browser never stores or echoes API keys."] },
    { title: ["选择帮助等级", "Choose an assistance level"], body: ["帮助等级决定可读取的受限上下文，但不会授予 Shell 或真实控制权限。", "Assistance level controls restricted context but never grants Shell or real-control access."] },
    { title: ["询问当前页面", "Ask about the current page"], body: ["Agent 会携带当前路由以及已知的 Project、Run、Scene 标识。悬浮球支持拖动并记住位置；点击打开或收起。", "The Agent receives the route and known Project, Run and Scene identifiers. Drag its floating button to reposition it; click to open or collapse. The position is remembered."] },
    { title: ["结合完整文档学习", "Use the full guide with the Agent"], body: ["文档是受控来源，AI 是个性化解释层。", "The guide is the controlled source; AI is the personalized explanation layer."] },
  ], [["问题包含当前页面或 Run 上下文", "The question includes current context"], ["回答没有超出文档和权限", "The answer stays within the guide and permissions"]], [["Agent 不会执行 Shell，也不会直接控制机器人。", "The Agent does not execute Shell or directly control a robot."], ["AI 建议不能替代正式评分。", "AI suggestions do not replace formal scores."]], "/agent", "teaching-agent"),
  section("troubleshooting", ["常见问题排查", "Troubleshooting"], ["从加载、权限、校验、Run 状态和证据缺口定位问题。", "Locate loading, permission, validation, Run and evidence problems."], ["遇到页面错误或按钮不可用的使用者。", "Users encountering errors or unavailable controls."], [
    { title: ["页面一直 Loading", "A page stays loading"], body: ["确认 API 服务和身份有效，再刷新。", "Confirm the API service and session, then refresh."] },
    { title: ["按钮不可用", "A button is unavailable"], body: ["检查组织、课程、Project、Scene、权限和 Run 状态，不要绕过 disabled。", "Check organization, course, Project, Scene, permissions and Run status; do not bypass disabled state."] },
    { title: ["没有评估分数", "No evaluation score"], body: ["检查 Run 是否终态、事件和指标是否完整。", "Check whether the Run is terminal and evidence is complete."] },
    { title: ["Agent 无法回答", "The Agent cannot answer"], body: ["在 Agent 设置中配置并验证连接。待验证不等于可连接；按错误检查 Key、余额、模型 ID、API 地址与服务器出站网络。", "Configure and verify the connection in Agent settings. Untested is not connected. Check the reported key, balance, model ID, API URL and server network errors."] },
  ], [["能区分前端等待、权限拒绝和证据不足", "You can distinguish loading, permission and insufficient evidence"]], [["不要在排查信息中包含 API Key 或 Authorization header。", "Do not include API keys or Authorization headers in diagnostics."], ["真实控制问题必须由硬件授权流程处理。", "Real-control issues require the hardware authorization process."]]),
  section("glossary", ["术语表", "Glossary"], ["统一理解 Project、Scene、Run、Evidence 和 Sim2Real。", "Use consistent meanings for Project, Scene, Run, Evidence and Sim2Real."], ["所有平台使用者。", "All platform users."], [
    { title: ["Project", "Project"], body: ["服务端定义的教学或实验配置，包含任务、参数 Schema 和 rubric。", "A server-defined configuration containing task, parameter schema and rubric."] },
    { title: ["Scene Revision", "Scene Revision"], body: ["版本化场景记录，建立实验环境的可追溯关系。", "A versioned scene record that makes the environment traceable."] },
    { title: ["Run", "Run"], body: ["一次运行的服务端记录，包含状态、配置、事件、指标和制品。", "The server record of an execution, including status, configuration, events, metrics and artifacts."] },
    { title: ["Evidence", "Evidence"], body: ["来自 Run、支持评估结论的事件、指标、配置和制品。", "Events, metrics, configuration and artifacts from a Run supporting evaluation."] },
    { title: ["Sim2Real", "Sim2Real"], body: ["仿真到真实系统的迁移边界，本平台当前保持 BLOCKED。", "The simulation-to-real boundary, currently BLOCKED on this platform."] },
  ], [["能准确使用这些术语描述实验", "You can use these terms accurately"]], [["术语不能扩大平台当前能力声明。", "Terms must not expand declared capabilities."]]),
];

export const USER_GUIDE_BY_ID = Object.fromEntries(USER_GUIDE_SECTIONS.map((item) => [item.id, item])) as Record<string, GuideSection>;
export const guideForPath = (path: string): GuideSection => {
  const pathname = path.split(/[?#]/, 1)[0];
  if (pathname.startsWith("/guide/")) return USER_GUIDE_BY_ID[pathname.slice("/guide/".length)] ?? USER_GUIDE_BY_ID.overview;
  if (pathname === "/training") return USER_GUIDE_BY_ID["project-workspace"];
  if (pathname === "/training-code") return USER_GUIDE_BY_ID["project-workspace"];
  if (pathname.startsWith("/runs/")) return USER_GUIDE_BY_ID["run-analysis"];
  return USER_GUIDE_SECTIONS.find((item) => item.route === pathname) ?? USER_GUIDE_BY_ID.overview;
};
export const guidePathForSection = (sectionId: string) => `/guide/${sectionId}`;
