import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export type Locale = "zh-CN" | "en";

export const LOCALE_STORAGE_KEY = "embodied-arm-studio-locale";

const EXACT_TRANSLATIONS: Record<string, string> = {
  "Verified evidence · 已记录证据": "Verified evidence",
  "REQUIRED EVIDENCE · 完成证据": "Required evidence",
  "CAPABILITY BOUNDARY · 能力边界": "Capability boundary",
  "当前动力学为教学近似，不是数字孪生或物理精确仿真。Pipeline、复现实验与完整评估仍按真实运行证据进行。": "The current dynamics are a teaching approximation, not a digital twin or physically accurate simulation. Pipeline execution, replication, and full evaluation still rely on recorded run evidence.",
  "Mock 参数实验": "Mock parameter experiment",
  "GPU Worker 验证": "GPU worker verification",
  "训练提供方": "Training provider",
  "运行 Mock 实验": "Run Mock experiment",
  "验证 GPU Worker": "Verify GPU worker",
  "事件": "Events",
  "运行记录 API 不可用，无法建立证据评估。": "The run records API is unavailable, so an evidence-based evaluation cannot be created.",
  "先完成一次正式教学短训练。评估中心只读取实际运行证据，不会生成推测分数。": "Complete a formal short training run first. The evaluation center reads recorded run evidence and does not invent scores.",
  "每项结论都绑定到 Run 事件或制品。选择同环境、同策略的多个 seed 可评估同一运行配置的复现性；这仍不能证明泛化或真实部署安全。": "Each conclusion is linked to Run events or artifacts. Multiple seeds with the same environment and policy can test the reproducibility of a run configuration; they do not establish generalization or safe deployment on a real robot.",
  "等待生成报告": "Waiting to generate report",
  "选择现有运行并生成报告": "Select recorded runs and generate a report",
  "Evaluator 将检查实际可用通道。缺失复现性、泛化或安全证据时不会补造分数。": "The evaluator checks the available evidence channels. It does not invent scores when reproducibility, generalization, or safety evidence is missing.",
  "在运行记录中查看证据": "View evidence in run records",
  "基线评估报告": "Baseline evaluation report",
  "选择同环境历史报告": "Select a past report for the same environment",
  "完整度": "Completeness",
  "总分": "Overall score",
  "下一次需补证据": "Evidence needed in the next experiment",
  "完成一个 Mock 训练运行后，才能生成教学 Manifest 或提交模型格式门禁检查。": "Complete a Mock training run before generating a teaching manifest or requesting model-format validation.",
  "进入训练": "Go to training",
  "导出只封装当前 Mock 运行的策略契约、关节顺序、限制与校验和。模型格式必须关联通过证据门禁的持久化报告，任何结果都不会启用真实机械臂控制。": "Export packages the current Mock run's policy contract, joint order, limits, and checksums. Model formats require a persisted report that passes evidence validation. Export does not enable physical robot control.",
  "完成运行": "Completed run",
  "评估报告": "Evaluation report",
  "导出格式": "Export format",
  "Manifest 可在无报告时生成。ONNX 和 TorchScript 请求必须关联满足完整度、总评和 mandatory criteria 的报告。": "A manifest can be generated without a report. ONNX and TorchScript requests must reference a report that meets the completeness, overall score, and mandatory criteria requirements.",
  "通过": "Passed",
  "未通过": "Failed",
  "面向可复现实验的训练入口。编辑器支持 Python 语法内容保存和安全静态检查，服务端不会执行提交代码。": "Training entry point for reproducible experiments. The editor saves Python source and performs static checks; the server does not execute the submitted code.",
  "绑定场景": "Linked scene",
  "Demo 执行结果": "Demo execution result",
  "策略评估证据": "Policy evaluation evidence",
  "Demo 不训练或评估策略；没有 Policy 证据不代表失败。": "Demos do not train or evaluate policies. Missing policy evidence does not indicate failure.",
  "该运行未生成正式评估。Run completed 不代表 Policy passed，也不会显示为 0%。": "This run has no formal evaluation. A completed Run does not mean the policy passed, and missing results are not shown as 0%.",
  "固定评估 seed 中明确成功的回合数 / 已完成评估回合数": "Explicitly successful episodes / completed evaluation episodes using fixed evaluation seeds",
  "成功回合": "Successful episodes",
  "二项成功率的 Wilson score 95% 置信区间": "95% Wilson score confidence interval for a binomial success rate",
  "区间": "Interval",
  "单位": "Unit",
  "仅使用 metrics_summary.evaluation 中的持久化确定性证据。": "Uses only persisted deterministic evidence from metrics_summary.evaluation.",
  "正在读取实验工作台…": "Loading experiment workbench…",
  "回滚": "Roll back",
  "控制演示": "Control demos",
  "基线版本": "Baseline revision",
  "当前版本": "Current revision",
  "顶部": "Top",
  "文件末尾": "End of file",
  "入门方案用于检查流程；预算很短，不适合判断学习效果。": "The introductory plan checks the workflow. Its short budget is unsuitable for judging learning performance.",
  "正式训练方案用于观察策略学习。": "The formal training plan is intended for observing policy learning.",

  "入门学习": "Introduction to Embodied AI",
  "可用": "Available",
  "规划中": "Planned",
  "受限": "Restricted",
  "认识平台并按步骤理解一次六轴 S1 仿真实验": "Understand the platform and work through a six-axis S1 simulation experiment",
  "正在读取入门学习内容…": "Loading learning materials…",
  "正在保存，失败后重试": "Saving; retry if it fails",
  "S1 抓取": "S1 Grasp",
  "S1 抓取抬起": "S1 Grasp and Lift",
  "S1 抓取放下": "S1 Lower and Release",
  "S1 搬运放置": "S1 Transport and Place",
  "S1 平面到达": "S1 Reach",
  "S1 推动": "S1 Push",
  "API 服务器的出站代理连接被拒绝，请检查代理服务和端口": "The API server's outbound proxy refused the connection. Check the proxy service and port.",
  "模型供应商拒绝请求，请核对模型 ID 和 API 地址": "The provider rejected the request. Check the model ID and API URL.",
  "API Key 无效或已撤销，请重新配置": "The API key is invalid or revoked. Configure a valid key.",
  "模型账户余额不足，请检查供应商账户": "Insufficient provider account balance. Check your account.",
  "模型供应商拒绝访问，请检查密钥权限和地区限制": "Access denied. Check key permissions and regional restrictions.",
  "模型或 API 路径不存在，请核对模型 ID 和根地址": "Model or API path not found. Check the model ID and base URL.",
  "模型供应商不支持当前请求参数": "The provider does not support these request parameters.",
  "模型供应商请求限流，请稍后重试": "Provider rate limit reached. Try again later.",
  "模型供应商暂不可用，请稍后重试": "The provider is temporarily unavailable. Try again later.",
  "模型供应商请求超时，请检查地址和网络": "Provider request timed out. Check the URL and server network.",
  "模型供应商 TLS 证书验证失败，请检查 API 服务器证书配置": "Provider TLS certificate verification failed. Check the API server certificate configuration.",
  "API 服务器无法解析模型供应商域名，请检查 DNS": "The API server cannot resolve the provider hostname. Check DNS.",
  "API 服务器无法连接模型供应商，请检查出站网络和代理配置": "The API server cannot reach the provider. Check outbound networking and proxy configuration.",
  "静态契约与样例计算已通过；尚未执行物理仿真，也不保证训练成功率。": "Static contract and sample calculations passed. Physics and learning performance have not been verified.",
  "会将系统提示和当前会话消息发送到管理员或当前会话配置的 DeepSeek API。": "System instructions and conversation messages are sent to the configured DeepSeek API.",
  "当前仅支持 Push / Reach / Place 奖励实验": "Reward experiments support Push, Reach and Place only.",
  "搬运放置": "Transport and hold",
  "触达训练": "Reach training",
  "入门训练与演示": "Introductory training and demos",
  "推动训练": "Push training",
  "抓取训练": "Grasp training",
  "抓取抬起": "Grasp and lift",
  "抓取放下": "Lower and release",
  "读取奖励草稿失败": "Failed to load reward draft",
  "草稿已在其他窗口更新，请重新加载后对比。": "This draft changed in another window. Reload and compare revisions.",
  "兼容 API 地址必须是 HTTPS 根地址，不能包含 chat/completions": "Use an HTTPS API base URL without chat/completions, credentials, query parameters or fragments.",
  "无法解析兼容 API 地址": "The API server cannot resolve the base URL hostname.",
  "模型名称不能为空且不能超过 160 个字符": "The model ID must contain 1\u2013160 characters.",
  "兼容 API 地址不能指向本机或内网地址": "API URLs cannot point to local or private networks.",
  "API Key 不能为空": "The API key cannot be empty.",

  "学习路线与实验流程": "Learning path and experiment flow",
  "学习路线": "Learning Roadmap",
  "项目工作区": "Project Workspace",
  "实验工作台": "Experiment Workbench",
  "课程与高级任务": "Courses and advanced tasks",
  "任务与技能": "Tasks and skills",
  "S1 环境编辑": "S1 Environment",
  "参数沙箱": "Parameter Sandbox",
  "启动实验": "Launch Experiment",
  "运行分析": "Run Analysis",
  "运行结果对比": "Run Result Comparison",
  "Run 与当前任务不一致": "The Run does not match the current task",
  "Run 不存在或无读取权限": "The Run does not exist or you do not have read permission",
  "Run 证据暂不可用": "Run evidence is temporarily unavailable",
  "未知 Agent Provider": "Unknown Agent Provider",
  "场景不存在": "Scene not found",
  "训练代码不存在": "Training code not found",
  "未知课程": "Unknown course",
  "未知模型供应商": "Unknown model provider",
  "该 Provider 没有网页端配置": "This Provider has no web configuration",
  "未知 Agent 会话": "Unknown Agent session",
  "Patch 必须关联当前项目的有效 Agent 会话": "The Patch must be linked to a valid Agent session for the current Project",
  "当前 Assistance Level 不允许提出 Patch": "The current Assistance Level does not allow Patch proposals",
  "提出 Patch 前需提交完整的学生设计草案": "Submit a complete student design draft before proposing a Patch",
  "至少选择一个运行": "Select at least one Run",
  "运行结束后才能生成评估报告": "An evaluation report can be generated only after the Run ends",
  "评估报告不存在": "Evaluation report not found",
  "运行或评估报告不存在": "Run or evaluation report not found",
  "S1 Push · 推物": "S1 Push",
  "S1 Reach · 平面到达": "S1 Reach",
  "24 维：关节、末端/物体/目标相对位置和速度": "24D: joints and relative positions and velocities of the end effector, object, and goal",
  "18 维：关节状态、末端与目标位置及运动状态": "18D: joint state plus end-effector and goal positions and motion state",
  "二维末端 XY 位移；Z 和姿态固定": "2D end-effector XY displacement; Z and orientation are fixed",
  "二维末端 XY 位移；不是抓取": "2D end-effector XY displacement; this is not grasping",
  "simulation/src/simulation_core/reach_adapter.py（在线运行副本）": "simulation/src/simulation_core/reach_adapter.py (online runtime copy)",
  "策略评估": "Policy Evaluation",
  "运行制品": "Run Artifacts",
  "学习": "Learn",
  "实验": "Experiment",
  "分析": "Analyze",
  "教学 Agent": "Teaching Agent",
  "认识平台": "Understand the platform",
  "准备六轴场景": "Prepare the six-axis scene",
  "理解任务与策略": "Understand tasks and policies",
  "配置训练": "Configure training",
  "启动并观察 Run": "Start and observe the Run",
  "评估和复盘": "Evaluate and review",
  "先认清哪些内容会进入训练": "First identify what enters training",
  "先保存场景，再配置训练": "Save the scene before configuring training",
  "选择 Push 或 Reach，不混入控制 Demo": "Choose Push or Reach; do not mix in control demos",
  "项目版本决定正式训练约束": "The Project version defines formal training constraints",
  "从课程和教学班创建正式运行": "Create a formal Run from a course and class",
  "以证据决定下一次修改": "Use evidence to decide the next change",
  "S1 推动任务": "S1 Push task",
  "S1 平面到达任务": "S1 Planar Reach task",
  "S1 推物任务": "S1 Push task",
  "六轴场景": "Six-axis scene",
  "任务选择": "Task selection",
  "已保存的 S1 场景": "Saved S1 scene",
  "算法": "Algorithm",
  "训练档位": "Training profile",
  "前往 S1 环境编辑": "Open S1 Environment",
  "进入参数沙箱": "Open parameter sandbox",
  "打开对应工作区": "Open workspace",
  "两条训练路线": "Two training paths",
  "补齐两套任务草稿（不覆盖）": "Add both task drafts without overwriting",
  "我的工作区文件": "My workspace files",
  "搭建步骤": "Build steps",
  "查看输入与代码骨架": "Inspect inputs and code outline",
  "观测 / 动作": "Observation / action",
  "奖励 / 终止": "Reward / termination",
  "PPO 训练": "PPO training",
  "运行与评估": "Run and evaluate",
  "场景和机械臂": "Scene and robot arm",
  "任务定义": "Task definition",
  "到达任务": "Reach task",
  "正式 Run": "Formal Run",
  "训练产物": "Training artifact",
  "正式训练": "Formal training",
  "草稿可编辑": "Draft editable",
  "场景可编辑": "Scene editable",
  "运行后生成": "Generated after run",
  "服务端定义": "Server-defined",
  "平台源码": "Platform source",
  "源码参考": "Source reference",
  "不了解?": "",
  "登录研究平台": "Sign in to Research Studio",
  "使用平台账号进入课程、项目和运行证据工作区。": "Use your platform account to access courses, projects, and run evidence.",
  "用户名": "Username",
  "密码": "Password",
  "登录中...": "Signing in...",
  "登录": "Sign in",
  "登录失败": "Sign-in failed",
  "打开导航": "Open navigation",
  "关闭导航": "Close navigation",
  "主导航": "Main navigation",
  "工作模式": "Workspace mode",
  "退出登录": "Sign out",
  "正在恢复登录状态...": "Restoring your session...",
  "登录状态已失效，请重新登录。": "Your session has expired. Please sign in again.",
  "当前用户已被禁用": "This user account has been disabled",
  "身份恢复失败": "Could not restore your session",
  "平均奖励": "Average reward",
  "回合长度": "Episode length",
  "策略损失": "Policy loss",
  "价值损失": "Value loss",
  "策略熵": "Policy entropy",
  "近似 KL 散度": "Approximate KL divergence",
  "裁剪比例": "Clip fraction",
  "解释方差": "Explained variance",
  "学习率": "Learning rate",
  "训练吞吐": "Training throughput",
  "奖励/步": "reward/step",
  "动作步": "action steps",
  "无量纲": "dimensionless",
  "每次更新": "per update",
  "步/秒": "steps/second",
  "科研阶段": "Research stages",
  "完成证据": "Completion evidence",
  "项目与代码结构": "Project and code structure",
  "建立项目、工作区、文件树和可追踪的实验上下文。": "Establish a project, workspace, file tree, and traceable experiment context.",
  "项目创建并生成初始工作区": "Project created with an initial workspace",
  "理论与安全": "Theory and safety",
  "理解坐标系、关节、仿真边界、急停和真实部署风险。": "Understand coordinate frames, joints, simulation boundaries, emergency stops, and real-deployment risks.",
  "完成课程安全检查点": "Complete the course safety checkpoint",
  "机器人与场景": "Robot and scene",
  "选择机械臂、场景版本、目标物和可交互对象。": "Select the robot arm, scene revision, target, and interactive objects.",
  "场景通过结构校验并生成修订版本": "Scene passes structural validation and produces a revision",
  "Observation 设计": "Observation design",
  "定义关节、目标、视觉和归一化观测输入。": "Define joint, target, visual, and normalized observation inputs.",
  "保存 observation schema 与设计说明": "Save the observation schema and design notes",
  "Action 设计": "Action design",
  "选择关节增量、关节目标或末端执行器控制方式。": "Choose joint-delta, joint-target, or end-effector control.",
  "保存 action schema、范围和缩放": "Save the action schema, ranges, and scaling",
  "Reward 设计": "Reward design",
  "组合距离、姿态、成功、平滑度和惩罚项。": "Combine distance, pose, success, smoothness, and penalty terms.",
  "保存 reward 组成与预期现象": "Save the reward composition and expected behavior",
  "定义成功、超时、碰撞、无效状态和重置条件。": "Define success, timeout, collision, invalid-state, and reset conditions.",
  "环境逻辑通过契约验证": "Environment logic passes contract validation",
  "观察 action 到物理步进、奖励和终止的完整链路。": "Inspect the full path from action through physics stepping, reward, and termination.",
  "Pipeline 节点产生运行状态": "Pipeline nodes produce runtime state",
  "选择基线控制器、RL 策略或模仿学习策略契约。": "Choose a baseline controller, RL policy, or imitation-learning policy contract.",
  "策略配置与环境版本绑定": "Policy configuration is bound to an environment revision",
  "数据采集": "Data collection",
  "规划遥操作、episode、标注和数据质量检查。": "Plan teleoperation, episodes, annotations, and data-quality checks.",
  "数据集协议已定义": "Dataset protocol is defined",
  "训练执行": "Training execution",
  "绑定训练提供方、资源预算、seed 和 checkpoint。": "Bind the training provider, resource budget, seed, and checkpoint.",
  "产生可追踪运行记录": "Produce a traceable run record",
  "实时仿真": "Live simulation",
  "观察机器人、目标、轨迹、action、observation 和 reward。": "Observe the robot, target, trajectory, action, observation, and reward.",
  "Simulation Viewer 会话可复现": "Simulation Viewer session is reproducible",
  "实时指标": "Live metrics",
  "查看任务指标、优化指标和基础设施状态。": "Review task metrics, optimization metrics, and infrastructure status.",
  "指标事件与运行 ID 关联": "Metric events are associated with a run ID",
  "运行对比": "Run comparison",
  "比较代码、场景、配置、seed、策略和结果变化。": "Compare changes in code, scenes, configuration, seeds, policies, and results.",
  "生成多运行对比证据": "Generate multi-run comparison evidence",
  "自动评估": "Automated evaluation",
  "结合静态检查、运行测试、多 seed、基线和可复现性。": "Combine static checks, runtime tests, multiple seeds, baselines, and reproducibility.",
  "评估标准逐项关联 Evidence": "Associate each evaluation criterion with evidence",
  "Policy / Dataset 导出": "Policy / Dataset export",
  "封装策略、场景、映射、指标、报告和校验和。": "Package the policy, scene, mappings, metrics, report, and checksums.",
  "导出 manifest 并保留门禁结果": "Export a manifest and retain gate results",
  "检查关节顺序、标定、延迟、限位、急停和映射。": "Check joint order, calibration, latency, limits, emergency stops, and mappings.",
  "完成独立人工 readiness checklist": "Complete an independent manual readiness checklist",
  "受控部署": "Controlled deployment",
  "在独立 Robot Gateway、lease、watchdog 和操作员授权下运行。": "Operate through an independent Robot Gateway with a lease, watchdog, and operator authorization.",
  "真实机器人部署默认禁止": "Real-robot deployment is disabled by default",
  "机械臂智能，从可复现实验开始": "Robot Intelligence Starts with Reproducible Experiments",
  "已记录证据": "Evidence recorded",
  "起点": "Starting point",
  "证据必须关联到版本化制品或运行记录。仅勾选状态不会使研究阶段自动通过。": "Evidence must be linked to a versioned artifact or run record. Checking a status alone does not pass a research stage.",
  "真实控制": "Real control",
  "能力边界": "Capability boundary",
  "进入快速课程": "Open quick course",
  "移除证据标记": "Remove evidence mark",
  "记录已有证据": "Record existing evidence",
  "返回科研路线": "Back to research roadmap",
  "项目与文件工作区": "Project and file workspace",
  "工作区为空": "Workspace is empty",
  "选择或创建文件": "Select or create a file",
  "所有路径都限制在当前项目工作区内。": "All paths are restricted to the current project workspace.",
  "搜索工作区文本": "Search workspace text",
  "搜索": "Search",
  "创建快照": "Create snapshot",
  "恢复本次快照": "Restore this snapshot",
  "恢复 Starter": "Restore Starter",
  "训练能力": "training capabilities",
  "训练目录 API 不可用。": "The training catalog API is unavailable.",
  "环境": "Environment",
  "策略": "Policy",
  "最大步数": "Maximum steps",
  "动作缩放": "Action scale",
  "运行记录": "Run records",
  "尚无运行事件": "No run events yet",
  "查看运行记录": "View run records",
  "停止": "Stop",
  "随机种子": "Random seed",
  "预计步数": "Expected steps",
  "配置版本": "Config version",
  "标准姿态": "Default pose",
  "冻结训练配置": "Frozen training configuration",
  "选择 Project 后显示只读配置。": "Read-only configuration appears after selecting a Project.",
  "运行记录与对比": "Run records and comparison",
  "查看详情": "View details",
  "查看完整运行详情": "View full run details",
  "重试对比": "Retry comparison",
  "对比选中运行": "Compare selected runs",
  "进入证据评估": "Open evidence evaluation",
  "重试下载": "Retry download",
  "下载 JSON": "Download JSON",
  "尚无可评估运行": "No evaluable runs yet",
  "选择课程": "Select a course",
  "生成证据报告": "Generate evidence report",
  "评估中": "Evaluating",
  "报告生成后，这里会显示证据完整度、总分状态和下一次实验。": "After the report is generated, this area will show evidence completeness, score status, and the next experiment.",
  "比较评估": "Compare evaluations",
  "尚无可导出运行": "No exportable runs yet",
  "执行门禁并导出": "Run gate and export",
  "检查中": "Checking",
  "导出响应已生成": "Export response generated",
  "等待导出请求": "Waiting for export request",
  "策略导出与复现清单": "Policy export and reproducibility manifest",
  "神经网络训练代码": "Neural network training code",
  "静态检查": "Static check",
  "保存 revision": "Save revision",
  "代码结构通过": "Code structure passed",
  "代码被阻止": "Code blocked",
  "运行静态检查后显示 AST 安全结果。": "AST safety results appear after running the static check.",
  "环境与动作设计评估": "Environment and action design evaluation",
  "生成评估": "Generate evaluation",
  "尚无终态 Run": "No terminal Run yet",
  "运行证据评估": "Run evidence evaluation",
  "这份结果说明什么": "What this result means",
  "AI 个性化策略点评": "AI personalized policy review",
  "生成个性化点评": "Generate personalized review",
  "分析中": "Analyzing",
  "限制": "Limitations",
  "运行证据导出": "Run evidence export",
  "尚无已完成 Run": "No completed Run yet",
  "下载服务端生成的原始 JSON bytes 与权威响应头；导出不代表真实控制许可。": "Download the server-generated raw JSON bytes and authoritative response headers; export does not grant real-control permission.",
  "入门场景设计": "Introductory scene design",
  "六自由度场景设计": "Six-DOF scene design",
  "版本化场景": "Versioned scenes",
  "回退": "Undo",
  "方块": "Box",
  "球体": "Sphere",
  "圆柱": "Cylinder",
  "保存修订": "Save revision",
  "导出": "Export",
  "添加方块到场景": "Add a box to the scene",
  "名称": "Name",
  "边长 (m)": "Side length (m)",
  "颜色": "Color",
  "加入 3D 场景": "Add to 3D scene",
  "当前可保存": "Currently saveable",
  "当前仅预览": "Preview only",
  "场景对象": "Scene objects",
  "对象名称": "Object name",
  "固定物体": "Static object",
  "可推动物体": "Pushable object",
  "高级流程数据": "Advanced pipeline data",
  "六步实验主线已就绪": "Six-step experiment path is ready",
  "查看高级流程数据": "View advanced pipeline data",
  "打开对应界面": "Open corresponding page",
  "问教学 Agent": "Ask Teaching Agent",
  "结构验证通过": "Structure validation passed",
  "正在验证结构": "Validating structure",
  "正式 Project": "Formal Project",
  "组织": "Organization",
  "课程": "Course",
  "教学 Project": "Teaching Project",
  "S1 场景": "S1 scene",
  "选择组织": "Select organization",
  "选择 Project": "Select Project",
  "选择六自由度 S1 场景": "Select a six-DOF S1 scene",
  "随机种子 Seed": "Random seed",
  "Seed 改变什么": "What Seed changes",
  "进入启动实验": "Continue to launch experiment",
  "没有可用的六自由度 S1 场景": "No six-DOF S1 scenes are available",
  "S1 六轴姿态预览": "S1 six-axis pose preview",
  "教学说明": "Teaching instructions",
  "开始训练": "Start training",
  "收起 Agent": "Collapse Agent",
  "询问当前页面": "Ask about this page",
  "正在生成回复...": "Generating response...",
  "发送消息": "Send message",
  "模型来源与教学边界": "Model sources and teaching boundaries",
  "选择会话使用的模型来源和帮助强度。API Key 可在此临时配置到当前 API 进程，浏览器不会保存、读取或导出密钥。": "Choose the model source and assistance level for the session. An API key can be configured temporarily for the current API process; the browser does not store, read, or export it.",
  "仅发送至当前 API，不会回显": "Sent only to the current API and never displayed",
  "输入后配置；不会保存到浏览器": "Enter to configure; it will not be saved in the browser",
  "模型 ID": "Model ID",
  "例如 gpt-4o-mini": "For example, gpt-4o-mini",
  "兼容 API 根地址": "Compatible API base URL",
  "必须 HTTPS，后端禁止内网地址": "HTTPS is required; private-network addresses are blocked by the backend",
  "配置中...": "Configuring...",
  "配置 Provider": "Configure Provider",
  "清除临时配置": "Clear temporary configuration",
  "服务端配置": "Server configuration",
  "对话": "Chat",
  "结构化输出": "Structured output",
  "Provider 工具调用": "Provider tool calls",
  "密钥隔离": "Credential isolation",
  "API 仅返回 configured 状态。密钥值、Authorization header\n                和服务端 secret reference 不进入此页面。": "The API returns only the configured status. Key values, Authorization headers, and server-side secret references never enter this page.",
  "解释": "Explain",
  "Socratic 提示": "Socratic hints",
  "审查学生方案": "Review student proposal",
  "读取当前页面": "Read current page",
  "读取受限 Workspace": "Read restricted Workspace",
  "读取 Scene": "Read Scene",
  "读取 Run evidence": "Read Run evidence",
  "配置建议": "Configuration suggestions",
  "生成受审批 Patch 提案": "Propose an approval-gated Patch",
  "请选择权限档位": "Select an assistance level",
  "保存会话偏好": "Save session preferences",
  "项目与 Workspace 上下文": "Project and Workspace context",
  "Starter template 与允许文件": "Starter template and allowed files",
  "代码、场景、运行和评估关联": "Code, scene, run, and evaluation links",
  "快照、变更和回滚": "Snapshots, changes, and rollback",
  "打开代码工作区": "Open code workspace",
  "仿真与数字孪生工作台": "Simulation and digital-twin workspace",
  "Three.js 场景预览": "Three.js scene preview",
  "机器人与目标状态": "Robot and target state",
  "Start、Pause、Reset、Stop 控制契约": "Start, Pause, Reset, and Stop control contract",
  "编辑三维环境": "Edit 3D environment",
  "数据集与 Episode 工作台": "Dataset and Episode workspace",
  "Task metadata 与 recording protocol": "Task metadata and recording protocol",
  "Episode / reset / success 标记": "Episode / reset / success markers",
  "Replay、curation 与外部数据集适配": "Replay, curation, and external dataset adapters",
  "查看数据采集路线": "View data-collection roadmap",
  "机器人与设备中心": "Robot and device center",
  "Robot definition 与角色": "Robot definition and roles",
  "Joint limits 与 calibration artifact": "Joint limits and calibration artifacts",
  "Camera role 与 observation mapping": "Camera roles and observation mappings",
  "使用 Mock 机器人场景": "Use a Mock robot scene",
  "部署审查与安全门禁": "Deployment review and safety gates",
  "查看安全导出工件": "View safety export artifacts",
  "这个工作区负责什么": "What this workspace is responsible for",
  "完成时应留下的证据": "Evidence required for completion",
  "等待关联实际工件或运行记录": "Waiting for an actual artifact or run record",
  "当前可执行范围": "Current executable scope",
  "当前接入状态": "Current integration status",
  "安全状态": "Safety status",
};

const EXACT_CHINESE_TRANSLATIONS: Record<string, string> = {
  "Authentication is not configured": "身份认证未配置",
  "Invalid or missing bearer credentials": "身份凭据无效或缺失",
  "Simulation Service is unavailable": "仿真服务不可用",
  "Simulation Service is temporarily unavailable": "仿真服务暂时不可用",
  "Simulation Service returned an invalid response": "仿真服务返回了无效响应",
  "Simulation Service returned an invalid task catalog": "仿真服务返回了无效任务目录",
  "Simulation Service rejected the task catalog request": "仿真服务拒绝了任务目录请求",
  "Simulation Service returned an invalid task snapshot": "仿真服务返回了无效任务快照",
  "Simulation Service returned an invalid stream descriptor": "仿真服务返回了无效视频流描述",
  "Simulation task catalog is temporarily unavailable": "仿真任务目录暂时不可用",
  "Simulation task binding conflicts with the persisted Run": "仿真任务绑定与已保存的 Run 冲突",
  "Simulation stream-session is unavailable": "仿真视频流会话不可用",
  "Simulation dispatch is not configured": "仿真调度未配置",
  "Simulation stop is not configured": "仿真停止服务未配置",
  "Simulation stop is temporarily unavailable": "仿真停止服务暂时不可用",
  "Teaching projects are unavailable": "教学 Project 不可用",
  "Teaching project not found": "教学 Project 不存在",
  "Project operation is not allowed": "不允许执行此 Project 操作",
  "At least one Project field is required": "至少需要填写一个 Project 字段",
  "Organization context is unavailable": "组织上下文不可用",
  "Organization not found": "组织不存在",
  "Teaching resources are unavailable": "教学资源不可用",
  "Teaching resource not found": "教学资源不存在",
  "At least one field is required": "至少需要填写一个字段",
  "Canonical Evaluation service is not configured": "权威评估服务未配置",
  "Evaluation target not found": "评估目标不存在",
  "Evaluation history is temporarily unavailable": "评估历史暂时不可用",
  "Evaluation creation is forbidden": "不允许创建评估",
  "Evaluation is temporarily unavailable": "评估暂时不可用",
  "Evaluation not found": "评估不存在",
  "Evaluation evidence is temporarily unavailable": "评估证据暂时不可用",
  "Unsupported metric": "不支持该指标",
  "Metric history is temporarily unavailable": "指标历史暂时不可用",
  "Run not found": "Run 不存在",
  "Class not found": "教学班不存在",
  "Run lookup is temporarily unavailable": "Run 查询暂时不可用",
  "Run deletion is temporarily unavailable": "Run 删除服务暂时不可用",
  "Run creation is forbidden": "不允许创建 Run",
  "Run control is forbidden": "不允许控制 Run",
  "Canonical Run creation service is not configured": "权威 Run 创建服务未配置",
  "Canonical Run comparison is not configured": "权威 Run 对比服务未配置",
  "Canonical Run comparison is temporarily unavailable": "权威 Run 对比服务暂时不可用",
  "Canonical Run export is not configured": "权威 Run 导出服务未配置",
  "Canonical Run export is temporarily unavailable": "权威 Run 导出服务暂时不可用",
  "Idempotency-Key must not be blank": "Idempotency-Key 不能为空",
  "Idempotency-Key must be at most 255 characters": "Idempotency-Key 不能超过 255 个字符",
  "Artifact metadata is temporarily unavailable": "Artifact 元数据暂时不可用",
  "Artifact download is not configured": "Artifact 下载服务未配置",
  "Artifact not found": "Artifact 不存在",
  "Artifact integrity check failed": "Artifact 完整性校验失败",
  "Artifact download is temporarily unavailable": "Artifact 下载服务暂时不可用",
  "Run task binding is temporarily unavailable": "Run 任务绑定暂时不可用",
  "Run stop persistence is temporarily unavailable": "Run 停止状态暂时无法保存",
  "RESEARCH STUDIO": "研究工作台",
  "SYSTEM STATUS": "系统状态",
  "S1 / LEARNING GUIDE": "S1 / 学习指南",
  "STEP": "步骤",
  "THREE WORKBENCH TASKS": "三项工作台任务",
  "RESEARCH ROADMAP": "研究路线",
  "EVIDENCE PROGRESS": "证据进度",
  "ACTIVE CONTEXT": "当前上下文",
  "Mock Reach Baseline": "Mock Reach 基线",
  "BLOCKED": "已阻止",
  "PLANNED CAPABILITY": "规划能力",
  "EMBODIED ARM · USER GUIDE": "EMBODIED ARM · 用户指南",
  "DOCUMENT VERSION": "文档版本",
  "REFERENCE": "参考资料",
  "CONCEPTS & PRINCIPLES": "概念与原理",
  "STEP BY STEP": "分步说明",
  "CHECK BEFORE CONTINUE": "继续前检查",
  "CAPABILITY BOUNDARIES": "能力边界",
  "TEACHING AGENT CONTROL": "教学 Agent 控制",
  "GENERIC SHELL DISABLED": "通用 Shell 已禁用",
  "REAL ROBOT CONTROL DISABLED": "真实机械臂控制已禁用",
  "AVAILABLE PROVIDERS": "可用 Provider",
  "PROVIDER PROFILE": "Provider 信息",
  "RUNTIME PROVIDER CONFIG": "运行时 Provider 配置",
  "ASSISTANCE LEVEL": "辅助等级",
  "Level": "等级",
  "TEACHER ONLY": "仅限教师",
  "PROJECT WORKSPACE · V1": "Project 工作区 · V1",
  "S1 · SIX AXIS / PPO": "S1 · 六轴 / PPO",
  "REFERENCE OUTLINE · NOT EXECUTED": "参考结构 · 未执行",
  "OBSERVATION": "观测",
  "VALUE": "价值",
  "XY ACTION": "XY 动作",
  "FILE MAP": "文件图谱",
  "CONTROL PARTS": "控制组件",
  "EXPLORER": "资源管理器",
  "SEARCH": "搜索",
  "SNAPSHOT": "快照",
  "FILE HISTORY": "文件历史",
  "VERSIONS": "版本",
  "Starter Diff": "Starter 差异",
  "RUN CONFIGURATION": "Run 配置",
  "LIVE METRICS": "实时指标",
  "RUN SUMMARY": "Run 摘要",
  "SIMULATION TASK · PARAMETER SANDBOX": "仿真任务 · 参数沙箱",
  "SIMULATION ONLY · REAL CONTROL DISABLED": "仅限仿真 · 真实控制已禁用",
  "EXPERIMENT CONTEXT": "实验上下文",
  "NO RUN RECORDS": "暂无 Run 记录",
  "RUN EVIDENCE": "Run 证据",
  "CREATED": "已创建",
  "PENDING": "等待中",
  "QUEUED": "已排队",
  "RUNNING": "运行中",
  "PAUSED": "已暂停",
  "COMPLETED": "已完成",
  "FAILED": "失败",
  "STOPPED": "已停止",
  "SELECTED RUN": "已选 Run",
  "PROJECT": "Project",
  "CLASS": "教学班",
  "SEED": "随机种子",
  "UPDATES": "更新次数",
  "FROZEN CONFIGURATION": "冻结配置",
  "SCENE": "Scene",
  "ROBOT": "机器人",
  "CONTROL": "控制",
  "EVENT EVIDENCE": "事件证据",
  "EVIDENCE ACTIONS": "证据操作",
  "PRIMARY": "主要操作",
  "SECONDARY": "其他操作",
  "MANIFEST DOES NOT ENABLE CONTROL": "Manifest 不启用控制",
  "RUN COMPARISON": "Run 对比",
  "NO EVALUATION EVIDENCE": "暂无评估证据",
  "EVIDENCE-BOUND EVALUATION": "证据绑定评估",
  "RUN COHORT": "Run 组",
  "SELECTED": "已选择",
  "MOCK ONLY": "仅限 Mock",
  "RUBRIC CRITERIA": "评分标准",
  "BEFORE / AFTER": "修改前 / 修改后",
  "REPORT SUMMARY": "报告摘要",
  "OVERALL": "总体",
  "NO COMPLETED RUN": "暂无已完成 Run",
  "POLICY EXPORT GATE": "策略导出门禁",
  "SIM2REAL BLOCKED": "Sim2Real 已阻止",
  "OPERATOR REVIEW REQUIRED": "需要操作员审核",
  "EXPORT REQUEST": "导出请求",
  "FORMAT": "格式",
  "DISABLED": "已禁用",
  "EVALUATION GATE": "评估门禁",
  "REPORT": "报告",
  "SCORE": "分数",
  "RUNS": "Runs",
  "REAL CONTROL DISABLED": "真实控制已禁用",
  "EXPORT IS NOT DEPLOYMENT": "导出不等于部署",
  "S1 SCENE STUDIO": "S1 场景工作台",
  "ARM + GRIPPER DOF": "机械臂与夹爪自由度",
  "6 ARM + 2 GRIPPER DOF": "6 机械臂 + 2 夹爪自由度",
  "NEXT STEP · TASK PARAMETERS": "下一步 · 任务参数",
  "REVISION": "修订版本",
  "ENVIRONMENT": "环境",
  "DETERMINISTIC": "确定性",
  "INPUTS": "输入",
  "OUTPUTS": "输出",
  "POLICY WORKSPACE · PYTHON": "策略工作区 · Python",
  "STATIC VALIDATION ONLY": "仅静态校验",
  "STORED DRAFT · NOT EXECUTED": "已保存草稿 · 未执行",
  "TRAINING ARTIFACTS": "训练 Artifact",
  "VALIDATION REPORT": "校验报告",
  "ENTRYPOINT": "入口点",
  "SCENE REF": "Scene 引用",
  "EXECUTION": "执行方式",
  "STATIC ONLY": "仅静态处理",
  "S1 PUSH / REACH / MANIPULATION · TEACHING FLOW": "S1 Push / Reach / 操作 · 教学流程",
  "FORMAL": "正式流程",
  "SIMULATION ONLY": "仅限仿真",
  "SERVER CATALOG": "服务端目录",
  "SERVER CATALOG · 能力以当前后端响应为准": "服务端目录 · 能力以当前后端响应为准",
  "LOCAL FALLBACK · 仅供浏览，不能提交任务": "本地备用目录 · 仅供浏览，不能提交任务",
  "TEACHING PROJECTS": "教学 Project",
  "PROJECT DETAIL": "Project 详情",
  "CANONICAL EVALUATION": "权威评估",
  "TERMINAL RUN": "终态 Run",
  "IMMUTABLE REPORT": "不可变报告",
  "HISTORY": "历史记录",
  "CANONICAL JSON EXPORT": "权威 JSON 导出",
  "RUN ARTIFACTS": "Run Artifact",
  "SIMULATION TASK": "仿真任务",
  "DEMO TASK": "Demo 任务",
  "TRAINING TASK": "训练任务",
  "TRAINING": "训练",
  "PARAMETERS": "参数",
  "STREAM": "视频流",
  "CAMERA": "相机",
  "TRANSPORT": "传输协议",
  "SIMULATION": "仿真",
  "STATUS": "状态",
  "DEFAULT": "默认值",
  "MIN": "最小值",
  "READY": "就绪",
  "YES": "是",
  "NO": "否",
  "completed": "已完成",
  "running": "运行中",
  "paused": "已暂停",
  "failed": "失败",
  "stopped": "已停止",
  "assessed": "已评估",
  "complete": "完整",
  "Unfold line": "展开此行",
  "Fold line": "折叠此行",
  "Teaching Agent 使用说明": "教学 Agent 使用说明",
  "Project 配置、指标和评估 rubric 由服务端提供。": "Project 配置、指标和评估标准由服务端提供。",
  "· LEVEL": "· 等级",
};

const PHRASE_TRANSLATIONS: Array<[string, string]> = [
  ["认识平台", "Understand the platform"],
  ["准备六轴场景", "Prepare the six-axis scene"],
  ["理解任务与策略", "Understand tasks and policies"],
  ["配置训练", "Configure training"],
  ["启动并观察 Run", "Start and observe the Run"],
  ["评估和复盘", "Evaluate and review"],
  ["先认清哪些内容会进入训练", "First identify what enters training"],
  ["先保存场景，再配置训练", "Save the scene before configuring training"],
  ["选择 Push 或 Reach，不混入控制 Demo", "Choose Push or Reach; do not mix in control demos"],
  ["项目版本决定正式训练约束", "The Project version defines formal training constraints"],
  ["从课程和教学班创建正式运行", "Create a formal Run from a course and class"],
  ["以证据决定下一次修改", "Use evidence to decide the next change"],
  ["在实验模式打开“项目工作区”", "Open Project Workspace in Experiment mode"],
  ["切换 S1 Push / S1 Reach", "switch between S1 Push / S1 Reach"],
  ["查看六阶段蓝图与文件地图", "view the six-stage blueprint and file map"],
  ["进入“S1 环境编辑”", "Open S1 Environment"],
  ["选择六轴 S1 场景", "select a six-axis S1 scene"],
  ["检查机械臂、目标与物体", "inspect the robot arm, target, and objects"],
  ["按需调整位置并保存修订", "adjust positions as needed and save a revision"],
  ["在项目蓝图查看观测、二维 XY 动作、奖励和 PPO Actor/Critic", "review observations, 2D XY actions, rewards, and the PPO Actor/Critic in the project blueprint"],
  ["打开文件区，按需补齐两套教学草稿", "open the file area and add the two teaching drafts when needed"],
  ["在“参数沙箱”选择 Push 或 Reach", "choose Push or Reach in Parameter Sandbox"],
  ["核对对应教学 Project、场景和 seed", "verify the matching teaching Project, scene, and seed"],
  ["其他训练设置以服务端 Project 快照为准", "the server-side Project snapshot defines the remaining training settings"],
  ["进入“启动实验”", "Open Launch Experiment"],
  ["选择组织、课程、教学班和对应 Push/Reach Project", "select the organization, course, class, and matching Push/Reach Project"],
  ["核对六轴场景修订后点击“开始训练”", "verify the six-axis scene revision and click Start training"],
  ["在“运行分析”中看到状态、事件和指标", "view the status, events, and metrics in Run Analysis"],
  ["在“分析”模式检查成功率、目标距离、日志与 checkpoint", "inspect success rate, target distance, logs, and checkpoints in Analyze mode"],
  ["终态 Run 可进入“策略评估”", "terminal Runs can be opened in Policy Evaluation"],
  ["再在“运行制品”导出证据", "then export evidence in Run Artifacts"],
  ["编辑草稿不会替换服务端的策略网络，也不会自动启动训练。", "Editing drafts does not replace the server policy network or start training automatically."],
  ["网页画面只是结构预览，不能证明碰撞安全或真实机械臂可用。", "The web view is only a structural preview; it does not prove collision safety or real-robot readiness."],
  ["五种控制 Demo 是可复用零件，不是完整 PPO 训练任务。", "The five control demos are reusable parts, not complete PPO training tasks."],
  ["提交会占用 GPU；completed 表示执行结束，不表示任务成功。", "Submission uses the GPU; completed means execution ended, not that the task succeeded."],
  ["教学 Agent 可以解释证据，不能生成缺失指标或修改正式评分。", "The Teaching Agent can explain evidence, but cannot invent missing metrics or change formal scores."],
  ["推动桌面方块进入目标区域", "Push a tabletop cube into the goal region"],
  ["末端到达目标区域并保持", "Move the end effector into the target region and hold"],
  ["推动方块到目标；24 维观测，二维平面动作。", "Push a cube to the goal; 24D observation, 2D planar action."],
  ["让末端到达目标区域；18 维观测，二维平面动作。", "Reach the target region; 18D observation, 2D planar action."],
  ["从场景到策略", "From scene to policy"],
  ["强化学习任务", "Reinforcement learning task"],
  ["场景和机械臂", "Scene and robot arm"],
  ["任务定义", "Task definition"],
  ["奖励 / 终止", "Reward / termination"],
  ["运行与评估", "Run and evaluate"],
  ["场景可编辑", "Scene editable"],
  ["草稿可编辑", "Draft editable"],
  ["运行后生成", "Generated after run"],
  ["查看输入与代码骨架", "Inspect inputs and code outline"],
  ["关联文件", "Related files"],
  ["补齐两套任务草稿（不覆盖）", "Add both task drafts without overwriting"],
  ["控制演示零件", "Control demo parts"],
  ["这些模块用于检查姿态、轨迹与关节运动；它们不进行 PPO 学习，不能作为训练任务提交。", "These modules inspect poses, trajectories, and joints; they do not train PPO and cannot be submitted as training tasks."],
  ["沿着环境、策略、训练与证据评估推进。当前可执行能力均明确标记为", "Progress through environment, policy, training, and evidence evaluation. Every executable capability is explicitly marked as"],
  ["真实机械臂控制保持禁用。", "Real robot-arm control remains disabled."],
  ["依赖阶段 ", "Depends on stage "],
  ["当前动力学为教学近似，不是数字孪生或物理精确仿真。Pipeline、复杂环境与完整评估尚未实现。", "The current dynamics are an instructional approximation, not a digital twin or physically accurate simulation. The Pipeline, complex environments, and complete evaluation are not implemented yet."],
  ["该工作区已进入升级路线，但尚无可执行实现。这里不会用界面占位来冒充已接入能力。", "This workspace is on the upgrade roadmap but has no executable implementation yet. The interface does not present placeholders as integrated capabilities."],
  ["已配置到当前 API 进程。密钥不会写入浏览器存储；重启 API 后需重新配置。", "Configured for the current API process. The key is not written to browser storage and must be configured again after the API restarts."],
  ["Provider 配置失败", "Provider configuration failed"],
  ["已清除当前 API 进程中的临时配置。", "Temporary configuration cleared from the current API process."],
  ["Provider 配置清除失败", "Failed to clear Provider configuration"],
  ["正在读取服务端 Agent 能力...", "Loading server-side Agent capabilities..."],
  ["Agent Provider API 不可用，请检查后端服务。", "The Agent Provider API is unavailable. Check the backend service."],
  ["内置基础 Agent", "Built-in basic Agent"],
  ["OpenAI 兼容 API", "OpenAI-compatible API"],
  ["仅使用当前会话中提供的教学上下文，不发送外部请求。", "Uses only the teaching context provided in the current session and sends no external requests."],
  ["会将系统提示和当前会话消息发送到管理员配置的兼容端点。", "Sends the system prompt and current session messages to the compatible endpoint configured by an administrator."],
  ["会将系统提示和当前会话消息发送到管理员或当前会话配置的 DeepSeek API。", "Sends the system prompt and current session messages to the DeepSeek API configured by an administrator or for the current session."],
  ["会将系统提示和当前会话消息发送到 Anthropic API。", "Sends the system prompt and current session messages to the Anthropic API."],
  ["当前 Project 未声明可提交的整数 seed。", "The current Project declares no submit-ready integer seed."],
  ["等待 S1 场景", "Waiting for an S1 scene"],
  ["工作区已连接；文件写入受类型、大小和路径策略限制。", "Workspace connected; file writes are restricted by type, size, and path policies."],
  ["工作区初始化失败", "Workspace initialization failed"],
  ["正在初始化受限工作区...", "Initializing restricted workspace..."],
  ["正在创建项目", "Creating project"],
  ["工作区为空", "Workspace is empty"],
  ["选择或创建文件", "Select or create a file"],
  ["所有路径都限制在当前项目工作区内。", "All paths are restricted to the current project workspace."],
  ["最多 100 项", "Up to 100 items"],
  ["搜索工作区文本", "Search workspace text"],
  ["搜索", "Search"],
  ["创建快照", "Create snapshot"],
  ["恢复本次快照", "Restore this snapshot"],
  ["恢复 Starter 模板？系统会先创建可回滚快照。", "Restore the Starter template? The system will first create a reversible snapshot."],
  ["当前未快照变更将丢失", "unsnapshotted changes will be lost"],
  ["已自动保存", "Auto-saved"],
  ["运行已创建，但实时事件连接中断；可在运行记录中检查结果。", "The run was created, but the live event connection was interrupted; check the run records for results."],
  ["训练启动失败", "Training failed to start"],
  ["GPU 训练状态读取失败", "Failed to read GPU training status"],
  ["等待启动", "Waiting to start"],
  ["正在读取运行记录...", "Loading run records..."],
  ["运行记录 API 不可用。", "The run records API is unavailable."],
  ["尚无运行记录", "No run records yet"],
  ["运行记录 · ", "Run records · "],
  ["项目 ", "Project "],
  ["随机种子 ", "Random seed "],
  ["未指定", "Not specified"],
  ["正在对比", "Comparing"],
  ["导出请求失败，请重试", "Export request failed; please retry"],
  ["对比请求失败，请重试", "Comparison request failed; please retry"],
  ["正在读取可评估运行...", "Loading evaluable runs..."],
  ["评估请求失败", "Evaluation request failed"],
  ["报告比较失败", "Report comparison failed"],
  ["正在读取导出证据...", "Loading export evidence..."],
  ["导出证据 API 不可用。", "The export evidence API is unavailable."],
  ["场景保存失败", "Scene save failed"],
  ["场景导出失败", "Scene export failed"],
  ["场景 manifest 已导出", "Scene manifest exported"],
  ["固定物体（取消后可推动）", "Static object (pushable after canceling)"],
  ["运行证据不可用", "Run evidence unavailable"],
  ["当前只保存场景对象和资产引用；observation、action、reward、termination、碰撞与 IK 均不从 Mock 环境推断。", "Only scene objects and asset references are saved; observation, action, reward, termination, collisions, and IK are not inferred from the Mock environment."],
  ["高级：环境证据与参数 Schema", "Advanced: environment evidence and parameter Schema"],
  ["运行证据通道", "Run evidence channels"],
  ["可调参数 Schema", "Tunable parameter Schema"],
  ["契约可执行", "Contract executable"],
  ["目录可用 · 版本已声明", "Catalog available · version declared"],
  ["当前不可用", "Currently unavailable"],
  ["正在解析 Pipeline 修订...", "Parsing Pipeline revision..."],
  ["Pipeline API 不可用，请确认服务已更新并启动。", "The Pipeline API is unavailable. Confirm that the service is updated and running."],
  ["Mock Pipeline 是教学近似，不代表物理精确仿真", "The Mock Pipeline is an instructional approximation and does not represent physically accurate simulation"],
  ["输入新文件路径（支持 py/json/yaml/md/cpp/txt）", "Enter a new file path (py/json/yaml/md/cpp/txt supported)"],
  ["输入新目录路径", "Enter a new directory path"],
  ["输入新的工作区路径", "Enter a new workspace path"],
  ["上传到工作区路径", "Upload to workspace path"],
  ["新建文件", "Create file"],
  ["新建目录", "Create directory"],
  ["上传文本文件", "Upload text file"],
  ["下载当前文件", "Download current file"],
  ["重命名当前文件", "Rename current file"],
  ["删除当前文件", "Delete current file"],
  ["对比 Starter", "Compare with Starter"],
  ["删除 ", "Delete "],
  ["恢复 ", "Restore "],
  ["已保存", "Saved"],
  ["正在自动保存", "Auto-saving"],
  ["自动保存失败", "Auto-save failed"],
  ["等待自动保存", "Waiting for auto-save"],
  ["已创建快照", "Snapshot created"],
  ["已恢复快照", "Snapshot restored"],
  ["已恢复 ", "Restored "],
  ["当前未快照变更将丢失", "unsnapshotted changes will be lost"],
  ["该操作会写入审计记录", "This action will be recorded in the audit log"],
  ["恢复 Starter 模板", "Restore the Starter template"],
  ["系统会先创建可回滚快照", "The system will first create a reversible snapshot"],
  ["回滚快照", "rollback snapshot"],
  ["正在读取训练能力", "Loading training capabilities"],
  ["使用三关节教学近似验证 seed、参数与事件可复现性；这里不训练神经网络，也不创建课程正式 Run。", "Use the three-joint teaching approximation to verify seed, parameter, and event reproducibility; this does not train a neural network or create a formal course Run."],
  ["运行固定的 synthetic regression 模板，用于验证 GPU Worker、反向传播与 checkpoint 链路；它不是 S1 或课程 PPO 训练。", "Run a fixed synthetic regression template to verify the GPU Worker, backpropagation, and checkpoint pipeline; this is not S1 or course PPO training."],
  ["启动后将实时显示 reward 与 success rate。", "Reward and success rate will appear in real time after launch."],
  ["启动后将显示真实 GPU train loss。", "The real GPU train loss will appear after launch."],
  ["正式课程或 S1 场景目录不可用", "The formal course or S1 scene catalog is unavailable"],
  ["此页只允许调整 Project schema 声明的随机种子，不直接创建 Run。", "This page only allows changing the random seed declared by the Project schema; it does not create a Run directly."],
  ["参数沙箱不会退回三关节 Mock", "The parameter sandbox will not fall back to the three-joint Mock"],
  ["参数沙箱不会用三关节 Mock 代替正式 S1 资产。", "The parameter sandbox will not replace formal S1 assets with the three-joint Mock."],
  ["姿态滑杆只调整网页预览", "Pose sliders only adjust the web preview"],
  ["不写入 Run 参数，不执行 IK、碰撞检测或真实机器人控制。", "They do not write Run parameters or execute IK, collision checks, or real robot control."],
  ["不同 seed 仅支持复现性证据，不构成 held-out 泛化或 Sim2Real 证明。", "Different seeds support reproducibility evidence only; they do not prove held-out generalization or Sim2Real."],
  ["完成一个 Mock 训练运行后，才能生成教学 Manifest", "A teaching Manifest can be generated only after completing a Mock training run"],
  ["导出只封装当前 Mock", "Export only packages the current Mock"],
  ["任何结果都不会启用真实机械臂控制", "No result enables real robot-arm control"],
  ["每份不可变报告只绑定一个正式 Run；证据不足时不生成推测分数。", "Each immutable report is bound to one formal Run; speculative scores are not generated when evidence is insufficient."],
  ["每项结论都绑定到 Run 事件或制品。", "Every conclusion is bound to a Run event or artifact."],
  ["证据完整度", "Evidence completeness"],
  ["当前 Provider 尚未在服务端配置。", "The current Provider is not configured on the server."],
  ["当前 Level ", "Current Level "],
  ["不允许读取 Run/Evaluation 证据。", "cannot read Run/Evaluation evidence."],
  ["正在读取 Agent 配置", "Loading Agent configuration"],
  ["当前页面，Project、Run 和 Scene 标识", "this page, Project, Run, and Scene identifiers"],
  ["不会执行 Shell 或直接控制机器人。", "It will not execute Shell commands or directly control the robot."],
  ["请先在 Agent 设置中选择可用 Provider", "Select an available Provider in Agent settings first"],
  ["当前运行未启用视频", "Video was not enabled for this run"],
  ["正在准备训练录像", "Preparing the training recording"],
  ["本次运行没有可用的训练录像", "No training recording is available for this run"],
  ["训练录像信息读取失败", "Failed to read training recording information"],
  ["实时训练已结束，正在准备录像", "Live training has ended; preparing the recording"],
  ["评估回放信息读取失败", "Failed to read evaluation replay information"],
  ["本次运行没有可用的成功策略评估回放", "No successful-policy evaluation replay is available for this run"],
  ["未关联报告", "No report linked"],
  ["服务端门禁结果已写入制品", "The server gate result was written to the artifact"],
  ["结果将包含 checksum、safety checks 与 export gate。", "The result will include checksum, safety checks, and the export gate."],
  ["导出不代表真实控制许可", "Export does not grant real-control permission"],
  ["按六个步骤完成一次可复现的教学实验。", "Complete a reproducible teaching experiment in six steps."],
  ["当前流程绑定 Mock Reach 教学近似。", "The current workflow is bound to the Mock Reach teaching approximation."],
  ["流程可验证、可复现，但动力学是近似教学行为；外部 worker 和真实控制尚未接入。", "The workflow is verifiable and reproducible, but its dynamics are an instructional approximation; external workers and real control are not connected."],
  ["当前教学步骤未绑定底层节点", "The current teaching step is not bound to a backend node"],
  ["这一步由对应业务页面完成", "This step is completed by the corresponding product page"],
  ["训练结果与导出", "Training results and export"],
  ["状态以服务端 Run 为准", "The server-side Run is the source of truth for status"],
  ["浏览器刷新或实时连接中断不会停止训练。", "Refreshing the browser or losing the live connection will not stop training."],
  ["服务端估算进度", "Server-estimated progress"],
  ["该进度直接来自 Run 快照，不按浏览器停留时间推算。", "This progress comes directly from the Run snapshot and is not inferred from browser time."],
  ["冻结快照未声明可展示参数。", "The frozen snapshot declares no displayable parameters."],
  ["本次运行未上报该指标", "This run did not report this metric"],
  ["尚未由本次 Run 的 Event 输出；页面不推算、不补 0。", "was not emitted by this Run's events; the page does not infer or fill in zero."],
  ["下载已开始", "Download started"],
  ["正在准备下载", "Preparing download"],
  ["下载失败，请重试", "Download failed; please retry"],
  ["确认停止这次训练？", "Stop this training?"],
  ["已保存的事件和制品会保留，停止后不能继续本次 Run。", "Saved events and artifacts will be retained, but this Run cannot continue after stopping."],
  ["停止运行", "Stop run"],
  ["正在停止", "Stopping"],
  ["重试读取", "Retry reading"],
  ["正在回传训练结果", "Returning training results"],
  ["制品目录读取失败。", "Failed to read the artifact catalog."],
  ["继续学习路线", "Continue the learning roadmap"],
  ["完成此检查点", "Complete this checkpoint"],
  ["运行前预测", "Pre-run prediction"],
  ["仿真器已启动", "The simulator has started"],
  ["演示已停止", "The demo was stopped"],
  ["演示运行超时", "The demo run timed out"],
  ["Patch 已审批应用", "Patch approved and applied"],
  ["已回滚到审批前快照。", "Rolled back to the pre-approval snapshot."],
  ["沿学习路线理解完整实验流程与 Agent 指导", "Follow the roadmap to understand the complete experiment workflow and Agent guidance"],
  ["沿着环境、策略、训练与证据评估推进。当前可执行能力均明确标记为", "Progress through environment, policy, training, and evidence evaluation. Every executable capability is explicitly marked as"],
  ["真实机械臂控制保持禁用。", "Real robot-arm control remains disabled."],
  ["依赖阶段 ", "Depends on stage "],
  ["当前动力学为教学近似，不是数字孪生或物理精确仿真。Pipeline、复杂环境与完整评估尚未实现。", "The current dynamics are an instructional approximation, not a digital twin or physically accurate simulation. The Pipeline, complex environments, and complete evaluation are not implemented yet."],
  ["该工作区已进入升级路线，但尚无可执行实现。这里不会用界面占位来冒充已接入能力。", "This workspace is on the upgrade roadmap but has no executable implementation yet. The interface does not present placeholders as integrated capabilities."],
  ["已配置到当前 API 进程。密钥不会写入浏览器存储；重启 API 后需重新配置。", "Configured for the current API process. The key is not written to browser storage and must be configured again after the API restarts."],
  ["Provider 配置失败", "Provider configuration failed"],
  ["已清除当前 API 进程中的临时配置。", "Temporary configuration cleared from the current API process."],
  ["Provider 配置清除失败", "Failed to clear Provider configuration"],
  ["正在读取服务端 Agent 能力...", "Loading server-side Agent capabilities..."],
  ["Agent Provider API 不可用，请检查后端服务。", "The Agent Provider API is unavailable. Check the backend service."],
  ["API 仅返回 configured 状态。", "The API returns only the configured status."],
  ["和服务端 secret reference 不进入此页面。", "and server-side secret references never enter this page."],
  ["以项目为边界组织课程、环境、代码、运行、数据和评估工件。当前工作区提供受限文件读取、模板初始化和快照回滚能力。", "Organize course, environment, code, run, data, and evaluation artifacts within project boundaries. The current workspace provides restricted file access, template initialization, and snapshot rollback."],
  ["统一承载轻量 Three.js 预览、运行状态、轨迹和任务指标。当前已接入 Mock Viewer；WebRTC 仿真流与真实 Simulator Adapter 保持独立扩展。", "Bring lightweight Three.js previews, run status, trajectories, and task metrics into one workspace. A Mock Viewer is connected; WebRTC simulation streams and real Simulator Adapters remain independent extensions."],
  ["目标产品支持遥操作录制、episode 管理、回放、筛选和模仿学习训练。当前版本先展示数据契约和接入边界，尚未伪造真实相机或机器人数据。", "The target product supports teleoperation recording, episode management, replay, filtering, and imitation-learning training. This version presents the data contract and integration boundaries without fabricating real camera or robot data."],
  ["统一管理 RobotAdapter、关节、相机、传感器、Teleoperator 和校准工件。当前平台不连接真实机器人，所有硬件能力均保持不可用状态。", "Manage RobotAdapters, joints, cameras, sensors, Teleoperators, and calibration artifacts in one place. The platform does not connect to real robots, and all hardware capabilities remain unavailable."],
  ["部署页面只负责检查策略与机器人之间的兼容性、映射、限位、延迟、急停和 watchdog，不允许普通业务服务直接驱动硬件。", "The deployment page checks compatibility, mappings, limits, latency, emergency stops, and watchdogs between policies and robots. Ordinary application services cannot drive hardware directly."],
  ["当前功能可以创建和查看教学工件，运行结果来自 Mock Adapter。它们可以验证工作流，但不代表真实动力学或硬件性能。", "The current capabilities can create and inspect teaching artifacts, with results produced by a Mock Adapter. They validate the workflow but do not represent real dynamics or hardware performance."],
  ["该模块已经纳入正式产品契约，但当前版本尚未接入真实数据源。页面不会用示例数据冒充已完成能力。", "This module is part of the formal product contract, but no real data source is connected in this version. The page does not present sample data as a completed capability."],
  ["真实机器人控制、Sim2Real 下发和未经授权的硬件动作均被阻断。需要独立 Robot Gateway、租约、限位、watchdog 和操作员批准。", "Real-robot control, Sim2Real deployment, and unauthorized hardware actions are blocked. An independent Robot Gateway, lease, limits, watchdog, and operator approval are required."],
  ["编辑项目、S1 场景与策略，配置后启动正式实验", "Edit projects, S1 scenes, and policies, then configure and launch experiments"],
  ["核对 Run、正式评估与持久化制品", "Review runs, formal evaluations, and persisted artifacts"],
  ["查看 GPU、设备与部署能力边界", "Review GPU, device, and deployment capability boundaries"],
  ["演示能力", "Demo capability"],
  ["正在读取", "Loading"],
  ["读取失败", "failed to load"],
  ["本次 rollout 中每个训练步的平均奖励", "Average reward per training step in this rollout"],
  ["本次 rollout 已完成回合的平均动作步数", "Average action steps in episodes completed during this rollout"],
  ["PPO 策略目标在本次更新中的平均损失", "Average PPO policy-objective loss in this update"],
  ["价值函数在本次更新中的平均损失", "Average value-function loss in this update"],
  ["策略动作分布在本次更新中的平均熵", "Average entropy of the policy action distribution in this update"],
  ["Simulation 在 PPO 更新后输出的新旧策略近似 KL 散度", "Approximate KL divergence between old and new policies reported by Simulation after the PPO update"],
  ["Simulation 输出的 PPO 比率被裁剪样本比例", "Fraction of samples whose PPO ratio was clipped, reported by Simulation"],
  ["Simulation 输出的价值函数对回报方差的解释程度", "Share of return variance explained by the value function, reported by Simulation"],
  ["当前 Simulation 尚未在训练记录或 Event 中输出该字段", "Simulation does not currently report this field in training records or events"],
  ["正在加载", "Loading"],
  ["加载失败", "failed to load"],
  ["正在连接", "Connecting"],
  ["正在重连", "Reconnecting"],
  ["正在训练", "Training"],
  ["正在保存", "Saving"],
  ["正在生成", "Generating"],
  ["正在配置", "Configuring"],
  ["正在初始化", "Initializing"],
  ["正在重试", "Retrying"],
  ["尚未实现", "Not implemented yet"],
  ["暂不可用", "Temporarily unavailable"],
  ["未记录", "Not recorded"],
  ["已完成", "Completed"],
  ["已停止", "Stopped"],
  ["失败", "Failed"],
  ["重试", "Retry"],
  ["取消", "Cancel"],
  ["确认", "Confirm"],
  ["保存", "Save"],
  ["删除", "Delete"],
  ["下载", "Download"],
  ["上传", "Upload"],
  ["返回", "Back"],
  ["关闭", "Close"],
];

function hasChinese(value: string) {
  return /[\u3400-\u9fff]/u.test(value);
}

export function translateUiText(value: string): string {
  if (!hasChinese(value)) return value;
  const leading = value.match(/^\s*/u)?.[0] ?? "";
  const trailing = value.match(/\s*$/u)?.[0] ?? "";
  const content = value.slice(leading.length, value.length - trailing.length || undefined);
  const exact = EXACT_TRANSLATIONS[content]
    ?? EXACT_TRANSLATIONS[content.replace(/\s+/gu, " ").trim()];
  if (exact) return `${leading}${exact}${trailing}`;

  const prefix = Object.entries(EXACT_TRANSLATIONS).find(([source]) => content.startsWith(source) && /HTTP|request_id/.test(content.slice(source.length)));
  if (prefix) return `${leading}${prefix[1]}${content.slice(prefix[0].length)}${trailing}`;

  let translated = content;
  for (const [source, target] of PHRASE_TRANSLATIONS) {
    translated = translated.split(source).join(target);
  }
  return `${leading}${translated}${trailing}`;
}

export function translateUiTextToChinese(value: string): string {
  const leading = value.match(/^\s*/u)?.[0] ?? "";
  const trailing = value.match(/\s*$/u)?.[0] ?? "";
  const content = value.slice(leading.length, value.length - trailing.length || undefined);
  const normalized = content.replace(/\s+/gu, " ").trim();
  const exact = EXACT_CHINESE_TRANSLATIONS[content] ?? EXACT_CHINESE_TRANSLATIONS[normalized];
  if (exact) return `${leading}${exact}${trailing}`;
  const step = normalized.match(/^STEP (\d{2}(?: \/ \d{2})?)$/u);
  if (step) return `${leading}步骤 ${step[1]}${trailing}`;
  const trainingLab = normalized.match(/^TRAINING LAB · (MOCK|EDGE GPU)$/u);
  if (trainingLab) return `${leading}训练实验室 · ${trainingLab[1] === "MOCK" ? "Mock" : "Edge GPU"}${trailing}`;
  const stage = normalized.match(/^STAGE (\d{2})$/u);
  if (stage) return `${leading}阶段 ${stage[1]}${trailing}`;
  if (hasChinese(content)) {
    const localized = content
      .replace(/\bLevel (?=\d)/gu, "等级 ")
      .replace(/等级为 complete\b/gu, "等级为完整")
      .replace(/恢复 catalog 默认值/gu, "恢复目录默认值");
    if (localized !== content) return `${leading}${localized}${trailing}`;
  }
  return value;
}

function localizeUiText(value: string, locale: Locale): string {
  return locale === "en" ? translateUiText(value) : translateUiTextToChinese(value);
}

function isKnownUiText(value: string): boolean {
  return hasChinese(value) || translateUiTextToChinese(value) !== value;
}

export function readStoredLocale(storage: Pick<Storage, "getItem">): Locale {
  return storage.getItem(LOCALE_STORAGE_KEY) === "en" ? "en" : "zh-CN";
}

export function applyLocale(
  locale: Locale,
  storage: Pick<Storage, "setItem">,
  documentElement: Pick<HTMLElement, "lang">,
) {
  storage.setItem(LOCALE_STORAGE_KEY, locale);
  documentElement.lang = locale;
}

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (chinese: string, english: string) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children, initialLocale }: { children: ReactNode; initialLocale?: Locale }) {
  const [locale, setLocale] = useState<Locale>(() =>
    initialLocale ?? (typeof window === "undefined" ? "zh-CN" : readStoredLocale(window.localStorage)),
  );

  useEffect(() => {
    applyLocale(locale, window.localStorage, document.documentElement);
  }, [locale]);

  return (
    <I18nContext.Provider
      value={{ locale, setLocale, t: (chinese, english) => locale === "en" ? english : chinese }}
    >
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) throw new Error("useI18n must be used inside I18nProvider");
  return context;
}

export function LanguageSwitcher({ compact = false }: { compact?: boolean }) {
  const { locale, setLocale, t } = useI18n();
  return (
    <div className={`language-switcher ${compact ? "compact" : ""}`} aria-label={t("界面语言", "Interface language")}>
      <button type="button" className={locale === "zh-CN" ? "active" : ""} aria-pressed={locale === "zh-CN"} onClick={() => setLocale("zh-CN")}>{t("中文", "CN")}</button>
      <button type="button" className={locale === "en" ? "active" : ""} aria-pressed={locale === "en"} onClick={() => setLocale("en")}>{t("英文", "EN")}</button>
    </div>
  );
}

const LOCALIZED_ATTRIBUTES = ["aria-label", "title", "placeholder", "alt"] as const;

export function LocalizedContent({ children }: { children: ReactNode }) {
  const { locale } = useI18n();
  const hostRef = useRef<HTMLDivElement>(null);
  const localeRef = useRef(locale);
  const sourceText = useRef(new WeakMap<Text, string>());
  const sourceAttributes = useRef(new WeakMap<Element, Map<string, string>>());
  localeRef.current = locale;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let active = true;
    let applying = false;

    const localizeNode = (node: Node) => {
      if (!active || localeRef.current !== locale) return;
      const element = node instanceof Element ? node : node.parentElement;
      const translationDisabled = Boolean(element?.closest('[translate="no"]'));
      if (node.nodeType === Node.TEXT_NODE) {
        if (translationDisabled) return;
        const textNode = node as Text;
        let knownSource = sourceText.current.get(textNode);
        const matchesKnownSource = knownSource && (
          textNode.data === knownSource
          || textNode.data === localizeUiText(knownSource, "en")
          || textNode.data === localizeUiText(knownSource, "zh-CN")
        );
        if (!matchesKnownSource && isKnownUiText(textNode.data)) {
          sourceText.current.set(textNode, textNode.data);
          knownSource = textNode.data;
        } else if (knownSource && !matchesKnownSource) {
          sourceText.current.delete(textNode);
          return;
        }
        const source = sourceText.current.get(textNode);
        if (source) {
          const localized = localizeUiText(source, locale);
          if (textNode.data !== localized) textNode.data = localized;
        }
        return;
      }
      if (!(node instanceof Element)) return;
      let attributes = sourceAttributes.current.get(node);
      for (const name of LOCALIZED_ATTRIBUTES) {
        const value = node.getAttribute(name);
        let source = attributes?.get(name);
        const matchesKnownSource = source && value && (
          value === source
          || value === localizeUiText(source, "en")
          || value === localizeUiText(source, "zh-CN")
        );
        if (value && !matchesKnownSource && isKnownUiText(value)) {
          attributes ??= new Map<string, string>();
          attributes.set(name, value);
          sourceAttributes.current.set(node, attributes);
          source = value;
        } else if (source && value && !matchesKnownSource) {
          attributes?.delete(name);
          source = undefined;
        }
        if (source) {
          const localized = localizeUiText(source, locale);
          if (node.getAttribute(name) !== localized) node.setAttribute(name, localized);
        }
      }
      node.childNodes.forEach(localizeNode);
    };

    const apply = (nodes: Iterable<Node>) => {
      applying = true;
      for (const node of nodes) localizeNode(node);
      applying = false;
    };
    apply([host]);

    const observer = new MutationObserver((mutations) => {
      if (!active || applying || localeRef.current !== locale) return;
      const changed = new Set<Node>();
      for (const mutation of mutations) {
        if (mutation.type === "characterData") changed.add(mutation.target);
        mutation.addedNodes.forEach((node) => changed.add(node));
        if (mutation.type === "attributes") changed.add(mutation.target);
      }
      apply(changed);
    });
    observer.observe(host, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: [...LOCALIZED_ATTRIBUTES] });
    return () => {
      active = false;
      observer.disconnect();
    };
  }, [locale]);

  useEffect(() => {
    if (locale !== "en") return;
    const nativePrompt = window.prompt;
    const nativeConfirm = window.confirm;
    const nativeAlert = window.alert;
    window.prompt = (message, defaultValue) => nativePrompt(translateUiText(message ?? ""), defaultValue);
    window.confirm = (message) => nativeConfirm(translateUiText(message ?? ""));
    window.alert = (message) => nativeAlert(typeof message === "string" ? translateUiText(message) : message);
    return () => {
      window.prompt = nativePrompt;
      window.confirm = nativeConfirm;
      window.alert = nativeAlert;
    };
  }, [locale]);

  return <div ref={hostRef} className="localized-app-root">{children}</div>;
}
