# S1 Push Midterm UI Contract

## 1. 现有页面与接口

- 登录页真实调用 `POST /api/v1/auth/login`，随后调用 `GET /api/v1/me`。
- JWT 当前保存在 `sessionStorage["studio_access_token"]`，不符合“仅内存保存”；HTTP `api()` 会自动添加 `Authorization: Bearer <token>`。
- 浏览器硬刷新时，App 从 sessionStorage 取 token 并调用 `/api/v1/me`；401 会清除 token、显示登录失效并返回登录页。
- `/me` 已返回真实 Organization context/role/permissions，但前端没有 Organization 选择页，也未调用 `/api/v1/organizations`。
- Backend 的 canonical Course/Class/Enrollment API 已存在；前端 `/courses` 实际仍调用 legacy `/api/courses/quick-start`，没有 canonical Course、Class 或 Enrollment 页面。
- `/projects` 当前是 Workspace 文件树/编辑器：调用 `/api/v1/projects` 与 `/api/v1/workspaces/*`，缺省时硬编码创建 `research-workspace`；它不是正式教学 Project 列表或实验详情。
- Training 页默认启动 Mock；按钮优先 `POST /api/v1/runs`，404/405 时 fallback `/api/runs/train`。Edge GPU 分支另走 `/api/sandbox/jobs`，不是 S1 Push。
- canonical Run 请求当前发送 `project_id=quick-start`、`parameters.environment_id/strategy_id/seed/max_steps`、任意 `project_config_snapshot`，且省略 `execution_mode`，因此使用 Backend 默认 `mock`。
- Event WebSocket 路径已是 canonical `/api/v1/runs/{run_id}/events`，HTTP 补拉仍有 legacy fallback；JWT 被放入 query string，未使用首消息 AUTH。
- RunDetail 已接 canonical Run 查询/停止、Stream Descriptor、浏览器 WHEP player和 Artifact metadata；未接 Input WebSocket，Artifact 未提供带 Bearer 的下载操作。
- Evaluation 页仍调用 legacy `/api/evaluations`、cohort/报告比较模型；尚未接单 Run 的 canonical `/api/v1/evaluations`。Run Compare/Export 已优先 canonical，但仍 fallback legacy。

## 2. 必须删除或替换的Mock字段

- 删除正式 S1 页面中的 `mock-reach`、`mock-pick-place`、`scene_mock_reach_lab`、`joint-p-control`、`max_steps`、`action_scale`、`reward_distance_weight`。
- 删除 `quick-start`、`research-workspace` 等硬编码 Project ID；必须来自已选择 Class 下的正式 Project。
- 删除 Mock/Edge-GPU provider 切换、sandbox template、synthetic train loss、Three.js Mock arm 和假 success rate 在正式 S1 页面中的入口。
- `environment_id/strategy_id` 旧展示替换为 Project 返回的 task/environment 与 algorithm；不得让学生编辑 Worker 的 16 个底层字段。
- 指标改用真实 `average_reward`、`policy_loss`、`value_loss`、`entropy`、`episode_length`、`episodes`、training/update steps；当前不得展示 success rate。
- legacy `/api/runs*` fallback 和 URL query token 不再用于正式 S1 流程，但旧页面兼容代码可与新页面隔离，不能冒充 canonical。

## 3. 完整学生页面流程

`登录→选择Organization→选择Course→选择Class→Project列表→S1 Push详情→设置允许参数→创建Run→Run观察→停止或完成→Evaluation/Artifact/Compare/Export`

- Organization、Course、Class 逐级由 canonical API 加载，并保留所选 ID 作为页面上下文；无权限/无 enrollment 时不显示不可访问资源。
- Project 列表只展示当前 Class 分配且学生可访问的 Project；正式 Project API 未冻结前不得复用 WorkspaceProject 代替。
- 创建 Run 后立即进入 `/runs/{run_id}`；浏览器刷新、路由离开、WS/WHEP 断开均不能调用 stop。
- terminal Run 可创建 Evaluation、查看/下载 Artifact、选择两次 Run Compare，并下载 canonical JSON Export。

## 4. Project实验页设计

- 实验说明区只读展示 Project 提供的 title、description、instructions、task/environment=`s1_push`、algorithm=`PPO` 和 5cm success 条件。
- 参数区完全消费 Backend `parameter_schema`；首版只生成 seed 控件。训练时长仅在 Project schema 明确提供“短/标准”枚举时显示，不提交 Worker 原始整数。
- 操作区包含开始、显式停止、错误提示及八状态：queued、initializing、running、finalizing、stopping、completed、stopped、failed。
- 观察区显示 Run.progress；若展示本地推算必须标为“估算”。指标使用真实字段，时间线按 canonical sequence 去重/补拉。
- 视频区使用 Stream Descriptor 的 overview WHEP URL；视频失败显示降级状态，不改变 Run。Input 只能标记为 recording-only，不宣称控制 PPO 或机器人。
- 结果区以 run_id 关联 canonical Evaluation、Artifact、Run Compare/Export；Evaluation 允许显示 `insufficient_evidence`，不得补造分数。

## 5. Backend接口需求

- 认证/上下文：`POST /api/v1/auth/login`、`GET /api/v1/me`、`GET /api/v1/organizations[/{id}]`。
- 教学层级：`GET /api/v1/courses?organization_id=...`、`GET /api/v1/courses/{id}`、`GET /api/v1/courses/{id}/classes`、`GET /api/v1/classes/{id}`。
- Project：需要三方冻结后的“按 Class 列表/读取 Project”能力；页面只需要关联 ID、说明文本、task/environment、algorithm、success 条件、parameter_schema 和权限，不在本报告定义请求字段。
- Run：`POST/GET /api/v1/runs`、`GET/POST /api/v1/runs/{id}`/`stop`，创建必须带 Idempotency-Key，并使用 Project 契约给出的 simulation 配置。
- Event：`GET /api/v1/runs/{id}/events?after_sequence=` 与同路径 WebSocket。
- 视频/Input：`GET /api/v1/runs/{id}/stream?camera_id=overview`、`WS /api/v1/runs/{id}/input`。
- Artifact：`GET /api/v1/runs/{id}/artifacts`、`GET /api/v1/runs/{id}/artifacts/{artifact_id}/download`。
- 结果：`POST /api/v1/evaluations`、`GET /api/v1/evaluations/{id}`、`POST /api/v1/runs/compare`、`POST /api/v1/runs/export`。
- Token 目标状态只保存在运行时内存，不进 localStorage/sessionStorage/URL；硬刷新后没有可恢复凭据，应回登录页，除非未来另行冻结安全的服务端会话机制。
- HTTP 统一映射：401 清内存并登录；403 权限不足；409 状态/幂等/版本冲突；422 按 detail 定位表单字段；503 显示仿真或服务暂不可用并保留 Run 查询入口。

## 6. WebSocket与视频

- 浏览器 WebSocket 无自定义 Authorization header：连接后第一条发送 `{"type":"AUTH","access_token":"..."}`，收到 `AUTH_ACK` 后才处理 Event 或发送 Input；token 不得进入 URL。
- Event 使用 `after_sequence` 补拉、event_id 去重、sequence 排序和有界重连；terminal 后关闭实时连接，HTTP Run 仍为权威。
- Input 仅在 simulation+running+已绑定 task 时建立；页面关闭、非法输入或连接失败不得停止训练。
- WHEP player消费服务端 descriptor，不显示内部 RTSP URL；断开可重连，页面卸载只关闭 PeerConnection。
- Vite 当前 `/api` proxy 已启用 `ws:true`，足以代理 canonical Event/Input；`/ws` 仅留给 legacy 页面。

## 7. 实现文件建议

- `src/api.ts`：改为内存 token、canonical 类型、状态码/422错误模型，并增加鉴权 Artifact blob 下载。
- `src/App.tsx`：后续拆分 Login、TeachingContext、ProjectList、S1Project、RunDetail、Evaluation/Compare 页面；本阶段不重构。
- `src/runEvents.ts`：保留 canonical normalize/merge/cursor，增加 AUTH_ACK 门控；不得兼容假 metric 名称。
- `src/store.ts`：移除 `joint-p-control/action_scale` 默认值，只保存选择上下文、Project schema生成值和当前 Run。
- 建议新增独立 `authSession.ts`、`websocketClient.ts`、`WhepPlayer.tsx`，避免三处重复 query-token WebSocket。

## 8. 当前缺口

- 正式 Project/Class assignment/parameter_schema 契约和前端读取接口尚未冻结，真实主流程无法闭环。
- 内存 token 与硬刷新免登录不可同时成立；当前 sessionStorage 恢复方式必须替换，是否增加服务端安全会话需另行决策。
- canonical Evaluation 没有列表/报告比较端点，当前 Evaluation UI 数据模型与 v1 单 Run报告不兼容。
- Input 前端完全未接；Artifact 只有列表没有认证下载；WHEP 已实现但尚无正式跨模块浏览器验收。
- Simulation 当前不提供 success rate 或连续真实 progress；页面必须删除假曲线并诚实标注可用指标。
- 单体 `App.tsx` 超过 4500 行且混合 Mock/legacy/canonical；实现时应分阶段隔离，不在本审计阶段正式重构。
