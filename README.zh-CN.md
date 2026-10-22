

# Embodied Arm Studio

[English](README.md) · [第三方依赖与发布说明](NOTICE.md)

面向新部署者的入口是 `python3 scripts/start.py`，详见简洁启动指南；`self_host.py` 是详细运维和显式源码构建入口。默认使用独立 Compose 项目和 8080 端口；完整 GPU 模式见上面的部署指南。下文保留已有本地开发/实验运行方式。

具身智能机械臂教学研究平台。当前版本围绕“课程设计、受限项目工作区、环境与仿真、训练证据、评估、Pipeline、导出和教学 Agent”形成可审计的教学闭环。所有运行状态均明确标记为 `ready`、`mock`、`planned` 或 `blocked`，不会将教学 Mock 描述为真实 GPU 训练、机器人控制或 Sim2Real。

## 本地启动

推荐使用统一的 Docker Compose 本地栈：

```bash
make stack-up
make stack-status
```

首次执行 `make stack-up` 会创建权限为 `0600` 的私有 `.env`，生成并保存随机 PostgreSQL 密码和 JWT 签名密钥。后续启动会复用该文件，不会因重启重新生成密钥。Compose 会构建固定依赖的 API/Web 镜像，依次等待 PostgreSQL 健康、Alembic 迁移完成、API 健康后启动网页；网页地址为 `http://localhost:5173`。

如果机器上已有本项目的 PostgreSQL 和 API 容器，但尚无 `.env`，不要生成新密码或删除旧卷。先执行 `./scripts/studio.sh adopt`；它会从现有 Compose 容器采纳数据库和 JWT 配置，以 `0600` 权限写入 `.env`，且不会在终端回显秘密。随后再执行 `make stack-up` 原地升级同一 Compose 项目。

三个长驻服务都使用 `restart: unless-stopped`。Docker daemon 在宿主机重启后恢复时，PostgreSQL、API 和网页会自动恢复；命名卷 `postgres_data` 和 `studio_data` 分别保存数据库与本地运行文件。常用运维命令：

```bash
make stack-status
make stack-logs
make stack-restart
make stack-down
```

`make stack-down` 只停止并移除容器，不删除命名卷。不要使用 `docker compose down -v`，除非明确要销毁本地数据。项目脚本不能启动宿主机 Docker daemon；如果命令报告 daemon unavailable，应先在宿主机恢复 Docker 服务。Compose 也不负责宿主机 SSH 或 VS Code Remote 服务，推荐由 VS Code Dev Containers 直接通过 Docker 连接工作区，避免依赖容器内手工启动的 `sshd`。

无需 PostgreSQL 的快速开发仍可使用宿主进程模式。宿主环境需要 Python 3.8+ 和项目内的 Node 20 工具链：

```bash
make install-web
make dev
```

`make dev` 会同时启动 API 和网页；任一服务异常退出时会停止另一服务，避免留下失效端口。API 默认在 `http://127.0.0.1:8000`，网页在 `http://localhost:5173`。Vite 会代理 `/api` 和 `/ws`，因此远程 VS Code 工作区也应访问前端端口，而不是直接暴露 API 端口。

`make dev` 是临时前台进程，不具备开机恢复能力，也不会读取或启动 Compose PostgreSQL。需要可恢复的登录与数据库环境时使用 `make stack-up`。

远程工作区可使用 Vite 启动日志中的 `Network` 地址，例如 `http://<workspace-host>:5173/evaluation`。使用 `Ctrl-C` 停止 `make dev` 时，API 和 Vite 会一起退出。可通过 `curl -f http://127.0.0.1:5173/api/health` 验证网页代理和 API 是否同时可用。

```bash
make test
make mock-demo
```

PostgreSQL 测试必须同时设置 `POSTGRES_TEST_OPT_IN=1` 和独立的 `TEST_DATABASE_URL`。测试库名必须包含 `test`，且代码会拒绝已知运行库或与当前运行库指纹相同的目标，避免测试清理 demo 数据。

使用其他模型有两种方式：生产部署推荐在服务端设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 或 `ANTHROPIC_API_KEY`；教学演示也可以打开网页中的 Agent 设置，选择 `DeepSeek`、填写自己的 API Key、模型（默认 `deepseek-chat`）和 `https://api.deepseek.com`，点击“配置 Provider”。网页只通过 HTTPS 将 Key 发送给当前 API，状态接口不会回显 Key，浏览器不会写入 Key；网页配置只保存在当前 API 进程内存中，API 重启后自动清空。生产环境不要通过聊天消息提交 Key。

Agent 权限固定为五档，并由服务端保存不可由浏览器覆盖的 capability snapshot：

| 档位 | 服务端能力 |
| --- | --- |
| Level 1 | 解释概念 |
| Level 2 | 解释概念、Socratic 提示 |
| Level 3 | Level 2、审查学生方案 |
| Level 4 | Level 3、读取当前页面上下文、配置建议 |
| Level 5 | Level 4、读取受限 Workspace、Scene、Run evidence，并生成受审批的 Patch 提案 |

Level 5 也不能直接写文件、自动审批 Patch、执行 Shell、伪造证据、控制真实机械臂或绕过关节限位和人工确认。Workspace 读取最多 12 个文件、单文件 4000 字符、总计 16000 字符；页面标识、Scene 和 Run 必须属于当前项目。悬浮 Agent 回复中的 `READ` 标签表示本次请求实际使用的服务端只读工具。

## 当前能力

| 模块 | 状态 | 当前实现 |
| --- | --- | --- |
| 研究路线、课程与练习 | `ready` | 研究阶段、能力边界和教学任务入口 |
| 项目工作区 | `ready` | 受限文件树、多标签 Monaco 编辑、自动保存、搜索、上传下载、CRUD、快照、文件历史、版本恢复、Starter Diff 与恢复 |
| 环境设计 | `ready` | Three.js 场景编辑与结构化场景配置 |
| 仿真工作台 | `mock` | 教学 Mock 仿真，不代表 Isaac Lab、ManiSkill 或真实动力学接入 |
| 代码工作区 | `ready` | Monaco 教学代码编辑与运行配置入口 |
| 数据集 | `planned` | 尚未接入真实数据集目录和采集链路 |
| 训练、运行证据、评估与对比 | `ready` | 可复现 Mock 运行、事件流、证据记录、评估和运行对比 |
| Pipeline 与工件导出 | `ready` | 结构化流程、证据门禁和教学 manifest 导出 |
| 教学 Agent | `ready` | 网页 Provider 配置、五档 capability、受控只读上下文、Patch 审批与回滚 |
| 机器人设备 | `planned` | 不提供真实设备控制 |
| 部署审查与 Sim2Real | `blocked` | 缺少真实证据时保持阻断 |

项目工作区通过 `/api/v1/projects` 和 `/api/v1/workspaces/*` 访问项目内文件。服务端执行扩展名、文件大小、路径穿越和符号链接检查；浏览器不会获得宿主文件系统路径。

## 验证

```bash
/usr/bin/python3 -m unittest discover -s tests -v
cd apps/web && npm run build
```

## 边界

Mock 是完整教学闭环，不代表 GPU、Shifu Adapter、WebRTC、真实数据集或机器人设备已经接入。Shifu、Isaac Lab、ManiSkill、LeRobot 等策略或仿真后端应实现 Adapter 并提供可验证证据后再接入。manifest 可作为教学制品生成；ONNX/TorchScript 模型导出需要通过证据评估门禁。任何导出都不驱动真实机械臂。

