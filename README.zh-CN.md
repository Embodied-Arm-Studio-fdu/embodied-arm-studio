# Embodied Arm Studio

[English](README.md) | 简体中文

面向具身智能课程的机械臂仿真学习平台。阅读教材与交互图示，配置课程实验，运行 S1 仿真，查看训练指标、录像和评估结果。现有教材以中文为主。

- **学习与实践**：课程、教学班、项目工作区，以及支持修改审核与快照的可选教学助手。
- **仿真与训练**：固定控制器演示和 PPO 训练，受任务可用状态、课程权限及部署资源约束。
- **结果分析**：运行历史、指标曲线、录像、模型权重、评估报告及认证制品下载。

## 界面预览

![学习页面：章节导航与感知、决策、动作交互图示](docs/images/learning.png)

图中为学习界面；仿真实验另提供指标以及实际生成的视频和制品。截图用于展示界面，不作为策略效果证明。

## 从这里开始

| 文档 | 适用场景 |
| --- | --- |
| [快速开始](docs/quickstart.md) | 安装平台并首次登录 |
| [使用指南](docs/user-guide.md) | 阅读课程、配置实验、查看结果 |
| [部署说明](docs/deployment.md) | GPU 要求、远程访问、日常维护及故障处理 |
| [版本下载](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/releases) | 部署包与校验值；当前需要仓库访问权限 |

Core 模式提供账号、课程和项目编辑，无需 GPU。仿真、训练和实验视频需要 GPU 模式：Linux x86_64、NVIDIA GPU，以及部署者自行合法取得的 Isaac Gym Preview 4 SDK 和完整 S1 资产。取得项目并按安装指南认证后，运行 `python3 scripts/start.py --config .env.team-core`。

仓库和预构建 GHCR 镜像当前均需访问权限与认证；仓库权限和镜像权限分别配置。目前不提供公众免登录镜像安装。镜像固定到 digest，SDK 和完整机器人资产作为外部依赖。分发与依赖条款见 [NOTICE](NOTICE.md)。

## 支持范围

任务目录实现了 Push、Reach、Grasp、Lift、Lower/Release、Place 和固定控制演示。能否使用取决于课程项目权限与 GPU 服务；已有实现不等于每项任务的训练质量均已验证。Grasp、Lift 和 Lower/Release 使用随包提供的参考初始化权重，属于从预训练策略初始化的训练。Run 完成后仍需结合本次完整评估和物理成功条件判断效果。

真实机器人控制关闭。Windows、macOS、WSL2 和 ARM64 宿主安装尚无完整验证。直播还取决于浏览器与网络配置；SSH 和媒体端口配置见部署说明。

## 团队与分工

**[adaa77](https://github.com/adaa77) — 后端平台与部署交付。** 主要负责平台后端设计与开发，包括用户与权限、课程及实验项目管理、Run 生命周期与数据持久化、仿真服务对接、实验结果与制品管理，以及部署、系统集成和发布交付。

**[Kexin333](https://github.com/Kexin333) — 机器人仿真与强化学习训练。** 主要负责机械臂仿真环境和训练任务开发，包括任务观测、动作与奖励设计，强化学习训练流程、策略评估及模型保存，以及仿真执行和结果输出对接。

**[lxlhhh](https://github.com/lxlhhh) — 前端设计与开发。** 主要负责平台界面与交互设计、前端开发，包括课程与学习页面、实验配置与运行界面、训练指标可视化，以及实时视频、录像和结果查看的前端集成。

## 问题反馈

通过 [GitHub Issues](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/issues)反馈错误和安装问题，提供版本、复现步骤及去敏后的错误信息。不要提交密码、令牌、私有配置或学习者数据。
