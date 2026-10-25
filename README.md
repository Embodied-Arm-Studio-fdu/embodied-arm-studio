# Embodied Arm Studio

English | [简体中文](README.zh-CN.md)

A robotic arm simulation learning platform for embodied AI courses. Read lessons and interactive diagrams, configure course experiments, run S1 simulations, and inspect training metrics, recordings and evaluation results. The existing course material is primarily in Chinese.

- **Learn and experiment:** courses, classes, project workspaces and an optional teaching assistant with reviewed edits and snapshots.
- **Simulate and train:** fixed-controller demonstrations and PPO training, subject to task availability, course permissions and deployment resources.
- **Inspect results:** run history, metric charts, recordings, checkpoints, evaluation reports and authenticated artifact downloads.

## A look inside

![Learning page with chapter navigation and an interactive perception–decision–action diagram](docs/images/learning.png)

The learning interface; simulation runs additionally provide metrics and available video and artifacts. A screenshot illustrates the interface and does not establish policy performance.

## Get started

| Guide | Purpose |
| --- | --- |
| [Installation](docs/quickstart.en.md) | Install the platform and sign in for the first time |
| [User guide (中文)](docs/user-guide.md) | Read courses, configure experiments and inspect results |
| [Deployment guide (中文)](docs/deployment.md) | GPU requirements, remote access, maintenance and troubleshooting |
| [Downloads](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/releases) | Deployment packages and checksums; repository access is currently required |

Core mode provides accounts, courses and project editing without a GPU. Simulation, training and experiment video require GPU mode on Linux x86_64 with an NVIDIA GPU, an operator-supplied Isaac Gym Preview 4 SDK and complete S1 assets. After obtaining the project and authenticating, run `python3 scripts/start.py --config .env.team-core` as described in the installation guide.

The repository and prebuilt GHCR images currently require access and authentication. Repository access and image access are separate permissions. Public, login-free image installation is not available. Images are pinned by digest; SDKs and complete robot assets are external dependencies. See [distribution and dependency notices](NOTICE.md).

## Supported scope

The task catalog implements Push, Reach, Grasp, Lift, Lower/Release, Place and fixed-controller demonstrations. Availability depends on project permissions and GPU services; implementation alone does not establish training quality for every task. Grasp, Lift and Lower/Release use bundled reference initialization checkpoints, so their training is initialized from pretrained policies. A completed run must be assessed against its own full evaluation and physical success criteria.

Physical robot control is disabled. Windows, macOS, WSL2 and ARM64 host installations have not been fully validated. Live video also depends on browser and network configuration; follow the deployment guide for SSH and media ports.

## Team

**[adaa77](https://github.com/adaa77) — Backend Platform & Deployment.** Primarily responsible for backend platform design and development: authentication and access control, course and experiment project management, run lifecycle and persistence, simulation integration, results and artifact management, deployment, system integration and release delivery.

**[Kexin333](https://github.com/Kexin333) — Simulation & Robot Learning.** Primarily responsible for robotic simulation environments and training tasks: observation, action and reward design, reinforcement learning workflows, policy evaluation, checkpointing, and integration of simulation execution and result output.

**[lxlhhh](https://github.com/lxlhhh) — Frontend Design & Development.** Primarily responsible for interface and interaction design and frontend development: learning and course pages, experiment configuration and run monitoring, training visualizations, and frontend integration of live video, replay and experiment results.

## Feedback

Report bugs and installation questions through [GitHub Issues](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/issues). Include the version, steps to reproduce and sanitized error details. Keep passwords, tokens, private configuration and learner data out of reports.
