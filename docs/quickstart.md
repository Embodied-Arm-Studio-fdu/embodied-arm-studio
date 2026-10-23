# 快速开始

本文使用预构建镜像安装 Core 模式，完成后即可登录、阅读课程和查看项目。Core 不需要 GPU、Isaac Gym 或宿主机 Node/npm。需要运行真实仿真实验时，继续阅读[部署说明](deployment.md)。

## 1. 准备环境与权限

在 Linux amd64 部署机器上准备：

- Python 3.8 或更高版本。
- Docker Engine、Docker Compose v2.20 或更高版本，以及当前用户的 Docker 访问权限。
- Git，以及团队私有仓库和 API、Web 镜像的读取权限。
- 可访问 GitHub、GHCR 和 Docker Hub 的网络，及用于镜像和运行数据的磁盘空间。镜像为 GB 量级，请勿按源码大小估算空间。

在部署机器的终端确认环境可用：

```bash
python3 --version
docker compose version
docker info
```

如果你只使用已经部署的平台，直接打开维护人提供的网址，按[使用指南](user-guide.md)登录即可，无需安装。

## 2. 获取项目并登录镜像仓库

使用已获仓库读取权限的 GitHub 身份克隆项目：

```bash
git clone https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio.git
cd embodied-arm-studio
```

私有仓库需要 GitHub 认证。按团队要求配置 Git 凭据或在提示时认证，不将令牌写入克隆地址。若已从维护人提供的私有渠道取得并解压部署文件，进入含 `scripts/` 和 `deploy/` 的项目根目录即可。当前没有正式发布的公开 Release 下载入口。

然后在同一台部署机器的 Bash 终端登录 GHCR。使用有相应镜像读取权的账号和具备 `read:packages` 权限的令牌；组织要求 SSO 时需完成相应授权。

```bash
read -rp 'GitHub login: ' GHCR_USER
read -rsp 'GHCR read token: ' GHCR_READ_TOKEN
echo
printf '%s' "$GHCR_READ_TOKEN" | docker login ghcr.io --username "$GHCR_USER" --password-stdin
unset GHCR_READ_TOKEN GHCR_USER
```

Git 仓库认证和 Docker 镜像认证分别配置。令牌只用于认证，不填写到平台登录页，也不发送给他人。

## 3. 运行启动引导

```bash
python3 scripts/start.py --config .env.team-core
```

首次安装按以下示例填写：

| 提示 | 建议填写 |
| --- | --- |
| 模式 core / gpu | `core` |
| 独立安装名称 | `team-core`，或本机未使用的其他名称 |
| 网页端口 | `8080`，如被占用则换一个空闲端口 |
| 监听 IPv4 | `127.0.0.1` |
| 浏览器访问的 IPv4 或主机名 | `127.0.0.1` |

引导会拉取固定版本镜像、启动服务并初始化课程账号。按提示为 `studio-administrator`、`studio-instructor`、`studio-student` 分别设置密码，没有默认密码。首次下载可能较慢，等待终端输出安装完成和访问网址。

保留生成的 `.env.team-core`，后续启动和维护都使用这份配置。它含私有凭据，不要提交到仓库或分享。

## 4. 登录并开始学习

在部署机器的浏览器中打开引导输出的网址；以上示例为 `http://127.0.0.1:8080`。使用 `studio-student` 和刚设置的密码登录，在“入门学习”中阅读章节，在“课程与高级任务”中查看所属课程、教学班和项目。

浏览器在另一台电脑时，在那台电脑上执行并保持 SSH 连接：

```bash
ssh -N -L 8080:127.0.0.1:8080 YOUR_SSH_USER@YOUR_SERVER
```

替换自己的 SSH 用户名和服务器地址，再打开 `http://127.0.0.1:8080`。如果安装时更改了网页端口，同步更改转发端口。

以后从项目根目录运行相同启动命令，即可复用配置、账号和数据：

```bash
python3 scripts/start.py --config .env.team-core
```

接下来阅读[使用指南](user-guide.md)。状态、日志、停止服务和安装问题见[部署说明](deployment.md)。
