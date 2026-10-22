# Embodied Arm Studio

面向课程学习、受限项目工作区和S1机械臂仿真的团队私有教学平台。真实机器人控制保持关闭。

## 新用户唯一推荐入口

先读 从零安装与使用说明，按顺序完成：

1. 获取私有仓库/交付文件与镜像读取权限。
2. 在部署机器的Bash终端完成Docker GHCR认证；GitHub登录、Docker认证、网页账号是三件不同的事。
3. 运行启动引导，首次建议选Core，按提示设置安装名/端口和你自己的账号密码。
4. 打开引导最后输出的网址，使用本次新建账号登录，查看课程和章节。

完成前两步后，唯一推荐的首次安装及再次启动命令是：

```bash
python3 scripts/start.py --config .env.team-core
```

Core不需要GPU、Isaac Gym或宿主Node/npm。不要把源码构建、make开发栈或逐条self_host init/up当作新用户的并列安装入口。GPU是同一引导中的可选模式，必须先具备合法外部依赖及空闲GPU；具体输入见安装说明。

## 能力边界

Core提供网页、课程、账号、教材与项目编辑；真实仿真、训练、直播和训练制品需要GPU模式。可选教学Agent的模型服务另由部署者配置，不是安装前置，也不应把GitHub令牌填入网页。

[English](README.md) · [依赖与分发边界](NOTICE.md)
