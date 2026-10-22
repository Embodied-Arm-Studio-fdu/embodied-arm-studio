# Embodied Arm Studio

A team-private teaching platform for course materials, controlled project workspaces and S1 simulation. Physical robot control stays disabled.

## One recommended new-user entry

Follow the installation and first-use guide in order:

1. Obtain private delivery/repository and image-read access.
2. Authenticate Docker to GHCR on the deployment machine. GitHub sign-in, Docker registry authentication and application accounts are separate.
3. Run the guided installer in an interactive Bash terminal. Choose Core for the first handoff, select a unique installation name and set your own account passwords.
4. Open the URL printed by the installer, sign in with the accounts you just created and inspect the course materials.

After the access and registry steps, use the same command for first installation and later startup:

```bash
python3 scripts/start.py --config .env.team-core
```

Core needs neither a GPU/Isaac Gym nor host Node/npm. Developer builds, make targets and manual self_host init/up sequences are not alternative recommended new-user entries. GPU is an optional selection in the same guide, subject to external dependencies and an idle GPU.

Core provides accounts, courses, textbook content and project editing. Real simulation, training, live video and training artifacts require GPU mode. Optional teaching-agent providers need separate operator configuration; they are not prerequisites for reading courses.

[中文](README.zh-CN.md) · [Dependency and distribution notices](NOTICE.md)
