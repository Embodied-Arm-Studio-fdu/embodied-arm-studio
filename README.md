

# Embodied Arm Studio

A self-hosted learning and research platform for embodied AI and S1 robot-arm simulation. Read the 12-chapter Chinese/English textbook, explore interactive diagrams, edit controlled rewards, run PPO experiments, inspect videos and artifacts, compare evaluations, and use the teaching assistant.

[中文说明](README.zh-CN.md) · [Dependency and distribution notices](NOTICE.md)

## Start on your own machine

Install Python 3.8+ on the host, Docker Engine and Docker Compose v2.20+. Use an ordinary terminal for the interactive account bootstrap.

```bash
# Run these commands from the repository root after cloning/extracting it.
python3 scripts/self_host.py init
python3 scripts/self_host.py doctor
python3 scripts/self_host.py up
python3 scripts/self_host.py bootstrap
```

Open **http://localhost:8080**. Bootstrap asks you to choose passwords for `studio-administrator`, `studio-instructor`, and `studio-student`; there are no default passwords. It creates an isolated course, class, enrollment, and all seven teaching Projects, including Place. Repeating bootstrap preserves existing users and compatible Projects.

The default `core` mode provides learning materials, accounts, courses, editing, and assistant configuration. **Real simulation training, recordings, and training-derived evaluations require GPU mode.** The deployment does not present Mock outputs as GPU results.

## Full GPU installation

Use a Linux x86_64 NVIDIA host with a compatible driver and NVIDIA Container Toolkit. The current simulator is **Isaac Gym Preview 4**, not Isaac Sim/Lab. The existing verified reference hardware is an RTX 4080; other hardware must pass the deployment checks. The worker container uses Ubuntu 20.04, Python 3.8, CUDA 12.1 and PyTorch 2.4.1.

Obtain Isaac Gym Preview 4 from [NVIDIA](https://developer.nvidia.com/isaac-gym/download), accept its terms, and supply a permitted S1 `yhrg_s1_description` directory with the URDF and all referenced meshes. No external SDK or entire research repository is bundled. The three required initialization checkpoints are supplied in a small checksum-verified reference bundle.

```bash
# Use a different config file if a core configuration already exists.
python3 scripts/self_host.py --config .env.gpu init --mode gpu \
  --public-host 192.168.1.50 --bind 0.0.0.0 \
  --isaac-gym-path /absolute/path/to/isaacgym \
  --s1-assets-path /absolute/path/to/yhrg_s1_description
python3 scripts/self_host.py --config .env.gpu doctor
python3 scripts/self_host.py --config .env.gpu up
python3 scripts/self_host.py --config .env.gpu gpu-check
python3 scripts/self_host.py --config .env.gpu bootstrap
```

Replace `192.168.1.50` with your deployment machine's browser-reachable LAN address or DNS name. Open `http://192.168.1.50:8080`. Stop an existing core stack before switching modes: both configurations default to the same project and persistent volumes. Use a distinct `--project` and port for an independent installation.

GPU mode adds Redis, the simulation API, one heavy GPU Worker, a persistent callback dispatcher, and MediaMTX. It supports the implemented Push, Reach, Grasp, Lift, Lower/Release and Place tasks and the six fixed control demonstrations. Physical robot control remains disabled; features explicitly marked planned/blocked remain so.

## Operate and distribute

```bash
python3 scripts/self_host.py status
python3 scripts/self_host.py logs
python3 scripts/self_host.py down
python3 scripts/export_release.py /tmp/embodied-arm-studio-source.tar.gz
```

For a custom config, place `--config PATH` before the command. `down` retains volumes. Protect `.env.selfhost` / `.env.gpu` (mode 0600), back up both the database and artifact volume, and never commit runtime credentials. See the guide for provider configuration, video networking, upgrades, backups, and troubleshooting.

The source exporter excludes dependencies, private environment files, users/databases, logs, videos, and non-reference training checkpoints. **Review `NOTICE.md` before public redistribution. The project-owned code license is pending the owner's decision.**

