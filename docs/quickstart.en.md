# Installation

English | [简体中文](quickstart.md)

Core mode provides accounts, course content and project editing without a GPU, SDK or host Node/npm installation. For GPU simulation and training, see the [deployment guide (中文)](deployment.md).

## Requirements

- Linux amd64, Python 3.8+, Git, Docker Engine and Docker Compose v2.20+, with Docker access for the current user.
- Access to this repository and its private API/Web images. GPU mode additionally needs Simulation/Worker image access.
- Network access to GitHub, GHCR and Docker Hub; disk space for GB-scale images and growing runtime data.

Check `python3 --version`, `docker compose version` and `docker info`. Existing platform users can open the URL supplied by their maintainer and follow the [user guide (中文)](user-guide.md).

## Obtain the project and authenticate

```bash
git clone https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio.git
cd embodied-arm-studio
```

The repository currently requires GitHub authentication. Do not put a token in the clone URL. Alternatively, obtain an archive and checksum from [Releases](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/releases) using an authorized account, verify it, extract it and enter its project root.

Authenticate Docker separately on the deployment machine, using an account and token allowed to read the images (`read:packages` and organization SSO authorization where required):

```bash
read -rp 'GitHub login: ' GHCR_USER
read -rsp 'GHCR read token: ' GHCR_READ_TOKEN
echo
printf '%s' "$GHCR_READ_TOKEN" | docker login ghcr.io --username "$GHCR_USER" --password-stdin
unset GHCR_READ_TOKEN GHCR_USER
```

Image access is required even when repository access is available. Do not enter a GitHub token on the platform login page.

## Start Core mode

```bash
python3 scripts/start.py --config .env.team-core
```

Choose `core`, an unused installation name such as `team-core`, an unused web port such as `8080`, bind address `127.0.0.1` and browser host `127.0.0.1`. The guide pulls digest-pinned images, starts services and initializes the course. Set and confirm separate passwords for `studio-administrator`, `studio-instructor` and `studio-student`; there are no default passwords.

Open the printed URL (for these settings, `http://127.0.0.1:8080`) and sign in as `studio-student`. Use the learning pages and the course/class/project navigation. Preserve `.env.team-core`: it contains private credentials and identifies this installation. Run the same startup command for subsequent starts to reuse its data and accounts.

For a browser on another computer, keep this tunnel open there, replacing the SSH user and server:

```bash
ssh -N -L 8080:127.0.0.1:8080 YOUR_SSH_USER@YOUR_SERVER
```

Open `http://127.0.0.1:8080` on that computer. Match the actual web port if you chose another one. GPU live video needs additional media settings and forwards from the deployment guide.

## GPU mode and maintenance

GPU mode additionally requires Linux x86_64 with an NVIDIA GPU, a CUDA 12.1-compatible driver, NVIDIA Container Toolkit, an operator-supplied Isaac Gym Preview 4 directory and complete S1 description/URDF/meshes. The installer prepares bundled reference initialization checkpoints; it does not install drivers, SDKs or change the firewall. Shared GPU use requires an idle window and a serial Worker handoff.

Follow the [GPU installation and maintenance guide (中文)](deployment.md). Use a separate configuration, installation name, web port and media ports for a separate installation. Never reuse another installation's database or private configuration. Real robot control is disabled; other host platforms have not been fully validated.

Report installation problems through [Issues](https://github.com/Embodied-Arm-Studio-fdu/embodied-arm-studio/issues), including the version, steps and sanitized errors. Keep credentials and private configuration out of reports.
