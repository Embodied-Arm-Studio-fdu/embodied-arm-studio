from __future__ import annotations

import hashlib
import base64
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence


class SandboxStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class SandboxTemplate:
    template_id: str
    image: str
    command: Sequence[str]
    requires_gpu: bool
    timeout_seconds: int
    cpu_limit: float
    memory_mb: int
    process_limit: int


@dataclass
class SandboxJob:
    job_id: str
    template_id: str
    status: SandboxStatus = SandboxStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int
    stdout: str
    stderr: str


class ContainerBackend(Protocol):
    def run(
        self,
        job_id: str,
        template: SandboxTemplate,
        arguments: Sequence[str],
        timeout_seconds: int,
    ) -> ContainerResult: ...

    def stop(self, container_name: str) -> None: ...


class DockerCliBackend:
    def run(
        self,
        job_id: str,
        template: SandboxTemplate,
        arguments: Sequence[str],
        timeout_seconds: int,
    ) -> ContainerResult:
        completed = subprocess.run(
            ["docker"] + list(arguments),
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return ContainerResult(completed.returncode, completed.stdout, completed.stderr)

    def stop(self, container_name: str) -> None:
        subprocess.run(
            ["docker", "stop", "--time", "2", container_name],
            capture_output=True,
            check=False,
            timeout=5,
        )


class RemoteWorkerBackend:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_environment(cls) -> Optional["RemoteWorkerBackend"]:
        base_url = os.environ.get("STUDIO_SANDBOX_WORKER_URL", "").strip()
        token = os.environ.get("STUDIO_SANDBOX_WORKER_TOKEN", "").strip()
        token_file = os.environ.get("STUDIO_SANDBOX_WORKER_TOKEN_FILE", "").strip()
        if token and token_file:
            raise ValueError("Sandbox Worker Token 和 Token 文件不能同时配置")
        if token_file:
            token_file_mode = os.stat(token_file).st_mode & 0o777
            if token_file_mode & 0o077:
                raise ValueError("Sandbox Worker Token 文件不能被组或其他用户访问")
            token = Path(token_file).read_text(encoding="utf-8").strip()
        if not base_url and not token and not token_file:
            return None
        if not base_url or not token:
            raise ValueError("Sandbox Worker URL 和 Token 必须同时配置")
        return cls(base_url, token)

    def run(
        self,
        job_id: str,
        template: SandboxTemplate,
        arguments: Sequence[str],
        timeout_seconds: int,
    ) -> ContainerResult:
        payload = self._request("/v1/run", {
            "job_id": job_id,
            "template_id": template.template_id,
        }, timeout_seconds + 5)
        return ContainerResult(
            int(payload["exit_code"]),
            str(payload.get("stdout", "")),
            str(payload.get("stderr", "")),
        )

    def stop(self, container_name: str) -> None:
        self._request("/v1/stop", {"job_id": container_name}, 8)

    def _request(self, path: str, payload: Dict[str, object], timeout_seconds: int) -> Dict[str, object]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("Sandbox Worker 拒绝请求: %s %s" % (exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("无法连接 Sandbox Worker: %s" % exc.reason) from exc


GPU_SMOKE_TEMPLATE = SandboxTemplate(
    template_id="gpu-smoke",
    image="nvidia/cuda:12.4.1-base-ubuntu22.04",
    command=(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ),
    requires_gpu=True,
    timeout_seconds=30,
    cpu_limit=1.0,
    memory_mb=512,
    process_limit=32,
)

GPU_TRAINING_SCRIPT = """import base64,json,torch
torch.manual_seed(42)
device=torch.device('cuda:0')
x=torch.linspace(-1,1,4096,device=device).unsqueeze(1)
y=2.5*x-0.4+0.05*torch.sin(12*x)
model=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.02)
loss_fn=torch.nn.MSELoss()
metrics=[]
for epoch in range(1,81):
    optimizer.zero_grad(set_to_none=True)
    prediction=model(x)
    loss=loss_fn(prediction,y)
    loss.backward()
    optimizer.step()
    if epoch==1 or epoch%10==0:
        metric={'epoch':epoch,'train_loss':round(float(loss.detach().cpu()),8)}
        metrics.append(metric)
        print('STUDIO_METRIC '+json.dumps(metric,separators=(',',':')),flush=True)
checkpoint={'schema_version':'1.0','framework':'pytorch','device':torch.cuda.get_device_name(0),'seed':42,'epochs':80,'final_loss':metrics[-1]['train_loss'],'state_dict':{name:tensor.detach().cpu().tolist() for name,tensor in model.state_dict().items()}}
encoded=base64.b64encode(json.dumps(checkpoint,separators=(',',':')).encode()).decode()
print('STUDIO_ARTIFACT checkpoint.json '+encoded,flush=True)
"""

GPU_PYTORCH_TRAIN_TEMPLATE = SandboxTemplate(
    template_id="gpu-pytorch-train",
    image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
    command=("python", "-c", GPU_TRAINING_SCRIPT),
    requires_gpu=True,
    timeout_seconds=180,
    cpu_limit=2.0,
    memory_mb=2048,
    process_limit=64,
)

SANDBOX_TEMPLATES = {
    template.template_id: template
    for template in (GPU_SMOKE_TEMPLATE, GPU_PYTORCH_TRAIN_TEMPLATE)
}


def build_docker_arguments(job_id: str, template: SandboxTemplate) -> List[str]:
    arguments = [
        "run",
        "--rm",
        "--name", job_id,
        "--hostname", "sandbox",
        "--network", "none",
        "--read-only",
        "--init",
        "--user", "65534:65534",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(template.process_limit),
        "--cpus", str(template.cpu_limit),
        "--memory", "%sm" % template.memory_mb,
        "--memory-swap", "%sm" % template.memory_mb,
        "--ulimit", "nofile=256:256",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
    ]
    if template.requires_gpu:
        arguments.extend(["--gpus", "device=0"])
    arguments.append(template.image)
    arguments.extend(template.command)
    return arguments


class SandboxRunner:
    def __init__(self, storage_root: Path, backend: Optional[ContainerBackend] = None) -> None:
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or DockerCliBackend()
        self.templates = dict(SANDBOX_TEMPLATES)
        self._jobs: Dict[str, SandboxJob] = {}
        self._lock = threading.Lock()

    def list_templates(self) -> List[Dict[str, object]]:
        return [
            {
                "template_id": template.template_id,
                "requires_gpu": template.requires_gpu,
                "timeout_seconds": template.timeout_seconds,
                "cpu_limit": template.cpu_limit,
                "memory_mb": template.memory_mb,
            }
            for template in self.templates.values()
        ]

    def list_jobs(self) -> List[Dict[str, object]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.to_dict() for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)]

    def submit(self, template_id: str) -> SandboxJob:
        if template_id not in self.templates:
            raise ValueError("未知 Sandbox 模板: %s" % template_id)
        digest = hashlib.sha1((template_id + datetime.now(timezone.utc).isoformat()).encode("utf-8")).hexdigest()[:12]
        job = SandboxJob(job_id="sandbox-" + digest, template_id=template_id)
        with self._lock:
            self._jobs[job.job_id] = job
        self._persist(job)
        threading.Thread(target=self._execute, args=(job,), daemon=True, name=job.job_id).start()
        return job

    def get(self, job_id: str) -> SandboxJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError("未知 Sandbox Job: %s" % job_id) from exc

    def stop(self, job_id: str) -> SandboxJob:
        job = self.get(job_id)
        if job.status in (SandboxStatus.QUEUED, SandboxStatus.RUNNING):
            self.backend.stop(job.job_id)
            job.status = SandboxStatus.STOPPED
        return job

    def logs(self, job_id: str) -> Dict[str, str]:
        self.get(job_id)
        job_dir = self.storage_root / job_id
        return {
            "stdout": self._read_log(job_dir / "stdout.log"),
            "stderr": self._read_log(job_dir / "stderr.log"),
        }

    def artifacts(self, job_id: str) -> List[Dict[str, object]]:
        self.get(job_id)
        manifest_path = self.storage_root / job_id / "artifacts.json"
        if not manifest_path.exists():
            return []
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def artifact_path(self, job_id: str, name: str) -> Path:
        self.get(job_id)
        if name != "checkpoint.json":
            raise ValueError("未知 Sandbox 工件")
        path = self.storage_root / job_id / name
        if not path.exists():
            raise KeyError("Sandbox 工件尚未生成")
        return path

    def audit(self, job_id: str) -> Dict[str, object]:
        job = self.get(job_id)
        template = self.templates[job.template_id]
        return {
            "job": job.to_dict(),
            "template": {
                "template_id": template.template_id,
                "image": template.image,
                "command": list(template.command),
                "requires_gpu": template.requires_gpu,
            },
            "policy": {
                "network": "none",
                "root_filesystem": "read-only",
                "user": "65534:65534",
                "capabilities": "drop-all",
                "no_new_privileges": True,
                "docker_socket": False,
                "host_mounts": False,
                "gpu_devices": ["0"] if template.requires_gpu else [],
                "cpu_limit": template.cpu_limit,
                "memory_mb": template.memory_mb,
                "process_limit": template.process_limit,
                "timeout_seconds": template.timeout_seconds,
            },
        }

    def _execute(self, job: SandboxJob) -> None:
        template = self.templates[job.template_id]
        job_dir = self.storage_root / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job.status = SandboxStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._persist(job)
        try:
            arguments = build_docker_arguments(job.job_id, template)
            result = self.backend.run(job.job_id, template, arguments, template.timeout_seconds)
            (job_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (job_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
            self._extract_artifacts(job_dir, result.stdout)
            job.exit_code = result.exit_code
            if job.status != SandboxStatus.STOPPED:
                job.status = SandboxStatus.COMPLETED if result.exit_code == 0 else SandboxStatus.FAILED
                if result.exit_code != 0:
                    job.error = "Sandbox 进程退出码: %s" % result.exit_code
        except subprocess.TimeoutExpired:
            self.backend.stop(job.job_id)
            job.status = SandboxStatus.FAILED
            job.error = "Sandbox 运行超时"
        except Exception as exc:
            job.status = SandboxStatus.FAILED
            job.error = str(exc)
        finally:
            job.ended_at = datetime.now(timezone.utc).isoformat()
            self._persist(job)

    @staticmethod
    def _read_log(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @staticmethod
    def _extract_artifacts(job_dir: Path, stdout: str) -> None:
        manifest: List[Dict[str, object]] = []
        for line in stdout.splitlines():
            if not line.startswith("STUDIO_ARTIFACT "):
                continue
            _, name, encoded = line.split(" ", 2)
            if name != "checkpoint.json":
                raise ValueError("Sandbox 返回了未允许的工件")
            content = base64.b64decode(encoded, validate=True)
            if len(content) > 1024 * 1024:
                raise ValueError("Sandbox 工件超过 1 MiB 限制")
            json.loads(content.decode("utf-8"))
            artifact_path = job_dir / name
            artifact_path.write_bytes(content)
            manifest.append({
                "name": name,
                "media_type": "application/json",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        (job_dir / "artifacts.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _persist(self, job: SandboxJob) -> None:
        job_dir = self.storage_root / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        temporary = job_dir / "job.json.tmp"
        temporary.write_text(json.dumps(self.audit(job.job_id), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(job_dir / "job.json")