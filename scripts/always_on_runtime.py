#!/usr/bin/env python3
"""Keep the local Mock-only teaching stack alive without idle time limits."""

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
CONFIG = Path("/home/user/.config/embodied-arm-studio/midterm-runtime.env")
LOG_ROOT = ROOT / "var" / "log" / "always-on-runtime"
STOPPING = False


def load_environment(path):
    values = dict(os.environ)
    configured = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, encoded = line.split("=", 1)
        parsed = shlex.split(encoded, posix=True)
        if len(parsed) != 1:
            raise ValueError("invalid runtime environment entry: %s" % name)
        values[name] = parsed[0]
        configured[name] = parsed[0]
    # urllib gives lowercase proxy variables precedence. An explicit runtime
    # setting must override both spellings of inherited shell configuration.
    for scheme in ("http", "https", "all"):
        lower, upper = scheme + "_proxy", scheme.upper() + "_PROXY"
        if lower in configured or upper in configured:
            value = configured.get(lower, configured.get(upper))
            values[lower] = values[upper] = value
    values["PYTHONUNBUFFERED"] = "1"
    simulation_host = values.get("SIMULATION_HTTP_HOST", "127.0.0.1")
    no_proxy = "127.0.0.1,localhost,%s" % simulation_host
    values["NO_PROXY"] = no_proxy
    values["no_proxy"] = no_proxy
    return values


def service_specs(environment):
    simulation_environment = dict(environment)
    simulation_environment["PYTHONPATH"] = str(SIMULATION / "src")
    simulation_environment["SIMULATION_BACKEND_BASE_URL"] = "http://127.0.0.1:18000"
    worker_environment = dict(simulation_environment)
    worker_environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    worker_environment.pop("DISPLAY", None)
    backend_environment = dict(environment)
    backend_environment["PYTHONPATH"] = str(ROOT)
    backend_environment["SIMULATION_BASE_URL"] = "http://%s:%s" % (
        environment.get("SIMULATION_HTTP_HOST", "127.0.0.1"),
        environment.get("SIMULATION_HTTP_PORT", "8090"),
    )
    web_environment = dict(environment)
    web_environment["VITE_API_PROXY_TARGET"] = "http://127.0.0.1:18000"
    return (
        (
            "mediamtx",
            [environment["MEDIAMTX_BINARY_PATH"], str(SIMULATION / "config" / "mediamtx_midterm.yml")],
            ROOT,
            environment,
        ),
        (
            "simulation-adapter",
            ["/usr/bin/python3", "scripts/http_adapter.py"],
            SIMULATION,
            simulation_environment,
        ),
        (
            "gpu-worker",
            [
                "/home/user/.local/bin/rq", "worker", "-u",
                environment["SIMULATION_REDIS_URL"],
                environment["SIMULATION_RQ_QUEUE"],
            ],
            SIMULATION,
            worker_environment,
        ),
        (
            "callback-dispatcher",
            [
                "/usr/bin/python3", "scripts/callback_dispatcher.py",
                "--redis-url", environment["SIMULATION_REDIS_URL"],
                "--namespace", environment["SIMULATION_REDIS_NAMESPACE"],
                "--backend-base-url", "http://127.0.0.1:18000",
                "--request-timeout", environment.get("SIMULATION_BACKEND_REQUEST_TIMEOUT", "5"),
            ],
            SIMULATION,
            simulation_environment,
        ),
        (
            "backend",
            [
                "/usr/bin/python3", "-m", "uvicorn", "apps.api.main:app",
                "--host", "127.0.0.1", "--port", "18000",
            ],
            ROOT,
            backend_environment,
        ),
        (
            "frontend",
            ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "4173"],
            ROOT / "apps" / "web",
            web_environment,
        ),
    )


def supervise(name, command, cwd, environment):
    log_path = LOG_ROOT / (name + ".log")
    while not STOPPING:
        with log_path.open("ab", buffering=0) as output:
            output.write(("\n[%s] starting %s\n" % (time.strftime("%FT%TZ", time.gmtime()), name)).encode())
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while not STOPPING and process.poll() is None:
                time.sleep(1)
            if STOPPING and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if not STOPPING:
            time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate configuration without starting services")
    args = parser.parse_args(argv)
    if not CONFIG.is_file() or CONFIG.stat().st_mode & 0o077:
        parser.error("private runtime config must exist with mode 0600")
    environment = load_environment(CONFIG)
    specs = service_specs(environment)
    missing = [command[0] for _, command, _, _ in specs if not Path(command[0]).is_file() and "/" in command[0]]
    if missing:
        parser.error("missing runtime executable(s): %s" % ", ".join(missing))
    if args.check:
        print("always-on runtime configuration: PASS")
        print("services: " + ", ".join(item[0] for item in specs))
        print("policy: restart-always, no idle or runtime time limit, Mock Simulator only")
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(str(LOG_ROOT), 0o700)
    children = []

    def stop(_signum, _frame):
        global STOPPING
        STOPPING = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    import threading
    for spec in specs:
        thread = threading.Thread(target=supervise, args=spec, name=spec[0], daemon=True)
        thread.start()
        children.append(thread)
        time.sleep(1)
    while not STOPPING:
        time.sleep(1)
    for thread in children:
        thread.join(timeout=20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
