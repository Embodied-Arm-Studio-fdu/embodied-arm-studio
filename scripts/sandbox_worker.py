#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple

from robotics.sandbox import SANDBOX_TEMPLATES, DockerCliBackend, build_docker_arguments


JOB_ID_PATTERN = re.compile(r"^sandbox-[a-f0-9]{12}$")


class WorkerHandler(BaseHTTPRequestHandler):
    token = ""
    backend = DockerCliBackend()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._respond(404, {"detail": "not found"})
            return
        self._respond(200, {"status": "healthy", "worker": "sandbox-host-v1"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._respond(401, {"detail": "unauthorized"})
            return
        try:
            payload = self._read_payload()
            job_id = str(payload.get("job_id", ""))
            if not JOB_ID_PATTERN.fullmatch(job_id):
                raise ValueError("invalid job_id")
            if self.path == "/v1/run":
                self._run(job_id, payload)
                return
            if self.path == "/v1/stop":
                self.backend.stop(job_id)
                self._respond(200, {"status": "stopped"})
                return
            self._respond(404, {"detail": "not found"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._respond(422, {"detail": str(exc)})
        except subprocess.TimeoutExpired:
            self._respond(504, {"detail": "sandbox timeout"})
        except Exception as exc:
            self._respond(500, {"detail": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print("sandbox-worker:", format % args)

    def _run(self, job_id: str, payload: Dict[str, object]) -> None:
        allowed_fields = {"job_id", "template_id"}
        if set(payload) != allowed_fields:
            raise ValueError("only job_id and template_id are accepted")
        template = SANDBOX_TEMPLATES.get(str(payload["template_id"]))
        if template is None:
            raise ValueError("unknown template_id")
        arguments = build_docker_arguments(job_id, template)
        result = self.backend.run(job_id, template, arguments, template.timeout_seconds)
        self._respond(200, {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })

    def _authorized(self) -> bool:
        expected = "Bearer " + self.token
        return bool(self.token) and hmac.compare_digest(self.headers.get("Authorization", ""), expected)

    def _read_payload(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 2 or length > 4096:
            raise ValueError("invalid content length")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def _respond(self, status: int, payload: Dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_address(value: str) -> Tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("listen address must be HOST:PORT")
    return host, int(port)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trusted host-side Sandbox Worker")
    parser.add_argument("--listen", type=parse_address, required=True)
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args()
    token_file_mode = os.stat(args.token_file).st_mode & 0o777
    if token_file_mode & 0o077:
        raise SystemExit("token file must not be accessible by group or others")
    token = open(args.token_file, encoding="utf-8").read().strip()
    if len(token) < 32:
        raise SystemExit("token must contain at least 32 characters")
    WorkerHandler.token = token
    server = ThreadingHTTPServer(args.listen, WorkerHandler)
    print("Sandbox Worker listening on http://%s:%s" % args.listen)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())