"""Tiny local HTTP mock for contract tests; not a platform backend."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time
from urllib.parse import unquote


_PATH = re.compile(r"^/api/internal/v1/runs/([^/]+)/(events|callback)$")


class MockBackendState:
    def __init__(self, event_failures=0, callback_failures=0, delay_seconds=0.0):
        self.event_failures = int(event_failures)
        self.callback_failures = int(callback_failures)
        self.delay_seconds = float(delay_seconds)
        self.event_attempts = []
        self.callback_attempts = []
        self.events = []
        self.callbacks = []
        self.lock = threading.Lock()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            match = _PATH.match(self.path)
            if match is None:
                self._reply(404, {"error": "not_found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._reply(415, {"error": "content_type"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._reply(400, {"error": "invalid_json"})
                return
            encoded_run_id, kind = match.groups()
            run_id = unquote(encoded_run_id)
            if payload.get("run_id") != run_id:
                self._reply(400, {"error": "run_id_mismatch"})
                return
            if state.delay_seconds:
                time.sleep(state.delay_seconds)
            with state.lock:
                attempts = state.event_attempts if kind == "events" else state.callback_attempts
                attempts.append(payload)
                failures_name = "event_failures" if kind == "events" else "callback_failures"
                failures = getattr(state, failures_name)
                if failures > 0:
                    setattr(state, failures_name, failures - 1)
                    status = 500
                else:
                    (state.events if kind == "events" else state.callbacks).append(payload)
                    status = 202
            self._reply(status, {"accepted": status == 202})

        def log_message(self, format, *args):
            return None

        def _reply(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


class MockBackendServer:
    def __init__(self, state=None, host="127.0.0.1", port=0):
        self.state = state or MockBackendState()
        self._server = ThreadingHTTPServer((host, int(port)), make_handler(self.state))
        self._thread = None

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return "http://%s:%d" % (host, port)

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
