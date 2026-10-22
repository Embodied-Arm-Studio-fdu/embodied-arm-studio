#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NODE_DIR="$ROOT_DIR/.tools/node-v20.19.4-linux-x64/bin"

if [[ ! -x "$NODE_DIR/node" ]]; then
    echo "Node.js toolchain not found at $NODE_DIR" >&2
    echo "Run 'make install-web' after restoring .tools/node-v20.19.4-linux-x64." >&2
    exit 1
fi

cleanup() {
    trap - EXIT INT TERM
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
    [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
    wait "${API_PID:-}" "${WEB_PID:-}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
PYTHONPATH=. /usr/bin/python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

PATH="$NODE_DIR:$PATH" npm --prefix apps/web run dev &
WEB_PID=$!

echo "Embodied Arm Studio starting"
echo "  Web: http://localhost:5173"
echo "  API: http://127.0.0.1:8000"

wait -n "$API_PID" "$WEB_PID"
STATUS=$?
echo "A development service exited with status $STATUS; stopping the other service." >&2
exit "$STATUS"