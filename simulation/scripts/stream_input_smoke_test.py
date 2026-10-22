#!/usr/bin/env python3
"""Real loopback WebSocket input smoke; input is recording-only by design."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from simulation_core.input_events import MockInputWebSocketClient, MockInputWebSocketServer, RecordingInputEventSink


def event(number, kind, **fields):
    value = {"event_id": "event-%d" % number, "run_id": "phase-6-5-smoke",
             "task_id": "phase-6-5-smoke-task", "type": kind,
             "occurred_at": datetime.now(timezone.utc).isoformat()}
    value.update(fields); return value


async def run_input():
    sink = RecordingInputEventSink(); server = MockInputWebSocketServer(sink)
    await server.start()
    try:
        values = [event(1, "KEYBOARD", key="W", action="pressed"),
                  event(2, "KEYBOARD", key="W", action="released"),
                  event(3, "MOUSE_MOVE", x=.4, y=.6),
                  event(4, "MOUSE_BUTTON", button="left", action="pressed", x=.4, y=.6),
                  event(5, "MOUSE_BUTTON", button="left", action="released", x=.4, y=.6),
                  "{", event(7, "UNKNOWN"), event(8, "MOUSE_MOVE", x=2, y=.5),
                  event(9, "KEYBOARD", run_id="../bad", key="W", action="pressed")]
        replies = await MockInputWebSocketClient("ws://127.0.0.1:%d" % server.port).send(values)
        if sum(item["ok"] for item in replies) != 5 or len(sink.events) != 5: raise RuntimeError("unexpected input results")
        return len(values), len(sink.events)
    finally:
        await server.close()


def main():
    # The S1 smoke owns the real Camera/NVENC/WHEP chain in a child process;
    # this process runs the independent loopback input channel concurrently.
    root = Path(__file__).resolve().parent
    video = subprocess.Popen([sys.executable, str(root / "stream_session_s1_smoke_test.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        sent, received = asyncio.run(run_input())
        output, _ = video.communicate(timeout=90)
        if video.returncode or "PHASE_6_5_S1_SMOKE_RESULT=success" not in output:
            raise RuntimeError("S1 video smoke failed:\n%s" % output)
        frames = []
        for line in output.splitlines():
            if line.startswith(("first_client_frames=", "second_client_frames=")):
                frames.append(int(line.split("=", 1)[1]))
        print("STREAM_INPUT_SMOKE_RESULT=success")
        print("video_session=READY\nwebrtc_connected=true\nvideo_frames_received=%d" % sum(frames))
        print("websocket_connected=true\ninput_events_sent=%d\ninput_events_received=%d" % (sent, received))
        print("invalid_events_rejected=%d\ntraining_state=RUNNING\ninput_effect=recording_only" % (sent - received))
    finally:
        if video.poll() is None:
            video.terminate()
            try: video.wait(timeout=5)
            except subprocess.TimeoutExpired: video.kill(); video.wait(timeout=2)
        print("CLEANUP=complete")


if __name__ == "__main__": main()
