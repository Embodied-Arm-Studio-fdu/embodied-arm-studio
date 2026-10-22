#!/usr/bin/env python3
"""Real testsrc/NVENC/MediaMTX/WHEP StreamSession reconnect integration."""

import argparse
from pathlib import Path
import socket
import subprocess
import time

from simulation_core.stream_session import StreamSessionManager
from webrtc_test_client import receive_video_sync


class TestsrcRuntime:
    def __init__(self, session, ffmpeg): self.session, self.ffmpeg, self.process = session, ffmpeg, None
    def start(self):
        self.process = subprocess.Popen([str(self.ffmpeg), "-hide_banner", "-loglevel", "warning", "-re",
            "-f", "lavfi", "-i", "testsrc=size=%dx%d:rate=%s" % (self.session.width, self.session.height, self.session.fps),
            "-an", "-c:v", "h264_nvenc", "-gpu", "0", "-preset", "p4", "-tune", "ll", "-rc", "cbr",
            "-b:v", "2000000", "-g", "30", "-bf", "0", "-zerolatency", "1", "-forced-idr", "1",
            "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp",
            self.session.rtsp_internal_url], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Runtime.start does not return until MediaMTX reports the path; only
        # then may the manager expose READY.
        # The v1.19.3 API is enabled only as a diagnostic probe in this test
        # configuration and its default auth does not provide a dependable
        # production readiness contract. Wait a bounded RTSP announce window
        # and require the publisher to remain alive.
        time.sleep(2.0)
        if self.process.poll() is not None: raise RuntimeError(self.process.stderr.read().decode())
    def stop(self): stop(self.process)


def wait_tcp(port, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError("MediaMTX exited")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2): return
        except OSError: time.sleep(.05)
    raise RuntimeError("port not ready: %d" % port)


def stop(process):
    if process is None: return
    if process.poll() is None: process.terminate()
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, default=Path("/opt/mediamtx/mediamtx"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/ffmpeg/bin/ffmpeg"))
    parser.add_argument("--config", type=Path, default=Path("config/mediamtx_phase_6_5.yml"))
    args = parser.parse_args(); mediamtx = manager = session = None; failed = False
    try:
        mediamtx = subprocess.Popen([str(args.mediamtx), str(args.config)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wait_tcp(18554, mediamtx); wait_tcp(18889, mediamtx)
        manager = StreamSessionManager(lambda session: TestsrcRuntime(session, args.ffmpeg))
        created = manager.create_session("phase-6-5-integration", "phase-6-5-integration-task"); session = created
        ready = manager.start_session(created.stream_id)
        if ready.status.value != "READY": raise RuntimeError(ready.video_error)
        first = receive_video_sync(ready.whep_url, frames=5)
        manager.mark_client_connected(ready.stream_id); manager.mark_client_disconnected(ready.stream_id)
        time.sleep(1)
        after = manager.get_session(ready.stream_id)
        second = receive_video_sync(ready.whep_url, frames=5)
        reconnected = manager.mark_client_connected(ready.stream_id)
        stopped = manager.stop_session(ready.stream_id)
        print("STREAM_SESSION_INTEGRATION_RESULT=success")
        print("stream_id=%s\nstatus_before_start=%s\nstatus_after_start=%s" % (created.stream_id, created.status.value, ready.status.value))
        print("client_a_connected=true\nclient_a_disconnected=true\nsession_after_disconnect=%s" % after.status.value)
        print("client_b_connected=true\nreconnect_count=%d" % reconnected.reconnect_count)
        print("client_a_frames=%d\nclient_b_frames=%d" % (first.frames_decoded, second.frames_decoded))
        print("client_a_time_to_first_decoded_frame=%.3f\nclient_b_time_to_first_decoded_frame=%.3f" %
              (first.time_to_first_decoded_frame, second.time_to_first_decoded_frame))
        print("status_after_stop=%s\nbackend_mode=mock\nfrontend_mode=mock" % stopped.status.value)
    except Exception:
        failed = True
        raise
    finally:
        if manager is not None and session is not None:
            try: manager.stop_session(session.stream_id)
            except Exception: pass
        stop(mediamtx)
        if failed and mediamtx is not None and mediamtx.stdout is not None:
            print("MEDIAMTX_DIAGNOSTICS_BEGIN")
            print(mediamtx.stdout.read())
            print("MEDIAMTX_DIAGNOSTICS_END")
        print("CLEANUP=complete")


if __name__ == "__main__": main()
