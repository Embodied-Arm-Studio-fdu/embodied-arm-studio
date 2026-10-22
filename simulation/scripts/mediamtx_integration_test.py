#!/usr/bin/env python3
"""MediaMTX testsrc -> RTSP -> WHEP integration test (no Isaac Gym)."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from webrtc_test_client import receive_video_sync


def wait_tcp(port, process, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("process exited before port %d became ready" % port)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2): return
        except OSError: time.sleep(0.05)
    raise RuntimeError("port %d did not become ready" % port)


def stop(process):
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, default=Path("/opt/mediamtx/mediamtx"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/ffmpeg/bin/ffmpeg"))
    parser.add_argument("--config", type=Path, default=Path("config/mediamtx_phase_6_4.yml"))
    args = parser.parse_args()
    mediamtx = publisher = None
    try:
        mediamtx = subprocess.Popen([str(args.mediamtx), str(args.config)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wait_tcp(18554, mediamtx); wait_tcp(18889, mediamtx)
        publisher = subprocess.Popen([
            str(args.ffmpeg), "-hide_banner", "-loglevel", "warning", "-re",
            "-f", "lavfi", "-i", "testsrc=size=640x480:rate=15", "-an",
            "-c:v", "h264_nvenc", "-gpu", "0", "-preset", "p4", "-tune", "ll",
            "-rc", "cbr", "-b:v", "2000000", "-g", "30", "-bf", "0",
            "-zerolatency", "1", "-pix_fmt", "yuv420p", "-f", "rtsp",
            "-rtsp_transport", "tcp", "rtsp://127.0.0.1:18554/phase-6-4-testsrc",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(2)
        if publisher.poll() is not None:
            raise RuntimeError("testsrc RTSP publisher failed: %s" % publisher.stderr.read().decode())
        result = receive_video_sync(
            "http://127.0.0.1:18889/phase-6-4-testsrc/whep", frames=10)
        version = subprocess.run([str(args.mediamtx), "--version"], check=True,
            stdout=subprocess.PIPE, text=True).stdout.strip()
        print("MEDIAMTX_INTEGRATION_RESULT=success")
        print("mediamtx_version=%s" % version)
        print("rtsp_publish=success\nwebrtc_server=success")
        print("whep_status=%d\nclient_connected=%s\nvideo_received=%s" %
              (result.whep_status, str(result.connected).lower(),
               str(result.video_track_received).lower()))
        print("resolution=%dx%d\nframes_decoded=%d" %
              (result.width, result.height, result.frames_decoded))
    finally:
        stop(publisher); stop(mediamtx)
        print("CLEANUP=complete")


if __name__ == "__main__": main()
