#!/usr/bin/env python3
"""PHASE 6.4 TEST ONLY: minimal WHEP client with real frame decoding.

Runtime dependency is isolated under /home/user/opt/phase64-aiortc/site.
This is not frontend code and is not a production browser/session client.
"""

import asyncio
from dataclasses import dataclass
import time
import urllib.request

from aiortc import RTCPeerConnection, RTCSessionDescription


@dataclass(frozen=True)
class WebRtcTestResult:
    whep_status: int
    resource_location: str
    connected: bool
    video_track_received: bool
    width: int
    height: int
    frames_decoded: int
    first_pts: int
    last_pts: int
    time_to_first_decoded_frame: float


async def receive_video(whep_url: str, expected_width: int = 640,
                        expected_height: int = 480, frames: int = 10,
                        timeout: float = 15.0) -> WebRtcTestResult:
    pc = RTCPeerConnection()
    queue = asyncio.Queue()
    states = []
    track_received = False

    @pc.on("connectionstatechange")
    async def on_state():
        states.append(pc.connectionState)

    @pc.on("track")
    def on_track(track):
        nonlocal track_received
        if track.kind != "video":
            return
        track_received = True

        async def reader():
            while True:
                try:
                    await queue.put(await track.recv())
                except Exception:
                    return
        asyncio.create_task(reader())

    resource = ""
    status = 0
    decoded = []
    playback_started = None
    first_decoded_at = None
    try:
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        request = urllib.request.Request(
            whep_url, data=pc.localDescription.sdp.encode("utf-8"), method="POST",
            headers={"Content-Type": "application/sdp", "Accept": "application/sdp"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: opener.open(request, timeout=5),
        )
        status = response.status
        resource = response.headers.get("Location", "")
        answer = response.read().decode("utf-8")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        playback_started = time.monotonic()
        for _ in range(frames):
            decoded.append(await asyncio.wait_for(queue.get(), timeout=timeout))
            if first_decoded_at is None:
                first_decoded_at = time.monotonic()
        last = decoded[-1]
        if (last.width, last.height) != (expected_width, expected_height):
            raise RuntimeError("unexpected WebRTC resolution %dx%d" % (last.width, last.height))
        if decoded[-1].pts <= decoded[0].pts:
            raise RuntimeError("decoded video PTS did not advance")
        if "connected" not in states:
            raise RuntimeError("PeerConnection never reached connected")
        return WebRtcTestResult(status, resource, True, track_received,
                                last.width, last.height, len(decoded),
                                decoded[0].pts, decoded[-1].pts,
                                first_decoded_at - playback_started)
    finally:
        await pc.close()


def receive_video_sync(*args, **kwargs):
    return asyncio.run(receive_video(*args, **kwargs))
