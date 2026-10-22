import io
from pathlib import Path
import subprocess
import tempfile
import unittest

from simulation_core.media_streaming import (
    FakeStreamDescriptor, MediaMtxPublisherConfig, MediaMtxPublisherSink,
    MediaRuntimeConfig, VideoStreamState,
)
from simulation_core.video_encoder import EncodedPacket, VideoEncoderError


def packet(data=b"\x00\x00\x00\x01\x67abc"):
    return EncodedPacket("run", "task", "overview", 0, 0.0, None, True, True,
                         "h264", 640, 480, 15.0, data)


class FakeProcess:
    def __init__(self, command, stdin=None, stdout=None, stderr=None, bufsize=None,
                 fail=False):
        self.command = command
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"publisher failure\n" if fail else b"")
        self.returncode = 1 if fail else None
        self.terminated = False
        self.killed = False

    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode = 0 if self.returncode is None else self.returncode; return self.returncode
    def terminate(self): self.terminated = True; self.returncode = -15
    def kill(self): self.killed = True; self.returncode = -9


class MediaStreamingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ffmpeg = Path(self.directory.name) / "ffmpeg-fixture"
        # FakeProcess consumes packets; no executable or media hardware is run.
        self.ffmpeg.touch()

    def config(self, **kwargs):
        values = dict(stream_id="phase-6-4-smoke-overview",
                      ffmpeg_path=self.ffmpeg)
        values.update(kwargs)
        return MediaMtxPublisherConfig(**values)

    def test_config_and_url(self):
        cfg = self.config(rtsp_port=18554, fps=15)
        self.assertEqual(cfg.rtsp_url, "rtsp://127.0.0.1:18554/phase-6-4-smoke-overview")
        for value in ("", "/bad", "bad/path", "?bad"):
            with self.assertRaises(ValueError): self.config(stream_id=value)
        with self.assertRaises(ValueError): self.config(rtsp_port=0)
        with self.assertRaises(ValueError): self.config(fps=0)

    def test_runtime_config_requires_internal_loopback_and_public_lan(self):
        settings = MediaRuntimeConfig.from_environment({
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
            "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg",
        })
        self.assertEqual(settings.public_playback_base, "http://192.0.2.10:18889")
        for rtsp, public in (
            ("rtsp://192.0.2.10:18554", "http://192.0.2.10:18889"),
            ("rtsp://127.0.0.1:18554", "http://127.0.0.1:18889"),
            ("rtsp://127.0.0.1:18554", "http://192.0.2.10:18889?token=x"),
            ("rtsp://user:pass@127.0.0.1:18554", "http://192.0.2.10:18889"),
        ):
            with self.subTest(rtsp=rtsp, public=public), self.assertRaises(ValueError):
                MediaRuntimeConfig(rtsp, public, Path("/opt/ffmpeg"))

    def test_loopback_playback_requires_exact_explicit_opt_in(self):
        values = {
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://127.0.0.1:18890",
            "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg/bin/ffmpeg",
        }
        for value in (None, "0", "true", "yes"):
            source = dict(values)
            if value is not None: source["SIMULATION_STREAM_ALLOW_LOOPBACK_PLAYBACK"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                MediaRuntimeConfig.from_environment(source)
        values["SIMULATION_STREAM_ALLOW_LOOPBACK_PLAYBACK"] = "1"
        self.assertTrue(MediaRuntimeConfig.from_environment(values).allow_loopback_playback)
        for key, invalid in (
            ("SIMULATION_STREAM_RTSP_BASE", "rtsp://192.168.1.20:18554"),
            ("SIMULATION_STREAM_PUBLIC_BASE_URL", "http://user:pass@localhost:18890"),
            ("SIMULATION_STREAM_PUBLIC_BASE_URL", "http://localhost:18890/path"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MediaRuntimeConfig.from_environment(dict(values, **{key: invalid}))

    def test_missing_ffmpeg_is_still_rejected(self):
        self.ffmpeg.unlink()
        sink = MediaMtxPublisherSink(self.config(), FakeProcess)
        with self.assertRaisesRegex(VideoEncoderError, "FFmpeg executable does not exist"):
            sink.open()

    def test_descriptor_is_explicitly_fake(self):
        item = FakeStreamDescriptor("stream", "run", "task", "overview",
                                    "http://127.0.0.1:18889/stream")
        self.assertEqual(item.camera_id, "overview")
        with self.assertRaises(ValueError):
            FakeStreamDescriptor("bad/path", "run", "task", "overview", "url")

    def test_annex_b_chunks_are_written_losslessly_and_only_remuxed(self):
        made = []
        def factory(*args, **kwargs):
            proc = FakeProcess(*args, **kwargs); made.append(proc); return proc
        sink = MediaMtxPublisherSink(self.config(), factory)
        first = b"\x00\x00\x00\x01\x67part"
        second = b"ial\x00\x00\x01\x65rest"
        sink.write_packet(packet(first)); sink.write_packet(packet(second))
        self.assertEqual(made[0].stdin.getvalue(), first + second)
        self.assertEqual(sink.bytes_written, len(first + second))
        self.assertEqual(sink.chunks_written, 2)
        self.assertEqual(sink.state, VideoStreamState.READY)
        command = sink.command
        self.assertIn("copy", command)
        self.assertNotIn("libx264", command)
        self.assertNotIn("h264_nvenc", command)
        sink.close(); sink.close()
        self.assertEqual(sink.state, VideoStreamState.STOPPED)

    def test_subprocess_failure_is_video_error_only(self):
        training_state = {"status": "running"}
        sink = MediaMtxPublisherSink(self.config(),
            lambda *args, **kwargs: FakeProcess(*args, **kwargs, fail=True))
        with self.assertRaises(VideoEncoderError): sink.write_packet(packet())
        self.assertEqual(sink.state, VideoStreamState.ERROR)
        self.assertEqual(training_state["status"], "running")
        sink.close()

    def test_open_close_lifecycle(self):
        count = []
        sink = MediaMtxPublisherSink(self.config(),
            lambda *args, **kwargs: (count.append(1) or FakeProcess(*args, **kwargs)))
        self.assertEqual(sink.state, VideoStreamState.CREATED)
        sink.open(); sink.open()
        self.assertEqual(len(count), 1)
        self.assertEqual(sink.state, VideoStreamState.STARTING)
        sink.close(); sink.close()
        with self.assertRaises(VideoEncoderError): sink.open()


if __name__ == "__main__": unittest.main()
