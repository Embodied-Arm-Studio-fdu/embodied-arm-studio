import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from simulation_core.video_recorder import (
    Mp4ProbeResult,
    Mp4Recorder,
    Mp4RecorderConfig,
    Mp4RecorderError,
    Mp4RecorderState,
    RecordingFanoutSink,
    probe_mp4,
)


class FakeProcess:
    def __init__(
        self, output_path, returncode=0, wait_side_effect=None,
        exit_on_stdin_close=True,
    ):
        self.stdin = tempfile.TemporaryFile()
        self.returncode = None
        self.final_returncode = returncode
        self.wait_side_effect = list(wait_side_effect or [])
        self.exit_on_stdin_close = exit_on_stdin_close
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.output_path = Path(output_path)

    def poll(self):
        if self.returncode is None and self.exit_on_stdin_close and self.stdin.closed:
            self.returncode = self.final_returncode
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_side_effect:
            effect = self.wait_side_effect.pop(0)
            if effect == "timeout":
                raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = self.final_returncode
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


class RecorderFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ffmpeg = self.root / "bin" / "ffmpeg"
        self.ffprobe = self.root / "bin" / "ffprobe"
        self.ffmpeg.parent.mkdir()
        self.ffmpeg.touch()
        self.ffprobe.touch()
        self.output = self.root / "artifacts" / "video" / "overview.mp4"

    def tearDown(self):
        self.directory.cleanup()

    def config(self, **changes):
        values = {
            "ffmpeg_path": self.ffmpeg,
            "output_path": self.output,
            "width": 640,
            "height": 480,
            "fps": 15.0,
            "queue_size": 2,
            "write_timeout": 0.2,
            "close_timeout": 1.0,
            "probe_timeout": 0.2,
        }
        values.update(changes)
        return Mp4RecorderConfig(**values)

    @staticmethod
    def successful_probe(*args, **kwargs):
        return Mp4ProbeResult("h264", 640, 480, 14.99, 2.0)


class Mp4ProbeTestCase(RecorderFixture):
    def probe(self, frame_rate="15/1", duration="2.5", **stream_changes):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_bytes(b"mp4")
        stream = {
            "codec_name": "h264",
            "width": 640,
            "height": 480,
            "avg_frame_rate": frame_rate,
        }
        stream.update(stream_changes)
        runner = Mock(return_value=SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [stream], "format": {"duration": duration}}),
        ))
        return probe_mp4(
            self.ffprobe, self.output, 640, 480, 15.0,
            fps_tolerance=0.1, runner=runner,
        )

    def test_probe_records_measured_fps(self):
        result = self.probe("1499/100")
        self.assertAlmostEqual(14.99, result.actual_fps)
        self.assertEqual(2.5, result.duration)

    def test_probe_rejects_codec_dimensions_and_fps_mismatch(self):
        for changes, reason in (
            ({"codec_name": "hevc"}, "codec"),
            ({"width": 1280}, "dimensions"),
            ({"frame_rate": "25/1"}, "fps"),
        ):
            with self.subTest(changes=changes):
                frame_rate = changes.pop("frame_rate", "15/1")
                with self.assertRaisesRegex(Mp4RecorderError, reason):
                    self.probe(frame_rate, **changes)

    def test_probe_rejects_invalid_or_non_positive_fps_and_duration(self):
        for value in ("0/0", "N/A", "NaN", "inf", "-inf", "0", "-1/1"):
            with self.subTest(fps=value), self.assertRaises(Mp4RecorderError):
                self.probe(value)
        for value in ("N/A", "NaN", "inf", "-inf", "0", "-1"):
            with self.subTest(duration=value), self.assertRaises(Mp4RecorderError):
                self.probe(duration=value)

    def test_probe_rejects_empty_symlink_and_non_regular_targets(self):
        empty = self.root / "empty.mp4"
        empty.touch()
        with self.assertRaisesRegex(Mp4RecorderError, "probe_file_invalid"):
            probe_mp4(self.ffprobe, empty, 640, 480, 15.0)
        link = self.root / "linked.mp4"
        link.symlink_to(self.output)
        with self.assertRaisesRegex(Mp4RecorderError, "probe_file_invalid"):
            probe_mp4(self.ffprobe, link, 640, 480, 15.0)


class Mp4RecorderTestCase(RecorderFixture):
    def recorder(self, process=None, **config_changes):
        holder = {}

        def popen(args, **kwargs):
            holder["args"] = args
            holder["kwargs"] = kwargs
            holder["process"] = process or FakeProcess(args[-1])
            holder["process"].output_path.parent.mkdir(parents=True, exist_ok=True)
            holder["process"].output_path.write_bytes(b"fake-mp4")
            return holder["process"]

        recorder = Mp4Recorder(
            self.config(**config_changes), popen=popen, probe=self.successful_probe,
        )
        return recorder, holder

    def test_command_is_copy_mux_and_stderr_is_devnull(self):
        recorder, holder = self.recorder()
        recorder.start()
        recorder.offer(b"annex-b")
        result = recorder.close()

        self.assertEqual(Mp4RecorderState.READY, result.state)
        self.assertEqual(14.99, result.actual_fps)
        self.assertEqual(subprocess.DEVNULL, holder["kwargs"]["stderr"])
        self.assertEqual(subprocess.DEVNULL, holder["kwargs"]["stdout"])
        command = holder["args"]
        self.assertIn("-c:v", command)
        self.assertEqual("copy", command[command.index("-c:v") + 1])
        self.assertEqual("h264", command[command.index("-f") + 1])
        self.assertEqual("mp4", command[-2])
        self.assertTrue(command[-1].endswith("overview.mp4.partial"))
        self.assertFalse(recorder.writer_alive)
        self.assertTrue(self.output.is_file())

    def test_queue_full_never_blocks_primary(self):
        recorder = Mp4Recorder(self.config(queue_size=1))
        recorder._accepting = True
        recorder._set_state(Mp4RecorderState.RECORDING)
        primary = Mock()
        sink = RecordingFanoutSink(primary, recorder)
        packet = SimpleNamespace(data=b"packet")

        sink.write_packet(packet)
        started = time.monotonic()
        sink.write_packet(packet)

        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(2, primary.write_packet.call_count)
        self.assertEqual("recorder_queue_full", sink.recorder_diagnostic)

    def test_open_failure_is_diagnostic_and_primary_still_works(self):
        recorder = Mp4Recorder(self.config(ffmpeg_path=self.root / "missing"))
        primary = Mock()
        sink = RecordingFanoutSink(primary, recorder)
        self.assertFalse(sink.start_recorder_best_effort())
        sink.write_packet(SimpleNamespace(data=b"packet"))
        primary.write_packet.assert_called_once()
        self.assertEqual("ffmpeg_unavailable", recorder.failure_reason)
        self.assertNotIn(str(self.root), sink.recorder_diagnostic)

    def test_pipe_failure_racing_close_has_one_shutdown_owner(self):
        recorder, holder = self.recorder()
        cleanup_calls = []
        original_cleanup = recorder._cleanup_partial_once

        def cleanup_once():
            cleanup_calls.append(True)
            original_cleanup()

        recorder._cleanup_partial_once = cleanup_once
        entered = threading.Event()
        release = threading.Event()

        def fail_write(data):
            entered.set()
            release.wait(0.5)
            raise Mp4RecorderError("recorder_pipe_failed")

        recorder.start()
        recorder._write_packet = fail_write
        self.assertTrue(recorder.offer(b"packet"))
        self.assertTrue(entered.wait(0.5))
        close_thread = threading.Thread(target=recorder.close)
        close_thread.start()
        release.set()
        close_thread.join(1.5)

        self.assertFalse(close_thread.is_alive())
        self.assertFalse(recorder.writer_alive)
        self.assertEqual("writer", recorder.shutdown_owner)
        self.assertEqual(1, holder["process"].wait_calls)
        self.assertEqual(0, holder["process"].terminate_calls)
        self.assertEqual(0, holder["process"].kill_calls)
        self.assertEqual(1, len(cleanup_calls))
        self.assertFalse(recorder.config.partial_path.exists())

    def test_wait_escalation_is_bounded_and_each_action_occurs_once(self):
        process = FakeProcess(
            self.config().partial_path, exit_on_stdin_close=False
        )
        recorder, _ = self.recorder(process=process, close_timeout=0.3)
        recorder.start()
        started = time.monotonic()
        result = recorder.close()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(Mp4RecorderState.READY, result.state)
        self.assertEqual(1, process.wait_calls)
        self.assertEqual(1, process.terminate_calls)
        self.assertEqual(1, process.kill_calls)

    def test_blocked_pipe_obeys_close_deadline_and_writer_exits(self):
        recorder, holder = self.recorder(write_timeout=5.0, close_timeout=0.2)

        def never_writable(readers, writers, errors, timeout=None):
            time.sleep(min(timeout or 0.0, 0.02))
            return [], [], []

        with patch(
            "simulation_core.video_recorder.select.select",
            side_effect=never_writable,
        ):
            recorder.start()
            self.assertTrue(recorder.offer(b"annex-b"))
            started = time.monotonic()
            result = recorder.close()

        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(Mp4RecorderState.FAILED, result.state)
        self.assertEqual("recorder_write_timeout", result.failure_reason)
        self.assertFalse(recorder.writer_alive)
        self.assertEqual("writer", recorder.shutdown_owner)
        self.assertEqual(1, holder["process"].wait_calls)
        self.assertFalse(recorder.config.partial_path.exists())

    def test_close_is_idempotent_and_ready_survives_late_failure(self):
        recorder, holder = self.recorder()
        recorder.start()
        first = recorder.close()
        recorder.record_external_failure("recorder_late_failure")
        second = recorder.close()
        self.assertEqual(Mp4RecorderState.READY, first.state)
        self.assertEqual(Mp4RecorderState.READY, second.state)
        self.assertIsNone(second.failure_reason)
        self.assertTrue(self.output.is_file())
        self.assertEqual(1, holder["process"].wait_calls)

    def test_high_frequency_stderr_cannot_block_close(self):
        script = self.ffmpeg
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "sys.stderr.write('x' * (2 * 1024 * 1024))\n"
            "sys.stderr.flush()\n"
            "sys.stdin.buffer.read()\n"
            "pathlib.Path(sys.argv[-1]).write_bytes(b'fake-mp4')\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        recorder = Mp4Recorder(self.config(), probe=self.successful_probe)
        recorder.start()
        self.assertTrue(recorder.offer(b"packet"))
        started = time.monotonic()
        result = recorder.close()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(Mp4RecorderState.READY, result.state)


if __name__ == "__main__":
    unittest.main()
