import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/push_diagnostic_training.py"
TREE = ast.parse(SOURCE.read_text())
WRITER = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "RawVideoWriter")
NAMESPACE = {"Path": Path, "os": os, "shutil": shutil, "subprocess": subprocess}
exec(compile(ast.Module(body=[WRITER], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
RawVideoWriter = NAMESPACE["RawVideoWriter"]


class PushReplayWriterTests(unittest.TestCase):
    def create_writer(self, configured, discovered):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SIMULATION_FFMPEG_PATH": configured}):
                with patch.object(shutil, "which", return_value=discovered) as which:
                    with patch.object(Path, "is_file", return_value=True), patch.object(os, "access", return_value=True):
                        with patch.object(subprocess, "Popen") as popen:
                            writer = RawVideoWriter(Path(directory) / "episode.mp4", fps=12)
                            return writer, popen.call_args.args[0], which.call_count

    def test_configured_path_overrides_path_lookup(self):
        writer, command, lookups = self.create_writer(" /portable/ffmpeg ", "/other/ffmpeg")
        self.assertEqual("/portable/ffmpeg", command[0])
        self.assertEqual(0, lookups)

    def test_path_lookup_without_configured_path(self):
        writer, command, lookups = self.create_writer(" ", "/usr/bin/ffmpeg")
        self.assertEqual("/usr/bin/ffmpeg", command[0])
        self.assertEqual(1, lookups)

    def test_bundled_container_fallback(self):
        writer, command, lookups = self.create_writer("", None)
        self.assertEqual("/opt/ffmpeg/bin/ffmpeg", command[0])

    def test_missing_configured_executable_fails_without_silent_fallback(self):
        with patch.dict(os.environ, {"SIMULATION_FFMPEG_PATH": "/missing/ffmpeg"}):
            with patch.object(Path, "is_file", return_value=False), patch.object(subprocess, "Popen") as popen:
                with self.assertRaisesRegex(FileNotFoundError, "SIMULATION_FFMPEG_PATH"):
                    RawVideoWriter("unused.mp4")
                popen.assert_not_called()

    def test_non_executable_file_fails_before_spawning(self):
        with patch.dict(os.environ, {"SIMULATION_FFMPEG_PATH": "/unexecutable/ffmpeg"}):
            with patch.object(Path, "is_file", return_value=True), patch.object(os, "access", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    RawVideoWriter("unused.mp4")

    def test_nvenc_rgba_input_and_browser_compatible_mp4_output(self):
        writer, command, lookups = self.create_writer("/portable/ffmpeg", None)
        for flag, expected in [("-pixel_format", "rgba"), ("-video_size", "640x480"),
                               ("-framerate", "12"), ("-c:v", "h264_nvenc"),
                               ("-pix_fmt", "yuv420p"), ("-movflags", "+faststart")]:
            self.assertEqual(expected, command[command.index(flag) + 1])

    def test_encoder_failure_is_reported(self):
        writer, command, lookups = self.create_writer("/portable/ffmpeg", None)
        writer.process.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "ffmpeg failed: 1"):
            writer.close()
        writer.process.wait.assert_called_once_with(timeout=30)


if __name__ == "__main__":
    unittest.main()
