#!/usr/bin/env python3
"""Audit installed video runtimes and actually probe h264_nvenc initialization."""

import ctypes
import importlib.util
from pathlib import Path
import shutil
import subprocess

USER_FFMPEG = Path("/opt/ffmpeg/bin/ffmpeg")


def command_has_encoder(command, name):
    if not command: return False
    result = subprocess.run([command, "-hide_banner", "-encoders"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return name in result.stdout


def system_libav_has(name):
    try:
        lib = ctypes.CDLL("libavcodec.so.58")
        lib.avcodec_find_encoder_by_name.argtypes = [ctypes.c_char_p]
        lib.avcodec_find_encoder_by_name.restype = ctypes.c_void_p
        return bool(lib.avcodec_find_encoder_by_name(name.encode()))
    except OSError:
        return False


def ovstream_evidence():
    executable = shutil.which("ovstream")
    module = importlib.util.find_spec("ovstream")
    roots = (Path("/usr/local"), Path("/opt"), Path.home() / ".local")
    matches = []
    for root in roots:
        if root.exists():
            result = subprocess.run(
                ["find", str(root), "-maxdepth", "6", "-iname", "*ovstream*"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            matches.extend(result.stdout.splitlines())
    return executable, module is not None, matches[:10]


def main():
    ffmpeg = str(USER_FFMPEG) if USER_FFMPEG.is_file() else shutil.which("ffmpeg")
    gst_inspect = shutil.which("gst-inspect-1.0")
    gst_plugin = ""
    if gst_inspect:
        for candidate in ("nvh264enc", "nvautogpuh264enc"):
            if subprocess.run([gst_inspect, candidate], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                gst_plugin = candidate; break
    initialized = False; initialization_error = ""
    if ffmpeg:
        probe = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "verbose", "-f", "lavfi",
            "-i", "testsrc=size=640x480:rate=1", "-frames:v", "1", "-an",
            "-c:v", "h264_nvenc", "-gpu", "0", "-f", "h264", "-",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        initialized = probe.returncode == 0 and "h264_nvenc" in probe.stderr
        if not initialized:
            initialization_error = " | ".join(probe.stderr.splitlines()[-12:])
    ov_exec, ov_python, ov_paths = ovstream_evidence()
    print("VIDEO_ENCODER_CAPABILITIES")
    print("gpu=NVIDIA GeForce RTX 4080")
    print("ffmpeg_available=%s" % str(bool(ffmpeg)).lower())
    print("ffmpeg_path=%s" % (ffmpeg or ""))
    print("h264_nvenc_available=%s" % str(command_has_encoder(ffmpeg, "h264_nvenc")).lower())
    print("system_libavcodec_h264_nvenc=%s" % str(system_libav_has("h264_nvenc")).lower())
    print("nvenc_initialize_success=%s" % str(initialized).lower())
    print("nvenc_initialize_error=%s" % initialization_error)
    print("gstreamer_available=%s" % str(bool(shutil.which("gst-launch-1.0"))).lower())
    print("gstreamer_nvenc_plugin=%s" % gst_plugin)
    print("ovstream_available=%s" % str(bool(ov_exec or ov_python or ov_paths)).lower())
    print("ovstream_executable=%s" % (ov_exec or ""))
    print("ovstream_paths=%s" % ",".join(ov_paths))
    print("selected_backend=%s" % ("ffmpeg_cli_h264_nvenc" if initialized else "none"))


if __name__ == "__main__": main()
