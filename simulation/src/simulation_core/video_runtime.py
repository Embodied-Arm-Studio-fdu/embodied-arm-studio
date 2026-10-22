"""Task-neutral live-stream and recording lifecycle construction."""

from pathlib import Path
import sys
from urllib.parse import urlsplit

from .artifacts import ArtifactPaths


class TaskVideoRuntimeFactory:
    """Build the existing Camera/FramePipeline -> NVENC -> MediaMTX runtime."""

    def __init__(self, artifacts_root, stream_gateway=None, recorder_factory=None):
        self.artifacts_root = Path(artifacts_root)
        self.stream_gateway = stream_gateway
        self.recorder_factory = recorder_factory

    @staticmethod
    def _frame_pipeline(runtime):
        if hasattr(runtime, "frame_pipeline"):
            return runtime.frame_pipeline
        backend = getattr(runtime, "backend", None)
        pipeline = getattr(backend, "frame_pipeline", None)
        if pipeline is None:
            raise ValueError("video-enabled task runtime has no frame_pipeline")
        return pipeline

    def create(self, runtime, task):
        from .media_streaming import (
            MediaMtxPublisherConfig, MediaMtxPublisherSink, MediaRuntimeConfig,
        )
        from .stream_session import StreamSessionManager, TaskStreamLifecycle
        from .video_encoder import EncoderConfig, EncoderWorker, FFmpegNvencEncoder
        from .video_recorder import (
            Mp4Recorder, Mp4RecorderConfig, RecordingFanoutSink,
        )

        pipeline = self._frame_pipeline(runtime)
        media = MediaRuntimeConfig.from_environment()
        rtsp = urlsplit(media.internal_rtsp_base)
        outer = self

        class Runtime:
            def __init__(self, session):
                self.publisher = MediaMtxPublisherSink(MediaMtxPublisherConfig(
                    session.stream_id, ffmpeg_path=media.ffmpeg_path,
                    rtsp_host=rtsp.hostname or "", rtsp_port=rtsp.port or 0,
                    fps=session.fps,
                ))
                recorder_config = Mp4RecorderConfig(
                    ffmpeg_path=media.ffmpeg_path,
                    output_path=(
                        ArtifactPaths(outer.artifacts_root).run_root(
                            task.project_id, task.run_id
                        ) / "video" / "overview.mp4"
                    ),
                    width=session.width,
                    height=session.height,
                    fps=session.fps,
                )
                recorder = (
                    outer.recorder_factory(recorder_config)
                    if outer.recorder_factory is not None
                    else Mp4Recorder(recorder_config)
                )
                self.sink = RecordingFanoutSink(
                    self.publisher,
                    recorder,
                    diagnostic_callback=lambda reason: print(
                        "training video recorder: %s" % reason,
                        file=sys.stderr,
                        flush=True,
                    ),
                )
                self.worker = EncoderWorker(
                    pipeline,
                    FFmpegNvencEncoder(media.ffmpeg_path),
                    EncoderConfig(
                        width=session.width,
                        height=session.height,
                        fps=session.fps,
                        gop_size=max(1, int(session.fps / 2)),
                    ),
                    self.sink,
                )

            def start(self):
                try:
                    self.publisher.open()
                    self.sink.start_recorder_best_effort()
                    self.worker.start()
                except Exception:
                    self.worker.close()
                    raise

            def stop(self):
                self.worker.close()

            def set_error_callback(self, callback):
                self.worker.set_error_callback(callback)

        manager = StreamSessionManager(
            Runtime,
            rtsp_base=media.internal_rtsp_base,
            webrtc_base=media.public_playback_base,
            session_observer=(
                self.stream_gateway.publish
                if self.stream_gateway is not None else None
            ),
        )
        config = task.config
        return TaskStreamLifecycle(
            manager,
            camera_id=config.camera_id,
            width=config.video_width,
            height=config.video_height,
            fps=config.video_fps,
        )
