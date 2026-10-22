from datetime import datetime, timezone
import time
import unittest

import numpy as np

from simulation_core.video_encoder import (
    EncodedPacket, EncoderConfig, EncoderInputConverter, EncoderWorker,
    FakeVideoEncoder, RecordingEncodedSink, VideoEncoderError,
)
from simulation_core.video_frame_pipeline import FramePipeline, PixelFormat, VideoFrame


NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def frame(index=0, fmt=PixelFormat.RGB24, pixel=None):
    channels = 4 if fmt != PixelFormat.RGB24 else 3
    data = np.zeros((2, 3, channels), dtype=np.uint8)
    if pixel is not None: data[:] = pixel
    return VideoFrame("run", "task", "overview", index, NOW, index / 15,
                      3, 2, fmt, data)


class EncoderModelsTest(unittest.TestCase):
    def test_config_validation(self):
        config = EncoderConfig()
        self.assertEqual(0, config.gpu_index)
        for kwargs in ({"width": 0}, {"height": -1}, {"fps": 0},
                       {"bitrate": 0}, {"codec": "hevc"}, {"device": "cpu"}):
            with self.assertRaises(ValueError): EncoderConfig(**kwargs)

    def test_packet_validation_and_keyframe_metadata(self):
        packet = EncodedPacket("r", "t", "overview", 0, 0.0, None, True, True,
                               "h264", 640, 480, 15, b"packet")
        self.assertTrue(packet.is_keyframe)
        self.assertTrue(packet.contains_codec_config)
        with self.assertRaises(ValueError):
            EncodedPacket("r", "t", "overview", 0, 0, None, False, False,
                          "h264", 640, 480, 15, b"")

    def test_encoder_input_channel_conversion(self):
        rgba = EncoderInputConverter.to_bgra(frame(fmt=PixelFormat.RGBA8,
                                                    pixel=(10, 20, 30, 40)))
        rgb = EncoderInputConverter.to_bgra(frame(fmt=PixelFormat.RGB24,
                                                   pixel=(10, 20, 30)))
        self.assertEqual([30, 20, 10, 40], rgba[0, 0].tolist())
        self.assertEqual([30, 20, 10, 255], rgb[0, 0].tolist())
        self.assertTrue(rgb.flags.c_contiguous)


class EncoderWorkerTest(unittest.TestCase):
    def make_pipeline(self, max_frames=3):
        return FramePipeline("run", "task", PixelFormat.RGB24,
                             max_frames=max_frames, target_fps=15)

    def test_fake_encoder_lifecycle_flush_packets_and_stats(self):
        pipeline = self.make_pipeline(); encoder = FakeVideoEncoder()
        sink = RecordingEncodedSink(); worker = EncoderWorker(
            pipeline, encoder, EncoderConfig(width=3, height=2), sink)
        worker.start()
        for index in range(3): pipeline.push(frame(index))
        deadline = time.monotonic() + 1
        while worker.stats.encoded_frames < 3 and time.monotonic() < deadline: time.sleep(0.001)
        worker.close(); worker.close()
        self.assertTrue(encoder.flushed and encoder.closed and sink.closed)
        self.assertEqual(3, worker.stats.encoded_frames)
        self.assertEqual(3, worker.stats.encoded_packets)
        self.assertEqual(1, worker.stats.keyframes)
        self.assertGreater(worker.stats.encoded_bytes, 0)

    def test_bounded_queue_with_slow_encoder(self):
        pipeline = self.make_pipeline(max_frames=2)
        for index in range(20): pipeline.push(frame(index))
        self.assertEqual(2, pipeline.stats.queue_depth)
        self.assertEqual(18, pipeline.stats.dropped_frames)

    def test_encoder_error_isolated_and_cleanup(self):
        pipeline = self.make_pipeline(); encoder = FakeVideoEncoder(fail_at=0)
        sink = RecordingEncodedSink(); worker = EncoderWorker(
            pipeline, encoder, EncoderConfig(width=3, height=2), sink)
        worker.start(); pipeline.push(frame(0))
        deadline = time.monotonic() + 1
        while worker.error is None and time.monotonic() < deadline: time.sleep(0.001)
        worker.close()
        self.assertIsInstance(worker.error, VideoEncoderError)
        self.assertEqual(1, worker.stats.encode_errors)
        self.assertTrue(encoder.closed and sink.closed)

    def test_late_encoder_error_calls_video_only_observer(self):
        observed = []
        pipeline = self.make_pipeline(); encoder = FakeVideoEncoder(fail_at=0)
        worker = EncoderWorker(
            pipeline, encoder, EncoderConfig(width=3, height=2),
            RecordingEncodedSink(), error_callback=observed.append,
        )
        worker.start(); pipeline.push(frame(0))
        deadline = time.monotonic() + 1
        while not observed and time.monotonic() < deadline: time.sleep(0.001)
        worker.close()
        self.assertEqual(len(observed), 1)
        self.assertIsInstance(observed[0], VideoEncoderError)

    def test_sink_error_isolated(self):
        pipeline = self.make_pipeline(); encoder = FakeVideoEncoder()
        sink = RecordingEncodedSink(fail_at=0); worker = EncoderWorker(
            pipeline, encoder, EncoderConfig(width=3, height=2), sink)
        worker.start(); pipeline.push(frame(0))
        deadline = time.monotonic() + 1
        while worker.error is None and time.monotonic() < deadline: time.sleep(0.001)
        worker.close()
        self.assertIsInstance(worker.error, VideoEncoderError)
        self.assertTrue(sink.closed)


if __name__ == "__main__": unittest.main()
