from datetime import datetime, timezone
import unittest

import numpy as np

from simulation_core.isaac_gym_camera import CameraFrame, RGBA8
from simulation_core.push_adapter import IsaacGymPushBackend
from simulation_core.video_frame_pipeline import (
    BGRA8, RGB24, DropPolicy, FrameConverter, FramePipeline,
    FrameRateLimiter, PixelFormat, RecordingFrameConsumer, VideoFrame,
)


NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


class Clock:
    def __init__(self): self.now = 10.0
    def __call__(self): return self.now


def video(index=0, pts=0.0, fmt=PixelFormat.RGBA8, data=None):
    channels = 3 if fmt == PixelFormat.RGB24 else 4
    data = data if data is not None else np.zeros((2, 3, channels), dtype=np.uint8)
    return VideoFrame("run", "task", "overview", index, NOW, pts, 3, 2, fmt, data)


def camera(index=0, pixels=None):
    pixels = pixels if pixels is not None else np.zeros((2, 3, 4), dtype=np.uint8)
    return CameraFrame("overview", 3, 2, RGBA8, index, NOW, pixels)


class VideoFrameTest(unittest.TestCase):
    def test_fields_metadata_and_validation(self):
        frame = video()
        self.assertEqual("RGBA8", frame.metadata()["pixel_format"])
        self.assertNotIn("data", frame.metadata())
        with self.assertRaisesRegex(ValueError, "shape"):
            video(data=np.zeros((2, 3, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            video(data=np.zeros((2, 3, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            video(fmt="NV12")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            video(data=np.zeros((2, 6, 4), dtype=np.uint8)[:, ::2, :])

    def test_rgba_to_rgb_and_bgra_channel_order(self):
        pixels = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
        source = VideoFrame("r", "t", "overview", 0, NOW, 0, 1, 1,
                            PixelFormat.RGBA8, pixels)
        rgb = FrameConverter.convert(source, PixelFormat.RGB24)
        bgra = FrameConverter.convert(source, PixelFormat.BGRA8)
        self.assertEqual([10, 20, 30], rgb.data[0, 0].tolist())
        self.assertEqual([30, 20, 10, 40], bgra.data[0, 0].tolist())
        self.assertTrue(rgb.data.flags.c_contiguous)
        self.assertEqual((1, 1), (rgb.width, rgb.height))


class LimiterTest(unittest.TestCase):
    def test_non_blocking_sampling(self):
        clock = Clock()
        limiter = FrameRateLimiter(10, clock)
        self.assertTrue(limiter.should_capture())
        clock.now += 0.05
        self.assertFalse(limiter.should_capture())
        clock.now += 0.05
        self.assertTrue(limiter.should_capture())
        clock.now += 0.35
        self.assertTrue(limiter.should_capture())
        self.assertFalse(limiter.should_capture())

    def test_invalid_fps(self):
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError): FrameRateLimiter(value)


class PipelineTest(unittest.TestCase):
    def make_pipeline(self, **kwargs):
        clock = kwargs.pop("clock", Clock())
        return FramePipeline("run", "task", monotonic_clock=clock, **kwargs), clock

    def test_drop_oldest_bounded_order_and_stats(self):
        pipeline, _ = self.make_pipeline(max_frames=2)
        for index in range(4): pipeline.push(video(index, index / 10))
        stats = pipeline.stats
        self.assertEqual((4, 4, 2, 2),
                         (stats.produced_frames, stats.accepted_frames,
                          stats.dropped_frames, stats.queue_depth))
        self.assertEqual([2, 3], [pipeline.poll().frame_index, pipeline.poll().frame_index])
        self.assertEqual(2, pipeline.stats.consumed_frames)

    def test_drop_newest_does_not_grow(self):
        pipeline, _ = self.make_pipeline(max_frames=1, drop_policy=DropPolicy.DROP_NEWEST)
        self.assertTrue(pipeline.push(video(0)))
        self.assertFalse(pipeline.push(video(1, 0.1)))
        self.assertEqual((1, 1), (pipeline.stats.queue_depth, pipeline.stats.dropped_frames))
        self.assertEqual(0, pipeline.poll().frame_index)

    def test_strict_index_and_monotonic_pts(self):
        pipeline, _ = self.make_pipeline()
        pipeline.push(video(2, 1.0)); pipeline.poll()
        with self.assertRaisesRegex(ValueError, "frame_index"): pipeline.push(video(2, 2.0))
        with self.assertRaisesRegex(ValueError, "timestamp"): pipeline.push(video(3, 0.5))

    def test_camera_conversion_pts_and_consumer(self):
        pipeline, clock = self.make_pipeline(output_format=PixelFormat.RGB24)
        pixels = np.zeros((2, 3, 4), dtype=np.uint8); pixels[..., :3] = (1, 2, 3)
        first = pipeline.push_camera_frame(camera(0, pixels), monotonic_time=10.0)
        second = pipeline.push_camera_frame(camera(1, pixels), monotonic_time=10.1)
        self.assertEqual(RGB24, first.pixel_format.value)
        self.assertLessEqual(first.presentation_timestamp, second.presentation_timestamp)
        consumer = RecordingFrameConsumer()
        pipeline.consume_next(consumer); pipeline.consume_next(consumer)
        self.assertEqual([0, 1], [item["frame_index"] for item in consumer.records])
        self.assertEqual((2, 3, 3), consumer.records[0]["shape"])

    def test_maybe_produce_uses_limiter_without_sleep(self):
        pipeline, clock = self.make_pipeline(target_fps=10)
        calls = []
        producer = lambda: (calls.append(True) or camera(len(calls) - 1))
        self.assertIsNotNone(pipeline.maybe_produce(producer, 10.0))
        self.assertIsNone(pipeline.maybe_produce(producer, 10.05))
        self.assertIsNotNone(pipeline.maybe_produce(producer, 10.1))
        self.assertEqual(2, len(calls))

    def test_slow_consumer_never_causes_unbounded_growth(self):
        pipeline, _ = self.make_pipeline(max_frames=3)
        for index in range(100): pipeline.push(video(index, index / 15))
        self.assertEqual(3, pipeline.stats.queue_depth)
        self.assertEqual(97, pipeline.stats.dropped_frames)

    def test_push_video_hook_is_disabled_or_failure_isolated(self):
        backend = object.__new__(IsaacGymPushBackend)
        backend.camera = None; backend.frame_pipeline = None; backend.last_video_error = None
        self.assertIsNone(backend.maybe_capture_frame())

        class BrokenPipeline:
            def maybe_produce(self, producer, now): raise RuntimeError("video failed")
        backend.camera = object(); backend.frame_pipeline = BrokenPipeline()
        self.assertIsNone(backend.maybe_capture_frame())
        self.assertIsInstance(backend.last_video_error, RuntimeError)


if __name__ == "__main__": unittest.main()
