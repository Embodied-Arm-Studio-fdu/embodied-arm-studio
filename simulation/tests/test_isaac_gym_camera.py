import unittest
from datetime import datetime, timezone

import numpy as np

from simulation_core.isaac_gym_camera import (
    CameraFrame, CameraStateError, IsaacGymCamera, IsaacGymCameraConfig, RGBA8,
)


class FakeApi:
    IMAGE_COLOR = "color"
    class CameraProperties:
        pass
    class Vec3:
        def __init__(self, *values): self.values = values


class FakeGym:
    def __init__(self):
        self.destroy_calls = 0
        self.render_calls = 0
    def create_camera_sensor(self, env, properties):
        self.properties = properties
        return 7
    def set_camera_location(self, *args): pass
    def step_graphics(self, sim): pass
    def render_all_camera_sensors(self, sim): self.render_calls += 1
    def get_camera_image_gpu_tensor(self, *args):
        return FakeTensor(np.arange(4 * 3 * 4, dtype=np.uint8))
    def start_access_image_tensors(self, sim): pass
    def end_access_image_tensors(self, sim): pass
    def destroy_camera_sensor(self, *args): self.destroy_calls += 1


class FakeTensor:
    def __init__(self, data): self.data = data
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.data


class FakeGymTorch:
    @staticmethod
    def wrap_tensor(value): return value


class CameraTest(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual("overview", IsaacGymCameraConfig().camera_id)
        for kwargs in ({"width": 0}, {"height": -1}, {"horizontal_fov": 180}):
            with self.assertRaises(ValueError): IsaacGymCameraConfig(**kwargs)

    def test_frame_validates_shape_and_dtype(self):
        now = datetime.now(timezone.utc)
        frame = CameraFrame("overview", 4, 3, RGBA8, 0, now,
                            np.zeros((3, 4, 4), dtype=np.uint8))
        self.assertEqual((3, 4, 4), frame.data.shape)
        with self.assertRaisesRegex(ValueError, "uint8"):
            CameraFrame("overview", 4, 3, RGBA8, 0, now,
                        np.zeros((3, 4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            CameraFrame("overview", 4, 3, RGBA8, 0, now,
                        np.zeros((3, 4, 3), dtype=np.uint8))

    def test_capture_index_cleanup_and_closed_state(self):
        gym = FakeGym()
        camera = IsaacGymCamera(gym, "sim", "env", FakeApi, FakeGymTorch,
                                IsaacGymCameraConfig(width=4, height=3)).initialize()
        self.assertEqual(0, camera.capture().frame_index)
        self.assertEqual(1, camera.capture().frame_index)
        self.assertEqual(2, gym.render_calls)
        camera.close(); camera.close()
        self.assertEqual(1, gym.destroy_calls)
        with self.assertRaisesRegex(CameraStateError, "closed"): camera.capture()

if __name__ == "__main__": unittest.main()
