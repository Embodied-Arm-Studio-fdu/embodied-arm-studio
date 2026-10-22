"""Shared S1/Cartesian/Camera runtime for bounded control demos."""

import math
import time

import numpy as np

from ..cartesian_controller import ControllerAdapter
from ..isaac_gym_camera import CameraFrame, IsaacGymCamera, IsaacGymCameraConfig
from ..push_adapter import EETarget, PUSH_FIXED_ORIENTATION_XYZW, PUSH_FIXED_TARGET_Z, PUSH_INITIAL_DOF_POSITIONS
from ..s1_isaac_gym_adapter import S1IsaacGymAdapter, S1IsaacGymConfig
from ..video_frame_pipeline import FramePipeline

FIXED_Z = PUSH_FIXED_TARGET_Z
FIXED_ORIENTATION = PUSH_FIXED_ORIENTATION_XYZW
ROBOT_POSITION = (0.10, 0.0, 0.40925)
ROBOT_YAW = math.pi
DEMO_CAMERA_POSITION = (1.18, -1.18, 1.08)
DEMO_CAMERA_TARGET = (0.38, 0.0, 0.48)
DEMO_CAMERA_FOV_DEG = 52.0


class S1DemoRuntime:
    def __init__(self, config, task=None):
        self.config, self.task = config, task
        graphics = 0 if config.video_enabled else -1
        self.robot = S1IsaacGymAdapter(S1IsaacGymConfig(graphics_device_id=graphics, initial_dof_positions=PUSH_INITIAL_DOF_POSITIONS))
        self.controller = ControllerAdapter()
        self.camera_config = IsaacGymCameraConfig(
            camera_id=config.camera_id, width=config.width, height=config.height,
            horizontal_fov=DEMO_CAMERA_FOV_DEG,
            position=DEMO_CAMERA_POSITION, target=DEMO_CAMERA_TARGET,
        )
        self.camera = None
        self.frame_pipeline = (FramePipeline(task.run_id, task.task_id, target_fps=config.fps) if config.video_enabled and task is not None else None)
        self._started = None; self._errors = []; self._target = None; self._path = ()
        self._control_step = 0
        self._control_dt = self.robot.config.physics_dt * config.physics_steps_per_command

    def initialize(self):
        self.robot.initialize(); self.robot.load_asset(); self.robot.create_env(ROBOT_POSITION, ROBOT_YAW)
        self._create_table()
        if self.config.video_enabled:
            self.camera = IsaacGymCamera(self.robot.gym, self.robot.sim, self.robot.env, self.robot._gymapi, self.robot._gymtorch, self.camera_config).initialize()
        self.robot.prepare_and_acquire(); self.robot.reset_dofs(); self.robot.step(1)
        self._started = time.monotonic(); self._control_step = 0; self.on_initialize()

    def _create_table(self):
        api, gym = self.robot._gymapi, self.robot.gym
        options = api.AssetOptions(); options.fix_base_link = True
        asset = gym.create_box(self.robot.sim, 0.8, 0.8, 0.05, options)
        pose = api.Transform(); pose.p = api.Vec3(0.45, 0.0, 0.35)
        gym.create_actor(self.robot.env, asset, pose, "table", 0, 0)

    def on_initialize(self): pass
    def command(self, control_time): raise NotImplementedError

    def step(self):
        # Target phase is derived only from commands/physics, never Camera or
        # encoder wall-clock latency.  This exact target remains active for
        # controller dispatch and the subsequent overlay capture.
        control_time = self._control_step * self._control_dt
        values = self.command(control_time)
        self.robot.step(self.config.physics_steps_per_command)
        self.maybe_capture_frame()
        self._control_step += 1
        wall_elapsed = time.monotonic() - self._started
        values.update({
            "control_step": self._control_step,
            "control_time_seconds": self._control_step * self._control_dt,
            "physics_steps": self._control_step * self.config.physics_steps_per_command,
            "wall_time_seconds": wall_elapsed,
            "control_command_hz": self._control_step / wall_elapsed if wall_elapsed > 0 else 0.0,
        })
        return values

    def set_cartesian_target(self, xyz):
        self._target = tuple(float(v) for v in xyz)
        state = self.robot.read_state()
        command = self.controller.compute(EETarget(self._target, FIXED_ORIENTATION), state)
        self.robot.set_joint_position_targets(command.positions)
        return state

    def tracking_metrics(self, target, state=None):
        state = state or self.robot.read_state(); actual = tuple(float(v) for v in state.ee_link_state[:3])
        error = math.sqrt(sum((a-b)**2 for a,b in zip(actual, target)))
        self._errors.append(error)
        return {"target_xyz_m": list(target), "ee_xyz_m": list(actual), "tracking_error_m": error,
                "rms_tracking_error_m": math.sqrt(sum(v*v for v in self._errors)/len(self._errors)), "max_tracking_error_m": max(self._errors)}

    def maybe_capture_frame(self):
        if self.camera is None or self.frame_pipeline is None: return None
        return self.frame_pipeline.maybe_produce(self.capture_camera_frame)

    def capture_camera_frame(self):
        frame = self.camera.capture(); pixels = frame.data.copy()
        for point in self._path:
            pixel = self._project(point)
            if pixel: self._blend_dot(pixels, pixel, (45, 105, 210), 1, 0.32)
        if self._target:
            pixel = self._project(self._target)
            if pixel: self._target_ring(pixels, pixel)
        return CameraFrame(frame.camera_id, frame.width, frame.height, frame.pixel_format, frame.frame_index, frame.captured_at, np.ascontiguousarray(pixels))

    def _project(self, point):
        camera=np.asarray(self.camera_config.position); target=np.asarray(self.camera_config.target); forward=target-camera; forward/=np.linalg.norm(forward)
        right=np.cross(forward, np.asarray((0.,0.,1.))); right/=np.linalg.norm(right); up=np.cross(right,forward); relative=np.asarray(point)-camera; depth=float(relative@forward)
        if depth <= 1e-6: return None
        focal=self.camera_config.width/(2*math.tan(math.radians(self.camera_config.horizontal_fov)/2))
        return int(round(self.camera_config.width/2+focal*float(relative@right)/depth)), int(round(self.camera_config.height/2-focal*float(relative@up)/depth))

    @staticmethod
    def _blend_dot(pixels, pixel, color, radius, alpha):
        u,v=pixel; overlay=np.asarray(color,dtype=np.float32)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x,y=u+dx,v+dy
                if dx*dx+dy*dy <= radius*radius and 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
                    source=pixels[y,x,:3].astype(np.float32)
                    pixels[y,x,:3]=np.clip(source*(1-alpha)+overlay*alpha,0,255).astype(np.uint8); pixels[y,x,3]=255

    @classmethod
    def _target_ring(cls, pixels, pixel):
        u,v=pixel; overlay=np.asarray((235,45,45),dtype=np.float32)
        for dy in range(-4,5):
            for dx in range(-4,5):
                distance=dx*dx+dy*dy; x,y=u+dx,v+dy
                if 9 <= distance <= 16 and 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
                    source=pixels[y,x,:3].astype(np.float32)
                    pixels[y,x,:3]=np.clip(source*.25+overlay*.75,0,255).astype(np.uint8); pixels[y,x,3]=255
        for dx,dy in ((-6,0),(-5,0),(5,0),(6,0),(0,-6),(0,-5),(0,5),(0,6)):
            x,y=u+dx,v+dy
            if 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
                pixels[y,x,:3]=(235,45,45); pixels[y,x,3]=255

    def close(self):
        try:
            if self.camera is not None: self.camera.close()
        finally: self.robot.close()
