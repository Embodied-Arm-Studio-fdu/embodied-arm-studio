"""Production wrapper for the proven rule-based grasp/lift/place program."""

from ..video_frame_pipeline import DropPolicy, FramePipeline, PixelFormat
from ..experimental.manipulation.scene import (
    ManipulationScene, ManipulationSceneConfig,
)
from ..experimental.manipulation.scripted_runtime import (
    ScriptedManipulationRuntime, ScriptedRuntimeConfig,
)


class ScriptedPickPlaceRuntime:
    """Run the existing deterministic state machine once as a bounded Demo."""

    def __init__(self, config, task=None):
        self.config = config
        self.task = task
        self.frame_pipeline = (
            FramePipeline(
                task.run_id, task.task_id, PixelFormat.RGB24,
                max_frames=3, drop_policy=DropPolicy.DROP_OLDEST,
                target_fps=config.fps,
            )
            if config.video_enabled and task is not None else None
        )
        self.scene = None
        self.runtime = None
        self.result = None

    def initialize(self):
        self.scene = ManipulationScene(ManipulationSceneConfig(
            graphics_device_id=0 if self.config.video_enabled else -1,
            with_camera=self.config.video_enabled,
        )).initialize()
        callback = (
            self.frame_pipeline.push_camera_frame
            if self.frame_pipeline is not None else None
        )
        self.runtime = ScriptedManipulationRuntime(
            self.scene,
            ScriptedRuntimeConfig(
                physics_steps_per_control=self.config.physics_steps_per_command,
            ),
            frame_callback=callback,
        )

    def step(self):
        if self.result is None:
            raw = self.runtime.run_trial(self.config.seed)
            self.result = {
                key: value for key, value in raw.items()
                if key not in {"trace", "video_path"}
            }
            self.result["controller"] = "deterministic_rule_state_machine"
            self.result["program_completed"] = True
        return dict(self.result)

    def close(self):
        if self.scene is not None:
            self.scene.close()
            self.scene = None


def create_scripted_pick_place_runtime(config, task=None):
    return ScriptedPickPlaceRuntime(config, task)
