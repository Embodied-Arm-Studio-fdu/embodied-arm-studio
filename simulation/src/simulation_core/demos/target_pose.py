import math
from .base import FIXED_Z, S1DemoRuntime

class TargetPoseRuntime(S1DemoRuntime):
    def on_initialize(self):
        state=self.robot.read_state(); ee=tuple(float(v) for v in state.ee_link_state[:3]); self.goal=(ee[0]+self.config.target_offset_xy_m[0], ee[1]+self.config.target_offset_xy_m[1], FIXED_Z); self._path=(self.goal,); self.minimum_distance=float("inf")
    def command(self, elapsed):
        state=self.set_cartesian_target(self.goal); values=self.tracking_metrics(self.goal,state); self.minimum_distance=min(self.minimum_distance,values["tracking_error_m"]); values.update({"final_ee_target_distance_m":values["tracking_error_m"],"minimum_distance_m":self.minimum_distance}); return values

def create_target_pose_runtime(config, task=None): return TargetPoseRuntime(config, task)
