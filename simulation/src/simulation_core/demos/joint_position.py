import math
from .base import S1DemoRuntime
from .config import SAFE_JOINT_POSES

class JointPositionRuntime(S1DemoRuntime):
    RAMP_SECONDS = 2.5
    def on_initialize(self):
        self.goal=SAFE_JOINT_POSES[self.config.pose]
        self.start=tuple(float(value) for value in self.robot.read_state().dof_positions)
    def command(self,control_time):
        linear=min(1.0,max(0.0,control_time/self.RAMP_SECONDS))
        alpha=linear*linear*(3.0-2.0*linear)
        target=tuple(start+(goal-start)*alpha for start,goal in zip(self.start,self.goal))
        self.robot.set_joint_position_targets(target); state=self.robot.read_state(); errors=[abs(float(a)-b) for a,b in zip(state.dof_positions,self.goal)]; self._errors.append(max(errors)); return {"pose":self.config.pose,"ramp_fraction":alpha,"final_joint_error_rad":math.sqrt(sum(v*v for v in errors)/len(errors)),"max_joint_error_rad":max(self._errors)}

def create_joint_position_runtime(config, task=None): return JointPositionRuntime(config,task)
