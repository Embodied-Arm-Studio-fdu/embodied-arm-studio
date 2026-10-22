import math
from .base import FIXED_Z, S1DemoRuntime

def circle_target(center, radius, period, elapsed):
    angle=2*math.pi*elapsed/period; return center[0]+radius*math.cos(angle), center[1]+radius*math.sin(angle), FIXED_Z

class CircleTrackingRuntime(S1DemoRuntime):
    def on_initialize(self):
        self.center=(0.39,0.0); self._path=tuple(circle_target(self.center,self.config.radius_m,self.config.period_seconds,self.config.period_seconds*i/80) for i in range(80))
    def command(self,elapsed):
        target=circle_target(self.center,self.config.radius_m,self.config.period_seconds,elapsed); state=self.set_cartesian_target(target); return self.tracking_metrics(target,state)

def create_circle_tracking_runtime(config, task=None): return CircleTrackingRuntime(config,task)
