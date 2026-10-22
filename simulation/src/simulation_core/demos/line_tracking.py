import math
from .base import FIXED_Z, S1DemoRuntime

def line_target(center, length, direction, period, elapsed):
    axis=0 if direction=="x" else 1; result=[center[0],center[1],FIXED_Z]; result[axis] += 0.5*length*math.sin(2*math.pi*elapsed/period); return tuple(result)

class LineTrackingRuntime(S1DemoRuntime):
    def on_initialize(self):
        self.center=(0.40,0.0); axis=0 if self.config.direction=="x" else 1; points=[]
        for i in range(41):
            p=[*self.center,FIXED_Z]; p[axis]+=(i/40-0.5)*self.config.line_length_m; points.append(tuple(p))
        self._path=tuple(points)
    def command(self,elapsed):
        target=line_target(self.center,self.config.line_length_m,self.config.direction,self.config.period_seconds,elapsed); state=self.set_cartesian_target(target); return self.tracking_metrics(target,state)

def create_line_tracking_runtime(config, task=None): return LineTrackingRuntime(config,task)
