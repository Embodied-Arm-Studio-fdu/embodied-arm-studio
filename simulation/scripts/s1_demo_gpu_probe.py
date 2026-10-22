#!/usr/bin/env python3
"""GPU-only geometry/control probe for M3; no video or platform state."""
import json, math, time
from isaacgym import gymapi as _gymapi  # noqa: F401
from simulation_core.demos import (
    create_circle_tracking_runtime, create_joint_position_runtime,
    create_line_tracking_runtime, create_target_pose_runtime,
    parse_circle_tracking, parse_joint_position, parse_line_tracking,
    parse_target_pose,
)

CASES=(
 ("target",parse_target_pose,create_target_pose_runtime,{}),
 ("line",parse_line_tracking,create_line_tracking_runtime,{}),
 ("circle",parse_circle_tracking,create_circle_tracking_runtime,{}),
 ("joint_home",parse_joint_position,create_joint_position_runtime,{"pose":"home"}),
 ("joint_side",parse_joint_position,create_joint_position_runtime,{"pose":"side"}),
)
IDS={"target":"demo_s1_target_pose","line":"demo_s1_line_tracking","circle":"demo_s1_circle_tracking","joint_home":"demo_s1_joint_position","joint_side":"demo_s1_joint_position"}
def main():
 out={}
 for name,parser,factory,extra in CASES:
  values={"task_type":"demo","environment_type":IDS[name],"duration_seconds":5,"video_enabled":False,"physics_steps_per_command":1}; values.update(extra)
  runtime=factory(parser(values)); samples=[]
  try:
   runtime.initialize()
   for index in range(360):
    metric=runtime.step()
    if index%60==0 or index==359: samples.append(metric)
   finite=all(math.isfinite(float(value)) for sample in samples for key,value in sample.items() if isinstance(value,(int,float)))
   out[name]={"finite":finite,"samples":samples,"final":samples[-1]}
  finally: runtime.close()
 print("S1_DEMO_GPU_PROBE="+json.dumps(out,sort_keys=True))
 if not all(v["finite"] for v in out.values()): raise SystemExit(2)
if __name__=="__main__": main()
