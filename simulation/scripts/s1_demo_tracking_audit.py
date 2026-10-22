#!/usr/bin/env python3
"""Direct GPU timing/tracking comparison with Camera Sensor off and on."""
import json, math, time
from dataclasses import dataclass
from isaacgym import gymapi as _gymapi  # noqa: F401
from simulation_core.demos import (
    create_circle_tracking_runtime, create_line_tracking_runtime,
    parse_circle_tracking, parse_line_tracking,
)

@dataclass(frozen=True)
class Task:
    run_id: str
    task_id: str

CASES=(("line", "demo_s1_line_tracking", parse_line_tracking, create_line_tracking_runtime),
       ("circle", "demo_s1_circle_tracking", parse_circle_tracking, create_circle_tracking_runtime))

def run(name, task_id, parser, factory, video):
    config=parser({"task_type":"demo","environment_type":task_id,
        "duration_seconds":8,"video_enabled":video,"fps":15,
        "physics_steps_per_command":1})
    runtime=factory(config, Task("audit-%s-%s"%(name,"on" if video else "off"),task_id))
    started=time.monotonic(); ticks=0; last={}; command_count=240
    try:
        runtime.initialize(); started=time.monotonic()
        while ticks < command_count:
            last=runtime.step(); ticks+=1
            # Represent a live consumer without adding codec policy to this
            # Camera timing audit; the production gate separately runs NVENC.
            if runtime.frame_pipeline is not None:
                runtime.frame_pipeline.poll()
            time.sleep(1/60)
        wall=time.monotonic()-started
        stats=runtime.frame_pipeline.stats if runtime.frame_pipeline else None
        return {"rms_error_m":last["rms_tracking_error_m"],"max_error_m":last["max_tracking_error_m"],
            "final_error_m":last["tracking_error_m"],"control_commands":ticks,
            "command_hz":ticks/wall,"physics_steps":ticks*config.physics_steps_per_command,
            "wall_duration_s":wall,"control_duration_s":last.get("control_time_seconds"),
            "camera_fps":config.fps if video else 0,
            "frames_produced":stats.produced_frames if stats else 0,"frames_dropped":stats.dropped_frames if stats else 0}
    finally: runtime.close()

def main():
    output={}
    for name,task_id,parser,factory in CASES:
        output[name]={"video_off":run(name,task_id,parser,factory,False),
                      "video_on":run(name,task_id,parser,factory,True)}
    print("S1_DEMO_TRACKING_AUDIT="+json.dumps(output,sort_keys=True))
if __name__=="__main__":main()
