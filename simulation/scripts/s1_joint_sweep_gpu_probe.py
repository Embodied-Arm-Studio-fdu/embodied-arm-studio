#!/usr/bin/env python3
"""M4 per-joint GPU safety probe without Camera or production registration."""
import json, math
from isaacgym import gymapi as _gymapi  # noqa: F401
from simulation_core.demos import create_joint_sweep_runtime, parse_joint_sweep

def main():
    output={}
    for index in range(6):
        config=parse_joint_sweep({"task_type":"demo","environment_type":"demo_s1_joint_sweep",
            "duration_seconds":300,"joint_index":index,"speed_scale":.5,
            "physics_steps_per_command":1,"video_enabled":False})
        runtime=create_joint_sweep_runtime(config); last={}; max_velocity=0.0; steps=0
        try:
            runtime.initialize()
            while steps < 2400:
                last=runtime.step();steps+=1
                state=runtime.robot.read_state()
                velocities=[abs(float(v)) for v in state.dof_velocities[:6]]
                max_velocity=max(max_velocity,max(velocities))
                if runtime.sweep_complete and steps >= last["control_step"]+120: break
                if runtime.sweep_complete:
                    # Fixed post-completion settling count independent of wall time.
                    if not hasattr(runtime,"_probe_complete_step"):runtime._probe_complete_step=steps
                    if steps-runtime._probe_complete_step >= 120:break
            finite=all(math.isfinite(float(v)) for v in runtime.robot.read_state().dof_positions)
            result=last["per_joint_results"][str(index)]
            passed=bool(runtime.sweep_complete and finite and last["non_finite_count"]==0 and last["limit_clamp_count"]==0 and last["final_home_error_rad"]<.01 and result["actual_min"] is not None and result["actual_max"] is not None)
            output[str(index)]={"passed":passed,"steps":steps,"physical_lower":last["physical_lower_limits"][index],"physical_upper":last["physical_upper_limits"][index],"target_lower":result["target_min"],"target_upper":result["target_max"],"actual_min":result["actual_min"],"actual_max":result["actual_max"],"max_tracking_error_rad":last["max_tracking_error_rad"],"final_home_error_rad":last["final_home_error_rad"],"max_velocity_rad_s":max_velocity,"non_finite_count":last["non_finite_count"],"limit_clamp_count":last["limit_clamp_count"]}
        finally:runtime.close()
    print("S1_JOINT_SWEEP_GPU_PROBE="+json.dumps(output,sort_keys=True))
    if not all(item["passed"] for item in output.values()):raise SystemExit(2)
if __name__=="__main__":main()
