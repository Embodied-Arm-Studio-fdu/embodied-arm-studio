#!/usr/bin/env python3
"""P1 isolated cube-close and scripted manipulation gates; never runs PPO."""

import argparse
import json
import math
import os
from pathlib import Path
import random

from simulation_core.experimental.manipulation.scene import ManipulationScene, ManipulationSceneConfig
from simulation_core.experimental.manipulation.scripted_runtime import (
    ScriptedManipulationRuntime, ScriptedRuntimeConfig,
)
from simulation_core.experimental.manipulation.target_compositor import ExperimentalTargetCompositor, Workspace3D


def norm(values): return math.sqrt(sum(float(v)**2 for v in values))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def close_probe(scene, config):
    compositor=ExperimentalTargetCompositor(workspace=config.workspace,jaw_rate_limit=config.jaw_rate_limit)
    scene.reset(); snap=scene.snapshot(); compositor.reset(snap["q"]); initial_cube=snap["cube_state"][:3]
    cube=scene.config.cube_start
    grasp=tuple(cube[i]-config.grasp_frame_object_offset[i] for i in range(3))
    pre=(grasp[0],grasp[1],grasp[2]+config.pregrasp_clearance); trace=[]; failure=""
    def command(stage,target,jaw,limit,done):
        nonlocal failure
        for step in range(limit):
            state=scene.robot.read_state(); cmd=compositor.compose(target,config.fixed_orientation_xyzw,jaw,state)
            scene.robot.set_joint_position_targets(cmd.positions); scene.robot.step(config.physics_steps_per_control)
            s=scene.snapshot(); forces={k:norm(v["force"]) for k,v in s["bodies"].items()}
            cube_state=s["cube_state"]; bottom=cube_state[2]-scene.config.cube_size/2
            item={"stage":stage,"step":step,"q":s["q"],"dq":s["dq"],
                "target_8d":[float(v) for v in cmd.positions.detach().cpu().tolist()],
                "ee":s["ee_state"],"cube":cube_state,"forces":forces,
                "force_vectors":{k:v["force"] for k,v in s["bodies"].items()},
                "left_finger_pose":s["bodies"]["8_Link"]["state"][:7],
                "right_finger_pose":s["bodies"]["7_Link"]["state"][:7],
                "estimated_aperture":s["estimated_aperture"],
                "cartesian_error":norm(cmd.cartesian_error.detach().cpu().tolist()),
                "workspace_clamp":cmd.workspace_clamped,"joint_clamp":cmd.joint_limit_clamped,
                "table_clearance":bottom-scene.config.table_top}
            trace.append(item)
            if not all(math.isfinite(float(v)) for v in s["q"]+s["dq"]+s["ee_state"]+cube_state): failure="NON_FINITE"; return False
            if max(forces.get("8_Link",0),forces.get("7_Link",0),forces.get("cube",0))>config.force_abort_threshold: failure="UNSAFE_FORCE"; return False
            if bottom<scene.config.table_top-.004: failure="PENETRATION"; return False
            if cmd.workspace_clamped or cmd.joint_limit_clamped: failure="CONTROL_CLAMP"; return False
            if done(item): return True
        failure="TIMEOUT_"+stage; return False
    clearance=(snap["ee_state"][0],snap["ee_state"][1],pre[2])
    ok=command("PRE_GRASP_CLEARANCE",clearance,config.open_safe,180,
        lambda r:norm(a-b for a,b in zip(r["ee"][:3],clearance))<.006)
    if ok: ok=command("PRE_GRASP",pre,config.open_safe,240,lambda r:norm(a-b for a,b in zip(r["ee"][:3],pre))<.006)
    if ok: ok=command("DESCEND",grasp,config.open_safe,180,lambda r:norm(a-b for a,b in zip(r["ee"][:3],grasp))<.006)
    if ok:
        closure_window=[]
        def closure_done(r):
            closure_window.append(tuple(r["q"][6:8]))
            if len(closure_window)>10: closure_window.pop(0)
            stopped=(len(closure_window)==10 and
                max(v[0] for v in closure_window)-min(v[0] for v in closure_window)<.0003 and
                max(v[1] for v in closure_window)-min(v[1] for v in closure_window)<.0003)
            blocked=(r["q"][6]+r["q"][7])/2>config.close_safe+.002
            cube_force=r["force_vectors"].get("cube",(0,0,0))
            horizontal_cube=math.hypot(cube_force[0],cube_force[1])>.15
            finger_contact=min(r["forces"].get("8_Link",0),r["forces"].get("7_Link",0))>.15
            commanded=abs(r["target_8d"][6]-config.close_safe)<1e-6
            return commanded and stopped and (blocked or horizontal_cube or finger_contact)
        ok=command("CLOSE",grasp,config.close_safe,180,closure_done)
    final=scene.snapshot(); final_forces={k:norm(v["force"]) for k,v in final["bodies"].items()}
    obj_move=norm(a-b for a,b in zip(final["cube_state"][:3],initial_cube))
    result={"success":bool(ok and not failure),"failure_reason":failure,
        "open_safe":config.open_safe,"close_safe_command":config.close_safe,
        "jaw_rate_limit":config.jaw_rate_limit,"force_abort_threshold":config.force_abort_threshold,
        "actual_jaw_q":final["q"][6:8],"actual_aperture":final["estimated_aperture"],
        "forces":final_forces,"object_motion":obj_move,"trace":trace}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("close-probe","single","fixed","randomized"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--open-safe",type=float,required=True)
    parser.add_argument("--close-safe",type=float,required=True); parser.add_argument("--jaw-rate-limit",type=float,default=.0015)
    parser.add_argument("--force-abort",type=float,default=150.); parser.add_argument("--video",action="store_true")
    args=parser.parse_args(); os.environ.pop("DISPLAY",None); os.environ.setdefault("VK_ICD_FILENAMES","/usr/share/vulkan/icd.d/nvidia_icd.json")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with_camera=args.video and args.mode!="close-probe"
    scene=ManipulationScene(ManipulationSceneConfig(graphics_device_id=0 if with_camera else -1,with_camera=with_camera,initial_open=args.open_safe))
    config=ScriptedRuntimeConfig(open_safe=args.open_safe,close_safe=args.close_safe,
        jaw_rate_limit=args.jaw_rate_limit,force_abort_threshold=args.force_abort)
    try:
        scene.initialize(); write(args.output_dir/"properties.json",scene.property_dump)
        if args.mode=="close-probe":
            result=close_probe(scene,config); write(args.output_dir/"close_probe.json",result)
            summary={k:v for k,v in result.items() if k!="trace"}; passed=result["success"]
        else:
            runtime=ScriptedManipulationRuntime(scene,config)
            count=1 if args.mode=="single" else 10; trials=[]; rng=random.Random(20261002)
            for index in range(count):
                if args.mode=="randomized":
                    base=scene.config.cube_xy; xy=(base[0]+rng.uniform(-.006,.006),base[1]+rng.uniform(-.006,.006))
                else: xy=scene.config.cube_xy
                video=(args.output_dir/("trial_%02d.mp4"%index) if with_camera and index==0 else None)
                trial=runtime.run_trial(index if args.mode!="randomized" else 100+index,xy,video)
                trace=trial.pop("trace"); write(args.output_dir/("trial_%02d_trace.json"%index),trace)
                trials.append(trial); print("trial=%d success=%s reason=%s"%(index,trial["success"],trial["failure_reason"]),flush=True)
            success_count=sum(t["success"] for t in trials)
            required=1 if args.mode=="single" else (9 if args.mode=="fixed" else 8)
            passed=success_count>=required and all(t["non_finite_count"]==0 and not t["unsafe_abort"] and t["resource_cleanup"] for t in trials)
            result={"mode":args.mode,"success_count":success_count,"trial_count":count,"required":required,
                "passed":passed,"config":{"open_safe":args.open_safe,"close_safe":args.close_safe,
                "jaw_rate_limit":args.jaw_rate_limit,"force_abort_threshold":args.force_abort,
                "orientation":list(config.fixed_orientation_xyzw),"workspace":{"lower":list(config.workspace.lower),"upper":list(config.workspace.upper)}},
                "trials":trials}
            write(args.output_dir/"summary.json",result); summary=result
        print("P1_SCRIPTED_%s=%s"%(args.mode.upper().replace("-","_"),json.dumps(summary,sort_keys=True)),flush=True)
        if not passed: raise SystemExit(2)
    finally: scene.close()


if __name__=="__main__": main()
