#!/usr/bin/env python3
"""P1 real-GPU property, empty-jaw and fixed-orientation XYZ probes."""

import argparse
import json
import math
import os
from pathlib import Path

from simulation_core.experimental.manipulation.scene import (
    ManipulationScene, ManipulationSceneConfig,
)
from simulation_core.experimental.manipulation.target_compositor import (
    ExperimentalTargetCompositor, Workspace3D,
)


def norm(values): return math.sqrt(sum(float(v)**2 for v in values))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def jaw_sweep(scene):
    targets=(.05,.045,.04,.035,.03,.025,.02,.015,.01)
    records=[]; onset=None; max_force=0.; all_finite=True
    # The Production reset pose puts the long fingers close to the table.  Move
    # the complete open hand upward before measuring finger/finger contact so
    # table reaction is not mislabeled as self-contact.
    snap=scene.snapshot(); start=tuple(snap["ee_state"][:3])
    clearance=(start[0],start[1],min(.58,start[2]+.16))
    compositor=ExperimentalTargetCompositor(workspace=Workspace3D(),jaw_rate_limit=.002)
    compositor.reset(snap["q"])
    for _ in range(240):
        state=scene.robot.read_state(); cmd=compositor.compose(
            clearance,scene.config.fixed_orientation_xyzw,.05,state)
        scene.robot.set_joint_position_targets(cmd.positions); scene.robot.step(4)
        snap=scene.snapshot()
        if norm(a-b for a,b in zip(snap["ee_state"][:3],clearance))<.006: break
    for target in targets:
        for step in range(180):
            snap=scene.snapshot(); q=snap["q"]
            command=list(q); current_target=(scene.robot._position_targets[0].detach().clone()
                                              if scene.robot._position_targets is not None
                                              else scene.robot._dof_state[0,:,0].detach().clone())
            command_tensor=current_target
            before=float(command_tensor[6])
            next_q=max(target,before-.001) if before>target else min(target,before+.001)
            command_tensor[6:]=next_q
            scene.robot.set_joint_position_targets(command_tensor)
            scene.robot.step(4)
            snap=scene.snapshot()
            left=norm(snap["bodies"]["8_Link"]["force"]); right=norm(snap["bodies"]["7_Link"]["force"])
            max_force=max(max_force,left,right)
            finite=all(math.isfinite(float(v)) for v in snap["q"]+snap["dq"])
            all_finite=all_finite and finite
            tracking_asymmetry=abs(snap["q"][6]-snap["q"][7])
            abnormal=left>100. or right>100.
            records.append({"target":target,"step":step,"actual_q":snap["q"][6:8],
                "dq":snap["dq"][6:8],"target_error":[target-snap["q"][6],target-snap["q"][7]],
                "left_finger_pose":snap["bodies"]["8_Link"]["state"][:7],
                "right_finger_pose":snap["bodies"]["7_Link"]["state"][:7],
                "left_force":left,"right_force":right,"estimated_aperture":snap["estimated_aperture"],
                "tracking_asymmetry":tracking_asymmetry,
                "abnormal_self_contact":abnormal,"finite":finite})
            if abnormal and onset is None: onset=target
            if (abs(snap["q"][6]-target)<.0005 and abs(snap["q"][7]-target)<.0005
                    and max(abs(snap["dq"][6]),abs(snap["dq"][7]))<.005 and step>=30): break
    by_target=[]
    for target in targets:
        rows=[r for r in records if r["target"]==target]
        by_target.append({"target":target,"final_actual_q":rows[-1]["actual_q"],
            "max_force":max(max(r["left_force"],r["right_force"]) for r in rows),
            "max_target_error":max(max(abs(v) for v in r["target_error"]) for r in rows),
            "final_target_error":max(abs(target-v) for v in rows[-1]["actual_q"]),
            "max_tracking_asymmetry":max(r["tracking_asymmetry"] for r in rows),
            "abnormal":any(r["abnormal_self_contact"] for r in rows),
            "estimated_aperture":rows[-1]["estimated_aperture"]})
    safe=[r["target"] for r in by_target if not r["abnormal"]]
    return {"targets":by_target,"self_contact_onset":onset,"open_safe":max(safe) if safe else None,
            "close_safe":min(safe) if safe else None,"max_force":max_force,"all_finite":all_finite,
            "passed":bool(safe and min(safe)<=.01 and all_finite),"trace":records}


def xyz_probe(scene):
    workspace=Workspace3D(); compositor=ExperimentalTargetCompositor(workspace=workspace,jaw_rate_limit=.002)
    snap=scene.snapshot(); compositor.reset(snap["q"])
    cube=scene.config.cube_start; object_offset=(-.008,.009,.012)
    grasp=tuple(cube[i]-object_offset[i] for i in range(3))
    initial=scene.snapshot()["ee_state"][:3]
    targets=(
        ("pre_grasp_clearance",(initial[0],initial[1],grasp[2]+.08)),
        ("pre_grasp",(grasp[0],grasp[1],grasp[2]+.08)),
        ("descend",grasp),("lift_25mm",(grasp[0],grasp[1],grasp[2]+.025)),
        ("lower",grasp),("probe_x_low",(.36,0.,.52)),("probe_y_high",(.44,.08,.52)),
        ("probe_y_low",(.44,-.08,.52)),
    )
    results=[]; trace=[]
    for name,target in targets:
        clamp_count=joint_clamps=limit_clamps=0; reached=False
        for step in range(260):
            state=scene.robot.read_state(); command=compositor.compose(
                target,scene.config.fixed_orientation_xyzw,.05,state)
            scene.robot.set_joint_position_targets(command.positions); scene.robot.step(4)
            snap=scene.snapshot(); error=norm(a-b for a,b in zip(target,snap["ee_state"][:3]))
            clamp_count+=int(command.workspace_clamped); joint_clamps+=int(command.joint_delta_clamped)
            limit_clamps+=int(command.joint_limit_clamped)
            finite=all(math.isfinite(float(v)) for v in snap["q"]+snap["dq"]+snap["ee_state"])
            trace.append({"target_name":name,"step":step,"target":list(target),
                "actual":snap["ee_state"][:3],"error":error,"workspace_clamp":command.workspace_clamped,
                "joint_delta_clamp":command.joint_delta_clamped,"joint_limit_clamp":command.joint_limit_clamped,
                "finite":finite})
            if not finite: break
            if error<=.006: reached=True; break
        results.append({"name":name,"target":list(target),"reached":reached,
            "final_error":trace[-1]["error"],"steps":step+1,"workspace_clamps":clamp_count,
            "joint_delta_clamps":joint_clamps,"joint_limit_clamps":limit_clamps,
            "finite":trace[-1]["finite"]})
    return {"fixed_orientation_xyzw":list(scene.config.fixed_orientation_xyzw),
        "workspace":{"lower":list(workspace.lower),"upper":list(workspace.upper)},
        "targets":results,"passed":all(r["reached"] and r["finite"] and
            r["workspace_clamps"]==0 and r["joint_limit_clamps"]==0 for r in results),"trace":trace}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("properties","jaw","xyz"),required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    os.environ.pop("DISPLAY",None); os.environ.setdefault("VK_ICD_FILENAMES","/usr/share/vulkan/icd.d/nvidia_icd.json")
    scene=ManipulationScene(ManipulationSceneConfig(with_cube=args.mode!="jaw"))
    try:
        scene.initialize()
        if args.mode=="properties": result={"properties":scene.property_dump,"passed":True}
        elif args.mode=="jaw": result={"properties":scene.property_dump,"jaw_sweep":jaw_sweep(scene)}; result["passed"]=result["jaw_sweep"]["passed"]
        else: result={"properties":scene.property_dump,"xyz_probe":xyz_probe(scene)}; result["passed"]=result["xyz_probe"]["passed"]
        write(args.output,result); print("P1_%s=%s"%(args.mode.upper(),json.dumps({"passed":result["passed"],"output":str(args.output)})),flush=True)
        if not result["passed"]: raise SystemExit(2)
    finally: scene.close()


if __name__=="__main__": main()
