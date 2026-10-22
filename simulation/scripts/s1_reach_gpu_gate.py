#!/usr/bin/env python3
"""M2 workspace or scripted-controller GPU gate; never performs PPO training."""

import argparse
import json
import math

from isaacgym import gymapi as _gymapi  # noqa: F401

from simulation_core.reach_adapter import IsaacGymReachBackend, ReachAdapter, ReachConfig


def oracle_action(env, config):
    state = env.backend.read_robot_state()
    ee = tuple(float(v) for v in state.ee_link_state)
    delta = (env.target_xy[0]-ee[0], env.target_xy[1]-ee[1])
    norm = math.hypot(*delta)
    return (0.,0.) if norm < 1e-12 else (delta[0]/norm,delta[1]/norm)


def one_episode(env, config, target=None):
    env.reset()
    if target is not None:
        env._target_xy = tuple(target)
        env._previous_distance = math.hypot(target[0]-float(env.backend.read_robot_state().ee_link_state[0]),
                                            target[1]-float(env.backend.read_robot_state().ee_link_state[1]))
        env._initial_distance = env._previous_distance; env._minimum_distance = env._previous_distance
        env.backend.set_goal_overlay(target, config.success_distance_m)
    initial = tuple(float(v) for v in env.backend.read_robot_state().ee_link_state[:2])
    clamped = False; info = {}; done = False
    while not done:
        _, _, done, info = env.step(oracle_action(env, config))
        desired = env.backend.last_target.position[:2]
        clamped = clamped or any(abs(desired[i]-config.workspace_min_xy_m[i])<1e-8 or
                                 abs(desired[i]-config.workspace_max_xy_m[i])<1e-8 for i in range(2))
    state = env.backend.read_robot_state(); ee = tuple(float(v) for v in state.ee_link_state[:2])
    values = tuple(float(v) for v in state.dof_positions)+tuple(float(v) for v in state.dof_velocities)+tuple(float(v) for v in state.ee_link_state)
    return {"target":list(env.target_xy),"initial_ee":list(initial),"final_ee":list(ee),
            "min_distance":float(info["minimum_distance"]),"final_distance":float(info["distance_to_goal"]),
            "success":bool(info["success"]),"timeout":bool(info["timeout"]),
            "failure":bool(info["failure"]),"failure_reason":info["failure_reason"],
            "steps":int(info["step_count"]),"joint_limit_proximity":float(info["joint_limit_proximity"]),
            "clamped":clamped,"finite":all(math.isfinite(v) for v in values),
            "fixed_z":float(env.backend.last_target.position[2]),
            "fixed_orientation":list(env.backend.last_target.orientation_xyzw)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("workspace","scripted"),required=True)
    args=parser.parse_args(); config=ReachConfig(seed=7)
    env=ReachAdapter(config,backend=IsaacGymReachBackend(config)); records=[]
    try:
        env.initialize()
        if args.mode=="workspace":
            cx,cy=config.target_center_xy_m; hx,hy=config.target_half_extent_xy_m
            targets=((cx,cy),(cx+hx,cy),(cx-hx,cy),(cx,cy+hy),(cx,cy-hy),
                     (cx+hx,cy+hy),(cx+hx,cy-hy),(cx-hx,cy+hy),(cx-hx,cy-hy))
            records=[one_episode(env,config,target) for target in targets]
        else:
            records=[one_episode(env,config) for _ in range(10)]
    finally: env.close()
    result={"mode":args.mode,"config":{"center":config.target_center_xy_m,"half_extent":config.target_half_extent_xy_m,
            "fixed_z":config.fixed_target_z_m,"success_distance":config.success_distance_m},
            "episodes":records,"success_count":sum(r["success"] for r in records),
            "passed":all(r["success"] and r["finite"] and not r["failure"] for r in records)}
    print("S1_REACH_GPU_GATE="+json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]: raise SystemExit(2)


if __name__=="__main__": main()
