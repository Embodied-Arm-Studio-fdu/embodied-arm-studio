#!/usr/bin/env python3
"""Real-GPU motion smoke test for PushAdapter's Cartesian controller."""

import argparse
import math

from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig


def distance(left, right):
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument("--physics-steps", type=int, default=4)
    args = parser.parse_args()
    if args.actions <= 0 or args.physics_steps <= 0:
        parser.error("step counts must be positive")

    config = PushConfig(
        max_steps=max(300, args.actions + 1),
        physics_steps_per_action=args.physics_steps,
    )
    backend = IsaacGymPushBackend(config)
    adapter = PushAdapter(config=config, backend=backend)
    try:
        adapter.initialize()
        adapter.reset()
        initial_state = backend.read_robot_state()
        initial_ee = tuple(float(x) for x in initial_state.ee_link_state[:3])
        initial_q = tuple(float(x) for x in initial_state.dof_positions)
        action = (0.003, 0.0)
        info = None
        observation = initial
        for _ in range(args.actions):
            observation, _, done, info = adapter.step(action)
            if done:
                break
        final_state = backend.read_robot_state()
        final_ee = tuple(float(x) for x in final_state.ee_link_state[:3])
        final_q = tuple(float(x) for x in final_state.dof_positions)
        ee_displacement = distance(initial_ee, final_ee)
        joint_displacement = distance(initial_q[:3], final_q[:3])
        if not info["controller_applied"] or backend.last_command is None:
            raise RuntimeError("controller did not produce/apply a joint command")
        if ee_displacement <= 1.0e-4:
            raise RuntimeError("ee_link did not move enough: %s" % ee_displacement)
        print("CONTROLLER_SMOKE_RESULT=success")
        print("actions=%d physics_steps_per_action=%d" % (info["step_count"], args.physics_steps))
        print("initial_ee=%r" % (initial_ee,))
        print("final_ee=%r" % (final_ee,))
        print("ee_displacement=%s" % ee_displacement)
        print("joint_displacement_first3=%s" % joint_displacement)
        print("controller_applied=%s" % info["controller_applied"])
        print("last_joint_delta=%r" % (tuple(float(x) for x in backend.last_command.joint_delta),))
    finally:
        adapter.close()
        print("CLEANUP=adapter_closed")


if __name__ == "__main__":
    main()
