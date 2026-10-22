#!/usr/bin/env python3
"""Explicit real-GPU smoke test for the minimal virtual S1 Push episode."""

import argparse
import math

from simulation_core.push_adapter import (
    ACTION_DIM,
    IsaacGymPushBackend,
    PushAdapter,
    PushConfig,
)


EXPECTED_OBSERVATION_KEYS = {
    "arm_q", "arm_dq", "ee_to_object_xy", "object_to_goal_xy",
    "ee_velocity_xy", "object_velocity_xy", "target_tracking_error_xy",
    "previous_action_xy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    config = PushConfig(max_steps=max(args.steps + 1, 300))
    backend = IsaacGymPushBackend(config)
    adapter = PushAdapter(config=config, backend=backend)
    try:
        adapter.initialize()
        actor_count = backend.robot.gym.get_actor_count(backend.robot.env)
        if actor_count != 3:
            raise RuntimeError("expected robot/table/object actors, got %d" % actor_count)
        if backend.table_actor is None or backend.object_actor is None:
            raise RuntimeError("table/object actor handle was not created")

        observation = adapter.reset()
        if set(observation) != EXPECTED_OBSERVATION_KEYS:
            raise RuntimeError("unexpected observation keys: %r" % set(observation))
        expected_shapes = {
            "arm_q": 6, "arm_dq": 6, "ee_to_object_xy": 2,
            "object_to_goal_xy": 2, "ee_velocity_xy": 2,
            "object_velocity_xy": 2, "target_tracking_error_xy": 2,
            "previous_action_xy": 2,
        }
        for name, size in expected_shapes.items():
            if len(observation[name]) != size:
                raise RuntimeError("%s expected length %d" % (name, size))
            if not all(math.isfinite(value) for value in observation[name]):
                raise RuntimeError("%s contains a non-finite value" % name)

        action = (0.001, 0.0)
        total_reward = 0.0
        transition = None
        for _ in range(args.steps):
            transition = adapter.step(action)
            observation, reward, done, info = transition
            if not math.isfinite(reward):
                raise RuntimeError("reward is not finite")
            total_reward += reward
            if done:
                break

        observation, reward, done, info = transition
        print("SMOKE_RESULT=success")
        print("actors=%d table=%s object=%s" % (actor_count, backend.table_actor, backend.object_actor))
        print("steps=%d reward=%s total_reward=%s done=%s" % (
            info["step_count"], reward, total_reward, done
        ))
        print("controller_applied=%s" % info["controller_applied"])
        print("observation_dim=%d" % sum(len(field) for field in observation.values()))
        print("ee_to_object_xy=%r" % (observation["ee_to_object_xy"],))
        print("object_to_goal_xy=%r" % (observation["object_to_goal_xy"],))
        print("distance_to_goal=%s" % info["distance_to_goal"])
    finally:
        adapter.close()
        print("CLEANUP=adapter_closed")


if __name__ == "__main__":
    main()
