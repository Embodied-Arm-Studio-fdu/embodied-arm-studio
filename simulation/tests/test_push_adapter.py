import math
import unittest
from types import SimpleNamespace

from simulation_core.push_adapter import (
    ACTION_DIM,
    EETarget,
    PUSH_FIXED_ORIENTATION_XYZW,
    PUSH_FIXED_TARGET_Z,
    PUSH_INITIAL_DOF_POSITIONS,
    VALIDATED_PRECONTACT_BEHIND_OFFSET,
    VALIDATED_PRECONTACT_LATERAL_OFFSET,
    IsaacGymPushBackend,
    PushAdapter,
    PushConfig,
)
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig, S1State


class FakePushBackend:
    def __init__(self):
        self.object_x = 0.0
        self.object_step = 0.1
        self.object_y = 0.0
        self.object_y_step = 0.0
        self.object_z = 0.0
        self.ee_position = (0.4, 0.0, 0.5)
        self.reset_ee_position = self.ee_position
        self.contact = False
        self.setup_calls = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.target = None
        self.follow_target = True

    def setup(self):
        self.setup_calls += 1

    def reset(self):
        self.reset_calls += 1
        self.object_x = 0.0
        self.object_y = 0.0
        self.object_z = 0.0
        self.ee_position = self.reset_ee_position

    def read_robot_state(self):
        return S1State(
            dof_positions=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            dof_velocities=(0.0,) * 8,
            ee_link_state=self.ee_position + (0.0, 0.0, 0.0, 1.0) + (0.0,) * 6,
            arm_ee_jacobian=((0.0,) * 6,) * 6,
        )

    def read_object_state(self):
        return (
            self.object_x, self.object_y, self.object_z,
            0.0, 0.0, 0.0, 1.0,
        ) + (0.0,) * 6

    def set_ee_target(self, target):
        self.target = target
        return False

    def has_object_contact(self):
        return self.contact

    def step(self, n):
        self.step_calls += n
        if self.target is not None and self.follow_target:
            self.ee_position = self.target.position
        self.object_x += self.object_step
        self.object_y += self.object_y_step

    def close(self):
        self.close_calls += 1


class PushAdapterTestCase(unittest.TestCase):
    def test_goal_marker_uses_a_non_physical_collision_group(self):
        self.assertEqual(0, IsaacGymPushBackend.PHYSICS_COLLISION_GROUP)
        self.assertNotEqual(
            IsaacGymPushBackend.PHYSICS_COLLISION_GROUP,
            IsaacGymPushBackend.GOAL_VISUAL_COLLISION_GROUP,
        )

    def test_default_scene_mounts_robot_on_table_top(self):
        config = PushConfig()
        self.assertEqual((0.10, 0.0, 0.40925), config.robot_position)
        self.assertEqual(math.pi, config.robot_yaw)
        self.assertAlmostEqual(
            config.table_position[2] + config.table_size[2] / 2 + 0.03425,
            config.robot_position[2],
        )
        self.assertEqual(PUSH_INITIAL_DOF_POSITIONS, config.robot_initial_dof_positions)
        self.assertEqual((0.01, 0.01), config.robot_initial_dof_positions[-2:])

    def test_push_backend_applies_push_only_home_without_changing_shared_default(self):
        config = PushConfig()
        backend = IsaacGymPushBackend(config, s1_config=S1IsaacGymConfig())
        self.assertEqual(PUSH_INITIAL_DOF_POSITIONS, backend.robot.config.initial_dof_positions)
        self.assertNotEqual(
            S1IsaacGymConfig().initial_dof_positions,
            backend.robot.config.initial_dof_positions,
        )
        backend.close()

    def test_contact_fallback_rejects_finger_self_force_without_cube_dynamics(self):
        import torch

        backend = object.__new__(IsaacGymPushBackend)
        backend.config = PushConfig()
        backend._contact_body_indices = (0, 1, 2)
        backend._object_body_index = 3
        backend._net_contact_force = torch.tensor(
            [[0.0, 0.0, 0.0], [3000.0, 0.0, 0.0], [-3000.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
        )
        rigid = torch.zeros((4, 13))
        rigid[1, :3] = torch.tensor([0.41, -0.04, 0.42])
        rigid[2, :3] = torch.tensor([0.42, -0.03, 0.42])
        backend.robot = SimpleNamespace(
            gym=SimpleNamespace(refresh_net_contact_force_tensor=lambda sim: None),
            sim=object(),
            _refresh_tensors=lambda: None,
            _rigid_body_state=rigid,
        )
        backend.read_object_state = lambda: (
            0.45, 0.0, 0.417, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        self.assertFalse(backend.has_object_contact())

        backend._net_contact_force[3, 0] = 1.0
        self.assertFalse(backend.has_object_contact())
        backend.read_object_state = lambda: (
            0.45, 0.0, 0.417, 0.0, 0.0, 0.0, 1.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        self.assertTrue(backend.has_object_contact())

    def make_adapter(self, max_steps=5):
        backend = FakePushBackend()
        config = PushConfig(
            table_size=(4.0, 4.0, 0.05),
            table_position=(0.0, 0.0, -0.065),
            object_start=(0.0, 0.0, 0.0),
            goal_position=(1.0, 0.0, 0.0),
            success_distance=0.05,
            max_steps=max_steps,
        )
        return PushAdapter(config=config, backend=backend), backend

    def test_reset_returns_minimal_observation(self):
        adapter, backend = self.make_adapter()
        adapter.initialize()
        observation = adapter.reset()
        self.assertEqual(
            {
                "arm_q", "arm_dq", "ee_to_object_xy", "object_to_goal_xy",
                "ee_velocity_xy", "object_velocity_xy",
                "target_tracking_error_xy", "previous_action_xy",
            },
            set(observation),
        )
        self.assertEqual((0.0, 0.5, 0.5, 0.0, 0.0, 0.0), observation["arm_q"])
        self.assertEqual((0.0,) * 6, observation["arm_dq"])
        self.assertEqual((-0.4, 0.0), observation["ee_to_object_xy"])
        self.assertEqual((1.0, 0.0), observation["object_to_goal_xy"])
        self.assertEqual((0.0, 0.0), observation["target_tracking_error_xy"])
        self.assertEqual((0.0, 0.0), observation["previous_action_xy"])
        self.assertEqual(1, backend.reset_calls)
        adapter.close()

    def test_step_maps_xy_action_onto_fixed_push_manifold(self):
        adapter, backend = self.make_adapter()
        adapter.initialize()
        adapter.reset()
        action = (0.001, -0.002)
        observation, reward, done, info = adapter.step(action)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        self.assertFalse(done)
        self.assertFalse(info["controller_applied"])
        self.assertIsInstance(backend.target, EETarget)
        self.assertEqual((0.401, -0.002, PUSH_FIXED_TARGET_Z), backend.target.position)
        self.assertEqual(PUSH_FIXED_ORIENTATION_XYZW, backend.target.orientation_xyzw)
        self.assertEqual(
            {
                "precontact_progress", "wrong_side_or_alignment_term",
                "directional_push_progress", "lateral_push_penalty",
                "contact_sustain_bonus", "success_bonus", "action_penalty",
                "joint_limit_penalty",
            },
            set(info["reward_terms"]),
        )
        self.assertEqual((0.9, 0.0), observation["object_to_goal_xy"])
        self.assertEqual((0.2, -0.4), observation["previous_action_xy"])

    def test_tracking_error_observes_persistent_target_lag(self):
        adapter, backend = self.make_adapter()
        backend.follow_target = False
        adapter.initialize()
        reset_observation = adapter.reset()
        self.assertEqual((0.0, 0.0), reset_observation["previous_action_xy"])
        observation, _, _, _ = adapter.step((0.003, -0.002))
        self.assertAlmostEqual(0.003, observation["target_tracking_error_xy"][0])
        self.assertAlmostEqual(-0.002, observation["target_tracking_error_xy"][1])
        self.assertEqual((0.6, -0.4), observation["previous_action_xy"])
        adapter.reset()
        reset_again = adapter.get_observation()
        self.assertEqual((0.0, 0.0), reset_again["target_tracking_error_xy"])
        self.assertEqual((0.0, 0.0), reset_again["previous_action_xy"])
        adapter.close()

    def make_reward_adapter(self, object_step=0.0, contact=False, **overrides):
        backend = FakePushBackend()
        backend.object_step = object_step
        backend.contact = contact
        values = {
            "table_size": (4.0, 4.0, 0.05),
            "table_position": (0.0, 0.0, -0.065),
            "object_start": (0.0, 0.0, 0.0),
            "goal_position": (1.0, 0.0, 0.0),
            "success_distance": 0.01,
            "max_steps": 20,
            "action_penalty_scale": 0.0,
            "joint_limit_penalty_scale": 0.0,
        }
        values.update(overrides)
        adapter = PushAdapter(PushConfig(**values), backend=backend)
        adapter.initialize()
        adapter.reset()
        return adapter, backend

    def test_approaching_object_rewards_more_than_staying_still(self):
        still, still_backend = self.make_reward_adapter()
        approach, approach_backend = self.make_reward_adapter()
        target = approach.reward_geometry().precontact_xy
        for adapter, backend in ((still, still_backend), (approach, approach_backend)):
            backend.reset_ee_position = (target[0] - 0.02, target[1], PUSH_FIXED_TARGET_Z)
            adapter.reset()
        still_reward = still.step((0.0,) * ACTION_DIM)[1]
        approach_reward = approach.step((0.005, 0.0))[1]
        self.assertGreater(approach_reward, still_reward)
        still.close()
        approach.close()

    def test_contact_rewards_more_than_no_contact(self):
        no_contact, _ = self.make_reward_adapter(contact=False)
        contact, _ = self.make_reward_adapter(contact=True)
        no_contact_reward = no_contact.step((0.0,) * ACTION_DIM)[1]
        contact_reward = contact.step((0.0,) * ACTION_DIM)[1]
        self.assertGreater(contact_reward, no_contact_reward)
        self.assertAlmostEqual(0.002, contact_reward - no_contact_reward)
        no_contact.close()
        contact.close()

    def test_reward_success_and_contact_contract_constants_are_unchanged(self):
        config = PushConfig()
        self.assertEqual(2.0, config.precontact_reward_scale)
        self.assertEqual(1.0, config.wrong_side_penalty_scale)
        self.assertEqual(20.0, config.directional_push_reward_scale)
        self.assertEqual(5.0, config.lateral_push_penalty_scale)
        self.assertEqual(0.002, config.contact_bonus)
        self.assertEqual(1.0, config.success_bonus)
        self.assertEqual(0.001, config.action_penalty_scale)
        self.assertEqual(0.01, config.joint_limit_penalty_scale)
        self.assertEqual(0.05, config.success_distance)
        self.assertEqual(0.1, config.contact_force_threshold)

    def test_pushing_toward_goal_rewards_more_than_reverse(self):
        toward, _ = self.make_reward_adapter(
            object_step=0.05, precontact_reward_scale=0.0,
            wrong_side_penalty_scale=0.0, contact_bonus=0.0
        )
        reverse, _ = self.make_reward_adapter(
            object_step=-0.05, precontact_reward_scale=0.0,
            wrong_side_penalty_scale=0.0, contact_bonus=0.0
        )
        toward_reward = toward.step((0.0,) * ACTION_DIM)[1]
        reverse_reward = reverse.step((0.0,) * ACTION_DIM)[1]
        self.assertGreater(toward_reward, reverse_reward)
        self.assertGreater(toward_reward, 0.0)
        self.assertLess(reverse_reward, 0.0)
        toward.close()
        reverse.close()

    def test_validated_precontact_geometry_and_wrong_side_gate(self):
        adapter, backend = self.make_reward_adapter()
        geometry = adapter.reward_geometry()
        self.assertAlmostEqual(-VALIDATED_PRECONTACT_BEHIND_OFFSET, geometry.precontact_xy[0])
        self.assertAlmostEqual(-VALIDATED_PRECONTACT_LATERAL_OFFSET, geometry.precontact_xy[1])
        backend.ee_position = geometry.precontact_xy + (PUSH_FIXED_TARGET_Z,)
        adapter.get_observation()
        aligned = adapter.reward_geometry()
        self.assertAlmostEqual(0.0, aligned.precontact_error)
        self.assertAlmostEqual(1.0, aligned.behind_gate)
        self.assertAlmostEqual(1.0, aligned.alignment_gate)
        backend.ee_position = (VALIDATED_PRECONTACT_BEHIND_OFFSET, 0.0, PUSH_FIXED_TARGET_Z)
        adapter.get_observation()
        wrong = adapter.reward_geometry()
        self.assertEqual(0.0, wrong.behind_gate)
        backend.object_x = 0.1
        adapter.get_observation()
        moved = adapter.reward_geometry()
        self.assertAlmostEqual(geometry.precontact_xy[0] + 0.1, moved.precontact_xy[0])
        self.assertAlmostEqual(geometry.precontact_xy[1], moved.precontact_xy[1])
        adapter.close()

    def test_correct_approach_positive_wrong_side_approach_not_profitable(self):
        correct, backend = self.make_reward_adapter(object_step=0.0)
        target = correct.reward_geometry().precontact_xy
        backend.reset_ee_position = (target[0] - 0.02, target[1], PUSH_FIXED_TARGET_Z)
        correct.reset()
        correct_reward = correct.step((0.005, 0.0))[1]
        wrong, wrong_backend = self.make_reward_adapter(object_step=0.0)
        wrong_backend.reset_ee_position = (0.12, 0.0, PUSH_FIXED_TARGET_Z)
        wrong.reset()
        wrong_reward = wrong.step((-0.005, 0.0))[1]
        self.assertGreater(correct_reward, 0.0)
        self.assertLessEqual(wrong_reward, 0.0)
        correct.close(); wrong.close()

    def test_near_precontact_approach_fades(self):
        far, far_backend = self.make_reward_adapter(object_step=0.0, action_penalty_scale=0.0)
        target = far.reward_geometry().precontact_xy
        far_backend.reset_ee_position = (target[0] - 0.02, target[1], PUSH_FIXED_TARGET_Z)
        far.reset()
        far_term = far.step((0.002, 0.0))[3]["reward_terms"]["precontact_progress"]
        near, near_backend = self.make_reward_adapter(object_step=0.0, action_penalty_scale=0.0)
        near_backend.reset_ee_position = (target[0] - 0.004, target[1], PUSH_FIXED_TARGET_Z)
        near.reset()
        near_term = near.step((0.002, 0.0))[3]["reward_terms"]["precontact_progress"]
        self.assertGreater(far_term, near_term)
        self.assertGreater(near_term, 0.0)
        far.close(); near.close()

    def test_forward_lateral_backward_reward_order_and_contact_scale(self):
        overrides = dict(
            precontact_reward_scale=0.0, wrong_side_penalty_scale=0.0,
            action_penalty_scale=0.0, joint_limit_penalty_scale=0.0,
            contact_bonus=0.0,
        )
        forward, _ = self.make_reward_adapter(object_step=0.01, **overrides)
        lateral, lateral_backend = self.make_reward_adapter(object_step=0.0, **overrides)
        lateral_backend.object_y_step = 0.01
        backward, _ = self.make_reward_adapter(object_step=-0.01, **overrides)
        forward_reward = forward.step((0.0, 0.0))[1]
        lateral_reward = lateral.step((0.0, 0.0))[1]
        backward_reward = backward.step((0.0, 0.0))[1]
        self.assertGreater(forward_reward, lateral_reward)
        self.assertGreater(lateral_reward, backward_reward)
        self.assertGreater(forward_reward, 0.0)
        self.assertLess(backward_reward, 0.0)
        contact_only, _ = self.make_reward_adapter(object_step=0.0, contact=True)
        contact_reward = contact_only.step((0.0, 0.0))[3]["reward_terms"]["contact_sustain_bonus"]
        self.assertLess(contact_reward, 20.0 * 0.001)
        for item in (forward, lateral, backward, contact_only): item.close()

    def test_reward_progress_state_resets_without_episode_leakage(self):
        adapter, backend = self.make_reward_adapter(object_step=0.0)
        target = adapter.reward_geometry().precontact_xy
        backend.reset_ee_position = (target[0] - 0.02, target[1], PUSH_FIXED_TARGET_Z)
        adapter.reset()
        first = adapter.step((0.005, 0.0))[1]
        adapter.reset()
        second = adapter.step((0.005, 0.0))[1]
        self.assertAlmostEqual(first, second)
        self.assertTrue(math.isfinite(first))
        adapter.close()

    def test_success_bonus_is_emitted_only_once(self):
        adapter, backend = self.make_reward_adapter(success_bonus=1.5)
        backend.object_x = 0.995
        success_observation = adapter.get_observation()
        first = adapter.compute_reward_terms(success_observation, success=True)
        second = adapter.compute_reward_terms(success_observation, success=True)
        self.assertEqual(1.5, first["success_bonus"])
        self.assertEqual(0.0, second["success_bonus"])
        adapter.close()

    def test_object_drop_terminates_as_failure(self):
        adapter, backend = self.make_reward_adapter()
        backend.object_z = -1.0
        _, _, done, info = adapter.step((0.0,) * ACTION_DIM)
        self.assertTrue(done)
        self.assertTrue(info["failure"])
        self.assertEqual("object_drop", info["failure_reason"])
        self.assertFalse(info["success"])
        self.assertFalse(info["timeout"])
        adapter.close()

    def test_object_out_of_bounds_terminates_as_failure(self):
        adapter, backend = self.make_reward_adapter()
        backend.object_y = 3.0
        _, _, done, info = adapter.step((0.0,) * ACTION_DIM)
        self.assertTrue(done)
        self.assertTrue(info["failure"])
        self.assertEqual("object_out_of_bounds", info["failure_reason"])
        self.assertFalse(info["success"])
        self.assertFalse(info["timeout"])
        adapter.close()

    def test_action_shape_and_physical_bounds_are_strictly_2d(self):
        adapter, _ = self.make_adapter()
        adapter.initialize()
        adapter.reset()
        self.assertEqual(2, ACTION_DIM)
        adapter.step((0.0,) * ACTION_DIM)
        for invalid_size in (1, 3, 6, 18):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaisesRegex(ValueError, "exactly 2"):
                    adapter.step((0.0,) * invalid_size)
        with self.assertRaisesRegex(ValueError, "physical XY bounds"):
            adapter.step((0.005001, 0.0))
        adapter.close()

    def test_both_action_indices_are_consumed_and_pose_constraint_does_not_drift(self):
        adapter, backend = self.make_adapter()
        adapter.initialize()
        adapter.reset()
        adapter.step((0.001, 0.0))
        self.assertEqual((0.401, 0.0), backend.target.position[:2])
        adapter.step((0.0, -0.002))
        self.assertEqual((0.401, -0.002), backend.target.position[:2])
        for _ in range(20):
            adapter.step((0.0001, 0.0001))
            self.assertEqual(PUSH_FIXED_TARGET_Z, backend.target.position[2])
            self.assertEqual(PUSH_FIXED_ORIENTATION_XYZW, backend.target.orientation_xyzw)
        adapter.close()

    def test_action_metrics_have_only_normalized_2d_semantics(self):
        adapter, _ = self.make_adapter()
        adapter.initialize()
        adapter.reset()
        _, _, _, info = adapter.step((0.005, 0.0))
        self.assertAlmostEqual(1.0, info["action_norm"])
        self.assertAlmostEqual(0.5, info["saturation_rate"])
        adapter.close()

    def test_multiple_steps_timeout_without_crashing(self):
        adapter, backend = self.make_adapter(max_steps=3)
        adapter.initialize()
        adapter.reset()
        transition = None
        for _ in range(3):
            transition = adapter.step((0.0,) * ACTION_DIM)
        self.assertTrue(transition[2])
        self.assertTrue(transition[3]["timeout"])
        self.assertEqual(3, backend.step_calls)
        adapter.close()

    def test_invalid_action_is_not_treated_as_joint_target(self):
        adapter, _ = self.make_adapter()
        adapter.initialize()
        adapter.reset()
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            adapter.step((0.0,) * 8)
        adapter.close()

    def test_context_manager_closes_backend_idempotently(self):
        adapter, backend = self.make_adapter()
        with adapter as active:
            self.assertIs(adapter, active)
            self.assertEqual(1, backend.setup_calls)
            self.assertEqual(1, backend.reset_calls)
        adapter.close()
        self.assertEqual(1, backend.close_calls)

    def test_success_terminates_episode(self):
        adapter, _ = self.make_adapter(max_steps=20)
        adapter.initialize()
        adapter.reset()
        transition = None
        for _ in range(10):
            transition = adapter.step((0.0,) * ACTION_DIM)
        self.assertTrue(transition[2])
        self.assertTrue(transition[3]["success"])


if __name__ == "__main__":
    unittest.main()
