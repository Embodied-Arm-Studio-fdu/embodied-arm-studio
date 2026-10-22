import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from simulation_core.experimental.manipulation.evidence import evaluate_grasp
from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_A, CURRICULUM_STAGE_B, CURRICULUM_STAGE_C,
    GraspEnvConfig, GraspPPOEnvironment, progressive_reset_pose,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_DIM, GRASP_OBSERVATION_SCHEMA_ID,
    GRASP_POLICY_SCHEMA, GRASP_REWARD_VERSION, compute_grasp_reward,
)
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


EXPECTED_IDS = [
    "demo_s1_circle_tracking", "demo_s1_joint_position", "demo_s1_joint_sweep",
    "demo_s1_line_tracking", "demo_s1_target_pose", "s1_push", "s1_reach",
]
EXPECTED_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


def state_snapshot(aligned=False):
    cube = [.45, 0., .398, 0., 0., 0., 1.] + [0.] * 6
    ee = ([.458, -.009, .386] if aligned else [.391, -.082, .390]) + [0., 0., 0., 1.] + [0.] * 6
    force = [1., 0., 0.]
    return {
        "q": [0., 1., 1., 0., 0., 0., .033, .033], "dq": [0.] * 8,
        "ee_state": ee, "cube_state": cube, "estimated_aperture": .0561,
        "bodies": {name: {"force": list(force), "state": [0.] * 13}
                   for name in ("8_Link", "7_Link", "cube", "table", "ee_link")},
    }


class FakeEnvironment:
    def __init__(self, stage):
        self.config = GraspEnvConfig(curriculum_stage=stage)
    def reset(self):
        return {name: (0.,) * size for name, size in zip(
            GRASP_POLICY_SCHEMA.observation_fields, GRASP_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0.0, True, {"success": False, "timeout": True,
            "failure": True, "failure_reason": "TIMEOUT", "distance_to_goal": 1.0,
            "initial_distance_to_goal": 1.0, "action_norm": 0.0,
            "saturation_rate": 0.0, "reward_terms": {}}


class ExperimentalGraspCurriculumTests(unittest.TestCase):
    def test_p2_contract_is_locked(self):
        config = GraspEnvConfig()
        self.assertEqual("s1-grasp-observation/v1", GRASP_OBSERVATION_SCHEMA_ID)
        self.assertEqual(40, GRASP_OBSERVATION_DIM)
        self.assertEqual("s1-grasp-action/v1", GRASP_ACTION_SCHEMA_ID)
        self.assertEqual(4, GRASP_POLICY_SCHEMA.action_dim)
        self.assertEqual("s1-grasp-reward/v3", GRASP_REWARD_VERSION)
        self.assertEqual((100., 50., 50., 2., 3., 5., 20., .002, .01, .25, 5.), (
            config.reward.approach_progress, config.reward.alignment_progress,
            config.reward.controlled_descend, config.reward.closure_progress,
            config.reward.blockage_progress,
            config.reward.candidate_bonus, config.reward.verified_bonus,
            config.reward.action_magnitude, config.reward.saturation,
            config.reward.clamp, config.reward.failure))
        self.assertEqual(4, config.physics_steps_per_action)
        self.assertEqual(.005, config.xyz_delta_scale)
        self.assertEqual(.0025, config.near_table_z_scale)
        self.assertEqual(.001, config.jaw_rate_limit)

    def test_reset_profiles_change_only_initial_ee_target(self):
        snapshot = state_snapshot()
        a = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A))
        b = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_B))
        c = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C))
        for actual, expected in zip(a._curriculum_reset_target(snapshot), (.458, -.009, .431)):
            self.assertAlmostEqual(expected, actual)
        for actual, expected in zip(b._curriculum_reset_target(snapshot), (.426, -.045, .466)):
            self.assertAlmostEqual(expected, actual)
        self.assertIsNone(c._curriculum_reset_target(snapshot))
        with self.assertRaises(ValueError):
            GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A, randomize_xy=.001)

    def test_observation_is_stage_invariant_for_identical_state(self):
        vectors = []
        for stage in (CURRICULUM_STAGE_A, CURRICULUM_STAGE_B, CURRICULUM_STAGE_C):
            env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=stage))
            env._previous_action = (0.,) * 4
            vectors.append(flatten_observation(env._observation(state_snapshot()), GRASP_POLICY_SCHEMA))
        self.assertTrue(torch.equal(vectors[0], vectors[1]))
        self.assertTrue(torch.equal(vectors[1], vectors[2]))

    def test_progressive_reset_pose_is_monotonic_and_matches_spec(self):
        stage_b = (.426, -.045, .466)
        full = (.391, -.082, .390)
        expected = (
            (.4085, -.0635, .466),
            (.391, -.082, .466),
            (.391, -.082, .44066666666666665),
            (.391, -.082, .41533333333333333),
            full,
        )
        poses = tuple(progressive_reset_pose(stage, stage_b, full)
                      for stage in range(1, 6))
        for actual, wanted in zip(poses, expected):
            for a, b in zip(actual, wanted):
                self.assertAlmostEqual(a, b)
        distances = [sum((a - b) ** 2 for a, b in zip(pose, full)) ** .5
                     for pose in poses]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_reward_is_stage_invariant_for_identical_transition(self):
        values = []
        kwargs = dict(approach_progress=.002, alignment_progress=.001,
            descend_progress=.001, xy_aligned=True,
            closure_potential_progress=.04, blockage_potential_progress=.25,
            candidate_event=False, verified_event=False,
            action=(.1, -.2, .3, -.4), saturated=0., clamped=False, failure=False)
        for stage in (CURRICULUM_STAGE_A, CURRICULUM_STAGE_B, CURRICULUM_STAGE_C):
            values.append(compute_grasp_reward(
                config=GraspEnvConfig(curriculum_stage=stage).reward, **kwargs))
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])

    def test_candidate_and_formal_success_are_stage_invariant(self):
        command = SimpleNamespace(jaw_target=.025)
        candidates = []
        verified = []
        for stage in (CURRICULUM_STAGE_A, CURRICULUM_STAGE_B, CURRICULUM_STAGE_C):
            env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=stage))
            result = False
            for _ in range(6):
                result = env._candidate(state_snapshot(aligned=True), command)
            candidates.append(result)
            evidence = evaluate_grasp(
                config=env._evidence_config(), actual_jaw=.033,
                object_in_corridor=True, object_table_clearance=.010,
                object_rise=.015, ee_rise=.020,
                relative_translation_drift=.005, relative_rotation_drift=.1,
                state_values=[0.] * 20, left_force=1., right_force=1., object_force=1.)
            verified.append(evidence.grasp_verified)
        self.assertEqual([True, True, True], candidates)
        self.assertEqual([True, True, True], verified)

    def test_checkpoint_records_stage_lineage_and_preserves_optimizer(self):
        ppo = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = ExperimentalGraspPPOTrainer(FakeEnvironment(CURRICULUM_STAGE_A), config=ppo)
            a.training_steps = 20000
            first = a.save_checkpoint(root, "p", "a")
            payload = torch.load(first, map_location="cpu", weights_only=False)
            self.assertEqual(CURRICULUM_STAGE_A, payload["curriculum_stage"])
            self.assertEqual(20000, payload["training_transitions_total"])
            self.assertEqual(20000, payload["training_transitions_stage"])
            b = ExperimentalGraspPPOTrainer(FakeEnvironment(CURRICULUM_STAGE_B), config=ppo)
            b.load_staged_curriculum_checkpoint(first)
            self.assertEqual(20000, b.training_steps)
            self.assertEqual(20000, b.curriculum_stage_start_steps)
            second = b.save_checkpoint(root, "p", "b")
            child = torch.load(second, map_location="cpu", weights_only=False)
            self.assertEqual(CURRICULUM_STAGE_B, child["curriculum_stage"])
            self.assertIsNotNone(child["parent_checkpoint_sha256"])
            self.assertEqual(0, child["training_transitions_stage"])

    def test_production_catalog_is_unchanged(self):
        self.assertIn("s1_place", [d.id for d in PRODUCTION_TASK_REGISTRY.definitions()])
        document = task_catalog_document()
        self.assertEqual("task-catalog/v2", document["catalog_version"])



if __name__ == "__main__":
    unittest.main()
