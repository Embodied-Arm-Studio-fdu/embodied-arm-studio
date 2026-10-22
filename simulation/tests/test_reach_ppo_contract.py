import tempfile
import unittest
from pathlib import Path

import torch

from simulation_core.ppo.policy import MLPActorCritic, PUSH_POLICY_SCHEMA, REACH_POLICY_SCHEMA
from simulation_core.ppo.trainer import (
    PPOTaskContract, PPOConfig, PPOTrainer, validate_checkpoint_action_schema,
    validate_checkpoint_observation_schema,
)
from simulation_core.reach_adapter import (
    ReachConfig, REACH_REWARD_SCHEMA_VERSION, REACH_REWARD_SEMANTICS,
    REACH_REWARD_TERM_NAMES, reward_schema_metadata,
)


REACH_CONTRACT = PPOTaskContract(REACH_POLICY_SCHEMA, REACH_REWARD_SCHEMA_VERSION,
    REACH_REWARD_SEMANTICS, REACH_REWARD_TERM_NAMES, reward_schema_metadata)


class Environment:
    config = ReachConfig()
    def reset(self): return {"arm_q":(0.,)*6,"arm_dq":(0.,)*6,"ee_to_goal_xy":(.1,0.),"ee_velocity_xy":(0.,0.),"previous_action_xy":(0.,0.)}
    def step(self, action): return self.reset(), 0., False, {"distance_to_goal":.1}


class ReachPPOContractTests(unittest.TestCase):
    def test_policy_dimensions(self):
        reach = MLPActorCritic(REACH_POLICY_SCHEMA.observation_dim, 2, action_low=(-1.,-1.), action_high=(1.,1.))
        push = MLPActorCritic(PUSH_POLICY_SCHEMA.observation_dim, 2)
        self.assertEqual(18, reach.actor[0].in_features); self.assertEqual(24, push.actor[0].in_features)
        self.assertEqual((2,), tuple(reach.act(torch.zeros(18), True)[0].shape))

    def test_checkpoint_identity_prevents_cross_load(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = PPOTrainer(Environment(), config=PPOConfig(rollout_steps=2,update_epochs=1,batch_size=2), task_contract=REACH_CONTRACT)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_action_schema(payload, REACH_CONTRACT)
            validate_checkpoint_observation_schema(payload, REACH_CONTRACT)
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(payload)
            with self.assertRaises(ValueError): validate_checkpoint_observation_schema(payload)


if __name__ == "__main__": unittest.main()
