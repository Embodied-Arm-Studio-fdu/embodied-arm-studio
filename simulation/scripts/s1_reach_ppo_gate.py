#!/usr/bin/env python3
"""Run an explicit finite S1 Reach PPO gate through TrainingTaskRunner."""

import argparse
import json
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401
import torch

from simulation_core.domain import ReachTrainingTaskConfig, TaskStatus
from simulation_core.training_task_runner import TrainingTaskRunner, TrainingTaskSpec


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--updates",type=int,default=5)
    parser.add_argument("--run-id",default="m2-reach-ppo-smoke")
    parser.add_argument("--artifacts-root",type=Path,default=Path("artifacts")); args=parser.parse_args()
    config=ReachTrainingTaskConfig(
        task_type="training",environment_type="s1_reach",seed=7,
        rollout_steps=200,num_updates=args.updates,checkpoint_every=max(1,args.updates),
        episode_steps=128,physics_steps_per_action=8,update_epochs=1,batch_size=16,
        device="cuda:0",video_enabled=False,run_mode="smoke",
    )
    result=TrainingTaskRunner(args.artifacts_root).run(TrainingTaskSpec(
        "m2-smoke-task",args.run_id,"m2",config
    ))
    checkpoint=Path(result.checkpoint_path)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    values=list(result.final_metrics_summary.values())
    finite=all(not isinstance(v,(int,float)) or torch.isfinite(torch.tensor(float(v))) for v in values)
    output={"status":result.status.value,"training_steps":result.training_steps,
            "update_steps":result.update_steps,"checkpoint":str(checkpoint.resolve()),
            "checkpoint_exists":checkpoint.is_file(),"task_id":payload.get("task_id"),
            "observation_dim":payload.get("observation_dim"),"action_dim":payload.get("action_dim"),
            "finite":bool(finite),"final_metrics":dict(result.final_metrics_summary)}
    output["passed"]=(result.status==TaskStatus.COMPLETED and result.training_steps==200*args.updates
                      and result.update_steps==args.updates and checkpoint.is_file() and finite)
    print("S1_REACH_PPO_GATE="+json.dumps(output,sort_keys=True),flush=True)
    if not output["passed"]: raise SystemExit(2)


if __name__=="__main__": main()
