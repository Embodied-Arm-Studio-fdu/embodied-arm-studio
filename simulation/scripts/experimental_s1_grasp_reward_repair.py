#!/usr/bin/env python3
"""P2-R2 reward-v3 runner using the unchanged R1 curriculum runtime."""

import argparse
import json
import os
from pathlib import Path

import experimental_s1_grasp_curriculum as curriculum
from simulation_core.experimental.manipulation.grasp_env import CURRICULUM_STAGES


ROOT = Path(__file__).resolve().parents[1]
curriculum.ARTIFACT_ROOT = ROOT / "artifacts" / "experimental" / "p2_r2_s1_grasp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "video"), required=True)
    parser.add_argument("--stage", choices=CURRICULUM_STAGES, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--filename", default="policy.mp4")
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    if args.mode == "train":
        result = curriculum.train(args.stage, args.updates, args.resume)
    elif args.mode == "eval":
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        result = curriculum.evaluate(args.stage, args.checkpoint, args.episodes)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        result = curriculum.video(args.stage, args.checkpoint, args.filename)
    print("P2_R2_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
