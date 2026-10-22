#!/usr/bin/env python3
"""Fresh-process deterministic evaluation/replay for formal S1 Place."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from isaacgym import gymapi as _gymapi  # noqa: F401
import torch
from push_diagnostic_training import RawVideoWriter, seed_all
from simulation_core.experimental.manipulation.place_env import PlaceTransportEnvironment, place_config_from_snapshot
from simulation_core.experimental.manipulation.place_ppo import ExperimentalPlacePPOTrainer, deterministic_evaluate, canonical_place_evaluation


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_runtime(run_root, camera=False):
    checkpoint = run_root / 'checkpoints/model_latest.pt'
    payload = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    if payload.get('task_id') != 's1_place':
        raise ValueError('Place evaluation requires an s1_place checkpoint')
    run_config = run_root / 'config/training_config.json'
    values = json.loads(run_config.read_text()) if run_config.is_file() else {'environment_type': 's1_place', 'seed': 0}
    if values.get('environment_type') != 's1_place':
        raise ValueError('Evaluation task differs from the frozen Run config')
    seed_all(int(values.get('seed', 0)))
    environment = PlaceTransportEnvironment(place_config_from_snapshot(payload.get('environment_config')), with_camera=camera)
    trainer = ExperimentalPlacePPOTrainer(environment, device='cuda:0')
    trainer.load_checkpoint(checkpoint, evaluation_only=True)
    return environment, trainer, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--task-id', choices=['s1_place'], default='s1_place')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--replay-seed', type=int)
    parser.add_argument('--video-output', type=Path)
    parser.add_argument('--video-fps', type=int, default=12)
    args = parser.parse_args()
    if args.replay_seed is None and args.episodes != 100:
        raise ValueError('Production Place evaluation requires 100 episodes')
    if args.replay_seed is not None and (not 0 <= args.replay_seed < 100 or not args.video_output or not 10 <= args.video_fps <= 15):
        raise ValueError('Replay requires episode 0–99, --video-output and FPS 10–15')
    environment, trainer, checkpoint = build_runtime(args.run_root, camera=args.replay_seed is not None)
    writer = None
    try:
        environment.initialize()
        if args.replay_seed is not None:
            if args.replay_seed:
                deterministic_evaluate(environment, trainer, episodes=args.replay_seed)
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
            partial = args.video_output.with_name(args.video_output.name + '.partial.mp4')
            writer = RawVideoWriter(partial, width=640, height=480, fps=args.video_fps)
            environment.frame_callback = writer.write
            _, rows = deterministic_evaluate(environment, trainer, episodes=1)
            writer.close(); writer = None
            if not rows[0]['success']:
                raise RuntimeError('Selected Place episode did not replay successfully')
            os.replace(str(partial), str(args.video_output))
            print('EVALUATION_REPLAY=' + json.dumps(rows[0], sort_keys=True), flush=True)
            return
        domain_summary, domain_records = deterministic_evaluate(environment, trainer, episodes=args.episodes)
    finally:
        if writer is not None:
            writer.close()
        environment.close()
    summary, records = canonical_place_evaluation(domain_summary, domain_records, file_sha256(checkpoint))
    output = args.run_root / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('EVALUATION_RESULT=' + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
