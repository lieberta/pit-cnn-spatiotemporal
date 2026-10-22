"""Plan or execute the three-seed A–E experiment campaign sequentially."""
import argparse
import json
import math
import os
import statistics
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output-dir', type=Path, default=Path('reports/physics_comparison'))
    parser.add_argument('--execute', action='store_true', help='Run scale measurements and all 15 trainings.')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        raise ValueError('Use at least three distinct seeds.')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in args.seeds:
        calibration = args.output_dir / f'scales_seed{seed}.json'
        tasks.append(dict(kind='calibration', seed=seed, command=[sys.executable, '-m',
                          'evaluation.calibrate_physics', '--seed', str(seed), '--output', str(calibration)]))
    for seed in args.seeds:
        for variant in 'ABCDE':
            tasks.append(dict(kind='training', seed=seed, variant=variant,
                              soft_weight_from='median across all seed calibration reports' if variant == 'E' else None,
                              command=[sys.executable, 'main.py', '--config', 'configs/pitcnn_steel_comparison.py']))
    (args.output_dir / 'plan.json').write_text(json.dumps(tasks, indent=2))
    if not args.execute:
        print(f'Prepared {len(args.seeds)*5} trainings; plan in {args.output_dir}/plan.json. No runs started.')
        return
    # The real-data consistency gate must pass before any training is launched.
    data_root = root / 'data/steel_train_medium'
    if not (data_root / 'info.json').is_file() or not list(data_root.glob('experiment_*/*.zarr')):
        raise FileNotFoundError('Stored steel simulations are required; a skipped residual test is insufficient.')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_physics_residual.py', '-v'], check=True)
    soft_weight = None
    for task in tasks:
        env = dict(os.environ, PITCNN_SEED=str(task['seed']))
        if task['kind'] == 'training':
            env['PITCNN_VARIANT'] = task['variant']
            if task['variant'] == 'E':
                if soft_weight is None:
                    reports = [json.loads((args.output_dir / f'scales_seed{s}.json').read_text())
                               for s in args.seeds]
                    weights = [r['scales']['soft_physics']['candidate_weight'] for r in reports]
                    if any(w is None or not math.isfinite(w) or w <= 0 for w in weights):
                        raise ValueError('Soft-physics calibration did not produce valid positive weights.')
                    soft_weight = statistics.median(weights)
                    (args.output_dir / 'calibration_choice.json').write_text(json.dumps(dict(
                        seeds=args.seeds, per_seed_candidates=weights, common_soft_weight=soft_weight,
                        rule='median of training-only initial gradient-ratio candidates'), indent=2))
                env['PITCNN_SOFT_WEIGHT'] = str(soft_weight)
        subprocess.run(task['command'], env=env, check=True)
        with (args.output_dir / 'completed.jsonl').open('a') as f:
            f.write(json.dumps(task) + '\n')


if __name__ == '__main__':
    main()
