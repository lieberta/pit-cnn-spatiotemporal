"""Measure loss/gradient scales on training data without optimizer updates."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import HeatEquationMultiDataset_dynamic, prepare_experiment_split
from models.pitcnn_latenttime import PITCNN_dynamic
from training.loss import CombinedLoss_dynamic
from training.train_pitcnn_dynamic import _anchor_initial_state, _normalized_model_times
from training.train_utils import git_provenance, loss_gradient_scales, model_state_hash, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--target-gradient-ratio', type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.target_gradient_ratio) or args.target_gradient_ratio <= 0:
        raise ValueError('Target gradient ratio must be positive and finite.')
    torch.set_default_dtype(torch.float32)
    seed_everything(args.seed)
    dataset = HeatEquationMultiDataset_dynamic(base_path='data/steel_train_medium', use_index_cache=False)
    subsets, split = prepare_experiment_split(dataset, 'splits/steel_train_medium.json', 42, None)
    # Construct before the loader to match main.dynamic's initialization order.
    model = PITCNN_dynamic(c=16).to(args.device).train()
    initial_hash = model_state_hash(model)
    loader = DataLoader(subsets['train'], batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    inputs, target = next(iter(loader))
    x0, steps, past_gt, future_gt, mask, source = [x.to(args.device) for x in inputs]
    target = target.to(args.device)
    tau, tau_past = _normalized_model_times(steps, dataset.dt, dataset.prediction_horizon, target.dtype)
    output = model(x0, tau)
    past = _anchor_initial_state(x0, model(x0, tau_past), steps)
    records = {}
    for mode in ('model_past', 'soft_physics'):
        criterion = CombinedLoss_dynamic(
            lp_weight=1, mse_weight=1, device=args.device, alpha=dataset.alpha,
            min_temp=dataset.min_temp, max_temp=dataset.max_temp,
            dx=dataset.dx, dy=dataset.dy, dz=dataset.dz, dt=dataset.dt,
            ambient_temp=dataset.ambient_temp, source_region_weight=1,
            physics_residual_mode=mode)
        _, data, physics = criterion.compute_components(x0, output, past, past_gt, target,
                                                       future_gt, mask, source, measure_physics=True)
        record = loss_gradient_scales(model, data, physics, 1)
        gradient_ratio = record['physics_to_data_gradient']
        record['candidate_weight'] = (args.target_gradient_ratio / gradient_ratio
                                      if gradient_ratio is not None and gradient_ratio > 0 else None)
        records[mode] = record
    report = dict(seed=args.seed, samples=len(target), split=split, initial_state_sha256=initial_hash,
                  git=git_provenance(), target_gradient_ratio=args.target_gradient_ratio,
                  note='Single training-batch initialization probe; candidate weights require validation.',
                  scales=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Training-only scale measurement saved to {args.output}')


if __name__ == '__main__':
    main()
