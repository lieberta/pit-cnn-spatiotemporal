"""Evaluate untouched experiment holdouts using a common physical residual.

Example:
    python -m evaluation.evaluate_physics --runs runs/dynamic/.../RUN_ID \
        --output-dir reports/test_evaluation --device cpu
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import HeatEquationMultiDataset_dynamic, prepare_experiment_split, load_experiment_metadata
from training.loss import Laplacian3DLayer, physics_residual
from training.train_pitcnn_dynamic import _normalized_model_times, _anchor_initial_state
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_latenttime1, PITCNN_dynamic_batchnorm
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst

MODELS = {cls.__name__: cls for cls in (PITCNN_dynamic, PITCNN_dynamic_latenttime1,
                                       PITCNN_dynamic_batchnorm, PITCNN_dynamic_timefirst)}
METRICS = ('mae_c', 'rmse_c', 'pde_rmse_k_per_s', 'ic_rmse_c', 'ic_max_abs_c',
           'boundary_rmse_c', 'boundary_max_abs_c')


def evaluate_experiments(model, dataset, experiments, training_horizon, batch_size=16,
                         device='cpu', dtype=torch.float32, workers=0):
    """Return separate experiment and time records; do not average batches equally."""
    model = model.to(device=device, dtype=dtype).eval()
    scale = dataset.temp_range
    ambient = (dataset.ambient_temp - dataset.min_temp) / scale
    lap = Laplacian3DLayer(dataset.dx, dataset.dy, dataset.dz, ambient).to(device=device, dtype=dtype)
    grouped = defaultdict(list)
    for i, (folder, _, _) in enumerate(dataset.files):
        if Path(folder).name in experiments:
            grouped[Path(folder).name].append(i)
    if set(grouped) != set(experiments):
        raise ValueError('Requested experiments are missing from the evaluation dataset.')
    records, time_records = [], []
    with torch.no_grad():
        for experiment in sorted(grouped):
            indices = grouped[experiment]
            metadata = load_experiment_metadata(dataset.files[indices[0]][0])
            sums = dict(abs=0.0, squared=0.0, voxels=0, pde_squared=0.0,
                        pde_voxels=0, boundary_squared=0.0, boundary_voxels=0, boundary_max=0.0)
            initial = None
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                                shuffle=False, num_workers=workers)
            for inputs, target in loader:
                x0, steps, _, _, _, source = inputs
                x0, target = x0.to(device=device, dtype=dtype), target.to(device=device, dtype=dtype)
                steps, source = steps.to(device), source.to(device=device, dtype=dtype)
                tau, tau_past = _normalized_model_times(steps, dataset.dt, training_horizon, dtype)
                output = model(x0, tau)
                past = _anchor_initial_state(x0, model(x0, tau_past), steps)
                residual = physics_residual(output, past, dataset.dt, dataset.alpha, lap, source) * scale
                error = (output - target) * scale
                mask = torch.ones_like(output[0], dtype=torch.bool)
                mask[:, 1:-1, 1:-1, 1:-1] = False
                boundary = (output - ambient)[:, mask] * scale
                if initial is None:
                    delta = (model(x0[:1], torch.zeros_like(tau[:1])) - x0[:1, :1]) * scale
                    initial = dict(ic_rmse_c=delta.square().mean().sqrt().item(),
                                   ic_max_abs_c=delta.abs().max().item())
                sums['abs'] += error.abs().sum().item()
                sums['squared'] += error.square().sum().item()
                sums['voxels'] += error.numel()
                sums['pde_squared'] += residual.square().sum().item()
                sums['pde_voxels'] += residual.numel()
                sums['boundary_squared'] += boundary.square().sum().item()
                sums['boundary_voxels'] += boundary.numel()
                sums['boundary_max'] = max(sums['boundary_max'], boundary.abs().max().item())
                for i in range(len(steps)):
                    time_records.append(dict(experiment=experiment, source_count=metadata['num_fires'],
                        time_seconds=steps[i].item()*dataset.dt,
                        mae_c=error[i].abs().mean().item(), rmse_c=error[i].square().mean().sqrt().item(),
                        pde_rmse_k_per_s=residual[i].square().mean().sqrt().item()))
            records.append(dict(experiment=experiment, source_count=metadata['num_fires'],
                frames=len(indices), mae_c=sums['abs']/sums['voxels'],
                rmse_c=(sums['squared']/sums['voxels'])**0.5,
                pde_rmse_k_per_s=(sums['pde_squared']/sums['pde_voxels'])**0.5,
                boundary_rmse_c=(sums['boundary_squared']/sums['boundary_voxels'])**0.5,
                boundary_max_abs_c=sums['boundary_max'], **initial))
    return records, time_records


def write_csv(path, rows):
    if not rows:
        raise ValueError('No evaluation records generated.')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, metrics=METRICS):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys, key), count=len(group))
        for metric in metrics:
            values = np.asarray([x[metric] for x in group])
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--extrapolation-data-root', type=Path)
    parser.add_argument('--source-counts', type=int, nargs='+')
    args = parser.parse_args()
    all_records, all_times, provenance = [], [], []
    reference_split = None
    for run in args.runs:
        cfg = json.loads((run / 'config.json').read_text())
        protocol = json.loads((run / 'protocol.json').read_text())
        if reference_split is not None and reference_split != protocol['split']:
            raise ValueError('Compared runs must use exactly the same experiment split and normalization.')
        reference_split = protocol['split']
        dtype = torch.float64 if cfg['training_dtype'] == 'float64' else torch.float32
        torch.set_default_dtype(dtype)
        root = str(args.extrapolation_data_root or cfg['data_path'])
        dataset = HeatEquationMultiDataset_dynamic(
            base_path=root, modulo=cfg.get('data_modulo', 1) if not args.extrapolation_data_root else 1,
            max_experiments=cfg.get('data_max_experiments') if not args.extrapolation_data_root else None,
            experiment_offset=cfg.get('data_experiment_offset', 0) if not args.extrapolation_data_root else 0,
            use_index_cache=False,
        )
        if args.extrapolation_data_root:
            norm = protocol['split']['normalization']
            dataset.min_temp, dataset.max_temp, dataset.temp_range = norm['min_temp'], norm['max_temp'], norm['temp_range']
            names = sorted({Path(folder).name for folder, _, _ in dataset.files})
            seen = set(protocol['split']['train'] + protocol['split']['validation'])
            if set(names) & seen:
                raise ValueError('Extrapolation set includes known training/validation experiment IDs.')
        else:
            split = protocol['split']
            if not (run/'experiment_split.json').is_file():
                raise ValueError('Saved run split manifest is missing.')
            _, actual = prepare_experiment_split(dataset, run/'experiment_split.json',
                                                 split['seed'], split['training_source_counts'])
            if actual != split:
                raise ValueError('Run manifest and checkpoint protocol differ.')
            names = split['test']
        for key in ('alpha', 'dx', 'dy', 'dz', 'rho', 'cp', 'ambient_temp'):
            if not np.isclose(getattr(dataset, key), protocol['physics'][key], rtol=0, atol=1e-12):
                raise ValueError(f'Evaluation changes material/geometry parameter {key}.')
        if args.source_counts:
            names = [name for name in names if load_experiment_metadata(str(Path(root)/name))['num_fires'] in args.source_counts]
        if not names:
            raise ValueError('No independent experiments match the requested source counts.')
        model = MODELS[cfg['model_class']](c=cfg['channels'])
        checkpoint = torch.load(run/'best.pth', map_location='cpu')
        if checkpoint.get('protocol') != protocol:
            raise ValueError('Checkpoint provenance does not match the saved experiment protocol.')
        model.load_state_dict(checkpoint['model_state_dict'])
        records, times = evaluate_experiments(model, dataset, names, protocol['physics']['prediction_horizon'],
                                              args.batch_size, args.device, dtype, args.workers)
        variant = json.dumps(dict(model=cfg['model_class'], mode=cfg['physics_residual_mode'],
                                  schedule=cfg['loss_weight_schedule'], lr=cfg['lr'],
                                  batch=cfg['batch'], channels=cfg['channels'],
                                  dtype=cfg['training_dtype']), sort_keys=True)
        identity = dict(run_id=cfg['run_id'], seed=cfg['seed'], variant=variant)
        all_records.extend(dict(**identity, **r) for r in records)
        all_times.extend(dict(**identity, **r) for r in times)
        provenance.append(dict(run=str(run), checkpoint='best.pth', checkpoint_epoch=checkpoint['epoch']+1,
                               evaluation_data=root, experiments=names, protocol=protocol))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'experiments.csv', all_records)
    write_csv(out/'times.csv', all_times)
    write_csv(out/'seed_summary.csv', summarize(all_records, ('variant', 'run_id', 'seed')))
    write_csv(out/'source_count_summary.csv', summarize(all_records, ('variant', 'run_id', 'source_count')))
    write_csv(out/'time_summary.csv', summarize(all_times, ('variant', 'run_id', 'time_seconds'),
                                               ('mae_c','rmse_c','pde_rmse_k_per_s')))
    # First average repetitions within each independent experiment, then across
    # experiments. Standard deviations are descriptive, not confidence intervals.
    per_experiment = summarize(all_records, ('variant','experiment'))
    write_csv(out/'experiment_across_seeds.csv', per_experiment)
    averaged = [dict(variant=r['variant'], experiment=r['experiment'],
                    **{m:r[m+'_mean'] for m in METRICS}) for r in per_experiment]
    write_csv(out/'variant_summary.csv', summarize(averaged, ('variant',)))
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for run_id in sorted({r['run_id'] for r in all_records}):
        rows = [r for r in all_records if r['run_id'] == run_id]
        ax.scatter([r['pde_rmse_k_per_s'] for r in rows], [r['rmse_c'] for r in rows], label=run_id)
    ax.set(xlabel='PDE residual RMSE [K/s]', ylabel='Temperature RMSE [°C]',
           title='Independent test experiments')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'physics_vs_temperature.png', dpi=180)
    plt.close(fig)
    print(f'Evaluated {len(all_records)} run/experiment pairs; results in {out}')


if __name__ == '__main__':
    main()
