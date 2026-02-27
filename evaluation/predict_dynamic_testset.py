import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.train_config import TRAIN_DTYPE
from data import (
    SECONDS_PER_STEP,
    list_experiment_folders,
    load_normalization_values,
    load_temperature_full,
    resolve_temperature_store,
)
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst


MODEL_CLASS_REGISTRY = {
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}


def load_model_state(model, model_pth, device):
    checkpoint = torch.load(model_pth, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def find_run_dir_by_id(mode_root, run_id):
    mode_root = Path(mode_root)
    direct = mode_root / run_id
    if direct.is_dir():
        return direct
    for config_path in mode_root.rglob("config.json"):
        if config_path.parent.name == run_id:
            return config_path.parent
    return None


def resolve_dynamic_run(model_name_or_run_id, runs_root):
    mode_root = Path(runs_root)
    run_dir = find_run_dir_by_id(mode_root, model_name_or_run_id)
    if run_dir is not None:
        run_id = run_dir.name
        model_pth = run_dir / f"{run_id}.pth"
        if model_pth.exists():
            return run_id, run_dir, model_pth

    group_dir = mode_root / model_name_or_run_id
    if group_dir.is_dir():
        run_dirs = [p for p in group_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
        if run_dirs:
            run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            run_dir = run_dirs[0]
            run_id = run_dir.name
            model_pth = run_dir / f"{run_id}.pth"
            if model_pth.exists():
                return run_id, run_dir, model_pth
    return None, None, None


def load_run_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return json.load(f)


def denormalize_array(array, min_temp, temp_range):
    return array * temp_range + min_temp


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dynamic model on a full testset horizon and save denormalized predictions + MAE stats."
    )
    parser.add_argument("--run", required=True, help="Run ID or model group name in runs/dynamic.")
    parser.add_argument("--runs-root", default="./runs/dynamic", help="Root folder containing dynamic runs.")
    parser.add_argument("--testset-path", default="./data/new_detailed_heat_sim_f64/", help="Path to test dataset.")
    parser.add_argument("--out-root", default="./evaluation/predictions", help="Output root for predictions and metrics.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optional limit of evaluated experiments.")
    parser.add_argument("--experiment-offset", type=int, default=0, help="Skip first N experiment folders.")
    parser.add_argument("--time-stride", type=int, default=1, help="Evaluate every nth timestep.")
    args = parser.parse_args()

    if args.time_stride < 1:
        raise ValueError("--time-stride must be >= 1")
    if args.experiment_offset < 0:
        raise ValueError("--experiment-offset must be >= 0")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    torch.set_default_dtype(TRAIN_DTYPE)

    run_id, run_dir, model_pth = resolve_dynamic_run(args.run, args.runs_root)
    if run_id is None:
        raise FileNotFoundError(f"Could not resolve run '{args.run}' in {args.runs_root}")

    run_cfg = load_run_config(run_dir)
    model_class_name = run_cfg.get("model_class", "PITCNN_dynamic_latenttime1")
    channels = int(run_cfg.get("channels", 16))
    if model_class_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(f"Unsupported model_class '{model_class_name}'. Available: {list(MODEL_CLASS_REGISTRY.keys())}")

    model = MODEL_CLASS_REGISTRY[model_class_name](c=channels).to(device=device, dtype=TRAIN_DTYPE)
    load_model_state(model, str(model_pth), device)
    model.eval()

    min_temp, _, temp_range = load_normalization_values(args.testset_path)
    folders = sorted(list_experiment_folders(args.testset_path))
    folders = folders[args.experiment_offset :]
    if args.max_experiments is not None:
        folders = folders[: args.max_experiments]

    out_dir = Path(args.out_root) / run_id
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    per_t_abs_sum = {}
    per_t_count = {}
    global_abs_sum = 0.0
    global_count = 0
    per_experiment_rows = []
    processed_experiments = 0

    for folder in tqdm(folders, desc="Experiments"):
        store = resolve_temperature_store(folder)
        if store is None:
            continue

        data_norm = load_temperature_full(store, min_temp, temp_range)
        if data_norm.shape[0] < 2:
            continue

        exp_name = Path(folder).name
        nt, nx, ny, nz = data_norm.shape
        input0 = torch.tensor(data_norm[0], dtype=TRAIN_DTYPE, device=device).unsqueeze(0).unsqueeze(0)
        pred_denorm = np.full((nt, nx, ny, nz), np.nan, dtype=np.float32)
        pred_denorm[0] = denormalize_array(data_norm[0], min_temp, temp_range).astype(np.float32)

        exp_abs_sum = 0.0
        exp_count = 0

        with torch.no_grad():
            for t_idx in range(1, nt, args.time_stride):
                t_seconds = t_idx * SECONDS_PER_STEP
                time_tensor = torch.tensor([[t_seconds]], dtype=TRAIN_DTYPE, device=device)
                pred_norm_t = model(input0, time_tensor)

                target_norm_t = torch.tensor(data_norm[t_idx], dtype=TRAIN_DTYPE, device=device).unsqueeze(0).unsqueeze(0)
                pred_denorm_t = pred_norm_t * temp_range + min_temp
                target_denorm_t = target_norm_t * temp_range + min_temp

                abs_err = torch.abs(pred_denorm_t - target_denorm_t)
                abs_sum = float(abs_err.sum().item())
                cnt = int(abs_err.numel())

                pred_denorm[t_idx] = pred_denorm_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

                per_t_abs_sum[t_idx] = per_t_abs_sum.get(t_idx, 0.0) + abs_sum
                per_t_count[t_idx] = per_t_count.get(t_idx, 0) + cnt
                global_abs_sum += abs_sum
                global_count += cnt
                exp_abs_sum += abs_sum
                exp_count += cnt

        exp_out = pred_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            exp_out / "prediction_denorm.npz",
            temperature=pred_denorm,
            time_seconds=np.arange(nt, dtype=np.float32) * float(SECONDS_PER_STEP),
        )

        exp_mae = (exp_abs_sum / exp_count) if exp_count > 0 else float("nan")
        per_experiment_rows.append(
            {
                "experiment": exp_name,
                "num_timesteps": nt,
                "timesteps_evaluated": len(range(1, nt, args.time_stride)),
                "mae_abs": exp_mae,
            }
        )
        processed_experiments += 1

    if processed_experiments == 0 or global_count == 0:
        raise RuntimeError("No experiments were evaluated. Check testset path and data files.")

    per_t_rows = []
    for t_idx in sorted(per_t_abs_sum.keys()):
        mae_t = per_t_abs_sum[t_idx] / per_t_count[t_idx]
        per_t_rows.append(
            {
                "timestep_index": t_idx,
                "time_seconds": t_idx * float(SECONDS_PER_STEP),
                "mae_abs": mae_t,
            }
        )

    with (out_dir / "mae_per_timestep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestep_index", "time_seconds", "mae_abs"])
        writer.writeheader()
        writer.writerows(per_t_rows)

    with (out_dir / "mae_per_experiment.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "num_timesteps", "timesteps_evaluated", "mae_abs"])
        writer.writeheader()
        writer.writerows(per_experiment_rows)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_checkpoint": str(model_pth),
        "model_class": model_class_name,
        "channels": channels,
        "testset_path": args.testset_path,
        "processed_experiments": processed_experiments,
        "time_stride": args.time_stride,
        "global_mae_abs": global_abs_sum / global_count,
        "output_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
