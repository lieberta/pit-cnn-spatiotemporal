import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import zarr
except ImportError:
    zarr = None

from data import (
    SECONDS_PER_STEP,
    list_experiment_folders,
    load_normalization_values,
    load_temperature_full,
    resolve_temperature_store,
)
from evaluation.plot_room_slice import render_experiment
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst


MODEL_CLASS_REGISTRY = {
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}

LEGACY_DYNAMIC_TRAIN_DATA_PATH = "./data/new_detailed_heat_sim_f64/"


def get_run_dtype(run_cfg):
    return torch.float64 if run_cfg.get("training_dtype") == "float64" else torch.float32


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
    candidate_path = Path(model_name_or_run_id)
    if candidate_path.suffix == ".pth" and candidate_path.is_file():
        run_dir = candidate_path.parent
        run_id = run_dir.name
        return run_id, run_dir, candidate_path
    if candidate_path.is_dir():
        run_dir = candidate_path
        run_id = run_dir.name
        model_pth = run_dir / f"{run_id}.pth"
        if model_pth.exists():
            return run_id, run_dir, model_pth

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


def derive_testset_name(testset_path: str) -> str:
    testset_dir = Path(testset_path)
    if testset_dir.name:
        return testset_dir.name
    return testset_dir.resolve().parent.name


def denormalize_array(array, min_temp, temp_range):
    return array * temp_range + min_temp


def time_bucket_seconds(t_seconds: float, bucket_size: float = 0.1) -> float:
    return round(round(t_seconds / bucket_size) * bucket_size, 6)


def resolve_normalization_base_path(run_cfg, testset_path: str) -> str:
    # Dynamic runs should use the training normalization, not a separately scaled testset.
    data_path = run_cfg.get("data_path")
    if data_path:
        return data_path

    # Legacy dynamic runs did not persist data_path in config.json.
    # Those runs were trained against the project's standard dynamic dataset.
    legacy_default = Path(LEGACY_DYNAMIC_TRAIN_DATA_PATH)
    if legacy_default.exists():
        return str(legacy_default)

    raise FileNotFoundError(
        "Could not determine normalization source path. "
        f"Run config has no 'data_path', testset='{testset_path}', and legacy default "
        f"'{LEGACY_DYNAMIC_TRAIN_DATA_PATH}' is missing."
    )


def load_axes_from_store(store_path: str, nt: int, nx: int, ny: int, nz: int):
    if store_path.endswith(".zarr"):
        if zarr is None:
            raise ImportError("zarr is required to read .zarr datasets. Install with: pip install zarr")
        root = zarr.open_group(store_path, mode="r")
        time_axis = np.asarray(root["time"]) if "time" in root else np.arange(nt, dtype=np.float32) * float(SECONDS_PER_STEP)
        x_axis = np.asarray(root["x"]) if "x" in root else np.arange(nx, dtype=np.float32)
        y_axis = np.asarray(root["y"]) if "y" in root else np.arange(ny, dtype=np.float32)
        z_axis = np.asarray(root["z"]) if "z" in root else np.arange(nz, dtype=np.float32)
        return time_axis, x_axis, y_axis, z_axis

    with np.load(store_path) as npz_file:
        time_axis = npz_file["time"] if "time" in npz_file.files else np.arange(nt, dtype=np.float32) * float(SECONDS_PER_STEP)
        x_axis = npz_file["x"] if "x" in npz_file.files else np.arange(nx, dtype=np.float32)
        y_axis = npz_file["y"] if "y" in npz_file.files else np.arange(ny, dtype=np.float32)
        z_axis = npz_file["z"] if "z" in npz_file.files else np.arange(nz, dtype=np.float32)
    return time_axis, x_axis, y_axis, z_axis


def write_prediction_zarr(exp_out: Path, temperature, time_axis, x_axis, y_axis, z_axis):
    if zarr is None:
        raise ImportError("zarr is required to save prediction stores. Install with: pip install zarr")

    root = zarr.open_group(str(exp_out / "heat_equation_solution.zarr"), mode="w")
    root.create_dataset(
        "temperature",
        data=temperature,
        chunks=(1, temperature.shape[1], temperature.shape[2], temperature.shape[3]),
        overwrite=True,
    )
    root.create_dataset("time", data=time_axis, overwrite=True)
    root.create_dataset("x", data=x_axis, overwrite=True)
    root.create_dataset("y", data=y_axis, overwrite=True)
    root.create_dataset("z", data=z_axis, overwrite=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dynamic model on a full testset horizon and save denormalized predictions + MAE stats."
    )
    parser.add_argument("--run", required=True, help="Run ID or model group name in runs/dynamic.")
    parser.add_argument("--runs-root", default="./runs/dynamic", help="Root folder containing dynamic runs.")
    parser.add_argument("--testset-path", default="./data/new_detailed_heat_sim_f64/", help="Path to test dataset.")
    parser.add_argument(
        "--out-root",
        default="./evaluation/predictions",
        help="Deprecated. Predictions are always saved inside the resolved run directory.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optional limit of evaluated experiments.")
    parser.add_argument("--experiment-offset", type=int, default=0, help="Skip first N experiment folders.")
    parser.add_argument("--time-stride", type=int, default=1, help="Evaluate every nth timestep.")
    parser.add_argument("--plot-step-every", type=int, default=100, help="Render every n-th stored timestep after prediction (default 0.1s at dt=0.001).")
    parser.add_argument("--plot-output-subdir", default="plots_room_slice", help="Plot subfolder inside each prediction experiment folder.")
    args = parser.parse_args()

    if args.time_stride < 1:
        raise ValueError("--time-stride must be >= 1")
    if args.experiment_offset < 0:
        raise ValueError("--experiment-offset must be >= 0")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    run_id, run_dir, model_pth = resolve_dynamic_run(args.run, args.runs_root)
    if run_id is None:
        raise FileNotFoundError(f"Could not resolve run '{args.run}' in {args.runs_root}")

    run_cfg = load_run_config(run_dir)
    train_dtype = get_run_dtype(run_cfg)
    torch.set_default_dtype(train_dtype)
    model_class_name = run_cfg.get("model_class", "PITCNN_dynamic_latenttime1")
    channels = int(run_cfg.get("channels", 16))
    if model_class_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(f"Unsupported model_class '{model_class_name}'. Available: {list(MODEL_CLASS_REGISTRY.keys())}")

    model = MODEL_CLASS_REGISTRY[model_class_name](c=channels).to(device=device, dtype=train_dtype)
    load_model_state(model, str(model_pth), device)
    model.eval()

    normalization_base_path = resolve_normalization_base_path(run_cfg, args.testset_path)
    min_temp, _, temp_range = load_normalization_values(normalization_base_path)
    folders = sorted(list_experiment_folders(args.testset_path))
    folders = folders[args.experiment_offset :]
    if args.max_experiments is not None:
        folders = folders[: args.max_experiments]

    testset_name = derive_testset_name(args.testset_path)
    out_dir = Path(run_dir) / "predictions" / testset_name
    pred_dir = out_dir
    pred_dir.mkdir(parents=True, exist_ok=True)

    per_t_abs_sum = {}
    per_t_count = {}
    per_bucket_abs_sum = {}
    per_bucket_count = {}
    global_abs_sum = 0.0
    global_count = 0
    per_experiment_rows = []
    processed_experiments = 0
    total_predictions = 0
    total_inference_seconds = 0.0
    prediction_job_tic = time.perf_counter()

    for folder in tqdm(folders, desc="Experiments"):
        store = resolve_temperature_store(folder)
        if store is None:
            continue

        data_norm = load_temperature_full(store, min_temp, temp_range)
        if data_norm.shape[0] < 2:
            continue

        exp_name = Path(folder).name
        nt, nx, ny, nz = data_norm.shape
        input0 = torch.tensor(data_norm[0], dtype=train_dtype, device=device).unsqueeze(0).unsqueeze(0)
        pred_denorm = np.full((nt, nx, ny, nz), np.nan, dtype=np.float32)
        pred_denorm[0] = denormalize_array(data_norm[0], min_temp, temp_range).astype(np.float32)

        exp_abs_sum = 0.0
        exp_count = 0
        exp_predictions = 0
        exp_inference_seconds = 0.0
        exp_per_t_abs_sum = {}
        exp_per_t_count = {}
        exp_per_bucket_abs_sum = {}
        exp_per_bucket_count = {}

        with torch.no_grad():
            for t_idx in range(1, nt, args.time_stride):
                t_seconds = t_idx * SECONDS_PER_STEP
                time_tensor = torch.tensor([[t_seconds]], dtype=train_dtype, device=device)
                # Measure only the model forward pass so prediction runtime stays comparable.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    pred_norm_t = model(input0, time_tensor)
                    end_event.record()
                    torch.cuda.synchronize(device)
                    inference_seconds = start_event.elapsed_time(end_event) / 1000.0
                else:
                    tic = time.perf_counter()
                    pred_norm_t = model(input0, time_tensor)
                    inference_seconds = time.perf_counter() - tic

                total_predictions += 1
                total_inference_seconds += float(inference_seconds)
                exp_predictions += 1
                exp_inference_seconds += float(inference_seconds)

                target_norm_t = torch.tensor(data_norm[t_idx], dtype=train_dtype, device=device).unsqueeze(0).unsqueeze(0)
                pred_denorm_t = pred_norm_t * temp_range + min_temp
                target_denorm_t = target_norm_t * temp_range + min_temp

                abs_err = torch.abs(pred_denorm_t - target_denorm_t)
                abs_sum = float(abs_err.sum().item())
                cnt = int(abs_err.numel())

                pred_denorm[t_idx] = pred_denorm_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

                per_t_abs_sum[t_idx] = per_t_abs_sum.get(t_idx, 0.0) + abs_sum
                per_t_count[t_idx] = per_t_count.get(t_idx, 0) + cnt
                exp_per_t_abs_sum[t_idx] = exp_per_t_abs_sum.get(t_idx, 0.0) + abs_sum
                exp_per_t_count[t_idx] = exp_per_t_count.get(t_idx, 0) + cnt
                bucket_seconds = time_bucket_seconds(t_seconds, bucket_size=0.1)
                per_bucket_abs_sum[bucket_seconds] = per_bucket_abs_sum.get(bucket_seconds, 0.0) + abs_sum
                per_bucket_count[bucket_seconds] = per_bucket_count.get(bucket_seconds, 0) + cnt
                exp_per_bucket_abs_sum[bucket_seconds] = exp_per_bucket_abs_sum.get(bucket_seconds, 0.0) + abs_sum
                exp_per_bucket_count[bucket_seconds] = exp_per_bucket_count.get(bucket_seconds, 0) + cnt
                global_abs_sum += abs_sum
                global_count += cnt
                exp_abs_sum += abs_sum
                exp_count += cnt

        exp_out = pred_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)
        time_axis, x_axis, y_axis, z_axis = load_axes_from_store(store, nt, nx, ny, nz)
        write_prediction_zarr(exp_out, pred_denorm, time_axis, x_axis, y_axis, z_axis)
        render_experiment(
            exp_out,
            npz_name="heat_equation_solution.zarr",
            step_every=args.plot_step_every,
            vmax_clip=500.0,
            output_subdir=args.plot_output_subdir,
        )

        exp_mae = (exp_abs_sum / exp_count) if exp_count > 0 else float("nan")
        exp_per_t_rows = []
        for t_idx in sorted(exp_per_t_abs_sum.keys()):
            exp_per_t_rows.append(
                {
                    "timestep_index": t_idx,
                    "time_seconds": t_idx * float(SECONDS_PER_STEP),
                    "mae_abs": exp_per_t_abs_sum[t_idx] / exp_per_t_count[t_idx],
                }
            )
        with (exp_out / "mae_per_timestep.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestep_index", "time_seconds", "mae_abs"])
            writer.writeheader()
            writer.writerows(exp_per_t_rows)

        exp_per_bucket_rows = []
        for bucket_seconds in sorted(exp_per_bucket_abs_sum.keys()):
            exp_per_bucket_rows.append(
                {
                    "time_seconds": float(bucket_seconds),
                    "mae_abs": exp_per_bucket_abs_sum[bucket_seconds] / exp_per_bucket_count[bucket_seconds],
                }
            )
        with (exp_out / "mae_per_0p1s.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_seconds", "mae_abs"])
            writer.writeheader()
            writer.writerows(exp_per_bucket_rows)

        exp_summary = {
            "experiment": exp_name,
            "num_timesteps": nt,
            "timesteps_evaluated": len(range(1, nt, args.time_stride)),
            "mae_abs": exp_mae,
            "num_predictions": exp_predictions,
            "total_inference_seconds": float(exp_inference_seconds),
            "mean_inference_seconds": (
                float(exp_inference_seconds / exp_predictions) if exp_predictions > 0 else float("nan")
            ),
        }
        with (exp_out / "mae_summary.json").open("w") as f:
            json.dump(exp_summary, f, indent=2)

        per_experiment_rows.append(
            exp_summary
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

    per_bucket_rows = []
    for bucket_seconds in sorted(per_bucket_abs_sum.keys()):
        mae_bucket = per_bucket_abs_sum[bucket_seconds] / per_bucket_count[bucket_seconds]
        per_bucket_rows.append(
            {
                "time_seconds": float(bucket_seconds),
                "mae_abs": mae_bucket,
            }
        )

    with (out_dir / "mae_per_0p1s.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_seconds", "mae_abs"])
        writer.writeheader()
        writer.writerows(per_bucket_rows)

    with (out_dir / "mae_per_experiment.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "num_timesteps",
                "timesteps_evaluated",
                "mae_abs",
                "num_predictions",
                "total_inference_seconds",
                "mean_inference_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(per_experiment_rows)

    prediction_job_seconds = time.perf_counter() - prediction_job_tic

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_checkpoint": str(model_pth),
        "model_class": model_class_name,
        "channels": channels,
        "testset_path": args.testset_path,
        "testset_name": testset_name,
        "normalization_base_path": normalization_base_path,
        "processed_experiments": processed_experiments,
        "time_stride": args.time_stride,
        "mae_bucket_seconds": 0.1,
        "total_predictions": total_predictions,
        "total_inference_seconds": float(total_inference_seconds),
        "mean_inference_seconds": (
            float(total_inference_seconds / total_predictions) if total_predictions > 0 else float("nan")
        ),
        "prediction_job_seconds": float(prediction_job_seconds),
        "global_mae_abs": global_abs_sum / global_count,
        "output_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
