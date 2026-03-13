import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import zarr
except ImportError:
    zarr = None

from data import (
    list_experiment_folders,
    load_time_grid_from_metadata,
    load_normalization_values,
    load_temperature_full,
    resolve_temperature_store,
)

def denormalize(arr, min_temp, temp_range):
    return arr * temp_range + min_temp


def load_prediction_temperature(store_path: Path):
    if zarr is None:
        raise ImportError("zarr is required to read prediction stores. Install with: pip install zarr")
    root = zarr.open_group(str(store_path), mode="r")
    return np.asarray(root["temperature"])


def load_run_config(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r") as f:
        return json.load(f)


def resolve_dynamic_run(identifier: str, runs_root: Path):
    as_path = Path(identifier)
    if as_path.is_file() and as_path.suffix == ".pth":
        run_dir = as_path.parent
        return run_dir.name, as_path, run_dir

    # 1) identifier as run_id
    for cfg in runs_root.rglob("config.json"):
        run_dir = cfg.parent
        if run_dir.name == identifier:
            ckpt = run_dir / f"{run_dir.name}.pth"
            if ckpt.exists():
                return run_dir.name, ckpt, run_dir

    # 2) identifier as model group -> latest run
    group_dir = runs_root / identifier
    if group_dir.is_dir():
        runs = [p for p in group_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in runs:
            ckpt = run_dir / f"{run_dir.name}.pth"
            if ckpt.exists():
                return run_dir.name, ckpt, run_dir

    raise FileNotFoundError(f"Could not resolve run from identifier '{identifier}' under '{runs_root}'.")

def pick_fire_slice_y(input0_3d: np.ndarray):
    # Fire region is near z=0 with elevated temp at t=0.
    plane = input0_3d[:, :, 0]
    idx = np.unravel_index(np.argmax(plane), plane.shape)
    return int(idx[1])


def metrics(pred, gt):
    err = pred - gt
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(abs_err))
    maxae = float(np.max(abs_err))
    denom = float(np.linalg.norm(gt.ravel(), ord=2))
    rel_l2 = float(np.linalg.norm(err.ravel(), ord=2) / denom) if denom > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "maxae": maxae,
        "rel_l2": rel_l2,
        "mean_pred": float(np.mean(pred)),
        "mean_gt": float(np.mean(gt)),
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_time(rows):
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["t_seconds"]].append(r)

    out = []
    for t in sorted(by_t.keys()):
        bucket = by_t[t]
        out.append({
            "t_seconds": t,
            "n": len(bucket),
            "mae_mean": mean([b["mae"] for b in bucket]),
            "mae_std": pstdev([b["mae"] for b in bucket]) if len(bucket) > 1 else 0.0,
            "rmse_mean": mean([b["rmse"] for b in bucket]),
            "rmse_std": pstdev([b["rmse"] for b in bucket]) if len(bucket) > 1 else 0.0,
            "maxae_mean": mean([b["maxae"] for b in bucket]),
            "rel_l2_mean": mean([b["rel_l2"] for b in bucket]),
        })
    return out


def plot_metric_curves(agg_rows, out_dir: Path):
    if not agg_rows:
        return
    t = [r["t_seconds"] for r in agg_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(t, [r["mae_mean"] for r in agg_rows], label="MAE")
    plt.plot(t, [r["rmse_mean"] for r in agg_rows], label="RMSE")
    plt.plot(t, [r["maxae_mean"] for r in agg_rows], label="MaxAE")
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title("Error over prediction horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, [r["rel_l2_mean"] for r in agg_rows], label="Relative L2", color="tab:purple")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative L2")
    plt.title("Relative L2 over prediction horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rel_l2_over_time.png", dpi=150)
    plt.close()


def render_triplet_frame(gt, pred, abs_err, y_slice, t_seconds, exp_name, out_path: Path):
    vmax = max(float(np.max(gt)), float(np.max(pred)))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = axs[0].imshow(gt[:, y_slice, :].T, origin="lower", cmap="hot", vmin=float(np.min(gt)), vmax=vmax)
    axs[0].set_title(f"GT @ {t_seconds:.1f}s")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(pred[:, y_slice, :].T, origin="lower", cmap="hot", vmin=float(np.min(gt)), vmax=vmax)
    axs[1].set_title("Prediction")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(abs_err[:, y_slice, :].T, origin="lower", cmap="magma")
    axs[2].set_title("Abs Error")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    fig.suptitle(f"{exp_name} | y-slice={y_slice}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_video(frame_paths, out_video_path: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except Exception as e:
        print(f"[warn] imageio unavailable, skip video export: {e}")
        return False

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(out_video_path, fps=fps) as writer:
            for p in frame_paths:
                writer.append_data(imageio.imread(p))
        return True
    except Exception as e:
        print(f"[warn] video export failed ({out_video_path}): {e}")
        return False


def evaluate_dynamic_horizon(
    test_base_path: Path,
    prediction_root: Path,
    out_dir: Path,
    dt: float,
    min_seconds: float,
    max_seconds: float,
    eval_stride: int,
    make_video: bool,
    video_experiment_idx: int,
    video_fps: int,
):
    min_temp, _, temp_range = load_normalization_values(str(test_base_path))
    experiments = [Path(p) for p in sorted(list_experiment_folders(str(test_base_path)))]

    manifest_rows = []
    eligible = []
    for exp in experiments:
        store_path = resolve_temperature_store(str(exp))
        if store_path is None:
            continue
        arr = load_temperature_full(store_path, min_temp, temp_range)
        max_t = (arr.shape[0] - 1) * dt
        row = {"experiment": exp.name, "n_steps": int(arr.shape[0]), "max_seconds": float(max_t)}
        manifest_rows.append(row)
        if max_t >= min_seconds:
            eligible.append((exp, store_path))

    write_csv(out_dir / "manifest.csv", manifest_rows, ["experiment", "n_steps", "max_seconds"])
    if not eligible:
        raise RuntimeError(f"No experiment reaches min_seconds={min_seconds}. See {out_dir / 'manifest.csv'}")

    rows = []
    per_experiment_summary_rows = []
    frame_paths = []
    total_predictions = 0

    for exp_idx, (exp, store_path) in enumerate(eligible):
        gt_norm = load_temperature_full(store_path, min_temp, temp_range)
        nt = gt_norm.shape[0]
        y_slice = pick_fire_slice_y(gt_norm[0])

        pred_path = prediction_root / "predictions" / exp.name / "heat_equation_solution.zarr"
        if not pred_path.exists():
            continue

        pred_den_all = load_prediction_temperature(pred_path)

        if pred_den_all.shape[0] != nt:
            raise ValueError(
                f"Prediction timestep count mismatch for '{exp.name}': "
                f"pred={pred_den_all.shape[0]} vs gt={nt}."
            )

        experiment_predictions = 0
        end_idx = min(nt - 1, int(round(max_seconds / dt)))
        for t_idx in range(1, end_idx + 1, eval_stride):
            t_seconds = t_idx * dt
            gt_den = denormalize(gt_norm[t_idx], min_temp, temp_range)
            pred_den = pred_den_all[t_idx]
            total_predictions += 1
            experiment_predictions += 1

            m = metrics(pred_den, gt_den)
            m.update({
                "experiment": exp.name,
                "experiment_idx": exp_idx,
                "t_idx": t_idx,
                "t_seconds": round(float(t_seconds), 6),
            })
            rows.append(m)

            if make_video and exp_idx == video_experiment_idx:
                abs_err = np.abs(pred_den - gt_den)
                frame_path = out_dir / "frames" / f"frame_{t_idx:04d}.png"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                render_triplet_frame(gt_den, pred_den, abs_err, y_slice, t_seconds, exp.name, frame_path)
                frame_paths.append(frame_path)

        # This summary is now about loaded prediction coverage, not runtime.
        per_experiment_summary_rows.append(
            {
                "experiment": exp.name,
                "experiment_idx": exp_idx,
                "num_predictions": experiment_predictions,
            }
        )

    fieldnames = [
        "experiment", "experiment_idx", "t_idx", "t_seconds",
        "mae", "rmse", "maxae", "rel_l2", "mean_pred", "mean_gt",
    ]
    write_csv(out_dir / "per_experiment_per_time.csv", rows, fieldnames)

    agg_rows = aggregate_by_time(rows)
    write_csv(
        out_dir / "aggregate_by_time.csv",
        agg_rows,
        [
            "t_seconds", "n",
            "mae_mean", "mae_std",
            "rmse_mean", "rmse_std",
            "maxae_mean", "rel_l2_mean",
        ],
    )
    write_csv(
        out_dir / "per_experiment_summary.csv",
        per_experiment_summary_rows,
        ["experiment", "experiment_idx", "num_predictions"],
    )
    plot_metric_curves(agg_rows, out_dir)

    if make_video and frame_paths:
        mp4_ok = write_video(frame_paths, out_dir / "prediction_vs_gt.mp4", fps=video_fps)
        if not mp4_ok:
            write_video(frame_paths, out_dir / "prediction_vs_gt.gif", fps=video_fps)

    summary = {
        "eligible_experiments": len(eligible),
        "total_predictions": int(total_predictions),
        "prediction_root": str(prediction_root),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Eligible experiments: {len(eligible)}")
    print(f"[done] Results: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved dynamic-model predictions over long horizons and create plots/videos.")
    parser.add_argument("--run", required=True, help="Run ID, model group under runs/dynamic, or direct .pth path")
    parser.add_argument("--runs-root", default="./runs/dynamic")
    parser.add_argument("--test-base-path", default="./data/new_detailed_heat_sim_f64")
    parser.add_argument("--out-dir", default="./plots/evaluation_horizon")
    parser.add_argument(
        "--prediction-root",
        default="./evaluation/predictions",
        help="Root directory containing saved predictions, typically from evaluation/predict_dynamic_testset.py.",
    )

    parser.add_argument("--dt", type=float, default=None, help="time delta represented by one index step")
    parser.add_argument("--min-seconds", type=float, default=20.0, help="only evaluate experiments that reach at least this horizon")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="evaluate up to this horizon")
    parser.add_argument("--eval-stride", type=int, default=1, help="evaluate every n-th timestep")

    parser.add_argument("--video", action="store_true", help="export frame sequence + video for one example experiment")
    parser.add_argument("--video-experiment-idx", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)

    args = parser.parse_args()
    if args.dt is None:
        _, dt, _ = load_time_grid_from_metadata(args.test_base_path)
        args.dt = float(dt)
    if args.dt <= 0.0:
        raise ValueError(f"Invalid dt: {args.dt}")

    run_id, ckpt_path, run_dir = resolve_dynamic_run(args.run, Path(args.runs_root))
    cfg = load_run_config(run_dir)

    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg.get("name") or cfg.get("model_class") or run_id
    prediction_root = Path(args.prediction_root) / model_name / run_id
    if not prediction_root.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_root}. "
            f"Run evaluation/predict_dynamic_testset.py first."
        )

    run_meta = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "model_class": cfg.get("model_class"),
        "channels": cfg.get("channels"),
        "dt": args.dt,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "eval_stride": args.eval_stride,
        "prediction_root": str(prediction_root),
    }
    with (out_dir / "eval_config.json").open("w") as f:
        json.dump(run_meta, f, indent=2)

    evaluate_dynamic_horizon(
        test_base_path=Path(args.test_base_path),
        prediction_root=prediction_root,
        out_dir=out_dir,
        dt=args.dt,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        eval_stride=max(1, args.eval_stride),
        make_video=args.video,
        video_experiment_idx=max(0, args.video_experiment_idx),
        video_fps=max(1, args.video_fps),
    )


if __name__ == "__main__":
    main()
