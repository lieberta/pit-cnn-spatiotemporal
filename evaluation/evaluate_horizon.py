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
import torch

from configs.train_config import TRAIN_DTYPE
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst

# 

MODEL_REGISTRY = {
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}


def list_experiment_dirs(base_path: Path):
    return sorted([p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("experiment")])


def load_normalization(base_path: Path):
    norm_path = base_path / "normalization_values.json"
    if not norm_path.exists():
        return None, None
    with norm_path.open("r") as f:
        norm = json.load(f)
    return float(norm["min_temp"]), float(norm["max_temp"])


def denormalize(arr, min_temp, max_temp):
    if min_temp is None or max_temp is None:
        return arr
    return arr * (max_temp - min_temp) + min_temp


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


def load_model(model_class_name: str, channels: int, checkpoint_path: Path, device: torch.device):
    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model class '{model_class_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_class_name](c=channels).to(device=device, dtype=TRAIN_DTYPE)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


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
    model,
    test_base_path: Path,
    out_dir: Path,
    seconds_per_step: float,
    min_seconds: float,
    max_seconds: float,
    eval_stride: int,
    device: torch.device,
    make_video: bool,
    video_experiment_idx: int,
    video_fps: int,
):
    min_temp, max_temp = load_normalization(test_base_path)
    experiments = list_experiment_dirs(test_base_path)

    manifest_rows = []
    eligible = []
    for exp in experiments:
        npz_path = exp / "normalized_heat_equation_solution.npz"
        if not npz_path.exists():
            continue
        arr = np.load(npz_path)["temperature"]
        max_t = (arr.shape[0] - 1) * seconds_per_step
        row = {"experiment": exp.name, "n_steps": int(arr.shape[0]), "max_seconds": float(max_t)}
        manifest_rows.append(row)
        if max_t >= min_seconds:
            eligible.append(exp)

    write_csv(out_dir / "manifest.csv", manifest_rows, ["experiment", "n_steps", "max_seconds"])
    if not eligible:
        raise RuntimeError(f"No experiment reaches min_seconds={min_seconds}. See {out_dir / 'manifest.csv'}")

    rows = []
    frame_paths = []

    with torch.no_grad():
        for exp_idx, exp in enumerate(eligible):
            npz_path = exp / "normalized_heat_equation_solution.npz"
            temp = np.load(npz_path)["temperature"]  # [nt, nx, ny, nz]
            nt = temp.shape[0]

            input0 = torch.tensor(temp[0], dtype=TRAIN_DTYPE, device=device).unsqueeze(0).unsqueeze(0)
            y_slice = pick_fire_slice_y(temp[0])

            end_idx = min(nt - 1, int(round(max_seconds / seconds_per_step)))
            for t_idx in range(1, end_idx + 1, eval_stride):
                t_seconds = t_idx * seconds_per_step
                t_tensor = torch.tensor([[t_seconds]], dtype=TRAIN_DTYPE, device=device)

                pred = model(input0, t_tensor).detach().cpu().numpy()[0, 0]
                gt = temp[t_idx]

                pred_den = denormalize(pred, min_temp, max_temp)
                gt_den = denormalize(gt, min_temp, max_temp)
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

    fieldnames = [
        "experiment", "experiment_idx", "t_idx", "t_seconds",
        "mae", "rmse", "maxae", "rel_l2", "mean_pred", "mean_gt",
    ]
    write_csv(out_dir / "per_experiment_per_time.csv", rows, fieldnames)

    agg_rows = aggregate_by_time(rows)
    write_csv(
        out_dir / "aggregate_by_time.csv",
        agg_rows,
        ["t_seconds", "n", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "maxae_mean", "rel_l2_mean"],
    )
    plot_metric_curves(agg_rows, out_dir)

    if make_video and frame_paths:
        mp4_ok = write_video(frame_paths, out_dir / "prediction_vs_gt.mp4", fps=video_fps)
        if not mp4_ok:
            write_video(frame_paths, out_dir / "prediction_vs_gt.gif", fps=video_fps)

    print(f"[done] Eligible experiments: {len(eligible)}")
    print(f"[done] Results: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic PITCNN models over long horizons and create plots/videos.")
    parser.add_argument("--run", required=True, help="Run ID, model group under runs/dynamic, or direct .pth path")
    parser.add_argument("--runs-root", default="./runs/dynamic")
    parser.add_argument("--test-base-path", default="./data/testset")
    parser.add_argument("--out-dir", default="./plots/evaluation_horizon")

    parser.add_argument("--seconds-per-step", type=float, default=0.1, help="time delta represented by one index step")
    parser.add_argument("--min-seconds", type=float, default=20.0, help="only evaluate experiments that reach at least this horizon")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="evaluate up to this horizon")
    parser.add_argument("--eval-stride", type=int, default=1, help="evaluate every n-th timestep")

    parser.add_argument("--channels", type=int, default=None, help="override channels if not found in run config")
    parser.add_argument("--model-class", type=str, default=None, help="override model class if not found in run config")

    parser.add_argument("--video", action="store_true", help="export frame sequence + video for one example experiment")
    parser.add_argument("--video-experiment-idx", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, ckpt_path, run_dir = resolve_dynamic_run(args.run, Path(args.runs_root))
    cfg = load_run_config(run_dir)

    model_class_name = args.model_class or cfg.get("model_class")
    channels = args.channels if args.channels is not None else cfg.get("channels")

    if model_class_name is None:
        raise ValueError("Could not determine model class. Pass --model-class or ensure config.json has 'model_class'.")
    if channels is None:
        raise ValueError("Could not determine channels. Pass --channels or ensure config.json has 'channels'.")

    model = load_model(model_class_name, int(channels), ckpt_path, device)

    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "model_class": model_class_name,
        "channels": int(channels),
        "dtype": str(TRAIN_DTYPE),
        "device": str(device),
        "seconds_per_step": args.seconds_per_step,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "eval_stride": args.eval_stride,
    }
    with (out_dir / "eval_config.json").open("w") as f:
        json.dump(run_meta, f, indent=2)

    evaluate_dynamic_horizon(
        model=model,
        test_base_path=Path(args.test_base_path),
        out_dir=out_dir,
        seconds_per_step=args.seconds_per_step,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        eval_stride=max(1, args.eval_stride),
        device=device,
        make_video=args.video,
        video_experiment_idx=max(0, args.video_experiment_idx),
        video_fps=max(1, args.video_fps),
    )


if __name__ == "__main__":
    main()
