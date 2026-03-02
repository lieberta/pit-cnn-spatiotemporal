import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

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


MODEL_REGISTRY = {
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    for cfg in runs_root.rglob("config.json"):
        run_dir = cfg.parent
        if run_dir.name == identifier:
            ckpt = run_dir / f"{run_dir.name}.pth"
            if ckpt.exists():
                return run_dir.name, ckpt, run_dir

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

def parse_times(times_str: str):
    out = []
    for s in times_str.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    return sorted(set(out))


def generate_times(min_seconds: float, max_seconds: float, step_seconds: float):
    values = []
    t = min_seconds
    while t <= max_seconds + 1e-12:
        values.append(round(t, 6))
        t += step_seconds
    return values


def load_sim_runtime_map(manifest_csv: Path):
    if manifest_csv is None or not manifest_csv.exists():
        return {}

    out = {}
    with manifest_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment_name")
            runtime = row.get("runtime_seconds")
            if not name or runtime is None:
                continue
            try:
                out[name] = float(runtime)
            except ValueError:
                continue
    return out


def measure_one_forward(model, input0, t_tensor, warmup: int, repeats: int, device: torch.device):
    # Warmup helps with CUDA kernel initialization variance.
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(input0, t_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(max(1, repeats)):
            tic = time.perf_counter()
            _ = model(input0, t_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tac = time.perf_counter()
            times.append(tac - tic)

    return {
        "inference_seconds_mean": float(np.mean(times)),
        "inference_seconds_std": float(np.std(times)),
        "inference_seconds_min": float(np.min(times)),
        "inference_seconds_max": float(np.max(times)),
    }


def aggregate(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)

    out = []
    for k in sorted(buckets.keys()):
        vals = buckets[k]
        means = [v["inference_seconds_mean"] for v in vals]
        out.append(
            {
                key: k,
                "n": len(vals),
                "inference_seconds_mean": mean(means),
                "inference_seconds_std": pstdev(means) if len(means) > 1 else 0.0,
                "inference_seconds_min": min(means),
                "inference_seconds_max": max(means),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark dynamic model inference runtime and compare with simulation runtime.")
    parser.add_argument("--run", required=True, help="run_id | dynamic model group name | direct .pth path")
    parser.add_argument("--runs-root", default="./runs/dynamic")
    parser.add_argument("--test-base-path", default="./data/new_detailed_heat_sim_f64")
    parser.add_argument("--out-dir", default="./plots/inference_benchmark")

    parser.add_argument("--model-class", default=None, help="override model class")
    parser.add_argument("--channels", type=int, default=None, help="override channels")

    parser.add_argument("--times", default=None, help="comma-separated seconds, e.g. '1,5,10,20'")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--seconds-per-step", type=float, default=float(SECONDS_PER_STEP))

    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)

    parser.add_argument(
        "--simulation-manifest-csv",
        default=None,
        help="optional path to simulation_manifest.csv for speedup comparison",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, ckpt_path, run_dir = resolve_dynamic_run(args.run, Path(args.runs_root))
    run_cfg = load_run_config(run_dir)

    model_class = args.model_class or run_cfg.get("model_class")
    channels = args.channels if args.channels is not None else run_cfg.get("channels")
    if model_class is None:
        raise ValueError("Could not determine model class. Pass --model-class or ensure config.json has model_class.")
    if channels is None:
        raise ValueError("Could not determine channels. Pass --channels or ensure config.json has channels.")

    model = load_model(model_class, int(channels), ckpt_path, device)

    if args.times:
        times = parse_times(args.times)
    else:
        times = generate_times(args.min_seconds, args.max_seconds, args.step_seconds)
    if not times:
        raise ValueError("No evaluation times specified.")

    min_temp, _, temp_range = load_normalization_values(args.test_base_path)
    experiments = [Path(p) for p in sorted(list_experiment_folders(args.test_base_path))]
    if args.max_experiments is not None:
        experiments = experiments[: max(0, args.max_experiments)]
    if not experiments:
        raise RuntimeError(f"No experiment_* folders found in '{args.test_base_path}'.")

    sim_runtime_map = load_sim_runtime_map(Path(args.simulation_manifest_csv)) if args.simulation_manifest_csv else {}

    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for exp in experiments:
            store_path = resolve_temperature_store(str(exp))
            if store_path is None:
                print(f"[skip] missing temperature store: {exp}")
                continue

            temp = load_temperature_full(store_path, min_temp, temp_range)
            nt = temp.shape[0]

            input0 = torch.tensor(temp[0], dtype=TRAIN_DTYPE, device=device).unsqueeze(0).unsqueeze(0)
            sim_runtime = sim_runtime_map.get(exp.name)

            for t_seconds in times:
                t_idx = int(round(t_seconds / args.seconds_per_step))
                if t_idx <= 0 or t_idx >= nt:
                    continue

                t_tensor = torch.tensor([[float(t_seconds)]], dtype=TRAIN_DTYPE, device=device)
                m = measure_one_forward(
                    model=model,
                    input0=input0,
                    t_tensor=t_tensor,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                )

                row = {
                    "experiment": exp.name,
                    "t_seconds": float(t_seconds),
                    "t_idx": int(t_idx),
                    "dtype": str(TRAIN_DTYPE),
                    "device": str(device),
                    "warmup": int(args.warmup),
                    "repeats": int(args.repeats),
                    **m,
                }

                # Optional coarse speedup against full simulation runtime of this experiment.
                if sim_runtime is not None and m["inference_seconds_mean"] > 0:
                    row["simulation_runtime_seconds"] = float(sim_runtime)
                    row["speedup_sim_over_inference"] = float(sim_runtime / m["inference_seconds_mean"])
                else:
                    row["simulation_runtime_seconds"] = ""
                    row["speedup_sim_over_inference"] = ""

                rows.append(row)

    if not rows:
        raise RuntimeError("No benchmark rows created. Check testset path, times, and seconds-per-step.")

    per_case_fields = [
        "experiment", "t_seconds", "t_idx", "dtype", "device", "warmup", "repeats",
        "inference_seconds_mean", "inference_seconds_std", "inference_seconds_min", "inference_seconds_max",
        "simulation_runtime_seconds", "speedup_sim_over_inference",
    ]
    write_csv(out_dir / "inference_per_experiment_per_time.csv", rows, per_case_fields)

    by_time = aggregate(rows, "t_seconds")
    write_csv(
        out_dir / "inference_aggregate_by_time.csv",
        by_time,
        ["t_seconds", "n", "inference_seconds_mean", "inference_seconds_std", "inference_seconds_min", "inference_seconds_max"],
    )

    by_experiment = aggregate(rows, "experiment")
    write_csv(
        out_dir / "inference_aggregate_by_experiment.csv",
        by_experiment,
        ["experiment", "n", "inference_seconds_mean", "inference_seconds_std", "inference_seconds_min", "inference_seconds_max"],
    )

    summary = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "model_class": model_class,
        "channels": int(channels),
        "dtype": str(TRAIN_DTYPE),
        "device": str(device),
        "n_rows": len(rows),
        "n_experiments": len(set(r["experiment"] for r in rows)),
        "times": sorted(set(r["t_seconds"] for r in rows)),
        "mean_inference_seconds": float(np.mean([r["inference_seconds_mean"] for r in rows])),
    }
    with (out_dir / "benchmark_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] benchmark rows: {len(rows)}")
    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
