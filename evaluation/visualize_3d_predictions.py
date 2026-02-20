import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs.train_config import TRAIN_DTYPE
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst


MODEL_REGISTRY = {
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}


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

    raise FileNotFoundError(f"Could not resolve run from '{identifier}' under '{runs_root}'.")


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
    if not out:
        raise ValueError("No valid times parsed from --times")
    return out


def list_experiments(base_path: Path):
    return sorted([p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("experiment")])


def load_normalization(base_path: Path):
    norm_path = base_path / "normalization_values.json"
    if not norm_path.exists():
        return None, None
    with norm_path.open("r") as f:
        d = json.load(f)
    return float(d["min_temp"]), float(d["max_temp"])


def denormalize(arr, min_temp, max_temp):
    if min_temp is None or max_temp is None:
        return arr
    return arr * (max_temp - min_temp) + min_temp


def make_volume_toggle_html(gt, pred, err, title: str, out_html: Path, opacity=0.12, surface_count=18):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError(
            "plotly is required for interactive 3D HTML export. Install with `pip install plotly`."
        ) from e

    nx, ny, nz = gt.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]

    def _trace(arr, name, colorscale):
        lo = float(np.percentile(arr, 5))
        hi = float(np.percentile(arr, 99))
        if hi <= lo:
            hi = lo + 1e-8
        return go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=arr.flatten(),
            isomin=lo,
            isomax=hi,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            name=name,
            visible=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

    traces = [
        _trace(gt, "Ground Truth", "Hot"),
        _trace(pred, "Prediction", "Hot"),
        _trace(err, "Abs Error", "Magma"),
    ]
    traces[0].visible = True

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=1.08,
                buttons=[
                    dict(label="GT", method="update", args=[{"visible": [True, False, False]}]),
                    dict(label="Pred", method="update", args=[{"visible": [False, True, False]}]),
                    dict(label="Error", method="update", args=[{"visible": [False, False, True]}]),
                ],
            )
        ],
        margin=dict(l=0, r=0, t=45, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def make_isosurface_overlay_html(gt, pred, title: str, out_html: Path):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError(
            "plotly is required for interactive 3D HTML export. Install with `pip install plotly`."
        ) from e

    nx, ny, nz = gt.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]

    level_gt = float(np.percentile(gt, 97))
    level_pred = float(np.percentile(pred, 97))

    t_gt = go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=gt.flatten(),
        isomin=level_gt, isomax=level_gt,
        surface_count=1, opacity=0.45, colorscale="Reds", name="GT Isosurface",
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    t_pred = go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=pred.flatten(),
        isomin=level_pred, isomax=level_pred,
        surface_count=1, opacity=0.45, colorscale="Blues", name="Pred Isosurface",
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    fig = go.Figure(data=[t_gt, t_pred])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser(description="Create interactive 3D visualizations for dynamic model predictions.")
    parser.add_argument("--run", required=True, help="Run ID, model group under runs/dynamic, or direct .pth path")
    parser.add_argument("--runs-root", default="./runs/dynamic")
    parser.add_argument("--test-base-path", default="./data/testset")
    parser.add_argument("--experiment-name", default=None, help="exact experiment folder name")
    parser.add_argument("--experiment-idx", type=int, default=0, help="used if --experiment-name is not set")
    parser.add_argument("--times", default="1,5,10,20", help="comma-separated seconds, e.g. '1,5,10,20'")
    parser.add_argument("--seconds-per-step", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--model-class", default=None)
    parser.add_argument("--out-dir", default="./plots/volumetric_eval")

    args = parser.parse_args()

    run_id, ckpt_path, run_dir = resolve_dynamic_run(args.run, Path(args.runs_root))
    run_cfg = load_run_config(run_dir)

    model_class = args.model_class or run_cfg.get("model_class")
    channels = args.channels if args.channels is not None else run_cfg.get("channels")
    if model_class is None:
        raise ValueError("Could not determine model class. Pass --model-class or ensure config.json has model_class.")
    if channels is None:
        raise ValueError("Could not determine channels. Pass --channels or ensure config.json has channels.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_class, int(channels), ckpt_path, device)

    test_base = Path(args.test_base_path)
    exps = list_experiments(test_base)
    if not exps:
        raise RuntimeError(f"No experiment_* folders found in '{test_base}'")

    if args.experiment_name:
        exp = test_base / args.experiment_name
        if not exp.exists():
            raise FileNotFoundError(f"Experiment '{args.experiment_name}' not found in '{test_base}'")
    else:
        if args.experiment_idx < 0 or args.experiment_idx >= len(exps):
            raise IndexError(f"experiment-idx {args.experiment_idx} out of range [0, {len(exps)-1}]")
        exp = exps[args.experiment_idx]

    npz_path = exp / "normalized_heat_equation_solution.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    temp = np.load(npz_path)["temperature"]  # [nt, nx, ny, nz]
    nt = temp.shape[0]

    min_temp, max_temp = load_normalization(test_base)

    input0 = torch.tensor(temp[0], dtype=TRAIN_DTYPE, device=device).unsqueeze(0).unsqueeze(0)

    out_dir = Path(args.out_dir) / run_id / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_times = parse_times(args.times)
    exported = []

    with torch.no_grad():
        for t_seconds in wanted_times:
            t_idx = int(round(t_seconds / args.seconds_per_step))
            if t_idx <= 0 or t_idx >= nt:
                print(f"[skip] t={t_seconds}s -> idx={t_idx} out of range (nt={nt})")
                continue

            t_tensor = torch.tensor([[float(t_seconds)]], dtype=TRAIN_DTYPE, device=device)
            pred = model(input0, t_tensor).detach().cpu().numpy()[0, 0]
            gt = temp[t_idx]
            err = np.abs(pred - gt)

            pred_den = denormalize(pred, min_temp, max_temp)
            gt_den = denormalize(gt, min_temp, max_temp)
            err_den = np.abs(pred_den - gt_den)

            base = f"t{t_seconds:.2f}s_idx{t_idx:04d}".replace(".", "p")
            np.savez_compressed(
                out_dir / f"{base}.npz",
                gt=gt_den,
                pred=pred_den,
                abs_err=err_den,
                t_seconds=float(t_seconds),
                t_idx=int(t_idx),
            )

            make_volume_toggle_html(
                gt_den,
                pred_den,
                err_den,
                title=f"{run_id} | {exp.name} | t={t_seconds:.2f}s",
                out_html=out_dir / f"{base}_volume_toggle.html",
            )
            make_isosurface_overlay_html(
                gt_den,
                pred_den,
                title=f"{run_id} | {exp.name} | t={t_seconds:.2f}s | GT vs Pred Isosurfaces",
                out_html=out_dir / f"{base}_isosurface_overlay.html",
            )
            exported.append({"t_seconds": float(t_seconds), "t_idx": int(t_idx), "base": base})
            print(f"[ok] exported t={t_seconds:.2f}s -> {base}")

    with (out_dir / "export_manifest.json").open("w") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "experiment": exp.name,
                "dtype": str(TRAIN_DTYPE),
                "model_class": model_class,
                "channels": int(channels),
                "seconds_per_step": args.seconds_per_step,
                "exports": exported,
            },
            f,
            indent=2,
        )

    print(f"[done] outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
