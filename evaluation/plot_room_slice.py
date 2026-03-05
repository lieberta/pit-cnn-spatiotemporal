import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import zarr
except ImportError:
    zarr = None


def create_subfolder(folder_path: Path, subfolder_name: str):
    subfolder_path = folder_path / subfolder_name
    subfolder_path.mkdir(parents=True, exist_ok=True)
    return subfolder_path


def find_fire_slice_y(temp_t0):
    found_fire = False
    y_fire = 1
    for x_coord in range(temp_t0.shape[0]):
        for y_coord in range(temp_t0.shape[1]):
            if temp_t0[x_coord, y_coord, 0] > 21:
                y_fire = y_coord
                found_fire = True
                break
        if found_fire:
            break
    return y_fire


def render_experiment(experiment_folder: Path, npz_name: str, step_every: int, vmax_clip: float, output_subdir: str):
    store_path = experiment_folder / npz_name
    if store_path.suffix == ".zarr":
        if not store_path.exists():
            print(f"[skip] missing file: {store_path}")
            return
        if zarr is None:
            raise ImportError("zarr is required to read .zarr stores. Install with: pip install zarr")
        root = zarr.open_group(str(store_path), mode="r")
        temperature = np.asarray(root["temperature"])
        time_axis = np.asarray(root["time"]) if "time" in root else np.arange(temperature.shape[0], dtype=np.float64)
    else:
        if not store_path.exists():
            print(f"[skip] missing file: {store_path}")
            return
        data = np.load(store_path)
        time_axis = data["time"] if "time" in data.files else np.arange(data["temperature"].shape[0], dtype=np.float64)
        temperature = data["temperature"]

    global_min_temp = float(np.min(temperature))
    global_max_temp = float(np.max(temperature)) if vmax_clip <= 0 else min(float(np.max(temperature)), vmax_clip)

    y_fire = find_fire_slice_y(temperature[0])
    plots_folder = create_subfolder(experiment_folder, output_subdir)

    for timestep in range(temperature.shape[0]):
        if timestep % max(1, step_every) != 0:
            continue

        plt.figure()
        plt.imshow(
            temperature[timestep, :, y_fire, :].T,
            cmap="hot",
            extent=(0, temperature.shape[1], 0, temperature.shape[3]),
            origin="lower",
            vmin=global_min_temp,
            vmax=global_max_temp,
        )
        plt.colorbar(label="Temperature")
        t_val = time_axis[timestep] if timestep < len(time_axis) else float(timestep)
        plt.title(f"{experiment_folder.name} | timestep {timestep} | t={t_val:.3f}")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.savefig(plots_folder / f"timestep_{timestep:05d}.png")
        plt.close()

    print(f"[done] plots saved in {plots_folder}")


def main():
    parser = argparse.ArgumentParser(description="Plot x-z room slices for one or all experiments.")
    parser.add_argument("--base-path", default="./data/testdummy")
    parser.add_argument("--experiment", default="experiment_15_20260223_140219", help="optional exact experiment folder name")
    parser.add_argument("--normalized", action="store_true", help="use normalized store instead of raw")
    parser.add_argument("--step-every", type=int, default=10, help="render every n-th stored timestep")
    parser.add_argument("--vmax-clip", type=float, default=500.0, help="<=0 disables clip")
    parser.add_argument("--output-subdir", default="plots", help="subfolder inside the experiment folder")
    args = parser.parse_args()

    base = Path(args.base_path)
    npz_name = "normalized_heat_equation_solution.zarr" if args.normalized else "heat_equation_solution.zarr"

    if args.experiment:
        experiment_dirs = [base / args.experiment]
    else:
        experiment_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("experiment")])

    if not experiment_dirs:
        raise RuntimeError(f"No experiment_* folders found in {base}")

    for exp in experiment_dirs:
        if not exp.exists():
            print(f"[skip] experiment not found: {exp}")
            continue
        render_experiment(
            exp,
            npz_name=npz_name,
            step_every=args.step_every,
            vmax_clip=args.vmax_clip,
            output_subdir=args.output_subdir,
        )


if __name__ == "__main__":
    main()
