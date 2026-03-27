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


def find_fire_slice_y(source_mask):
    """
    Select a y-index that intersects at least one heat source.

    The new dataset stores source locations explicitly in source_mask, so the
    visualization no longer needs to infer source placement from T0 heuristics.
    This is more robust because T0 and heat sources are intentionally stored as
    separate fields.
    """

    source_voxels = np.argwhere(source_mask > 0.0)
    if source_voxels.size == 0:
        # Fall back to the first interior slice if no source voxels are present.
        return 1

    # source_mask uses the simulator axis order (Nx, Ny, Nz), so index 1 is the
    # y-coordinate of the source voxels.
    y_values = source_voxels[:, 1]
    unique_y, counts = np.unique(y_values, return_counts=True)
    return int(unique_y[np.argmax(counts)])


def render_experiment(
    experiment_folder: Path,
    npz_name: str,
    step_every: int,
    vmax_clip: float,
    output_subdir: str,
    vmin_clip: float = None,
    subtract_ambient: bool = False,
):
    store_path = experiment_folder / npz_name
    if store_path.suffix == ".zarr":
        if not store_path.exists():
            print(f"[skip] missing file: {store_path}")
            return
        if zarr is None:
            raise ImportError("zarr is required to read .zarr stores. Install with: pip install zarr")
        root = zarr.open_group(str(store_path), mode="r")
        temperature = np.asarray(root["temperature"])
        source_mask = np.asarray(root["source_mask"]) if "source_mask" in root else None
        ambient_temp = float(np.asarray(root["T0"]).min()) if "T0" in root else 20.0
        time_axis = np.asarray(root["time"]) if "time" in root else np.arange(temperature.shape[0], dtype=np.float64)
    else:
        if not store_path.exists():
            print(f"[skip] missing file: {store_path}")
            return
        data = np.load(store_path)
        time_axis = data["time"] if "time" in data.files else np.arange(data["temperature"].shape[0], dtype=np.float64)
        temperature = data["temperature"]
        source_mask = data["source_mask"] if "source_mask" in data.files else None
        ambient_temp = float(np.min(data["T0"])) if "T0" in data.files else 20.0

    if subtract_ambient:
        # Plot temperature rise above ambient instead of absolute temperature.
        # This makes weak spatial variations around 20 degC much easier to see.
        temperature = temperature - ambient_temp

    data_min_temp = float(np.min(temperature))
    data_max_temp = float(np.max(temperature))
    global_min_temp = data_min_temp if vmin_clip is None else max(data_min_temp, float(vmin_clip))
    global_max_temp = data_max_temp if vmax_clip <= 0 else min(data_max_temp, vmax_clip)
    if global_max_temp <= global_min_temp:
        global_max_temp = global_min_temp + 1e-6

    if source_mask is not None:
        y_fire = find_fire_slice_y(source_mask)
    else:
        # Backward-compatible fallback for older stores that do not contain
        # source_mask. This keeps the script usable across legacy datasets.
        y_fire = 1

    plots_folder = create_subfolder(experiment_folder, output_subdir)
    cmap = plt.get_cmap("hot").copy()
    cmap.set_under("blue")

    for timestep in range(temperature.shape[0]):
        if timestep % max(1, step_every) != 0:
            continue

        plt.figure()
        plt.imshow(
            temperature[timestep, :, y_fire, :].T,
            cmap=cmap,
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
    parser.add_argument("--vmin-clip", type=float, default=0.0, help="fixed lower color limit; values below are shown in blue")
    parser.add_argument("--vmax-clip", type=float, default=500.0, help="<=0 disables clip")
    parser.add_argument("--output-subdir", default="plots", help="subfolder inside the experiment folder")
    parser.add_argument(
        "--subtract-ambient",
        action="store_true",
        help="plot temperature rise above ambient instead of absolute temperature",
    )
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
            vmin_clip=args.vmin_clip,
            subtract_ambient=args.subtract_ambient,
        )


if __name__ == "__main__":
    main()
