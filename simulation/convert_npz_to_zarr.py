import argparse
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import zarr
except ImportError:
    zarr = None


def convert_experiment(exp_dir: Path, delete_npz: bool, chunk_t: int) -> Tuple[bool, str, str]:
    npz_path = exp_dir / "heat_equation_solution.npz"
    zarr_path = exp_dir / "heat_equation_solution.zarr"
    old_norm_npz = exp_dir / "normalized_heat_equation_solution.npz"

    if zarr_path.exists():
        if delete_npz and npz_path.exists():
            npz_path.unlink()
        if delete_npz and old_norm_npz.exists():
            old_norm_npz.unlink()
        return True, "already_converted", exp_dir.name

    if not npz_path.exists():
        return False, "missing_npz", exp_dir.name

    with np.load(npz_path) as data:
        temp = data["temperature"]
        time = data["time"] if "time" in data.files else None
        x = data["x"] if "x" in data.files else None
        y = data["y"] if "y" in data.files else None
        z = data["z"] if "z" in data.files else None

    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_dataset(
        "temperature",
        data=temp,
        chunks=(int(chunk_t), temp.shape[1], temp.shape[2], temp.shape[3]),
        overwrite=True,
    )
    if time is not None:
        root.create_dataset("time", data=time, overwrite=True)
    if x is not None:
        root.create_dataset("x", data=x, overwrite=True)
    if y is not None:
        root.create_dataset("y", data=y, overwrite=True)
    if z is not None:
        root.create_dataset("z", data=z, overwrite=True)

    if delete_npz:
        npz_path.unlink()
        if old_norm_npz.exists():
            old_norm_npz.unlink()

    return True, "converted", exp_dir.name


def main():
    if zarr is None:
        raise ImportError("zarr is required. Install with: pip install zarr")

    parser = argparse.ArgumentParser(description="Convert experiment NPZ files to Zarr and optionally delete NPZ.")
    parser.add_argument("--base-path", default="./data/new_detailed_heat_sim_f64")
    parser.add_argument("--delete-npz", action="store_true", help="delete original .npz files after successful conversion")
    parser.add_argument("--chunk-t", type=int, default=1, help="time chunk size for Zarr temperature array")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel conversion workers")
    args = parser.parse_args()

    base = Path(args.base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    exps = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("experiment")])
    converted = 0
    already = 0
    missing = 0

    if args.workers <= 1:
        for exp in exps:
            _, status, name = convert_experiment(exp, delete_npz=args.delete_npz, chunk_t=args.chunk_t)
            if status == "converted":
                converted += 1
            elif status == "already_converted":
                already += 1
            else:
                missing += 1
            print(f"{name}: {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_experiment, exp, args.delete_npz, args.chunk_t): exp.name
                for exp in exps
            }
            for fut in as_completed(futures):
                _, status, name = fut.result()
                if status == "converted":
                    converted += 1
                elif status == "already_converted":
                    already += 1
                else:
                    missing += 1
                print(f"{name}: {status}", flush=True)

    print(f"base_path={base}")
    print(f"experiments={len(exps)} converted={converted} already_converted={already} missing_npz={missing}")
    print(f"delete_npz={args.delete_npz}")


if __name__ == "__main__":
    main()
