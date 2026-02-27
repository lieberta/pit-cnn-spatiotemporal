import argparse

import torch
import zarr


def main():
    parser = argparse.ArgumentParser(description="Inspect a heat_equation_solution.zarr store.")
    parser.add_argument(
        "--zarr-path",
        default="data/new_detailed_heat_sim_f64/experiment_15_20260223_140219/heat_equation_solution.zarr",
        help="Path to heat_equation_solution.zarr",
    )
    parser.add_argument("--t-index", type=int, default=0, help="Time index to load (default: 0)")
    args = parser.parse_args()

    root = zarr.open_group(args.zarr_path, mode="r")
    temp = root["temperature"]  # shape: (nt, nx, ny, nz)

    print(f"temperature shape = {temp.shape}, dtype = {temp.dtype}")
    print(f"available arrays = {list(root.array_keys())}")

    t_idx = max(0, min(args.t_index, temp.shape[0] - 1))
    t_tensor = torch.tensor(temp[t_idx, :, :, :], dtype=torch.float32)

    print(f"\nLoaded t={t_idx} -> tensor shape = {tuple(t_tensor.shape)}, dtype = {t_tensor.dtype}")
    print(f"min={t_tensor.min().item():.4f}, max={t_tensor.max().item():.4f}, mean={t_tensor.mean().item():.4f}")

    # Small peek into the tensor values (z=0 plane, top-left 5x5)
    print("\nPreview t[z=0, x:0..4, y:0..4]:")
    print(t_tensor[:5, :5, 0])


if __name__ == "__main__":
    path_root = "data/new_detailed_heat_sim_f64/experiment_15_20260223_140219/heat_equation_solution.zarr"
    tensor = zarr.open_group(path_root, mode='r')["temperature"]
    print(f"the shape is {tensor.shape}")


    tensor3d = tensor[0]

    firecount0 =0
    firecount1=0

    for i in range(tensor3d.shape[0]):
        for j in range(tensor3d.shape[1]):
            if tensor3d[i,j,2] > 21.0:
                firecount0 += 1
            if tensor3d[i,j,3] > 22.0:
                firecount1 += 1 
            f0 = tensor3d[i, j].mean().item()

    print(f"firecount0 = {firecount0}, firecount1 = {firecount1}")
