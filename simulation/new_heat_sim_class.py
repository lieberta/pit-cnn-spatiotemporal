import os
import random
import time
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import zarr
except ImportError:
    zarr = None

'''
This file contains the HeatSimulation class which simulates the heat equation with multiple fireplaces and saves the results.
f64
''' 


class Laplacian3D(nn.Module):
    def __init__(self, dx, dy, dz, device, dtype=torch.float64):
        super(Laplacian3D, self).__init__()
        central_value = -2.0 * ((1.0 / (dx ** 2)) + (1.0 / (dy ** 2)) + (1.0 / (dz ** 2)))

        kernel = torch.zeros((3, 3, 3), dtype=dtype)
        kernel[1, 1, 1] = central_value
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0 / (dz ** 2)
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0 / (dy ** 2)
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0 / (dx ** 2)

        kernel = kernel.to(device).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv3d(x, self.kernel, padding=1, groups=1)


class HeatSimulation:
    def __init__(
        self,
        num_fires,
        alpha=0.0257,
        Lx=6.3,
        Ly=3.1,
        Lz=1.5,
        Nx=64,
        Ny=32,
        Nz=16,
        T=10.0,
        Nt=10000,
        device="cpu",
        work_dtype=torch.float64,
        save_dtype=np.float64,
        source_intensity=100000.0,
        ambient_temp=20.0,
        initial_fire_temp=1100.0,
    ):
        self.num_fires = num_fires
        self.device = device
        self.work_dtype = work_dtype
        self.save_dtype = save_dtype

        self.alpha = torch.tensor(alpha, dtype=work_dtype, device=device)
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.T, self.Nt = T, Nt
        self.dx, self.dy, self.dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)
        self.dt = T / Nt

        self.ambient_temp = float(ambient_temp)
        self.initial_fire_temp = float(initial_fire_temp)
        self.source_intensity = float(source_intensity)

        self.u = torch.zeros((Nt + 1, Nx, Ny, Nz), dtype=work_dtype, device=device)
        self.laplacian3D = Laplacian3D(self.dx, self.dy, self.dz, device, dtype=work_dtype)

        self.fireplaces = self.create_fireplace_experiments(self.num_fires)
        self.setup_initial_conditions()
        self.setup_source_term()

    @staticmethod
    def place_fireplace(max_x, max_y):
        width = random.randint(1, 4)
        height = random.randint(1, 4)
        # Keep source patches away from x/y boundaries (Dirichlet walls).
        top_left_x = random.randint(1, max_x - width - 1)
        top_left_y = random.randint(1, max_y - height - 1)
        return (top_left_x, top_left_y, width, height)

    def create_fireplace_experiments(self, num_fires):
        max_x, max_y = self.Nx, self.Ny
        return [self.place_fireplace(max_x, max_y) for _ in range(num_fires)]

    def setup_initial_conditions(self):
        self.u[0, :, :, :] = self.ambient_temp
        for x_start, y_start, x_size, y_size in self.fireplaces:
            # Source occupies interior z layers only (avoid z=0 and z=Nz-1 boundaries).
            self.u[0, x_start : x_start + x_size, y_start : y_start + y_size, 1:3] = self.initial_fire_temp

    def setup_source_term(self):
        self.source_term = torch.zeros_like(self.u[0], dtype=self.work_dtype, device=self.device)
        for x_start, y_start, x_size, y_size in self.fireplaces:
            # Match initial hot zone placement: interior z layers only.
            self.source_term[x_start : x_start + x_size, y_start : y_start + y_size, 1:3] = self.source_intensity

    def run_simulation(self):
        self.tic = time.perf_counter()
        for n in range(self.Nt):
            laplacian = self.laplacian3D(self.u[n : n + 1])
            self.u[n + 1] = self.u[n] + self.alpha * self.dt * laplacian + self.dt * self.source_term

            self.u[n + 1, 0, :, :] = self.ambient_temp
            self.u[n + 1, self.Nx - 1, :, :] = self.ambient_temp
            self.u[n + 1, :, 0, :] = self.ambient_temp
            self.u[n + 1, :, self.Ny - 1, :] = self.ambient_temp
            self.u[n + 1, :, :, 0] = self.ambient_temp
            self.u[n + 1, :, :, self.Nz - 1] = self.ambient_temp

        self.tac = time.perf_counter()

        self.min_temp = float(torch.min(self.u).item())
        self.max_temp = float(torch.max(self.u).item())

    def _build_summary_text(self, saved_steps):
        lines = []
        for i, (x_start, y_start, x_size, y_size) in enumerate(self.fireplaces):
            lines.append(
                f"Fireplace {i + 1}: Location: x={x_start}, y={y_start}, "
                f"Dimensions: width={x_size}, depth={y_size}"
            )
        lines.append("")
        lines.append(f"Simulated Time={self.T}s, dt={self.dt}s, amount of simulated timesteps={self.Nt}")
        lines.append(f"Alpha={self.alpha.item()}, amount of saved timesteps={saved_steps}")
        lines.append(f"Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}")
        lines.append(f"min Temperature={self.min_temp}, max Temperature={self.max_temp}")
        lines.append(f"Processing time={self.tac - self.tic}s")
        return "\n".join(lines) + "\n"

    def save_results(
        self,
        out_root="./data/new_detailed_heat_sim_f64",
        include_axes=True,
        chunk_t=1,
    ):
        if zarr is None:
            raise ImportError("zarr is required to save simulation output. Install with: pip install zarr")

        temperature_np = self.u.detach().cpu().numpy().astype(self.save_dtype, copy=False)
        saved_steps = int(temperature_np.shape[0])

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = os.path.join(out_root, f"experiment_{self.num_fires}_{current_time}")
        os.makedirs(experiment_folder, exist_ok=True)

        zarr_path = os.path.join(experiment_folder, "heat_equation_solution.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_dataset(
            "temperature",
            data=temperature_np,
            chunks=(int(chunk_t), self.Nx, self.Ny, self.Nz),
            overwrite=True,
        )
        if include_axes:
            root.create_dataset(
                "time",
                data=np.linspace(0.0, self.T, self.Nt + 1, dtype=self.save_dtype),
                overwrite=True,
            )
            root.create_dataset("x", data=np.linspace(0.0, self.Lx, self.Nx, dtype=self.save_dtype), overwrite=True)
            root.create_dataset("y", data=np.linspace(0.0, self.Ly, self.Ny, dtype=self.save_dtype), overwrite=True)
            root.create_dataset("z", data=np.linspace(0.0, self.Lz, self.Nz, dtype=self.save_dtype), overwrite=True)

        info_text = self._build_summary_text(saved_steps=saved_steps)
        with open(os.path.join(experiment_folder, "fireplace_simulation_results.txt"), "w") as file:
            file.write(info_text)


def run_experiment(num_fires, device):
    simulation = HeatSimulation(num_fires, device=device)
    simulation.run_simulation()
    simulation.save_results()


def load_runtime_config(config_path):
    spec = importlib.util.spec_from_file_location("new_sim_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    num_fires_list = list(getattr(module, "NUM_FIRES_LIST", [15, 20, 25]))
    experiments_per_fire_count = int(getattr(module, "EXPERIMENTS_PER_FIRE_COUNT", 10))
    data_root = str(getattr(module, "DATA_ROOT", "./data"))
    dataset_name = str(getattr(module, "DATASET_NAME", "new_detailed_heat_sim_f64"))
    total_time = float(getattr(module, "T", 10.0))
    dt = float(getattr(module, "DT", 0.001))
    include_axes = bool(getattr(module, "INCLUDE_AXES", True))
    chunk_t = int(getattr(module, "CHUNK_T", 1))
    device_name = str(getattr(module, "DEVICE", "auto"))

    if not num_fires_list:
        raise ValueError("NUM_FIRES_LIST must not be empty.")
    if any(int(n) <= 0 for n in num_fires_list):
        raise ValueError("All entries in NUM_FIRES_LIST must be > 0.")
    if experiments_per_fire_count <= 0:
        raise ValueError("EXPERIMENTS_PER_FIRE_COUNT must be > 0.")
    if total_time <= 0.0:
        raise ValueError("T must be > 0.")
    if dt <= 0.0:
        raise ValueError("DT must be > 0.")

    nt_float = total_time / dt
    nt = int(round(nt_float))
    if nt <= 0:
        raise ValueError(f"Invalid Nt computed from T/DT: T={total_time}, DT={dt}, Nt={nt}")
    if not np.isclose(nt_float, nt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"T/DT must be an integer (within tolerance). Got T={total_time}, DT={dt}, T/DT={nt_float}"
        )

    return {
        "num_fires_list": [int(n) for n in num_fires_list],
        "experiments_per_fire_count": experiments_per_fire_count,
        "out_root": os.path.join(data_root, dataset_name),
        "dataset_name": dataset_name,
        "T": total_time,
        "dt": dt,
        "Nt": nt,
        "num_time_points": nt + 1,
        "include_axes": include_axes,
        "chunk_t": chunk_t,
        "device_name": device_name,
    }


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heat simulations using a python config file.")
    parser.add_argument("--config", default="configs/new_sim_config.py", help="Path to simulation config file.")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config)
    device = resolve_device(cfg["device_name"])
    os.makedirs(cfg["out_root"], exist_ok=True)

    print(f"Using config: {args.config}")
    print(f"Dataset: {cfg['dataset_name']} -> {cfg['out_root']}")
    print(
        f"T={cfg['T']}s, dt={cfg['dt']}s, Nt={cfg['Nt']} steps, "
        f"saved time points (including t=0): {cfg['num_time_points']}"
    )

    for num_fires in cfg["num_fires_list"]:
        print(f"Number of fires: {num_fires}")
        for i in range(cfg["experiments_per_fire_count"]):
            print(f"Calculating experiment {i + 1}/{cfg['experiments_per_fire_count']}...")
            simulation = HeatSimulation(
                num_fires=num_fires,
                T=cfg["T"],
                Nt=cfg["Nt"],
                device=device,
            )
            simulation.run_simulation()
            simulation.save_results(
                out_root=cfg["out_root"],
                include_axes=cfg["include_axes"],
                chunk_t=cfg["chunk_t"],
            )
