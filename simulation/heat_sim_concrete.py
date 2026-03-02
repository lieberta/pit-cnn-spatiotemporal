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


class Laplacian3D(nn.Module):
    def __init__(self, dx, dy, dz, device):
        super(Laplacian3D, self).__init__()
        central_value = -2.0 * ((1.0 / (dx ** 2)) + (1.0 / (dy ** 2)) + (1.0 / (dz ** 2)))
        kernel = torch.zeros((3, 3, 3), dtype=torch.float64)
        kernel[1, 1, 1] = central_value
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0 / (dz ** 2)
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0 / (dy ** 2)
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0 / (dx ** 2)
        self.register_buffer("kernel", kernel.to(device).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return F.conv3d(x, self.kernel, padding=1, groups=1)


class HeatSimulationConcrete:
    """
    Concrete heat-conduction simulation with volumetric heat source:
        dT/dt = alpha * laplace(T) + Q / (rho * cp)
    """

    def __init__(
        self,
        num_fires,
        # Geometry / grid
        Lx=6.3,
        Ly=3.1,
        Lz=1.5,
        Nx=64,
        Ny=32,
        Nz=16,
        # Time
        T=600.0,      # 10 minutes
        Nt=6000,      # dt = 0.1 s
        # Material (concrete-like)
        k=1.7,        # W/(m*K)
        rho=2300.0,   # kg/m^3
        cp=880.0,     # J/(kg*K)
        # Source / temperature
        ambient_temp=20.0,
        ignition_temp=600.0,
        source_power_density=3.0e5,  # W/m^3 in fireplace cells
        device="cpu",
    ):
        self.num_fires = num_fires
        self.device = device

        self.k = float(k)
        self.rho = float(rho)
        self.cp = float(cp)
        self.alpha = torch.tensor(self.k / (self.rho * self.cp), dtype=torch.float64, device=device)

        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.T, self.Nt = float(T), int(Nt)
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.dz = self.Lz / (self.Nz - 1)
        self.dt = self.T / self.Nt

        self.ambient_temp = float(ambient_temp)
        self.ignition_temp = float(ignition_temp)
        self.source_power_density = float(source_power_density)

        self.u = torch.full((self.Nt + 1, self.Nx, self.Ny, self.Nz), self.ambient_temp, dtype=torch.float64, device=device)
        self.laplacian3d = Laplacian3D(self.dx, self.dy, self.dz, device)

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
        return top_left_x, top_left_y, width, height

    def create_fireplace_experiments(self, num_fires):
        return [self.place_fireplace(self.Nx, self.Ny) for _ in range(num_fires)]

    def setup_initial_conditions(self):
        # Ambient initial field, with hotter ignition region on interior z layers only.
        self.u[0, :, :, :] = self.ambient_temp
        for x_start, y_start, x_size, y_size in self.fireplaces:
            self.u[0, x_start:x_start + x_size, y_start:y_start + y_size, 1:3] = self.ignition_temp

    def setup_source_term(self):
        # Q [W/m^3] field; later converted via Q/(rho*cp) to K/s in update equation.
        self.source_q = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float64, device=self.device)
        for x_start, y_start, x_size, y_size in self.fireplaces:
            self.source_q[x_start:x_start + x_size, y_start:y_start + y_size, 1:3] = self.source_power_density

    def run_simulation(self):
        self.tic = time.perf_counter()
        source_term_ks = self.source_q / (self.rho * self.cp)  # K/s

        for n in range(self.Nt):
            laplacian = self.laplacian3d(self.u[n:n + 1])
            self.u[n + 1] = self.u[n] + self.alpha * self.dt * laplacian + self.dt * source_term_ks

            # Dirichlet boundary: fixed ambient temperature at all outer faces.
            self.u[n + 1, 0, :, :] = self.ambient_temp
            self.u[n + 1, self.Nx - 1, :, :] = self.ambient_temp
            self.u[n + 1, :, 0, :] = self.ambient_temp
            self.u[n + 1, :, self.Ny - 1, :] = self.ambient_temp
            self.u[n + 1, :, :, 0] = self.ambient_temp
            self.u[n + 1, :, :, self.Nz - 1] = self.ambient_temp

        self.tac = time.perf_counter()
        self.min_temp = float(torch.min(self.u).item())
        self.max_temp = float(torch.max(self.u).item())

    def save_results(self, out_root="./data/concrete"):
        # Save every 10th step by default (keeps time resolution at 1.0 s for dt=0.1 s).
        temperature_np = self.u[::10].cpu().numpy()
        os.makedirs(out_root, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{out_root}/experiment_{self.num_fires}_{current_time}"
        os.makedirs(experiment_folder, exist_ok=True)

        np.savez(
            f"{experiment_folder}/heat_equation_solution.npz",
            time=np.linspace(0.0, self.T, self.Nt + 1)[::10],
            x=np.linspace(0.0, self.Lx, self.Nx),
            y=np.linspace(0.0, self.Ly, self.Ny),
            z=np.linspace(0.0, self.Lz, self.Nz),
            temperature=temperature_np,
        )

        all_fireplaces_info = ""
        for i, (x_start, y_start, x_size, y_size) in enumerate(self.fireplaces):
            all_fireplaces_info += (
                f"Fireplace {i + 1}: x={x_start}, y={y_start}, width={x_size}, depth={y_size}\n"
            )

        simulation_info = (
            f"\nSimulated Time={self.T}s, dt={self.dt}s, Nt={self.Nt}"
            f"\nLx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}"
            f"\nConcrete: k={self.k} W/(m*K), rho={self.rho} kg/m^3, cp={self.cp} J/(kg*K), alpha={self.k/(self.rho*self.cp)} m^2/s"
            f"\nSource power density Q={self.source_power_density} W/m^3"
            f"\nMin Temp={self.min_temp}, Max Temp={self.max_temp}"
            f"\nProcessing time={self.tac - self.tic}s"
        )

        with open(f"{experiment_folder}/fireplace_simulation_results.txt", "w") as file:
            file.write(all_fireplaces_info + simulation_info)


def run_experiment(num_fires, device):
    simulation = HeatSimulationConcrete(num_fires=num_fires, device=device)
    simulation.run_simulation()
    simulation.save_results()


def load_runtime_config(config_path):
    spec = importlib.util.spec_from_file_location("heat_sim_concrete_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    num_fires_list = list(getattr(module, "NUM_FIRES_LIST", [10, 20, 30]))
    experiments_per_fire_count = int(getattr(module, "EXPERIMENTS_PER_FIRE_COUNT", 1))
    data_root = str(getattr(module, "DATA_ROOT", "./data"))
    dataset_name = str(getattr(module, "DATASET_NAME", "concrete"))
    total_time = float(getattr(module, "T", 600.0))
    dt = float(getattr(module, "DT", 0.1))
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
        "device_name": device_name,
    }


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run concrete heat simulations using a python config file.")
    parser.add_argument("--config", default="configs/heat_sim_concrete_config.py", help="Path to simulation config file.")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config)
    device = resolve_device(cfg["device_name"])
    os.makedirs(cfg["out_root"], exist_ok=True)

    print(f"Using config: {args.config}")
    print(f"Dataset: {cfg['dataset_name']} -> {cfg['out_root']}")
    print(f"T={cfg['T']}s, dt={cfg['dt']}s, Nt={cfg['Nt']} steps")

    for num_fires in cfg["num_fires_list"]:
        print(f"Number of fires: {num_fires}")
        for i in range(cfg["experiments_per_fire_count"]):
            print(f"Calculating experiment {i + 1}/{cfg['experiments_per_fire_count']}...")
            simulation = HeatSimulationConcrete(
                num_fires=num_fires,
                T=cfg["T"],
                Nt=cfg["Nt"],
                device=device,
            )
            simulation.run_simulation()
            simulation.save_results(out_root=cfg["out_root"])
