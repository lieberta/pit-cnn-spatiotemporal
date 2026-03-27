import os
import random
import time
import argparse
import importlib.util
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import zarr
except ImportError:
    zarr = None


class Laplacian3D(nn.Module):
    """Finite-difference 3D Laplacian implemented via a fixed convolution kernel."""

    def __init__(self, dx, dy, dz, ambient_temp, device):
        super(Laplacian3D, self).__init__()
        # Store the Dirichlet boundary value used for explicit constant padding.
        self.ambient_temp = float(ambient_temp)
        # Build the standard 7-point stencil for the anisotropic 3D Laplacian.
        central_value = -2.0 * ((1.0 / (dx ** 2)) + (1.0 / (dy ** 2)) + (1.0 / (dz ** 2)))
        kernel = torch.zeros((3, 3, 3), dtype=torch.float64)
        kernel[1, 1, 1] = central_value
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0 / (dz ** 2)
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0 / (dy ** 2)
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0 / (dx ** 2)
        self.register_buffer("kernel", kernel.to(device).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # Pad explicitly with the ambient temperature so that the convolution
        # matches the intended Dirichlet boundary condition at the outer faces.
        x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant", value=self.ambient_temp)
        return F.conv3d(x_pad, self.kernel, padding=0, groups=1)


class HeatSimulation:
    """
    Generic 3D heat-conduction simulation with volumetric heat sources.

    The simulated PDE is:
        dT/dt = alpha * laplace(T) + Q / (rho * cp)

    The dataset is stored with a clear separation between:
    - T0: initial temperature field
    - source_mask: binary map showing where heat sources are located
    - source_field: volumetric source intensity field in W/m^3
    - temperature: full temperature trajectory over time
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
        dt=0.1,
        # Material parameters
        k=1.7,        # W/(m*K)
        rho=2300.0,   # kg/m^3
        cp=880.0,     # J/(kg*K)
        # Source / temperature
        ambient_temp=20.0,
        ignition_temp=600.0,
        source_power_density=3.0e5,  # W/m^3 in fireplace cells
        t0_noise_sigma=0.5,
        t0_noise_smoothing_passes=2,
        device="cpu",
    ):
        self.num_fires = num_fires
        self.device = device

        # Material parameters are stored explicitly so that metadata can expose
        # both the primitive values and the derived diffusivity alpha.
        self.k = float(k)
        self.rho = float(rho)
        self.cp = float(cp)
        self.alpha = torch.tensor(self.k / (self.rho * self.cp), dtype=torch.float64, device=device)

        # Store the physical domain and grid resolution.
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)

        # Time is parameterized by T and dt, mirroring the earlier 2D setup.
        #
        # Important semantic distinction:
        # - T is the requested prediction horizon the model should learn up to.
        # - We simulate one extra future step beyond T so that all three physics
        #   residual variants can be evaluated on the exact same target times,
        #   including the final target at t = T.
        #
        # Concretely, if T = 600 s and dt = 0.1 s:
        # - prediction steps cover 0.1 s .. 600.0 s
        # - one additional simulation step is stored at 600.1 s
        #   solely to provide target_future for the final training sample.
        self.T = float(T)
        self.dt = float(dt)
        nt_float = self.T / self.dt
        self.num_prediction_steps = int(round(nt_float))
        if self.num_prediction_steps <= 0:
            raise ValueError(
                f"Invalid num_prediction_steps computed from T/dt: T={self.T}, dt={self.dt}, "
                f"num_prediction_steps={self.num_prediction_steps}"
            )
        if not np.isclose(nt_float, self.num_prediction_steps, rtol=0.0, atol=1e-12):
            raise ValueError(f"T/dt must be an integer (within tolerance). Got T={self.T}, dt={self.dt}, T/dt={nt_float}")
        self.num_simulation_steps = self.num_prediction_steps + 1
        self.num_stored_frames = self.num_simulation_steps + 1

        # Compute grid spacing from the domain and the number of grid points.
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.dz = self.Lz / (self.Nz - 1)

        # Store thermodynamic reference values used for initialization and forcing.
        self.ambient_temp = float(ambient_temp)
        self.ignition_temp = float(ignition_temp)
        self.source_power_density = float(source_power_density)
        self.t0_noise_sigma = float(t0_noise_sigma)
        self.t0_noise_smoothing_passes = int(t0_noise_smoothing_passes)

        # The full trajectory is stored in float64 for consistency and simplicity.
        self.u = torch.full(
            (self.num_stored_frames, self.Nx, self.Ny, self.Nz),
            self.ambient_temp,
            dtype=torch.float64,
            device=device,
        )

        # T0 is stored separately so it can be written to disk independently from
        # the full time trajectory. This makes the initial condition explicit.
        self.T0 = torch.full((self.Nx, self.Ny, self.Nz), self.ambient_temp, dtype=torch.float64, device=device)

        self.laplacian3d = Laplacian3D(self.dx, self.dy, self.dz, self.ambient_temp, device)

        # Each experiment samples a random set of source patches in x/y while
        # keeping the source away from the Dirichlet boundaries.
        self.fireplaces = self.create_fireplace_experiments(self.num_fires)
        self.setup_initial_conditions()
        self.setup_source_term()

    @staticmethod
    def place_fireplace(max_x, max_y):
        """Sample one rectangular fireplace footprint away from the boundary."""

        # Keep fireplace patches visually and physically meaningful by avoiding
        # single-cell source footprints in x/y.
        width = random.randint(2, 4)
        height = random.randint(2, 4)
        # Keep source patches away from x/y boundaries to preserve a clean
        # separation between source cells and fixed-temperature walls.
        top_left_x = random.randint(1, max_x - width - 1)
        top_left_y = random.randint(1, max_y - height - 1)
        return top_left_x, top_left_y, width, height

    def create_fireplace_experiments(self, num_fires):
        """Create the list of randomly placed fireplace footprints."""

        return [self.place_fireplace(self.Nx, self.Ny) for _ in range(num_fires)]

    def _apply_dirichlet_boundaries(self, field):
        """Clamp all outer faces of a 3D field to the ambient temperature."""

        field[0, :, :] = self.ambient_temp
        field[self.Nx - 1, :, :] = self.ambient_temp
        field[:, 0, :] = self.ambient_temp
        field[:, self.Ny - 1, :] = self.ambient_temp
        field[:, :, 0] = self.ambient_temp
        field[:, :, self.Nz - 1] = self.ambient_temp
        return field

    def _create_smoothed_temperature_noise(self):
        """
        Create a small smooth perturbation field for T0.

        A minimal realistic model is used here:
        - sample Gaussian voxel noise
        - smooth it with repeated average pooling
        - enforce zero noise at the boundaries afterward

        This keeps the initial field mostly ambient while avoiding an unrealistically
        perfectly flat temperature distribution.
        """

        if self.t0_noise_sigma <= 0.0:
            return torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float64, device=self.device)

        noise = torch.randn((1, 1, self.Nx, self.Ny, self.Nz), dtype=torch.float64, device=self.device)
        noise = noise * self.t0_noise_sigma

        # Repeated local averaging is a minimal way to turn white noise into a
        # spatially smooth perturbation field without introducing extra machinery.
        for _ in range(max(0, self.t0_noise_smoothing_passes)):
            noise = F.avg_pool3d(noise, kernel_size=3, stride=1, padding=1)

        noise = noise.squeeze(0).squeeze(0)
        self._apply_dirichlet_boundaries(noise)
        return noise

    def setup_initial_conditions(self):
        """
        Build the initial temperature field T0.

        T0 is composed of:
        - a uniform ambient baseline
        - small smooth spatial fluctuations
        - hotter ignition regions at the sampled source locations
        """

        self.T0[:, :, :] = self.ambient_temp
        self.T0 += self._create_smoothed_temperature_noise()
        self._apply_dirichlet_boundaries(self.T0)

        # Source cells are initialized at a higher ignition temperature on a
        # shallow interior slab near the floor.
        for x_start, y_start, x_size, y_size in self.fireplaces:
            self.T0[x_start:x_start + x_size, y_start:y_start + y_size, 1:3] = self.ignition_temp

        self._apply_dirichlet_boundaries(self.T0)
        self.u[0] = self.T0

    def setup_source_term(self):
        """
        Build the source representation used both for simulation and storage.

        source_mask:
            Binary occupancy map. It shows where a heat source exists.
        source_field:
            Physical volumetric source intensity in W/m^3. It is zero outside
            source cells and equal to source_power_density inside source cells.
        """

        self.source_mask = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float64, device=self.device)
        self.source_field = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float64, device=self.device)
        for x_start, y_start, x_size, y_size in self.fireplaces:
            self.source_mask[x_start:x_start + x_size, y_start:y_start + y_size, 1:3] = 1.0
            self.source_field[x_start:x_start + x_size, y_start:y_start + y_size, 1:3] = self.source_power_density

    def run_simulation(self):
        """Run the explicit Euler time integration for the 3D heat equation."""

        self.tic = time.perf_counter()

        # Convert volumetric heating from W/m^3 into a temperature-rate term in K/s.
        source_term_ks = self.source_field / (self.rho * self.cp)

        # Simulate one extra step beyond the requested horizon T so that the
        # final target at t = T also has a valid target_future = u(T + dt).
        for n in range(self.num_simulation_steps):
            # Clamp the old state before applying the Laplacian so that the
            # update uses the intended Dirichlet boundary values.
            self._apply_dirichlet_boundaries(self.u[n])

            laplacian = self.laplacian3d(self.u[n:n + 1])
            self.u[n + 1] = self.u[n] + self.alpha * self.dt * laplacian + self.dt * source_term_ks

            # Clamp the new state after the explicit Euler update as well.
            self._apply_dirichlet_boundaries(self.u[n + 1])

        self.tac = time.perf_counter()
        self.min_temp = float(torch.min(self.u).item())
        self.max_temp = float(torch.max(self.u).item())
        self.t0_min = float(torch.min(self.T0).item())
        self.t0_max = float(torch.max(self.T0).item())

    def _build_summary_text(self, saved_steps):
        """Create a compact human-readable experiment summary."""

        return (
            "Heat Simulation Results\n"
            f"saved_steps: {saved_steps}\n"
            f"grid: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}\n"
            f"dt: {self.dt}\n"
            f"prediction_horizon: {self.T}\n"
            f"num_prediction_steps: {self.num_prediction_steps}\n"
            f"num_simulation_steps: {self.num_simulation_steps}\n"
            f"num_stored_frames: {self.num_stored_frames}\n"
            f"T0 min_temp: {self.t0_min}\n"
            f"T0 max_temp: {self.t0_max}\n"
            f"min_temp: {self.min_temp}\n"
            f"max_temp: {self.max_temp}\n"
            f"processing_time: {self.tac - self.tic}\n"
        )

    def save_results(self, out_root="./data/heat_sim", include_axes=True, chunk_t=1):
        """Write one experiment to disk using a zarr-based dataset layout."""

        if zarr is None:
            raise ImportError("zarr is required to save simulation results in zarr format.")

        temperature_np = self.u.detach().cpu().numpy()
        t0_np = self.T0.detach().cpu().numpy()
        source_mask_np = self.source_mask.detach().cpu().numpy()
        source_field_np = self.source_field.detach().cpu().numpy()
        saved_steps = int(temperature_np.shape[0])
        os.makedirs(out_root, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{out_root}/experiment_{self.num_fires}_{current_time}"
        os.makedirs(experiment_folder, exist_ok=True)

        zarr_path = os.path.join(experiment_folder, "heat_equation_solution.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_dataset(
            "temperature",
            data=temperature_np,
            chunks=(int(chunk_t), self.Nx, self.Ny, self.Nz),
            overwrite=True,
        )
        # Store the initial condition separately so downstream code does not need
        # to infer it from temperature[0].
        root.create_dataset(
            "T0",
            data=t0_np,
            chunks=(self.Nx, self.Ny, self.Nz),
            overwrite=True,
        )
        # Store both the binary occupancy map and the physical source intensity field.
        root.create_dataset(
            "source_mask",
            data=source_mask_np,
            chunks=(self.Nx, self.Ny, self.Nz),
            overwrite=True,
        )
        root.create_dataset(
            "source_field",
            data=source_field_np,
            chunks=(self.Nx, self.Ny, self.Nz),
            overwrite=True,
        )
        if include_axes:
            root.create_dataset(
                "time",
                data=np.linspace(0.0, self.T + self.dt, self.num_stored_frames, dtype=np.float64),
                overwrite=True,
            )
            root.create_dataset("x", data=np.linspace(0.0, self.Lx, self.Nx, dtype=np.float64), overwrite=True)
            root.create_dataset("y", data=np.linspace(0.0, self.Ly, self.Ny, dtype=np.float64), overwrite=True)
            root.create_dataset("z", data=np.linspace(0.0, self.Lz, self.Nz, dtype=np.float64), overwrite=True)

        summary_text = self._build_summary_text(saved_steps=saved_steps)
        with open(f"{experiment_folder}/fireplace_simulation_results.txt", "w") as file:
            file.write(summary_text)

        metadata = {
            "k": float(self.k),
            "rho": float(self.rho),
            "cp": float(self.cp),
            "alpha": float(self.alpha.item()),
            "min_temp": float(self.min_temp),
            "max_temp": float(self.max_temp),
            "t_min": float(self.min_temp),
            "t_max": float(self.max_temp),
            "t0_min": float(self.t0_min),
            "t0_max": float(self.t0_max),
            "prediction_horizon": float(self.T),
            "dt": float(self.dt),
            "num_prediction_steps": int(self.num_prediction_steps),
            "num_simulation_steps": int(self.num_simulation_steps),
            "num_stored_frames": int(self.num_stored_frames),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dz": float(self.dz),
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "Nz": int(self.Nz),
            "ambient_temp": float(self.ambient_temp),
            "ignition_temp": float(self.ignition_temp),
            "source_power_density": float(self.source_power_density),
            "t0_noise_sigma": float(self.t0_noise_sigma),
            "t0_noise_smoothing_passes": int(self.t0_noise_smoothing_passes),
            "num_fires": int(self.num_fires),
            "has_source_mask": True,
            "has_source_field": True,
            "fireplaces": [
                {
                    "x_start": int(x_start),
                    "y_start": int(y_start),
                    "width": int(x_size),
                    "depth": int(y_size),
                    "z_start": 1,
                    "z_end_exclusive": 3,
                }
                for x_start, y_start, x_size, y_size in self.fireplaces
            ],
        }
        with open(os.path.join(experiment_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return experiment_folder


def write_dataset_info(out_root):
    """Aggregate global dataset statistics from all experiment metadata files."""

    experiment_dirs = []
    for name in os.listdir(out_root):
        path = os.path.join(out_root, name)
        if os.path.isdir(path) and name.startswith("experiment_"):
            experiment_dirs.append(path)

    global_min = None
    global_max = None
    global_t0_min = None
    global_t0_max = None
    dt = None
    prediction_horizon = None
    num_prediction_steps = None
    num_simulation_steps = None
    num_stored_frames = None
    alpha = None
    rho = None
    cp = None
    dx = None
    dy = None
    dz = None
    Nx = None
    Ny = None
    Nz = None
    ambient_temp = None
    source_power_density = None

    for exp_dir in sorted(experiment_dirs):
        metadata_path = os.path.join(exp_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        exp_min = float(meta["t_min"])
        exp_max = float(meta["t_max"])
        exp_t0_min = float(meta["t0_min"])
        exp_t0_max = float(meta["t0_max"])
        if global_min is None or exp_min < global_min:
            global_min = exp_min
        if global_max is None or exp_max > global_max:
            global_max = exp_max
        if global_t0_min is None or exp_t0_min < global_t0_min:
            global_t0_min = exp_t0_min
        if global_t0_max is None or exp_t0_max > global_t0_max:
            global_t0_max = exp_t0_max

        if dt is None:
            dt = float(meta["dt"])
        if prediction_horizon is None and "prediction_horizon" in meta:
            prediction_horizon = float(meta["prediction_horizon"])
        if num_prediction_steps is None and "num_prediction_steps" in meta:
            num_prediction_steps = int(meta["num_prediction_steps"])
        if num_simulation_steps is None and "num_simulation_steps" in meta:
            num_simulation_steps = int(meta["num_simulation_steps"])
        if num_stored_frames is None and "num_stored_frames" in meta:
            num_stored_frames = int(meta["num_stored_frames"])
        if alpha is None and "alpha" in meta:
            alpha = float(meta["alpha"])
        if rho is None and "rho" in meta:
            rho = float(meta["rho"])
        if cp is None and "cp" in meta:
            cp = float(meta["cp"])
        if dx is None and "dx" in meta:
            dx = float(meta["dx"])
        if dy is None and "dy" in meta:
            dy = float(meta["dy"])
        if dz is None and "dz" in meta:
            dz = float(meta["dz"])
        if Nx is None and "Nx" in meta:
            Nx = int(meta["Nx"])
        if Ny is None and "Ny" in meta:
            Ny = int(meta["Ny"])
        if Nz is None and "Nz" in meta:
            Nz = int(meta["Nz"])
        if ambient_temp is None and "ambient_temp" in meta:
            ambient_temp = float(meta["ambient_temp"])
        if source_power_density is None and "source_power_density" in meta:
            source_power_density = float(meta["source_power_density"])

    if global_min is None or global_max is None:
        return

    info = {
        "alpha": None if alpha is None else float(alpha),
        "rho": None if rho is None else float(rho),
        "cp": None if cp is None else float(cp),
        "min_temp": float(global_min),
        "max_temp": float(global_max),
        "temp_range": float(global_max - global_min),
        "t_min": float(global_min),
        "t_max": float(global_max),
        "t0_min": float(global_t0_min),
        "t0_max": float(global_t0_max),
        "prediction_horizon": None if prediction_horizon is None else float(prediction_horizon),
        "dt": float(dt),
        "num_prediction_steps": None if num_prediction_steps is None else int(num_prediction_steps),
        "num_simulation_steps": None if num_simulation_steps is None else int(num_simulation_steps),
        "num_stored_frames": None if num_stored_frames is None else int(num_stored_frames),
        "dx": None if dx is None else float(dx),
        "dy": None if dy is None else float(dy),
        "dz": None if dz is None else float(dz),
        "Nx": None if Nx is None else int(Nx),
        "Ny": None if Ny is None else int(Ny),
        "Nz": None if Nz is None else int(Nz),
        "ambient_temp": None if ambient_temp is None else float(ambient_temp),
        "source_power_density": None if source_power_density is None else float(source_power_density),
    }
    with open(os.path.join(out_root, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def run_experiment(num_fires, device):
    """Run one default experiment and write/update the dataset metadata."""

    simulation = HeatSimulation(num_fires=num_fires, device=device)
    simulation.run_simulation()
    out_root = "./data/heat_sim"
    simulation.save_results(out_root=out_root)
    write_dataset_info(out_root)


def load_runtime_config(config_path):
    """Load a Python config module and validate the runtime parameters."""

    spec = importlib.util.spec_from_file_location("heat_sim_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    num_fires_list = list(getattr(module, "NUM_FIRES_LIST", [10, 20, 30]))
    experiments_per_fire_count = int(getattr(module, "EXPERIMENTS_PER_FIRE_COUNT", 1))
    data_root = str(getattr(module, "DATA_ROOT", "./data"))
    dataset_name = str(getattr(module, "DATASET_NAME", "heat_sim"))
    total_time = float(getattr(module, "T", 600.0))
    dt = float(getattr(module, "DT", 0.1))
    device_name = str(getattr(module, "DEVICE", "auto"))
    k = float(getattr(module, "K", 1.7))
    rho = float(getattr(module, "RHO", 2300.0))
    cp = float(getattr(module, "CP", 880.0))
    ignition_temp = float(getattr(module, "IGNITION_TEMP", 600.0))
    source_power_density = float(getattr(module, "SOURCE_POWER_DENSITY", 3.0e5))
    lx = float(getattr(module, "Lx", 6.3))
    ly = float(getattr(module, "Ly", 3.1))
    lz = float(getattr(module, "Lz", 1.5))
    nx = int(getattr(module, "Nx", 64))
    ny = int(getattr(module, "Ny", 32))
    nz = int(getattr(module, "Nz", 16))

    # Validate the config early so failed batch runs produce clear messages.
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
    if k <= 0.0 or rho <= 0.0 or cp <= 0.0:
        raise ValueError("K, RHO, and CP must all be > 0.")
    if ignition_temp < 0.0:
        raise ValueError("IGNITION_TEMP must be >= 0.")
    if source_power_density < 0.0:
        raise ValueError("SOURCE_POWER_DENSITY must be >= 0.")
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise ValueError("Lx, Ly, and Lz must all be > 0.")
    if nx <= 1 or ny <= 1 or nz <= 1:
        raise ValueError("Nx, Ny, and Nz must all be > 1.")

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
        "k": k,
        "rho": rho,
        "cp": cp,
        "ignition_temp": ignition_temp,
        "source_power_density": source_power_density,
        "Lx": lx,
        "Ly": ly,
        "Lz": lz,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "device_name": device_name,
    }


def resolve_device(device_name):
    """Resolve the requested device string into a torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heat simulations using a python config file.")
    parser.add_argument("--config", default="configs/heat_sim_config.py", help="Path to simulation config file.")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config)
    device = resolve_device(cfg["device_name"])
    os.makedirs(cfg["out_root"], exist_ok=True)

    print(f"Using config: {args.config}")
    print(f"Dataset: {cfg['dataset_name']} -> {cfg['out_root']}")
    print(
        f"T={cfg['T']}s, dt={cfg['dt']}s, "
        f"prediction_steps={int(round(cfg['T'] / cfg['dt']))}, "
        f"stored_frames={int(round(cfg['T'] / cfg['dt'])) + 2}"
    )

    for num_fires in cfg["num_fires_list"]:
        print(f"Number of fires: {num_fires}")
        for i in range(cfg["experiments_per_fire_count"]):
            print(f"Calculating experiment {i + 1}/{cfg['experiments_per_fire_count']}...")
            simulation = HeatSimulation(
                num_fires=num_fires,
                T=cfg["T"],
                dt=cfg["dt"],
                k=cfg["k"],
                rho=cfg["rho"],
                cp=cfg["cp"],
                ignition_temp=cfg["ignition_temp"],
                source_power_density=cfg["source_power_density"],
                Lx=cfg["Lx"],
                Ly=cfg["Ly"],
                Lz=cfg["Lz"],
                Nx=cfg["Nx"],
                Ny=cfg["Ny"],
                Nz=cfg["Nz"],
                device=device,
            )
            simulation.run_simulation()
            simulation.save_results(out_root=cfg["out_root"])
    write_dataset_info(cfg["out_root"])
