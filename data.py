import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr


def _require_zarr():
    """Fail explicitly when zarr is unavailable."""

    if zarr is None:
        raise ImportError("zarr is required to read simulation datasets stored as .zarr groups.")


def get_np_dtype():
    """Mirror the active PyTorch default dtype when exporting NumPy arrays."""

    return np.float64 if torch.get_default_dtype() == torch.float64 else np.float32


def list_experiment_folders(base_path):
    """List all experiment folders inside one dataset root."""

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Dataset root does not exist: {base_path}")

    folders = []
    for name in os.listdir(base_path):
        path = os.path.join(base_path, name)
        if os.path.isdir(path) and name.startswith("experiment_"):
            folders.append(path)
    return sorted(folders)


def load_json_required(path):
    """Load a JSON file and fail if it is missing."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Required JSON file is missing: {path}")

    with open(path, "r") as f:
        return json.load(f)


def require_keys(payload, keys, path):
    """Validate that a JSON payload contains all required keys."""

    missing = [key for key in keys if key not in payload or payload[key] is None]
    if missing:
        raise KeyError(f"Missing required keys in '{path}': {missing}")


def load_dataset_info(base_path):
    """
    Load dataset-level metadata from info.json.

    No default values are allowed here. Training and data loading should fail
    early if dataset generation did not write a complete metadata file.
    """

    info_path = os.path.join(base_path, "info.json")
    info = load_json_required(info_path)
    required_keys = (
        "min_temp",
        "max_temp",
        "temp_range",
        "t_min",
        "t_max",
        "t0_min",
        "t0_max",
        "prediction_horizon",
        "dt",
        "num_prediction_steps",
        "num_simulation_steps",
        "num_stored_frames",
        "alpha",
        "rho",
        "cp",
        "dx",
        "dy",
        "dz",
        "Nx",
        "Ny",
        "Nz",
        "ambient_temp",
        "source_power_density",
    )
    require_keys(info, required_keys, info_path)

    min_temp = float(info["min_temp"])
    max_temp = float(info["max_temp"])
    temp_range = float(info["temp_range"])
    dt = float(info["dt"])
    prediction_horizon = float(info["prediction_horizon"])
    num_prediction_steps = int(info["num_prediction_steps"])
    num_simulation_steps = int(info["num_simulation_steps"])
    num_stored_frames = int(info["num_stored_frames"])
    alpha = float(info["alpha"])
    rho = float(info["rho"])
    cp = float(info["cp"])

    expected_temp_range = max_temp - min_temp
    if expected_temp_range <= 0.0:
        raise ValueError(f"Invalid normalization bounds in '{info_path}': min={min_temp}, max={max_temp}")
    if not np.isclose(temp_range, expected_temp_range, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Inconsistent temp_range in '{info_path}': temp_range={temp_range}, expected={expected_temp_range}"
        )
    if dt <= 0.0:
        raise ValueError(f"Invalid dt in '{info_path}': dt={dt}")
    if prediction_horizon <= 0.0:
        raise ValueError(f"Invalid prediction_horizon in '{info_path}': prediction_horizon={prediction_horizon}")
    if num_prediction_steps <= 0:
        raise ValueError(
            f"Invalid num_prediction_steps in '{info_path}': num_prediction_steps={num_prediction_steps}"
        )
    if num_simulation_steps != num_prediction_steps + 1:
        raise ValueError(
            f"Inconsistent simulation-step metadata in '{info_path}': "
            f"num_prediction_steps={num_prediction_steps}, num_simulation_steps={num_simulation_steps}"
        )
    if num_stored_frames != num_simulation_steps + 1:
        raise ValueError(
            f"Inconsistent stored-frame metadata in '{info_path}': "
            f"num_simulation_steps={num_simulation_steps}, num_stored_frames={num_stored_frames}"
        )
    if alpha <= 0.0:
        raise ValueError(f"Invalid alpha in '{info_path}': alpha={alpha}")
    if rho <= 0.0:
        raise ValueError(f"Invalid rho in '{info_path}': rho={rho}")
    if cp <= 0.0:
        raise ValueError(f"Invalid cp in '{info_path}': cp={cp}")

    return {
        "min_temp": min_temp,
        "max_temp": max_temp,
        "temp_range": temp_range,
        "t_min": float(info["t_min"]),
        "t_max": float(info["t_max"]),
        "t0_min": float(info["t0_min"]),
        "t0_max": float(info["t0_max"]),
        "prediction_horizon": prediction_horizon,
        "dt": dt,
        "num_prediction_steps": num_prediction_steps,
        "num_simulation_steps": num_simulation_steps,
        "num_stored_frames": num_stored_frames,
        "alpha": alpha,
        "rho": rho,
        "cp": cp,
        "dx": float(info["dx"]),
        "dy": float(info["dy"]),
        "dz": float(info["dz"]),
        "Nx": int(info["Nx"]),
        "Ny": int(info["Ny"]),
        "Nz": int(info["Nz"]),
        "ambient_temp": float(info["ambient_temp"]),
        "source_power_density": float(info["source_power_density"]),
    }


def load_experiment_metadata(folder):
    """Load one experiment's metadata.json without allowing partial records."""

    metadata_path = os.path.join(folder, "metadata.json")
    metadata = load_json_required(metadata_path)
    required_keys = (
        "dt",
        "prediction_horizon",
        "num_prediction_steps",
        "num_simulation_steps",
        "num_stored_frames",
        "alpha",
        "rho",
        "cp",
        "dx",
        "dy",
        "dz",
        "Nx",
        "Ny",
        "Nz",
        "ambient_temp",
        "source_power_density",
        "num_fires",
        "has_source_mask",
        "has_source_field",
        "fireplaces",
    )
    require_keys(metadata, required_keys, metadata_path)

    if not bool(metadata["has_source_mask"]):
        raise ValueError(f"Experiment metadata requires source_mask to be present: {metadata_path}")
    if not bool(metadata["has_source_field"]):
        raise ValueError(f"Experiment metadata requires source_field to be present: {metadata_path}")

    return {
        "dt": float(metadata["dt"]),
        "prediction_horizon": float(metadata["prediction_horizon"]),
        "num_prediction_steps": int(metadata["num_prediction_steps"]),
        "num_simulation_steps": int(metadata["num_simulation_steps"]),
        "num_stored_frames": int(metadata["num_stored_frames"]),
        "alpha": float(metadata["alpha"]),
        "rho": float(metadata["rho"]),
        "cp": float(metadata["cp"]),
        "dx": float(metadata["dx"]),
        "dy": float(metadata["dy"]),
        "dz": float(metadata["dz"]),
        "Nx": int(metadata["Nx"]),
        "Ny": int(metadata["Ny"]),
        "Nz": int(metadata["Nz"]),
        "ambient_temp": float(metadata["ambient_temp"]),
        "source_power_density": float(metadata["source_power_density"]),
        "num_fires": int(metadata["num_fires"]),
        "fireplaces": metadata["fireplaces"],
    }


def load_normalization_values(base_path):
    """Expose the dataset-wide normalization bounds."""

    info = load_dataset_info(base_path)
    return info["min_temp"], info["max_temp"], info["temp_range"]


def load_time_grid_from_metadata(base_path):
    """Expose dt and the implied number of simulation steps per second."""

    info = load_dataset_info(base_path)
    dt = info["dt"]
    return 1.0 / dt, dt, info["num_prediction_steps"]


def normalize_temperature(data, min_temp, temp_range):
    """Normalize temperature values into the training range."""

    return (data - min_temp) / temp_range


def normalize_source_term(source_field, rho, cp, temp_range):
    """
    Convert volumetric heating Q [W/m^3] into the normalized source term.

    The simulator evolves:
        dT/dt = alpha * laplace(T) + Q / (rho * cp)

    Training uses normalized temperatures:
        T_norm = (T - T_min) / temp_range

    Therefore the normalized source term is:
        (Q / (rho * cp)) / temp_range
    """

    return source_field / (rho * cp * temp_range)


def resolve_temperature_store(folder):
    """Resolve the required zarr store inside one experiment folder."""

    store_path = os.path.join(folder, "heat_equation_solution.zarr")
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Missing zarr store in experiment folder: {store_path}")
    return store_path


def open_required_zarr_array(root, key, store_path):
    """Open one required array from a zarr group."""

    if key not in root:
        raise KeyError(f"Missing required zarr dataset '{key}' in store '{store_path}'")
    return root[key]


def load_temperature_full(path, min_temp, temp_range):
    """Load the full temperature trajectory and normalize it."""

    if not path.endswith(".zarr"):
        raise ValueError(f"Unsupported store format (expected .zarr): {path}")

    _require_zarr()
    root = zarr.open_group(path, mode="r")
    data = np.asarray(open_required_zarr_array(root, "temperature", path))
    return normalize_temperature(data, min_temp, temp_range)


class HeatEquationMultiDataset(Dataset):
    """Static dataset returning T0 and one target time slice."""

    def __init__(self, base_path="./data/laplace_convolution/", predicted_time=3):
        dataset_info = load_dataset_info(base_path)
        self.min_temp = dataset_info["min_temp"]
        self.max_temp = dataset_info["max_temp"]
        self.temp_range = dataset_info["temp_range"]
        self.sim_steps_per_second, self.dt, _ = load_time_grid_from_metadata(base_path)
        folders = list_experiment_folders(base_path)

        self.inputs = []
        self.targets = []

        # Convert the requested prediction time in seconds into a discrete frame index.
        target_idx = int(predicted_time * self.sim_steps_per_second)
        for folder in folders:
            store = resolve_temperature_store(folder)
            data = load_temperature_full(store, self.min_temp, self.temp_range)
            if target_idx >= data.shape[0]:
                continue
            inputs = torch.tensor(data[0, :, :, :], dtype=torch.get_default_dtype()).unsqueeze(0).unsqueeze(1)
            targets = torch.tensor(data[target_idx, :, :, :], dtype=torch.get_default_dtype()).unsqueeze(0).unsqueeze(1)
            self.inputs.append(inputs)
            self.targets.append(targets)

        if not self.inputs:
            raise RuntimeError(f"No valid experiments found in '{base_path}'.")

        self.inputs = torch.cat(self.inputs, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class HeatEquationPINNDataset(Dataset):
    """
    PINN dataset that samples random space-time coordinates from full trajectories.

    Source supervision is read from the stored source_field instead of being inferred
    from temperature thresholds.
    """

    def __init__(self, base_path="./data/laplace_convolution/", points_per_sample=8192, modulo=1):
        self.points_per_sample = int(points_per_sample)
        self.files = []
        self.base_path = base_path

        dataset_info = load_dataset_info(base_path)
        self.min_temp = dataset_info["min_temp"]
        self.temp_range = dataset_info["temp_range"]
        self.rho = dataset_info["rho"]
        self.cp = dataset_info["cp"]
        self.dt = dataset_info["dt"]

        for i, folder in enumerate(list_experiment_folders(base_path)):
            if i % int(modulo) != 0:
                continue
            store = resolve_temperature_store(folder)
            self.files.append((folder, store))

        if not self.files:
            raise RuntimeError(f"No valid experiment stores found in '{base_path}'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        folder, store_path = self.files[idx]
        metadata = load_experiment_metadata(folder)
        _require_zarr()
        root = zarr.open_group(store_path, mode="r")

        temperature = np.asarray(open_required_zarr_array(root, "temperature", store_path))
        source_field = np.asarray(open_required_zarr_array(root, "source_field", store_path))

        temperature = normalize_temperature(temperature, self.min_temp, self.temp_range)
        source_term = normalize_source_term(source_field, metadata["rho"], metadata["cp"], self.temp_range)

        nt, nx, ny, nz = temperature.shape
        n = self.points_per_sample
        t_idx = np.random.randint(0, nt, size=n)
        x_idx = np.random.randint(0, nx, size=n)
        y_idx = np.random.randint(0, ny, size=n)
        z_idx = np.random.randint(0, nz, size=n)

        np_dtype = get_np_dtype()
        coords = np.stack(
            [
                x_idx / max(1, nx - 1),
                y_idx / max(1, ny - 1),
                z_idx / max(1, nz - 1),
                t_idx / max(1, nt - 1),
            ],
            axis=1,
        ).astype(np_dtype)
        target = temperature[t_idx, x_idx, y_idx, z_idx].astype(np_dtype).reshape(-1, 1)
        source = source_term[x_idx, y_idx, z_idx].astype(np_dtype).reshape(-1, 1)
        return torch.from_numpy(coords), torch.from_numpy(target), torch.from_numpy(source)


class HeatEquationMultiDataset_dynamic(Dataset):
    """
    Dynamic dataset for time-conditioned supervised training.

    Each sample contains:
    - model_input with two spatial channels: [T0, source_field]
    - integer prediction step index n (torch.long, shape (1,))
    - ground-truth past state u(t_{n-1})
    - ground-truth future state u(t_{n+1})
    - source_mask
    - source_field
    - ground-truth target state u(t_n)

    The simulator stores one additional frame at T + dt. This is intentional:
    it allows every prediction target from dt .. T to have both a valid
    target_past and a valid target_future. That keeps all three
    physics-residual variants comparable on the exact same target times:
    - model_past
    - gt_past
    - gt_future
    """

    def __init__(
        self,
        modulo=1,
        base_path="./data/new_detailed_heat_sim_f64/",
        max_experiments=None,
        experiment_offset=0,
        cache_size=4,
        use_index_cache=True,
    ):
        self.files = []
        self.data_cache = OrderedDict()
        self.cache_size = int(cache_size)
        self.base_path = base_path
        self.modulo = int(modulo)
        self.max_experiments = None if max_experiments is None else int(max_experiments)
        self.experiment_offset = int(experiment_offset)
        self.use_index_cache = bool(use_index_cache)

        # Dataset-level metadata is the single source of truth for normalization
        # and global physical constants. No default fallbacks are allowed.
        dataset_info = load_dataset_info(base_path)
        self.min_temp = dataset_info["min_temp"]
        self.max_temp = dataset_info["max_temp"]
        self.temp_range = dataset_info["temp_range"]
        self.t_min = dataset_info["t_min"]
        self.t_max = dataset_info["t_max"]
        self.t0_min = dataset_info["t0_min"]
        self.t0_max = dataset_info["t0_max"]
        self.prediction_horizon = dataset_info["prediction_horizon"]
        self.dt = dataset_info["dt"]
        self.num_prediction_steps = dataset_info["num_prediction_steps"]
        self.num_simulation_steps = dataset_info["num_simulation_steps"]
        self.num_stored_frames = dataset_info["num_stored_frames"]
        self.alpha = dataset_info["alpha"]
        self.rho = dataset_info["rho"]
        self.cp = dataset_info["cp"]
        self.dx = dataset_info["dx"]
        self.dy = dataset_info["dy"]
        self.dz = dataset_info["dz"]
        self.Nx = dataset_info["Nx"]
        self.Ny = dataset_info["Ny"]
        self.Nz = dataset_info["Nz"]
        self.ambient_temp = dataset_info["ambient_temp"]
        self.source_power_density = dataset_info["source_power_density"]

        # Track how many experiment folders were selected after applying modulo/limits.
        self.num_selected_experiments = 0

        file_infos = self._load_or_build_file_infos()
        self.num_selected_experiments = len(file_infos)
        for folder, store_path, num_prediction_steps in file_infos:
            # Because the dataset stores one extra future frame at T + dt, every
            # prediction target from dt .. T can use the same sample structure:
            # target_past = u(t-dt), target = u(t), target_future = u(t+dt).
            for predicted_time in range(1, num_prediction_steps + 1):
                self.files.append((folder, store_path, predicted_time))

        if not self.files:
            raise RuntimeError(f"No valid dynamic samples found in '{base_path}'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        folder, store_path, predicted_time = self.files[idx]
        experiment_payload = self._get_or_load_experiment_payload(folder, store_path)
        temperature = experiment_payload["temperature"]
        t0 = experiment_payload["T0"]
        source_mask = experiment_payload["source_mask"]
        source_field = experiment_payload["source_field"]

        # The model now receives both the initial temperature field and the
        # explicit source field as two aligned spatial input channels.
        input_np = np.stack([t0, source_field], axis=0)
        target_past_np = temperature[predicted_time - 1]
        target_np = temperature[predicted_time]
        target_future_np = temperature[predicted_time + 1]

        dtype = torch.get_default_dtype()
        input_tensor = torch.tensor(input_np, dtype=dtype)
        target_past_tensor = torch.tensor(target_past_np, dtype=dtype).unsqueeze(0)
        target_tensor = torch.tensor(target_np, dtype=dtype).unsqueeze(0)
        target_future_tensor = torch.tensor(target_future_np, dtype=dtype).unsqueeze(0)
        source_mask_tensor = torch.tensor(source_mask, dtype=dtype).unsqueeze(0)
        source_field_tensor = torch.tensor(source_field, dtype=dtype).unsqueeze(0)
        # Keep the discrete index exact; the trainer derives model times using
        # this dataset's physical dt and prediction horizon.
        step_idx_tensor = torch.tensor([predicted_time], dtype=torch.long)
        return (
            input_tensor,
            step_idx_tensor,
            target_past_tensor,
            target_future_tensor,
            source_mask_tensor,
            source_field_tensor,
        ), target_tensor

    def _get_or_load_experiment_payload(self, folder, store_path):
        """Cache the normalized tensors needed by the dynamic training loop."""

        if store_path in self.data_cache:
            self.data_cache.move_to_end(store_path)
            return self.data_cache[store_path]

        payload = self._load_experiment_payload(folder, store_path)
        if self.cache_size > 0 and len(self.data_cache) >= self.cache_size:
            self.data_cache.popitem(last=False)
        self.data_cache[store_path] = payload
        return payload

    def _load_experiment_payload(self, folder, store_path):
        """Load all required tensors for one experiment from zarr and metadata."""

        metadata = load_experiment_metadata(folder)
        self._validate_experiment_metadata(metadata, folder)

        _require_zarr()
        root = zarr.open_group(store_path, mode="r")
        temperature = np.asarray(open_required_zarr_array(root, "temperature", store_path))
        t0 = np.asarray(open_required_zarr_array(root, "T0", store_path))
        source_mask = np.asarray(open_required_zarr_array(root, "source_mask", store_path))
        source_field = np.asarray(open_required_zarr_array(root, "source_field", store_path))

        temperature_norm = normalize_temperature(temperature, self.min_temp, self.temp_range)
        t0_norm = normalize_temperature(t0, self.min_temp, self.temp_range)
        source_term_norm = normalize_source_term(source_field, metadata["rho"], metadata["cp"], self.temp_range)

        return {
            "temperature": temperature_norm,
            "T0": t0_norm,
            "source_mask": source_mask.astype(get_np_dtype(), copy=False),
            "source_field": source_term_norm.astype(get_np_dtype(), copy=False),
        }

    def _validate_experiment_metadata(self, metadata, folder):
        """Ensure each experiment agrees with the dataset-level metadata."""

        comparisons = {
            "dt": (metadata["dt"], self.dt),
            "prediction_horizon": (metadata["prediction_horizon"], self.prediction_horizon),
            "num_prediction_steps": (metadata["num_prediction_steps"], self.num_prediction_steps),
            "num_simulation_steps": (metadata["num_simulation_steps"], self.num_simulation_steps),
            "num_stored_frames": (metadata["num_stored_frames"], self.num_stored_frames),
            "alpha": (metadata["alpha"], self.alpha),
            "rho": (metadata["rho"], self.rho),
            "cp": (metadata["cp"], self.cp),
            "dx": (metadata["dx"], self.dx),
            "dy": (metadata["dy"], self.dy),
            "dz": (metadata["dz"], self.dz),
            "Nx": (metadata["Nx"], self.Nx),
            "Ny": (metadata["Ny"], self.Ny),
            "Nz": (metadata["Nz"], self.Nz),
        }
        for key, (experiment_value, dataset_value) in comparisons.items():
            if isinstance(dataset_value, float):
                if not np.isclose(experiment_value, dataset_value, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"Experiment metadata mismatch for '{key}' in '{folder}': "
                        f"experiment={experiment_value}, dataset={dataset_value}"
                    )
            else:
                if experiment_value != dataset_value:
                    raise ValueError(
                        f"Experiment metadata mismatch for '{key}' in '{folder}': "
                        f"experiment={experiment_value}, dataset={dataset_value}"
                    )

    def _index_cache_path(self):
        """Build a deterministic index-cache filename for one dataset selection."""

        limit_tag = "all" if self.max_experiments is None else str(self.max_experiments)
        return os.path.join(
            self.base_path,
            f".dynamic_index_mod{self.modulo}_off{self.experiment_offset}_lim{limit_tag}.json",
        )

    def _load_or_build_file_infos(self):
        """Reuse a cached experiment index when possible."""

        cache_path = self._index_cache_path()
        if self.use_index_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                payload = json.load(f)
            require_keys(payload, ("modulo", "file_infos"), cache_path)
            infos = []
            for entry in payload["file_infos"]:
                require_keys(entry, ("folder", "path", "num_prediction_steps"), cache_path)
                folder = str(entry["folder"])
                path = str(entry["path"])
                num_prediction_steps = int(entry["num_prediction_steps"])
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"Cached experiment folder is missing: {folder}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Cached zarr store is missing: {path}")
                if num_prediction_steps <= 0:
                    raise ValueError(
                        f"Cached num_prediction_steps must be > 0 in '{cache_path}', got {num_prediction_steps}"
                    )
                infos.append((folder, path, num_prediction_steps))
            return infos

        infos = self._build_file_infos()
        if self.use_index_cache:
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "modulo": self.modulo,
                        "file_infos": [
                            {"folder": folder, "path": path, "num_prediction_steps": int(num_prediction_steps)}
                            for folder, path, num_prediction_steps in infos
                        ],
                    },
                    f,
                    indent=2,
                )
        return infos

    def _build_file_infos(self):
        """Enumerate selected experiment folders and read their frame counts."""

        infos = []
        selected_count = 0
        for index, folder in enumerate(list_experiment_folders(self.base_path)):
            if index < self.experiment_offset:
                continue
            if index % self.modulo != 0:
                continue

            store_path = resolve_temperature_store(folder)
            metadata = load_experiment_metadata(folder)
            num_prediction_steps = int(metadata["num_prediction_steps"])
            num_stored_frames = int(metadata["num_stored_frames"])
            if num_prediction_steps <= 0:
                raise ValueError(
                    f"Experiment '{folder}' must contain at least one prediction step, got {num_prediction_steps}"
                )
            if num_stored_frames != num_prediction_steps + 2:
                raise ValueError(
                    f"Experiment '{folder}' has inconsistent frame metadata: "
                    f"num_prediction_steps={num_prediction_steps}, num_stored_frames={num_stored_frames}"
                )

            infos.append((folder, store_path, num_prediction_steps))
            selected_count += 1
            if self.max_experiments is not None and selected_count >= self.max_experiments:
                break
        return infos


def prepare_experiment_split(dataset, manifest_path=None, seed=42, training_source_counts=None):
    """Persist disjoint experiment groups and fit normalization on training only.

    A manifest is immutable for a given selection and metadata snapshot. Separate
    model seeds reuse it. Optional source counts restrict training/validation;
    all other counts become held-out extrapolation experiments.
    """
    import hashlib
    import fcntl
    from pathlib import Path
    from torch.utils.data import Subset

    folders = {Path(folder).name: Path(folder) for folder, _, _ in dataset.files}
    names = sorted(folders)
    metadata = {name: load_json_required(str(folders[name] / 'metadata.json')) for name in names}
    fingerprint = hashlib.sha256(json.dumps({
        'info': load_json_required(str(Path(dataset.base_path) / 'info.json')),
        'experiments': metadata,
    }, sort_keys=True).encode()).hexdigest()
    allowed = None if training_source_counts is None else sorted(set(training_source_counts))
    path = Path(manifest_path) if manifest_path else Path(dataset.base_path) / 'experiment_split.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            manifest = load_json_required(str(path))
        else:
            eligible = [n for n in names if allowed is None or metadata[n]['num_fires'] in allowed]
            if len(eligible) < 3:
                raise ValueError('At least three eligible independent experiments are required for train/val/test.')
            rng = torch.Generator().manual_seed(int(seed))
            shuffled = [eligible[i] for i in torch.randperm(len(eligible), generator=rng).tolist()]
            n_train = min(len(eligible) - 2, max(1, int(0.7 * len(eligible))))
            n_val = max(1, int(0.15 * len(eligible)))
            train = sorted(shuffled[:n_train])
            val = sorted(shuffled[n_train:n_train + n_val])
            test = sorted(set(names) - set(train) - set(val))
            lower = min(float(metadata[n]['min_temp']) for n in train)
            upper = max(float(metadata[n]['max_temp']) for n in train)
            if not np.isfinite([lower, upper]).all() or upper <= lower:
                raise ValueError('Invalid training-only temperature bounds in experiment metadata.')
            manifest = dict(schema_version=1, seed=int(seed), metadata_sha256=fingerprint,
                            training_source_counts=allowed, train=train, validation=val, test=test,
                            normalization=dict(min_temp=lower, max_temp=upper,
                                               temp_range=upper-lower, fitted_on='train'))
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            os.replace(temporary, path)
    if manifest.get('schema_version') != 1 or manifest['metadata_sha256'] != fingerprint:
        raise ValueError('Split metadata changed: use the original dataset or an explicitly new split manifest.')
    if manifest['seed'] != int(seed) or manifest.get('training_source_counts') != allowed:
        raise ValueError('Requested split seed/source-count policy differs from the saved manifest.')
    groups = [manifest[key] for key in ('train', 'validation', 'test')]
    flat = [name for group in groups for name in group]
    if any(not group for group in groups) or len(flat) != len(set(flat)) or set(flat) != set(names):
        raise ValueError('Split must partition all selected experiments exactly once into nonempty groups.')
    expected_min = min(float(metadata[n]['min_temp']) for n in manifest['train'])
    expected_max = max(float(metadata[n]['max_temp']) for n in manifest['train'])
    norm = manifest['normalization']
    if (norm.get('fitted_on') != 'train' or norm['min_temp'] != expected_min or
            norm['max_temp'] != expected_max or norm['temp_range'] != expected_max-expected_min):
        raise ValueError('Manifest normalization must match the training experiments only.')
    dataset.min_temp, dataset.max_temp, dataset.temp_range = expected_min, expected_max, expected_max-expected_min
    dataset.data_cache.clear()
    subsets = {}
    for key, group in zip(('train', 'validation', 'test'), groups):
        selected = set(group)
        subsets[key] = Subset(dataset, [i for i, (folder, _, _) in enumerate(dataset.files)
                                       if Path(folder).name in selected])
    return subsets, manifest
