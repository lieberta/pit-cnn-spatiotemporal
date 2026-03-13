import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr


def get_np_dtype():
    return np.float64 if torch.get_default_dtype() == torch.float64 else np.float32


def list_experiment_folders(base_path):
    return [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and f.startswith("experiment")
    ]


def load_dataset_info(base_path):
    info_path = os.path.join(base_path, "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    required_keys = ("min_temp", "max_temp", "temp_range", "dt", "num_timesteps")
    missing = [k for k in required_keys if k not in info]
    if missing:
        raise KeyError(f"Missing keys in '{info_path}': {missing}")

    min_temp = float(info["min_temp"])
    max_temp = float(info["max_temp"])
    temp_range = float(info["temp_range"])
    dt = float(info["dt"])
    num_timesteps = int(info["num_timesteps"])

    expected_temp_range = max_temp - min_temp
    if expected_temp_range <= 0.0:
        raise ValueError(f"Invalid min/max in '{info_path}': min={min_temp}, max={max_temp}")
    if not np.isclose(temp_range, expected_temp_range, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"Inconsistent temp_range in '{info_path}': temp_range={temp_range}, expected={expected_temp_range}"
        )
    if dt <= 0.0:
        raise ValueError(f"Invalid dt in '{info_path}': {dt}")
    if num_timesteps <= 0:
        raise ValueError(f"Invalid num_timesteps in '{info_path}': {num_timesteps}")

    return {
        "min_temp": min_temp,
        "max_temp": max_temp,
        "temp_range": temp_range,
        "dt": dt,
        "num_timesteps": num_timesteps,
    }


def load_normalization_values(base_path):
    info = load_dataset_info(base_path)
    return info["min_temp"], info["max_temp"], info["temp_range"]


def load_time_grid_from_metadata(base_path):
    info = load_dataset_info(base_path)
    dt = info["dt"]
    sim_steps_per_second = 1.0 / dt
    return sim_steps_per_second, dt, info["num_timesteps"]


def normalize_temperature(data, min_temp, temp_range):
    return (data - min_temp) / temp_range


def resolve_temperature_store(folder):
    candidates = [
        "heat_equation_solution.zarr",
        "heat_equation_solution.npz",
        "normalized_heat_equation_solution.zarr",
        "normalized_heat_equation_solution.npz",
    ]
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def is_normalized_store(path):
    return os.path.basename(path).startswith("normalized_")


def load_temperature_full(path, min_temp, temp_range):
    if path.endswith(".zarr"):
        _require_zarr()
        arr = zarr.open_group(path, mode="r")["temperature"]
        data = np.asarray(arr)
    else:
        data = np.load(path)["temperature"]

    if is_normalized_store(path):
        return data
    return normalize_temperature(data, min_temp, temp_range)


class HeatEquationMultiDataset(Dataset):
    def __init__(self, base_path="./data/laplace_convolution/", predicted_time=3):
        min_temp, _, temp_range = load_normalization_values(base_path)
        self.sim_steps_per_second, self.dt, _ = load_time_grid_from_metadata(base_path)  # Load timing from metadata for correct time indexing.
        folders = list_experiment_folders(base_path)

        self.inputs = []
        self.targets = []

        target_idx = int(predicted_time * self.sim_steps_per_second)  # Convert requested prediction time (seconds) to discrete index.
        for folder in folders:
            store = resolve_temperature_store(folder)
            if store is None:
                continue
            data = load_temperature_full(store, min_temp, temp_range)
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
    def __init__(
        self,
        base_path="./data/laplace_convolution/",
        points_per_sample=8192,
        modulo=1,
        source_threshold_raw=1000.0,
        source_intensity_raw=100000.0,
    ):
        self.points_per_sample = int(points_per_sample)
        self.files = []

        min_temp, _, temp_range = load_normalization_values(base_path)
        self.min_temp = min_temp
        self.temp_range = temp_range
        self.source_threshold_norm = (float(source_threshold_raw) - min_temp) / temp_range
        self.source_intensity_norm = float(source_intensity_raw) / temp_range

        for i, folder in enumerate(sorted(list_experiment_folders(base_path))):
            if i % modulo != 0:
                continue
            store = resolve_temperature_store(folder)
            if store is not None:
                self.files.append(store)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = load_temperature_full(path, self.min_temp, self.temp_range)

        nt, nx, ny, nz = data.shape
        n = self.points_per_sample
        t_idx = np.random.randint(0, nt, size=n)
        x_idx = np.random.randint(0, nx, size=n)
        y_idx = np.random.randint(0, ny, size=n)
        z_idx = np.random.randint(0, nz, size=n)

        t = t_idx / max(1, nt - 1)
        x = x_idx / max(1, nx - 1)
        y = y_idx / max(1, ny - 1)
        z = z_idx / max(1, nz - 1)

        np_dtype = get_np_dtype()
        coords = np.stack([x, y, z, t], axis=1).astype(np_dtype)
        target = data[t_idx, x_idx, y_idx, z_idx].astype(np_dtype).reshape(-1, 1)
        source_mask = data[0, x_idx, y_idx, z_idx] > self.source_threshold_norm
        source = np.where(source_mask, self.source_intensity_norm, 0.0).astype(np_dtype).reshape(-1, 1)
        return torch.from_numpy(coords), torch.from_numpy(target), torch.from_numpy(source)


class HeatEquationMultiDataset_dynamic(Dataset):
    def __init__(
        self,
        modulo=1,
        base_path="./data/new_detailed_heat_sim_f64/",
        max_experiments=None,
        experiment_offset=0,
        cache_size=4,
        use_index_cache=True,
        default_num_timesteps=10001,
    ):
        self.files = []
        self.data_cache = OrderedDict()
        self.cache_size = int(cache_size)
        self.base_path = base_path
        # Keep every `modulo`th experiment folder (e.g. modulo=5 -> every 5th folder).
        self.modulo = int(modulo)
        # Optional absolute limit and offset to sweep dataset size deterministically.
        self.max_experiments = None if max_experiments is None else int(max_experiments)
        self.experiment_offset = int(experiment_offset)
        self.use_index_cache = bool(use_index_cache)
        self.default_num_timesteps = int(default_num_timesteps)
        self.min_temp, self.max_temp, self.temp_range = load_normalization_values(base_path)
        self.sim_steps_per_second, self.dt, self.meta_num_timesteps = load_time_grid_from_metadata(base_path)  # Resolve dataset dt/frequency once for dynamic time tensors.
        # Number of selected experiment folders (tracked for run metadata).
        self.num_selected_experiments = 0

        file_infos = self._load_or_build_file_infos()
        self.num_selected_experiments = len(file_infos)
        for path, num_timesteps in file_infos:
            for predicted_time in range(1, num_timesteps):
                self.files.append((path, predicted_time))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        store_path, predicted_time = self.files[idx]
        store_name = os.path.basename(store_path)
        already_normalized = store_name.startswith("normalized_")

        if store_path not in self.data_cache:
            if store_path.endswith(".zarr"):
                _require_zarr()
                temp_store = zarr.open_group(store_path, mode="r")["temperature"]
            else:
                temp_store = np.load(store_path)["temperature"]
                if not already_normalized:
                    temp_store = normalize_temperature(temp_store, self.min_temp, self.temp_range)

            if self.cache_size > 0 and len(self.data_cache) >= self.cache_size:
                self.data_cache.popitem(last=False)
            self.data_cache[store_path] = temp_store
        else:
            self.data_cache.move_to_end(store_path)

        temp_store = self.data_cache[store_path]
        if store_path.endswith(".zarr"):
            input_np = np.asarray(temp_store[0, :, :, :])
            target_np = np.asarray(temp_store[predicted_time, :, :, :])
            if not already_normalized:
                input_np = normalize_temperature(input_np, self.min_temp, self.temp_range)
                target_np = normalize_temperature(target_np, self.min_temp, self.temp_range)
        else:
            input_np = temp_store[0, :, :, :]
            target_np = temp_store[predicted_time, :, :, :]

        dtype = torch.get_default_dtype()
        input_tensor = torch.tensor(input_np, dtype=dtype).unsqueeze(0)
        target_tensor = torch.tensor(target_np, dtype=dtype).unsqueeze(0)
        predicted_time_tensor = torch.tensor([predicted_time * self.dt], dtype=dtype)  # Convert integer timestep index to physical time in seconds.
        return (input_tensor, predicted_time_tensor), target_tensor

    def _index_cache_path(self):
        limit_tag = "all" if self.max_experiments is None else str(self.max_experiments)
        return os.path.join(
            self.base_path,
            f".dynamic_index_mod{self.modulo}_off{self.experiment_offset}_lim{limit_tag}.json",
        )

    def _load_or_build_file_infos(self):
        cache_path = self._index_cache_path()
        if self.use_index_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    payload = json.load(f)
                infos = payload.get("file_infos", [])
                valid_infos = [
                    (entry["path"], int(entry["num_timesteps"]))
                    for entry in infos
                    if os.path.exists(entry["path"]) and int(entry["num_timesteps"]) > 1
                ]
                if valid_infos:
                    return valid_infos
            except (OSError, ValueError, KeyError, TypeError):
                pass

        infos = self._build_file_infos()
        if self.use_index_cache:
            try:
                with open(cache_path, "w") as f:
                    json.dump(
                        {
                            "modulo": self.modulo,
                            "file_infos": [
                                {"path": path, "num_timesteps": int(num_timesteps)}
                                for path, num_timesteps in infos
                            ],
                        },
                        f,
                    )
            except OSError:
                pass
        return infos

    def _build_file_infos(self):
        infos = []
        folders = sorted(list_experiment_folders(self.base_path))
        selected_count = 0
        for i, folder in enumerate(folders):
            if i < self.experiment_offset:
                continue
            if i % self.modulo != 0:
                continue

            store = resolve_temperature_store(folder)
            if store is None:
                continue

            num_timesteps = self._read_num_timesteps(folder, store, self.default_num_timesteps)
            if num_timesteps > 1:
                infos.append((store, num_timesteps))
                selected_count += 1
                if self.max_experiments is not None and selected_count >= self.max_experiments:
                    break
        return infos

    @staticmethod
    def _read_num_timesteps(folder, store_path, default_num_timesteps):
        # 1) Prefer per-experiment metadata when available (highest priority).
        metadata_path = os.path.join(folder, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                saved_frames = int(meta.get("saved_frames", 0))
                if saved_frames > 1:
                    return saved_frames
            except (OSError, ValueError, TypeError):
                pass

        # 2) Fallback to dataset-level info file:
        #    <dataset_root>/info.json
        #    where dataset_root is the parent of the experiment folder.
        dataset_meta_path = os.path.join(os.path.dirname(folder), "info.json")
        if os.path.exists(dataset_meta_path):
            try:
                with open(dataset_meta_path, "r") as f:
                    dataset_meta = json.load(f)
                dataset_num_timesteps = int(dataset_meta.get("num_timesteps", 0))
                # num_timesteps is Nt in simulation terms; stored arrays have Nt + 1 frames.
                # We return frame count here because predicted_time indexes into saved frames.
                if dataset_num_timesteps > 0:
                    return dataset_num_timesteps + 1
            except (OSError, ValueError, TypeError):
                pass

        # 3) Keep backward-compatible config fallback.
        if default_num_timesteps > 1:
            return default_num_timesteps

        # 4) Last resort: inspect array shape directly.
        if store_path.endswith(".zarr"):
            _require_zarr()
            return int(zarr.open_group(store_path, mode="r")["temperature"].shape[0])

        with np.load(store_path) as npz:
            return int(npz["temperature"].shape[0])
