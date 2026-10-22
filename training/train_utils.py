import csv
import json
import os
import time

import torch


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            try:
                param_dtype = next(model.parameters()).dtype
            except StopIteration:
                param_dtype = None
            for state in optimizer.state.values():
                for key, value in state.items():
                    if not torch.is_tensor(value):
                        continue
                    if value.is_floating_point() and param_dtype is not None:
                        state[key] = value.to(device=device, dtype=param_dtype)
                    else:
                        state[key] = value.to(device=device)
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        start_epoch = checkpoint.get("epoch", -1) + 1
        return start_epoch
    model.load_state_dict(checkpoint)
    return 0


def append_metrics_row(csv_path, header, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_loss_history_from_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return [], [], []

    by_epoch = {}
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            by_epoch[epoch] = (train_loss, val_loss)

    epochs = sorted(by_epoch.keys())
    train_losses = [by_epoch[e][0] for e in epochs]
    val_losses = [by_epoch[e][1] for e in epochs]
    return epochs, train_losses, val_losses


def fallback_loss_history(num_epochs, train_losses, val_losses):
    count = min(len(train_losses), len(val_losses))
    epochs = list(range(1, count + 1))
    return epochs, train_losses[:count], val_losses[:count]


def accumulate_training_duration(config_path, additional_seconds):
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    previous = config.get("training_duration_seconds", 0.0)
    try:
        previous = float(previous)
    except (TypeError, ValueError):
        previous = 0.0

    total = previous + float(additional_seconds)
    config["training_duration_seconds"] = total
    config["training_duration_hms"] = time.strftime("%H:%M:%S", time.gmtime(total))

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except OSError:
        return False

    return True


def update_run_config(config_path, updates):
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    config.update(updates)

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except OSError:
        return False

    return True


def seed_everything(seed):
    """Seed independent runs identically before model construction."""
    import random
    import numpy as np
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Seed Python/NumPy from the DataLoader's independent generator."""
    import random
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def git_provenance():
    """Record the committed revision and whether local files differ."""
    import hashlib
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    status = git('status', '--porcelain').decode()
    return dict(commit=git('rev-parse', 'HEAD').decode().strip(), dirty=bool(status),
                status=status, tracked_diff_sha256=hashlib.sha256(git('diff', 'HEAD')).hexdigest())


def state_dict_hash(state_dict):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_state_hash(model):
    return state_dict_hash(model.state_dict())
