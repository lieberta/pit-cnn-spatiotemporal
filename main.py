
import argparse
import importlib.util

from models.picnn_static import PICNN_static
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst
from dataset import HeatEquationMultiDataset, HeatEquationMultiDataset_dynamic
import torch
from training.loss import CombinedLoss
import os
import json
import datetime
import uuid
import random
from pathlib import Path
from types import SimpleNamespace
from torchsummary import summary
from scripts.list_run_ids import iter_run_configs, matches_filters

TRAIN_DTYPE = torch.float32 # Default training data type, can be overridden by config files.


MODEL_CLASS_REGISTRY = {
    "PICNN_static": PICNN_static,
    "PITCNN_dynamic": PITCNN_dynamic,
    "PITCNN_dynamic_batchnorm": PITCNN_dynamic_batchnorm,
    "PITCNN_dynamic_latenttime1": PITCNN_dynamic_latenttime1,
    "PITCNN_dynamic_timefirst": PITCNN_dynamic_timefirst,
}


def load_config_module(config_path):
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    spec = importlib.util.spec_from_file_location("job_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Function to create a unique run ID based on a prefix, current timestamp, and a short random UUID.
def make_run_id(prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{short_id}"

# Function to write the run configuration to a JSON file in the specified run directory.
def write_run_config(run_dir, config):
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def load_run_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def find_run_dir_by_id(runs_root, run_id):
    # Supports both old layout (runs_root/run_id) and nested model folders.
    direct = os.path.join(runs_root, run_id)
    if os.path.isdir(direct):
        return direct

    for root, dirs, files in os.walk(runs_root):
        if os.path.basename(root) == run_id and "config.json" in files:
            return root

    return direct


def collect_run_ids(runs_root="./runs", mode="both", a=None, model_name=None, model_class=None, seed=None, epochs=None):
    args = SimpleNamespace(
        a=a,
        model_name=model_name,
        model_class=model_class,
        seed=seed,
        epochs=epochs,
    )
    run_ids = []
    for run_id, _, config in iter_run_configs(Path(runs_root), mode):
        if matches_filters(run_id, mode, config, args):
            run_ids.append(run_id)
    return run_ids

# Function to train the static model. It sets up the parameters, creates the model and dataset, and trains the model while saving the configuration.
def static(predicted_time, a, lr, batch, epochs, channels, runs_root, resume_run_id_static, device, seed, comment=None):
    model_group = "PICNN_static"
    model_root = os.path.join(runs_root, model_group)
    resume_checkpoint_path = None
    if resume_run_id_static:
        run_dir = find_run_dir_by_id(runs_root, resume_run_id_static)
        model_root = os.path.dirname(run_dir)
        config = load_run_config(run_dir)
        if config:
            lr = config.get("lr", lr)
            batch = config.get("batch", batch)
            channels = config.get("channels", channels)
            a = config.get("a", a)
            predicted_time = config.get("predicted_time", predicted_time)
        resume_checkpoint_path = os.path.join(run_dir, f"{resume_run_id_static}.pth")


    run_id = resume_run_id_static or make_run_id("static")
    run_dir = os.path.join(model_root, run_id)
    if not resume_run_id_static:
        config = {
            "run_id": run_id,
            "model_class": "PICNN_static",
            "predicted_time": predicted_time,
            "a": a,
            "lr": lr,
            "batch": batch,
            "epochs": epochs,
            "channels": channels,
            "seed": seed,
            "dataset_version": "unknown",
            "tags": ["static", f"a{a}"],
        }
        if comment:
            config["comment"] = comment
        write_run_config(run_dir, config)

    required = {
        "predicted_time": predicted_time,
        "a": a,
        "lr": lr,
        "batch": batch,
        "epochs": epochs,
        "channels": channels,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Missing required static training params: {missing}. "
            f"For resume runs, ensure config.json exists in '{run_dir}' with these fields."
        )

    loss_fn = CombinedLoss(a=a, predicted_time=predicted_time, device=device)
    #loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

    model = PICNN_static(loss_fn=loss_fn, channels=channels).to(device=device, dtype=TRAIN_DTYPE)
    model_name = run_id
    dataset = HeatEquationMultiDataset(predicted_time=predicted_time)

    model.train_model(dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=model_root,
                      run_id=run_id, a=a, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)

def dynamic(a, lr, batch, epochs, channels, model_class, name, runs_root, resume_run_id, device, seed, comment=None):

    # a function that takes all important parameter as input, creates and trains the model

    run_id = resume_run_id or make_run_id("dynamic")
    model_name = run_id
    model_root = os.path.join(runs_root, name)
    model_dir = os.path.join(model_root, model_name)
    model_pth = os.path.join(model_dir, f"{model_name}.pth")

    resume_checkpoint_path = None
    if resume_run_id:
        model_dir = find_run_dir_by_id(runs_root, resume_run_id)
        model_root = os.path.dirname(model_dir)
        model_pth = os.path.join(model_dir, f"{resume_run_id}.pth")
        model_name = resume_run_id
        run_id = resume_run_id

    # Check if the model directory and the specific model file exist
    if os.path.exists(model_dir) and os.path.isfile(model_pth):
        resume_checkpoint_path = model_pth
        config = load_run_config(model_dir)
        if config:
            lr = config.get("lr", lr)
            batch = config.get("batch", batch)
            channels = config.get("channels", channels)
            a = config.get("a", a)
            name = config.get("name", name)
        print(f"The model '{model_name}' already exists and will be trained further.")
    else:
        # If the directory or model file doesn't exist, print this message
        print(f"A new folder and model '{model_name}' will be created.")
        config = {
            "run_id": run_id,
            "model_class": model_class.__name__,
            "a": a,
            "lr": lr,
            "batch": batch,
            "epochs": epochs,
            "channels": channels,
            "seed": seed,
            "dataset_version": "unknown",
            "tags": ["dynamic", f"a{a}"],
            "name": name,
        }
        if comment:
            config["comment"] = comment
        write_run_config(model_dir, config)

    # Validate that all required parameters are present before training
    required = {
        "a": a,
        "lr": lr,
        "batch": batch,
        "epochs": epochs,
        "channels": channels,
        "name": name,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Missing required dynamic training params: {missing}. "
            f"For resume runs, ensure config.json exists in '{model_dir}' with these fields."
        )

    print(f'Create Dataset HeatEquationMultiDatset_dynamic')
    # this is for version 2 training package:
    dataset = HeatEquationMultiDataset_dynamic()
    model = model_class(c=channels).to(device=device, dtype=TRAIN_DTYPE)
    print(f'Train Model:')
    model.train_model(a=a, dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=model_root,
                      run_id=run_id, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train static/dynamic model from a config file.")
    parser.add_argument(
        "--config",
        default=os.environ.get("TRAIN_CONFIG", "train_config.py"),
        help="Path to Python config file (default: $TRAIN_CONFIG or train_config.py).",
    )
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    required_cfg_fields = ["TRAIN_DTYPE", "epochs", "a_list", "model_class_name", "model_name", "run_comment"]
    missing_cfg_fields = [field for field in required_cfg_fields if not hasattr(cfg, field)]
    if missing_cfg_fields:
        raise ValueError(
            f"Missing required config fields in '{args.config}': {missing_cfg_fields}"
        )

    TRAIN_DTYPE = cfg.TRAIN_DTYPE
    epochs = cfg.epochs
    a_list = cfg.a_list
    model_class_name = cfg.model_class_name
    model_name = cfg.model_name
    run_comment = cfg.run_comment

    torch.set_default_dtype(TRAIN_DTYPE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    print('Okaaay - Let\'s go...')

    seed = None  # Set to an int for reproducible runs.
    
    if seed is None:
        seed = random.randint(1, 1000000)
        print(f'No seed provided. Generated random seed = {seed}')
    else:
        print(f'Using provided seed = {seed}')
    torch.manual_seed(seed)

    # Validate model_class_name and determine mode
    if model_class_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown model_class_name '{model_class_name}'. "
            f"Available: {list(MODEL_CLASS_REGISTRY.keys())}"
        )
    selected_model_class = MODEL_CLASS_REGISTRY[model_class_name]
    inferred_mode = "static" if selected_model_class is PICNN_static else "dynamic"

    runs_root_static = "./runs/static"
    runs_root_dynamic = "./runs/dynamic"

    # Shared training parameters
    channels = 16
    lr = 0.001
    batch = 32 * 8

    # Static parameters
    predicted_times = [0.5, 3, 10]
    resume_run_ids_static = []  # z.B. ["static_20260101-120000_ab12cd", "static_20260102-130000_ef34gh"]
    auto_collect_static = False

    # Dynamic parameters
    resume_run_ids_dynamic = []  # z.B. ["dynamic_20260101-120000_ab12cd", "dynamic_20260102-130000_ef34gh"]
    auto_collect_dynamic = False    # True if i want to further train my existing models

    # collects all ids from input values, i.e. a = 1
    if auto_collect_static:
        resume_run_ids_static = collect_run_ids(
            runs_root="./runs",
            mode="static",
            a=None,
            model_class="PICNN_static",
        )
        print(f"Auto-collected static run IDs: {resume_run_ids_static}")

    if auto_collect_dynamic:
        resume_run_ids_dynamic = collect_run_ids(
            runs_root="./runs",
            mode="dynamic",
            model_name=model_name,
        )
        print(f"Auto-collected dynamic run IDs: {resume_run_ids_dynamic}")

    if inferred_mode == "static":
        if resume_run_ids_static:
            for run_id in resume_run_ids_static:
                static(predicted_time=None, a=None, lr=None, batch=None, epochs=epochs,
                       channels=None, runs_root=runs_root_static, resume_run_id_static=run_id,
                       device=device, seed=seed, comment=run_comment)
        else:
            for a in a_list:
                for t in predicted_times:
                    static(predicted_time=t, a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                           runs_root=runs_root_static, resume_run_id_static=None,
                           device=device, seed=seed, comment=run_comment)

    if inferred_mode == "dynamic":
        if resume_run_ids_dynamic:
            for run_id in resume_run_ids_dynamic:
                dynamic(a=None, lr=None, batch=None, epochs=epochs, channels=None,
                        model_class=selected_model_class, name=None,
                        runs_root=runs_root_dynamic, resume_run_id=run_id,
                        device=device, seed=seed, comment=run_comment)
        else:
            for a in a_list:
                dynamic(a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                        model_class=selected_model_class, name=model_name,
                        runs_root=runs_root_dynamic, resume_run_id=None,
                        device=device, seed=seed, comment=run_comment)
