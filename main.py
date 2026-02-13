
from models import PICNN_static, PECNN_dynamic, PECNN_dynamic_batchnorm, PECNN_dynamic_timefirst, PECNN_dynamic_latenttime1
from dataset import HeatEquationMultiDataset, HeatEquationMultiDataset_dynamic
import torch
from training_class import CombinedLoss, CombinedLoss_dynamic
import os
import json
import datetime
import uuid
import random
from pathlib import Path
from types import SimpleNamespace
from torchsummary import summary
from list_run_ids import iter_run_configs, matches_filters

'This is the main file to run the training of the models. It includes functions to create unique run IDs, write configuration files, '
'and train both static and dynamic models.'

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
def static(predicted_time, a, lr, batch, epochs, channels, runs_root, resume_run_id_static, device, seed):
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
        write_run_config(run_dir, config)
    loss_fn = CombinedLoss(a=a, predicted_time=predicted_time, device=device)
    #loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

    model = PICNN_static(loss_fn=loss_fn, channels=channels).to(device)
    model_name = run_id
    dataset = HeatEquationMultiDataset(predicted_time=predicted_time)

    model.train_model(dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=model_root,
                      run_id=run_id, a=a, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)

def dynamic(a, lr, batch, epochs, channels, model_class, name, runs_root, resume_run_id, device, seed):

    # a function that takes all important parameter as input, creates and trains the model
    print('Create Combined loss')
    # CombinedLoss is a combination of MSE and Physics Loss  
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

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
        write_run_config(model_dir, config)

    # CombinedLoss is a combination of MSE and Physics Loss
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

    print(f'Create Dataset HeatEquationMultiDatset_dynamic')
    # this is for version 2 'training_class':
    dataset = HeatEquationMultiDataset_dynamic()
    model = model_class(c=channels).to(device)
    print(f'Train Model:')
    model.train_model(a=a, dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=model_root,
                      run_id=run_id, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    print('Okaaay - Let\'s go...')

    seed = None  # Set to an int for reproducible runs.
    if seed is None:
        seed = random.randint(1, 100000)
        print(f'No seed provided. Generated random seed = {seed}')
    else:
        print(f'Using provided seed = {seed}')
    torch.manual_seed(seed)

    run_mode = "dynamic"  # "static" | "dynamic"
    runs_root_static = "./runs/static"
    runs_root_dynamic = "./runs/dynamic"

    # Shared training parameters
    channels = 16
    lr = 0.001
    batch = 32 * 8
    epochs = 25

    # Static parameters
    a_list_static = [1, 0]
    predicted_times = [0.5, 3, 10]
    resume_run_ids_static = []  # z.B. ["static_20260101-120000_ab12cd", "static_20260102-130000_ef34gh"]
    auto_collect_static = False

    # Dynamic parameters
    a_list_dynamic = [0, 1]
    model_class_dynamic = PECNN_dynamic_latenttime1
    model_name_dynamic = "PECNN_dynamic_latenttime1"
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
            model_name="PECNN_dynamic_latenttime1",
        )
        print(f"Auto-collected dynamic run IDs: {resume_run_ids_dynamic}")

    if run_mode == "static":
        if resume_run_ids_static:
            for run_id in resume_run_ids_static:
                static(predicted_time=predicted_times[0], a=a_list_static[0], lr=lr, batch=batch, epochs=epochs,
                       channels=channels, runs_root=runs_root_static, resume_run_id_static=run_id,
                       device=device, seed=seed)
        else:
            for a in a_list_static:
                for t in predicted_times:
                    static(predicted_time=t, a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                           runs_root=runs_root_static, resume_run_id_static=None,
                           device=device, seed=seed)

    if run_mode == "dynamic":
        if resume_run_ids_dynamic:
            for run_id in resume_run_ids_dynamic:
                dynamic(a=a_list_dynamic[0], lr=lr, batch=batch, epochs=epochs, channels=channels,
                        model_class=model_class_dynamic, name=model_name_dynamic,
                        runs_root=runs_root_dynamic, resume_run_id=run_id,
                        device=device, seed=seed)
        else:
            for a in a_list_dynamic:
                dynamic(a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                        model_class=model_class_dynamic, name=model_name_dynamic,
                        runs_root=runs_root_dynamic, resume_run_id=None,
                        device=device, seed=seed)
