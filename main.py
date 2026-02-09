
from models import PICNN_static, PECNN_dynamic, PECNN_dynamic_batchnorm, PECNN_dynamic_timefirst
from dataset import HeatEquationMultiDataset, HeatEquationMultiDataset_dynamic
import torch
from training_class import CombinedLoss, CombinedLoss_dynamic
import os
import json
import datetime
import uuid
from torchsummary import summary

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

# Function to train the static model. It sets up the parameters, creates the model and dataset, and trains the model while saving the configuration.
def static(predicted_time, a, lr, batch, epochs, channels, runs_root, resume_run_id_static, device, seed):
    resume_checkpoint_path = None
    if resume_run_id_static:
        run_dir = os.path.join(runs_root, resume_run_id_static)
        config = load_run_config(run_dir)
        if config:
            lr = config.get("lr", lr)
            batch = config.get("batch", batch)
            epochs = config.get("epochs", epochs)
            channels = config.get("channels", channels)
            a = config.get("a", a)
            predicted_time = config.get("predicted_time", predicted_time)
        resume_checkpoint_path = os.path.join(run_dir, f"{resume_run_id_static}.pth")


    run_id = resume_run_id_static or make_run_id("static")
    run_dir = os.path.join(runs_root, run_id)
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
                      learning_rate=lr, model_name=model_name, save_path=runs_root,
                      run_id=run_id, a=a, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)

def dynamic(a, lr, batch, epochs, channels, model_class, name, runs_root, resume_run_id, device, seed):

    # a function that takes all important parameter as input, creates and trains the model
    print('Create Combined loss')
    # CombinedLoss is a combination of MSE and Physics Loss  
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

    run_id = resume_run_id or make_run_id("dynamic")
    model_name = run_id
    model_dir = os.path.join(runs_root, model_name)
    model_pth = os.path.join(model_dir, f"{model_name}.pth")

    resume_checkpoint_path = None
    # Check if the model directory and the specific model file exist
    if os.path.exists(model_dir) and os.path.isfile(model_pth):
        resume_checkpoint_path = model_pth
        config = load_run_config(model_dir)
        if config:
            lr = config.get("lr", lr)
            batch = config.get("batch", batch)
            epochs = config.get("epochs", epochs)
            channels = config.get("channels", channels)
            a = config.get("a", a)
            name = config.get("name", name)
        print(f"The model '{model_name}' already exists and will be trained further.")
    else:
        # If the directory or model file doesn't exist, print this message
        print(f"A new folder and model '{model_name}' will be created.")
        config = {
            "run_id": run_id,
            "model_class": model.__class__.__name__,
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
                      learning_rate=lr, model_name=model_name, save_path=runs_root,
                      run_id=run_id, channels=channels, seed=seed,
                      resume_checkpoint_path=resume_checkpoint_path)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    print('Okaaay - Let\'s go...')

    seed = 3141
    torch.manual_seed(seed)

    run_mode = "static"  # "static" | "dynamic"
    runs_root_static = "./runs/static"
    runs_root_dynamic = "./runs/dynamic"

    # Shared training parameters
    channels = 16
    lr = 0.001
    batch = 32 * 8
    epochs = 10

    # Static parameters
    a_list_static = [1, 0]
    predicted_times = [0.5, 3, 10]
    resume_run_id_static = None

    # Dynamic parameters
    a_list_dynamic = [1, 0]
    model_class_dynamic = PECNN_dynamic
    model_name_dynamic = "PECNN_dynamic_smalldatasettest"
    resume_run_id_dynamic = None

    if run_mode == "static":
        for a in a_list_static:
            for t in predicted_times:
                static(predicted_time=t, a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                       runs_root=runs_root_static, resume_run_id_static=resume_run_id_static,
                       device=device, seed=seed)

    if run_mode == "dynamic":
        for a in a_list_dynamic:
            dynamic(a=a, lr=lr, batch=batch, epochs=epochs, channels=channels,
                    model_class=model_class_dynamic, name=model_name_dynamic,
                    runs_root=runs_root_dynamic, resume_run_id=resume_run_id_dynamic,
                    device=device, seed=seed)
