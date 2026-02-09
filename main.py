
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

# Function to train the static model. It sets up the parameters, creates the model and dataset, and trains the model while saving the configuration.
def static():
    lr = 0.001 # 0.001 for CNN1D3D, 0.0001 for CNN1D
    batch = 32   # open for testing
    epochs = 50


    a_list = [1]
    predicted_time_list= [1]
    lr_list = [0.001]
    channels=16

    runs_root = "./runs/static"
    for predicted_time in predicted_time_list:
        for a in a_list:
            run_id = make_run_id("static")
            run_dir = os.path.join(runs_root, run_id)
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
            loss_fn = CombinedLoss(a=a,predicted_time=predicted_time,device=device)
            #loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

            model = PICNN_static(loss_fn=loss_fn,channels=channels).to(device)
            model_name = run_id
            dataset = HeatEquationMultiDataset(predicted_time=predicted_time)

            model.train_model(dataset = dataset, num_epochs= epochs,batch_size= batch,
                              learning_rate=lr, model_name=model_name, save_path=runs_root,
                              run_id=run_id, a=a, channels=channels, seed=seed)

def dynamic():

    # a function that takes all important parameter as input, creates and trains the model
    print('Create Combined loss')
    # CombinedLoss is a combination of MSE and Physics Loss  
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

    run_id = resume_run_id or make_run_id("dynamic")
    model_name = run_id
    model_dir = os.path.join(path, model_name)
    model_pth = os.path.join(model_dir, f"{model_name}.pth")

    # Check if the model directory and the specific model file exist
    if os.path.exists(model_dir) and os.path.isfile(model_pth):
        # Load the model
        model.load_state_dict(torch.load(model_pth))
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

    # loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

    print(f'Create Dataset HeatEquationMultiDatset_dynamic')
    # this is for version 2 'training_class':
    dataset = HeatEquationMultiDataset_dynamic()
    print(f'Train Model:')
    model.train_model(a=a, dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=path,
                      run_id=run_id, channels=channels, seed=seed)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    print('Okaaay - Let\'s go...')

    seed = 42
    torch.manual_seed(seed)

    channels = 16
    lr = 0.001
    batch = 32 * 8
    epochs = 10
    a = 1
    # dataloader uses only 1/10 of the actual data!!!!!!!!!! -> small dataset
    name = f'PECNN_TIMEFIRST.V1.1_dynamic_smalltest'
    model = PECNN_dynamic_timefirst(c=channels).to(device)
    path = f'./runs/dynamic'
    resume_run_id = None

    for a in [1,0]:
        dynamic()
