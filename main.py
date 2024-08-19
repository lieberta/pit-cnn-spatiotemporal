
from models import PICNN_static, PECNN_dynamic, PECNN_dynamic_batchnorm, PECNN_dynamic_timefirst
from dataset import HeatEquationMultiDataset, HeatEquationMultiDataset_dynamic
import torch
from training_class import CombinedLoss, CombinedLoss_dynamic
import os

def static():
    lr = 0.001 # 0.001 for CNN1D3D, 0.0001 for CNN1D
    batch = 32   # open for testing
    epochs = 50


    a_list = [1]
    predicted_time_list= [1]
    lr_list = [0.001]
    channels=16

    for predicted_time in predicted_time_list:
        path = f'./models/predicted_time={predicted_time}'
        os.makedirs(path, exist_ok=True)
        for a in a_list:
            loss_fn = CombinedLoss(a=a,predicted_time=predicted_time,device=device)
            #loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

            model = PICNN_static(loss_fn=loss_fn,channels=channels).to(device)
            # this is for version 2 'training_class':
            model_name = f'PICNN_predictedtime{predicted_time}s_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}'
            dataset = HeatEquationMultiDataset(predicted_time=predicted_time)

            model.train_model(dataset = dataset, num_epochs= epochs,batch_size= batch,
                              learning_rate=lr,model_name=model_name,save_path=path)

def dynamic():

    # a function that takes all important parameter as input, creates and trains the model
    print('Create Combined loss')
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

    model_name = name+f'_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}'
    model_dir = path+'/'+ model_name            # path to the model_dictionary
    model_pth = model_dir+'/'+model_name+'.pth' # path to the pth file

    # Check if the model directory and the specific model file exist
    if os.path.exists(model_dir) and os.path.isfile(model_pth):
        # Load the model
        model.load_state_dict(torch.load(model_pth))
        print(f"The model '{model_name}' already exists and will be trained further.")
    else:
        # If the directory or model file doesn't exist, print this message
        print(f"A new folder and model '{model_name}' will be created.")

    # loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error

    print(f'Create Dataset HeatEquationMultiDatset_dynamic')
    # this is for version 2 'training_class':
    dataset = HeatEquationMultiDataset_dynamic()
    print(f'Train Model:')
    model.train_model(dataset=dataset, num_epochs=epochs, batch_size=batch,
                      learning_rate=lr, model_name=model_name, save_path=path)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    print('Okaaay - Let\'s go...')

    channels = 16
    lr = 0.001
    batch = 32 * 8
    epochs = 10
    a = 0
    loss_fn = CombinedLoss_dynamic(a=a, device=device)

    # dataloader uses only 1/10 of the actual data!!!!!!!!!! -> small dataset
    name = f'PECNN_dynamic_smalldataset_no_time_normalization_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}'
    model = PECNN_dynamic(loss_fn=loss_fn, c=channels).to(device)
    path = f'./models/dynamic'

    dynamic()
