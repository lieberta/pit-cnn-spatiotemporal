
from models import PICNN_static
from dataset import HeatEquationMultiDataset
import torch
from training_class import CombinedLoss
import os

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device ={device}')
    print('Okaaay - Let\'s go...')



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
            model_name = f'PICNN_predictedtime{predicted_time}s_loss={a}xPhysicsLoss+MSE_lr={lr}_epoch{epochs}_batch{batch}_channels={channels}'
            dataset = HeatEquationMultiDataset(predicted_time=predicted_time)

            model.train_model(dataset = dataset, num_epochs= epochs,batch_size= batch,
                              learning_rate=lr,model_name=model_name,save_path=path)

