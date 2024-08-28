import torch
import re
import os
from dataset import HeatEquationMultiDataset, HeatEquationMultiDataset_dynamic
from models import PICNN_static, PECNN_dynamic
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from training_class import CombinedLoss_dynamic

# with  make_evaluation_table(...) makes a folder in ./plots/[Modell] and plots for each Testexperiment different
# timestep deviations if the Model for corresponding timestep exists
def denormalize(tensor):
    tensor_denorm = tensor * dist +min
    return tensor_denorm
def predict_and_denormalize(model, input_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
    # Assuming output and input_tensor are already on the appropriate device (CPU or GPU)
    output_denorm = denormalize(output)
    input_denorm = denormalize(input_tensor)
    return input_denorm, output_denorm

def find_fire(input):
    # Search for the fire source y coordinate
    # Iterate through [x, y, 0] coordinates
    found_fire = False
    for x_coord in range(input.shape[-3]):
        for y_coord in range(input.shape[-2]):
            # Check if temperature at [x, y, 0] is greater than 500
            if input[0,0, x_coord, y_coord, 0] > 21: # here we check the y coordinate that indicates the firesource, since we chose the starting conditions at the fire source to be 22. degrees
                y_fire = y_coord  # Save the y coordinate
                x_fire = x_coord
                found_fire = True
                break  # This breaks out of the inner loop
        if found_fire:
            break  # This breaks out of the outer loop
    return x_fire, y_fire




# Custom colormap creation
def create_custom_colormap():
    colors = [
        (0, 0, 0),  # Black for 0
        (1, 1, 0),  # Yellow for lower positive values
        (1, 0, 0),  # Red for higher positive values
        (1, 1, 1)   # White for maximum value
    ]
    nodes = [0.0, 0.33, 0.67, 1.0]  # Adjust these nodes to refine the color transitions
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    return cmap

# Use the custom colormap
cmap = create_custom_colormap()



# Assuming target, output, and deviation are 4D tensors in the format [batch, x, y, z]
def plot_heat_distribution(input_tensor, target, output, folder_path, predicted_time):
    # Assuming some previous code defines x_fire, y_fire
    x_fire, y_fire = find_fire(input_tensor)  # You need to implement this based on your specific logic

    # Calculate deviation
    deviation = torch.abs(target - output)

    # Plot Target
    plt.figure()
    plt.imshow(target[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=-20, vmax=1000)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Target at time {predicted_time}s')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'target_at_time_{predicted_time}.png'))
    plt.close()

    # Plot Output
    plt.figure()
    plt.imshow(output[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=-20, vmax=1000)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Output at time {predicted_time}s')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'output_at_time_{predicted_time}.png'))
    plt.close()

    # Plot Deviation
    plt.figure()
    plt.imshow(deviation[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=0, vmax=np.max(deviation.cpu().numpy()))  # Keep the max deviation color mapping as it was
    plt.colorbar(label='Deviation (°C)')
    plt.title(f'Deviation at time {predicted_time}')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'deviation_at_time_{predicted_time}.png'))
    plt.close()
def plot_heat_distribution(input_tensor, target, output, folder_path, predicted_time):
    # Assuming some previous code defines x_fire, y_fire
    x_fire, y_fire = find_fire(input_tensor)  # You need to implement this based on your specific logic

    # Calculate deviation
    deviation = torch.abs(target - output)

    # Plot Target
    plt.figure()
    plt.imshow(target[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=-20, vmax=1000)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Target at time {predicted_time}s')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'target_at_time_{predicted_time}.png'))
    plt.close()

    # Plot Output
    plt.figure()
    plt.imshow(output[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=-20, vmax=1000)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Output at time {predicted_time}s')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'output_at_time_{predicted_time}.png'))
    plt.close()

    # Plot Deviation
    plt.figure()
    plt.imshow(deviation[0, 0, :, y_fire, :].cpu().numpy().T, cmap=cmap, extent=(0, 64, 0, 16), origin='lower',
               vmin=0, vmax=np.max(deviation.cpu().numpy()))  # Keep the max deviation color mapping as it was
    plt.colorbar(label='Deviation (°C)')
    plt.title(f'Deviation at time {predicted_time}')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(folder_path, f'deviation_at_time_{predicted_time}.png'))
    plt.close()

# Plots a comparison between target and output

# makes a table that evaluates the mean deviation of a model for multiple experiments
# uses function plot_heat_distribution to make plots of a room slice output / target / deviation for all test Experiments
def make_evaluation_table(a=1,lr=0.001,epochs=50,batch=32,channels=16):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = './models'
    eval_data = []
    base_subdir_list = os.listdir(base_dir)
    # Define the regular expression pattern for matching 'predicted_time=' followed by a number
    pattern = r"predicted_time=(?P<predicted_time>[\d.]+)"

    # load the model
    model = PICNN_static(torch.nn.MSELoss(),channels).to(device).double()

    # List to hold the paths of the matching subfolders
    matching_subfolders_with_times = []

    for subdir in base_subdir_list: # subdir = modelnames
        print(f'subdir={subdir}')
        if f'predicted_time=' in subdir:
            # Search the directory name with the pattern to find 'predicted_time'
            match = re.search(pattern, subdir)
            if match:
                # Extract the 'predicted_time' value from the match object
                predicted_time = match.group('predicted_time')
                # Store the directory name along with the extracted predicted_time
                matching_subfolders_with_times.append((subdir, predicted_time)) # predicted_time t
                # print(f'predicted time ={predicted_time}')

            predicted_time_dir = os.path.join(base_dir, subdir) # ./models/predicted_time=t

            # iterate through the predicted_time subfolder
            for model_name in os.listdir(predicted_time_dir):
                if f'PICNN_predictedtime{predicted_time}s_loss={a}xPhysicsLoss+MSE_lr={lr}_epoch{epochs}_batch{batch}_channels={channels}' in model_name:

                    model_subdir = os.path.join(predicted_time_dir, model_name) # ./models/predicted_time=t/PICNN
                    model_pth = os.path.join(model_subdir, f'{model_name}.pth')
                    model_state_dict = torch.load(model_pth, map_location=device) # load statedictionary
                    try:
                        model.load_state_dict(model_state_dict)
                        print(f'Model = {model_name} loaded')
                    except Exception as e:
                        print(f"Failed to load model from {model_pth}: {e}")
                        break  # Skip that model, if dictionary won't match


                    dataset = HeatEquationMultiDataset(base_path=testset_path, predicted_time = float(predicted_time))
                    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)


                    # Evaluate the model
                    model_name_timeless = f'PICNN_loss={a}xPhysicsLoss+MSE_lr={lr}_epoch{epochs}_batch{batch}_channels={channels}'
                    eval_subdir = f'./plots/'+model_name_timeless
                    os.makedirs(eval_subdir, exist_ok=True) # generates a folder for one type of Architecture for multiple time steps
                    deviation_file_path = os.path.join(eval_subdir, 'deviations.txt')

                    with open(deviation_file_path, 'w') as deviation_file:
                        deviation_file.write("Mean Deviations:\n")

                        for experiment_idx, (input_tensor, target) in enumerate(dataloader):
                            input_tensor = input_tensor.to(device)
                            target = target.to(device)
                            target_denorm = denormalize(target).to(device)
                            input_denorm, output_denorm = predict_and_denormalize(model, input_tensor)


                            # calculate mean deviation:
                            mean_deviation = torch.mean(torch.abs(target - output_denorm)).item()

                            # Collect evaluation data
                            eval_data.append({
                                "Model": "PECNN",
                                "Predicted Second": predicted_time,
                                "Physics Loss Scalar": a,
                                "Epoch": epochs,
                                "Batch": batch,
                                "Learning Rate": lr,
                                "Experiment": f"Experimentindex {experiment_idx}",
                                "Mean Deviation": mean_deviation
                            })

                            deviation_str = f'Mean Deviation for {model_name}, Data {experiment_idx}: {mean_deviation}\n'
                            print(deviation_str.strip())

                            deviation_file.write(deviation_str)

                            heat_plot_folder = os.path.join(eval_subdir, f'experiment{experiment_idx}')
                            os.makedirs(heat_plot_folder, exist_ok=True)
                            plot_heat_distribution(input_denorm, target_denorm, output_denorm, heat_plot_folder,
                                                   predicted_time)
    # Convert collected data to DataFrame
    #df_eval = pd.DataFrame(eval_data)

    # Optionally, you can save this DataFrame to a CSV file for further analysis
    #df_eval.to_csv("model_evaluation_results.csv", index=False)

    # Define the save path
    #save_path = "./plots/"

    #csv_path = os.path.join(save_path, "model_evaluation_results.csv")
    #df_eval.to_csv(csv_path, index=False)

def make_evaluation_table_dynamic(model_name,model, a=1,lr=0.001,batch=256,channels=16):
    # calculates for a dynamic model the temp. mean deviation for the testset for times 1 to 10 in seconds
    # plots heat and error destributions


    base_dir = './models/dynamic'
    eval_data = []

    loss_fn = CombinedLoss_dynamic(a=a, device=device)
    # loss_choice = f'{a}xPhysicsLoss+MSE' # delete when no error
    print(f'Combined Loss created \n Load Model')

    # load model
    model = PECNN_dynamic(c=channels).to(device)

    dir = base_dir+'/'+model_name

    model_pth = os.path.join(dir, f'{model_name}.pth')
    model_state_dict = torch.load(model_pth, map_location=device) # load statedictionary
    try:
        model.load_state_dict(model_state_dict)
        print(f'Model = {model_name} loaded')
    except Exception as e:
        print(f"Failed to load model from {model_pth}: {e}")



    dataset = HeatEquationMultiDataset_dynamic(modulo=1,base_path=testset_path) # modulo=1 st we evaluate every experiment not only every 10th
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)


    # Evaluate the model

    eval_subdir = f'./plots/'+model_name
    os.makedirs(eval_subdir, exist_ok=True) # generates a folder for one type of Architecture for multiple time steps
    deviation_file_path = os.path.join(eval_subdir, 'deviations.txt')

    with open(deviation_file_path, 'w') as deviation_file:
        deviation_file.write("Mean Deviations:\n")

        experiment_idx = 0 # set the exp index = 1
        t_old = 999 # high value for start
        for _, (input_tuple, target) in enumerate(dataloader):
            input_tensor, t_tensor = input_tuple

            # Transform tensor to the actual value (timesteps are from 0.1 to 10 measured in seconds, 100 per experiment)
            t =t_tensor.item()
            t = round(t, 1) # rounds t to the first decimal because some t had the form 9.10000000000001
            print(f'We are looking at timestep: {t}')


            # make a deviation list for every t that gets evaluated
            deviation_time_dic = {}
            if t %1 == 0:
                if t not in deviation_time_dic:
                    deviation_time_dic[t] = []

                # change exp index each time the time value drops back to the inital value
                if t<t_old:
                    experiment_idx +=1

                input_tensor = input_tensor.to(device)
                target = target.to(device)
                target_denorm = denormalize(target).to(device)
                # predict the output:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    output = model(input_tensor.float(), t_tensor.float())
                # Assuming output and input_tensor are already on the appropriate device (CPU or GPU)
                output_denorm = denormalize(output)
                input_denorm = denormalize(input_tensor)


                # calculate absolute mean deviation:
                mean_deviation = torch.mean(torch.abs(target_denorm - output_denorm)).item()

                deviation_time_dic[t].append(mean_deviation) # append the deviation for the exact time

                # Collect evaluation data
                eval_data.append({
                    "Model": "PECNN",
                    "Physics Loss Scalar": a,
                    "Batch": batch,
                    "Learning Rate": lr,
                    "Experiment": f"Experimentindex {experiment_idx}",
                    "Time and Mean Deviation": f'(time={t}, mean_deviation)'
                })

                deviation_str = f'Mean Deviation for {model_name}, Data {experiment_idx}, Time {t}: {mean_deviation}\n'
                print(deviation_str.strip())

                deviation_file.write(deviation_str)

                heat_plot_folder = os.path.join(eval_subdir, f'experiment{experiment_idx}')
                os.makedirs(heat_plot_folder, exist_ok=True)
                plot_heat_distribution(input_denorm, target_denorm, output_denorm, heat_plot_folder, t)
                t_old=t     # set to compare these

def make_scalar_comparison(save_dir="./plots/meandeviationplots/",csv_path="./plots/model_evaluation_results.csv"):
    # compares different loss scalars a for static models

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_eval = pd.read_csv(csv_path)
    else:
        print(f"File not found or is empty: {csv_path}")
        # Handle the error, perhaps exit the function or raise an exception

    # Filter for epoch 50
    df_epoch50 = df_eval[df_eval['Epoch'] == 50]

    # Iterate over each unique experiment index
    for experiment_idx in df_epoch50['Experiment'].unique():
        # Filter the DataFrame for the current experiment index
        df_experiment = df_epoch50[df_epoch50['Experiment'] == experiment_idx]

        # Iterate over each unique predicted second within the current experiment
        for predicted_second in df_experiment['Predicted Second'].unique():
            # Further filter the DataFrame for the current predicted second
            df_filtered = df_experiment[df_experiment['Predicted Second'] == predicted_second]

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(df_filtered['Physics Loss Scalar'], df_filtered['Mean Deviation'])
            plt.title(
                f"Experiment: {experiment_idx} - Mean Deviation vs Physics Loss Scalar (Predicted Second: {predicted_second})")
            plt.xlabel("Physics Loss Scalar (a)")
            plt.ylabel("Mean Deviation")

            # Construct the save path dynamically to include both experiment index and predicted second
            formatted_experiment_idx = experiment_idx.replace(" ", "").replace("Experimentindex",
                                                                               "")  # Removing spaces and standardizing
            save_path = os.path.join(save_dir,
                                     f"Exp{formatted_experiment_idx}_Epoch50_PredictedSec{predicted_second}_MeanDeviationPlot.png")

            plt.savefig(save_path)
            plt.close()  # Close the plot to free memory

    print("Plots for each experiment index and predicted second generated and saved.")



if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    testset_path = './data/testset/'
    train_path ='./data/laplace_convolution/'
    with open(testset_path + 'normalization_values.json', 'r') as json_file:
        normalization_values = json.load(json_file)
        max_temp = normalization_values['max_temp']
        min_temp = normalization_values['min_temp']

    min = min_temp
    dist = max_temp - min



    #make_evaluation_table(a=1,lr=0.001,epochs=100,batch=32,channels=16)
    # make_scalar_comparison()



    channels = 16
    lr = 0.001
    batch = 32 * 8
    #epochs = 10
    a = 1
    loss_fn = CombinedLoss_dynamic(a=a, device=device)
    autodiff = False


    # modelname list: pick 0 for group normalization, 1 for batchnormalization, 2 for gn and no time normalization
    mn_list = [f'PECNN_dynamic_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}',
               f'PECNN_dynamic_batchnorm_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}',
               f'PECNN_dynamic_no_time_normalization_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}',
               f'PECNN_dynamic_smalltest_loss={a}xPhysicsLoss+MSE_lr={lr}_batch{batch}_channels={channels}_autodiff={autodiff}']

    # dataloader uses only 1/10 of the actual data!!!!!!!!!! -> small dataset
    name = mn_list[3]
    model = PECNN_dynamic(c=channels).to(device)
    # pick 0 for group normalization, 1 for batchnormalization, 2 for gn and no time normalization
    make_evaluation_table_dynamic(name, model, a,lr,batch,channels)


