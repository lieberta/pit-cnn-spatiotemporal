import numpy as np
import torch
import os


base_path = './data/laplace_convolution/'
# create list of folders
folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
           os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]

inputs = []
targets = []

# load normalized data
for folder in folders:
    npz_file_path = os.path.join(folder, 'normalized_heat_equation_solution.npz')
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path)['temperature']

        for predicted_time in range(1,101): # these predicted times are 0.1 seconds long
            input = [torch.tensor(data[0, :, :, :], dtype=torch.float64).unsqueeze(0), predicted_time]
            target = torch.tensor(data[int(predicted_time), :, :, :], dtype=torch.float64).unsqueeze(0) # predicted second*10 since 10 timesteps in the data equals 1 second



            inputs.append(input)
            targets.append(target)



print(len(inputs))

# Concatenate all inputs and targets from different files
# inputs = torch.cat(inputs, dim=0)
# targets = torch.cat(targets, dim=0)
