import numpy as np
import torch
# Define the path to your .npz file
path = './data/testset/experiment_1_20240229_125217/normalized_heat_equation_solution.npz'

path2= './data/laplace_convolution/experiment_7_20240224_043930/heat_equation_solution.npz'

# Load the .npz file
data = np.load(path)

# Assuming 'temperature' is the key for your data array in the npz file
if 'temperature' in data:
    temperature = data['temperature']
    # The number of timesteps is the size of the first dimension of the array
    num_timesteps = temperature.shape[0]
    print("Number of timesteps:", num_timesteps)
else:
    print("The 'temperature' array was not found in the npz file.")

# Close the file
data.close()

predicted_time = 10
print(f'timestep={int(predicted_time*10)}')


data = np.load(path)['temperature']

targets = torch.tensor(data[int(predicted_time*10), :, :, :], dtype=torch.float64).unsqueeze(0).unsqueeze(1) # predicted second*10 since 10 timesteps in the data equals 1 second
