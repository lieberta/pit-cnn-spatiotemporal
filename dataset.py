import torch
from torch.utils.data import Dataset
import numpy as np
import os

def list_experiment_folders(base_path):
    """
    List all subfolders in the base_path starting with 'experiment'.
    """
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]
    return folders


class HeatEquationMultiDataset(Dataset):
    def __init__(self, base_path='./data/laplace_convolution/', predicted_time=3):

        # create list of folders
        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]

        self.inputs = []
        self.targets = []

        # load normalized data
        for folder in folders:
            npz_file_path = os.path.join(folder, 'normalized_heat_equation_solution.npz')
            if os.path.exists(npz_file_path):
                data = np.load(npz_file_path)['temperature']

                inputs = torch.tensor(data[0, :, :, :], dtype=torch.float64).unsqueeze(0).unsqueeze(1)

                targets = torch.tensor(data[int(predicted_time*10), :, :, :], dtype=torch.float64).unsqueeze(0).unsqueeze(1) # predicted second*10 since 10 timesteps in the data equals 1 second

                self.inputs.append(inputs)
                self.targets.append(targets)



        # Concatenate all inputs and targets from different files
        self.inputs = torch.cat(self.inputs, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



class HeatEquationMultiDataset_dynamic_memoryintensive(Dataset):
    def __init__(self, base_path='./data/laplace_convolution/'):

        # Create list of folders
        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]

        self.inputs = []
        self.targets = []

        # Load normalized data
        for folder in folders:
            npz_file_path = os.path.join(folder, 'normalized_heat_equation_solution.npz')
            if os.path.exists(npz_file_path):
                data = np.load(npz_file_path)['temperature']

                for predicted_time in range(1, 101):  # These predicted times are 0.1 seconds long
                    input_tensor = torch.tensor(data[0, :, :, :], dtype=torch.float64).unsqueeze(0)
                    self.inputs.append((input_tensor, predicted_time*0.1))          # predicted_time*0.1 because my timesteps are 0.1 and my predicted time iterates from 1 to 100
                    target_tensor = torch.tensor(data[int(predicted_time), :, :, :], dtype=torch.float64).unsqueeze(0)
                    self.targets.append(target_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor, predicted_time = self.inputs[idx]
        target_tensor = self.targets[idx]
        return (input_tensor, predicted_time), target_tensor

class HeatEquationMultiDataset_dynamic(Dataset):
    def __init__(self, modulo=10, base_path='./data/laplace_convolution/'):
        # Create list of folders
        self.files = []
        self.data_cache = {}            # new with cache
        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]
        i=0 # count variable
        # Collect file paths and time steps
        for folder in folders:
            if i%modulo==0: # take only every tenth folder
                npz_file_path = os.path.join(folder, 'normalized_heat_equation_solution.npz')
                if os.path.exists(npz_file_path):
                    data = np.load(npz_file_path)['temperature']
                    num_timesteps = data.shape[0]

                    for predicted_time in range(1, num_timesteps):  # assuming data has time steps as the first dimension
                        self.files.append((npz_file_path, predicted_time, num_timesteps))
            i += 1  # count up the variable

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_file_path, predicted_time, num_timesteps = self.files[idx]
        if npz_file_path not in self.data_cache:                                        # new with cache
            self.data_cache.clear()                                                     # new with cache clears the cache dictionary to not overload the memory
            self.data_cache[npz_file_path] = np.load(npz_file_path)['temperature']      # new with cache
        data = self.data_cache[npz_file_path] # new with cache, old: data = np.load(npz_file_path)['temperature']
        input_tensor = torch.tensor(data[0, :, :, :], dtype=torch.float64).unsqueeze(0)
        target_tensor = torch.tensor(data[predicted_time, :, :, :], dtype=torch.float64).unsqueeze(0)
        predicted_time_tensor = torch.tensor([predicted_time * 0.1], dtype=torch.float64)  # Convert predicted time to tensor
        return (input_tensor, predicted_time_tensor), target_tensor


if __name__ == '__main__':

    dataset = HeatEquationMultiDataset_dynamic(base_path=f'data/testset')

    for x,y in dataset:
        print(f'{x[0].shape} \n {x[1].shape} \n {y.shape}')
        break

    # Now, to get the shape of the inputs and targets, you can do:
    #input_shape = dataset.inputs.shape
    #target_shape = dataset.targets.shape

    print("Inputs shape:", len(dataset))

    #print("Targets shape:", target_shape)