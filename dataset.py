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

if __name__ == '__main__':

    dataset = HeatEquationMultiDataset(base_path=f'data/testset')

    # Now, to get the shape of the inputs and targets, you can do:
    input_shape = dataset.inputs.shape
    target_shape = dataset.targets.shape

    print("Input shape:", input_shape)
    print("Target shape:", target_shape)