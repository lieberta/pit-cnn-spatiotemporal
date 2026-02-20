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


class HeatEquationPINNDataset(Dataset):
    """
    PINN dataset for heat-equation fields stored in normalized_heat_equation_solution.npz.
    Returns random collocation points and supervised temperatures from one experiment.
    """

    def __init__(self, base_path='./data/laplace_convolution/', points_per_sample=8192, modulo=1):
        self.points_per_sample = int(points_per_sample)
        self.files = []

        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]

        for i, folder in enumerate(sorted(folders)):
            if i % modulo != 0:
                continue
            npz_file_path = os.path.join(folder, 'normalized_heat_equation_solution.npz')
            if os.path.exists(npz_file_path):
                self.files.append(npz_file_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_file_path = self.files[idx]
        data = np.load(npz_file_path)['temperature']  # (nt, nx, ny, nz)
        nt, nx, ny, nz = data.shape

        n = self.points_per_sample
        t_idx = np.random.randint(0, nt, size=n)
        x_idx = np.random.randint(0, nx, size=n)
        y_idx = np.random.randint(0, ny, size=n)
        z_idx = np.random.randint(0, nz, size=n)

        t = t_idx / max(1, nt - 1)
        x = x_idx / max(1, nx - 1)
        y = y_idx / max(1, ny - 1)
        z = z_idx / max(1, nz - 1)

        coords = np.stack([x, y, z, t], axis=1).astype(np.float64)  # (n, 4)
        target = data[t_idx, x_idx, y_idx, z_idx].astype(np.float64).reshape(-1, 1)  # (n, 1)

        coords_tensor = torch.from_numpy(coords)
        target_tensor = torch.from_numpy(target)
        return coords_tensor, target_tensor

# 
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
