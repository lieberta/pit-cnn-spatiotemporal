import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time
from datetime import datetime

class Laplacian3D(nn.Module):
    def __init__(self, dx, dy, dz, device):
        super(Laplacian3D, self).__init__()
        # Elements for the central cell
        central_value = -2 * ((1 / (dx ** 2)) + (1 / (dy ** 2)) + (1 / (dz ** 2)))

        # Create an empty kernel with zeros
        kernel = torch.zeros((3, 3, 3), dtype=torch.float32)

        # Set the central cell
        kernel[1, 1, 1] = central_value

        # Set the directly adjacent cells for each dimension
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1 / (dz ** 2)  # Front and Back
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1 / (dy ** 2)  # Top and Bottom
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1 / (dx ** 2)  # Left and Right

        # Move the kernel to the specified device and add required dimensions for conv3d
        kernel = kernel.to(device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3, 3]

        self.register_buffer('kernel', kernel)

    def forward(self, x):
        padding = 1  # Use padding=1 to maintain the dimensionality
        laplacian = F.conv3d(x, self.kernel, padding=padding, groups=1)
        return laplacian


class HeatSimulation:
    def __init__(self, num_fires, alpha=0.0257, Lx=6.3, Ly=3.1, Lz=1.5, Nx=64, Ny=32, Nz=16, T=10.,
                 Nt=10000, device='cpu'):
        self.num_fires = num_fires
        self.device = device
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.T, self.Nt = T, Nt
        self.dx, self.dy, self.dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)
        self.dt = T / Nt

        self.u = torch.zeros((Nt + 1, Nx, Ny, Nz), dtype=torch.float32, device=device)
        self.laplacian3D = Laplacian3D(self.dx, self.dy, self.dz, device)

        self.fireplaces = self.create_fireplace_experiments(self.num_fires)
        self.setup_initial_conditions()
        self.setup_source_term()

    @staticmethod
    def place_fireplace(max_x, max_y):
        width = random.randint(1, 4)
        height = random.randint(1, 4)
        top_left_x = random.randint(0, max_x - width)
        top_left_y = random.randint(0, max_y - height)
        return (top_left_x, top_left_y, width, height)

    def create_fireplace_experiments(self, num_fires):
        max_x, max_y = self.Nx, self.Ny
        fireplaces = [self.place_fireplace(max_x, max_y) for _ in range(num_fires)]
        return fireplaces

    def setup_initial_conditions(self):
        self.u[0, :, :, :] = 20.0
        for fireplace in self.fireplaces:
            x_start, y_start, x_size, y_size = fireplace
            self.u[0, x_start:x_start + x_size, y_start:y_start + y_size, :2] = 1100.0

    def setup_source_term(self):
        self.source_term = torch.zeros_like(self.u[0], dtype=torch.float32, device=self.device)
        for fireplace in self.fireplaces:
            x_start, y_start, x_size, y_size = fireplace
            self.source_term[x_start:x_start + x_size, y_start:y_start + y_size, :2] = 100000.0

    def run_simulation(self):
        self.tic = time.perf_counter()
        for n in range(self.Nt):
            laplacian = self.laplacian3D(self.u[n:n + 1])
            self.u[n + 1] = self.u[n] + self.alpha * self.dt * laplacian + self.dt * self.source_term

            # Update boundary conditions
            self.u[n + 1, 0, :, :] = 20.0
            self.u[n + 1, self.Nx - 1, :, :] = 20.0
            self.u[n + 1, :, 0, :] = 20.0
            self.u[n + 1, :, self.Ny - 1, :] = 20.0
            self.u[n + 1, :, :, 0] = 20.0
            self.u[n + 1, :, :, self.Nz - 1] = 20.0

        self.tac = time.perf_counter()

        # Assuming self.u is a PyTorch tensor
        self.min_temp = np.min(self.u.cpu().numpy())
        self.max_temp = np.max(self.u.cpu().numpy())

    def save_results(self):
        temperature_np = self.u[::100].cpu().numpy()

        # Initialize a variable to hold information for all fireplaces
        all_fireplaces_info = ""
        # Extract the location and dimensions of the first fireplace

        if self.fireplaces:
            for i, fireplace in enumerate(self.fireplaces):
                x_start, y_start, x_size, y_size = fireplace
                # Append each fireplace's info to the string
                all_fireplaces_info += (f'Fireplace {i + 1}: Location: x={x_start}, y={y_start}, '
                                        f'Dimensions: width={x_size}, depth={y_size}\n')


            simulation_info = (f'\nSimulated Time ={self.T}s, dt ={self.dt}s, '
                               f'amount of simulated timesteps ={self.Nt}, Alpha ={self.alpha} '
                                f'amount of saved timesteps ={temperature_np.shape[0]}'
                               f'\nLx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, '
                               f'Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}'
                               f'\nmin Temperature={self.min_temp}, max Temperature={self.max_temp}'
                               f'\nProcessing time={self.tac - self.tic}s')
        else:
            simulation_info = 'No fireplace information'

        full_info = all_fireplaces_info + simulation_info

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        #experiment_folder = f'./data/laplace_convolution/experiment_{self.num_fires}_{current_time}'
        experiment_folder = f'./data/testset/experiment_{self.num_fires}_{current_time}'
        os.makedirs(experiment_folder, exist_ok=True)
        np.savez(f'{experiment_folder}/heat_equation_solution.npz', time=np.linspace(0, self.T, self.Nt + 1),
                 x=np.linspace(0, self.Lx, self.Nx), y=np.linspace(0, self.Ly, self.Ny),
                 z=np.linspace(0, self.Lz, self.Nz), temperature=temperature_np)

        with open(f'{experiment_folder}/fireplace_simulation_results.txt', 'w') as file:
            file.write(full_info)


def run_experiment(num_fires, device):

    simulation = HeatSimulation(num_fires, device=device)
    simulation.run_simulation()
    simulation.save_results()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for num_fires in [3,4,5,6,7,8,9]:  # Number of fireplaces
        print(f'Number of fires: {num_fires}')
        for i in range(1):
            print(f'Calculating experiment number {i}...')
            run_experiment(num_fires, device)