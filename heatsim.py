import numpy as np
import os
import random
import time
from datetime import datetime



class HeatSimulation:
    def __init__(self, experiment_number, fireplaces, alpha=0.0257, Lx=6.3, Ly=3.1, Lz=1.5, Nx=64, Ny=32, Nz=16, T=3., Nt=3000): # alpha is suitable for air
        self.experiment_number = experiment_number
        self.fireplaces = fireplaces
        self.alpha = alpha
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.T, self.Nt = T, Nt
        self.dx, self.dy, self.dz = Lx/(Nx-1), Ly/(Ny-1) , Lz/(Nz-1)
        self.dt = T / Nt

        self.u = np.zeros((Nt + 1, Nx, Ny, Nz))
        self.setup_initial_conditions()
        self.setup_source_term()

    def setup_initial_conditions(self):
        self.u[0, :, :, :] = 20.0
        for fireplace in self.fireplaces:
            x_start, y_start, x_size, y_size = fireplace
            self.u[0, x_start:x_start+x_size, y_start:y_start+y_size, :2] = 1100.0 # fire with a z thickness of 1

    def setup_source_term(self):
        self.source_term = np.zeros_like(self.u[0])
        for fireplace in self.fireplaces:
            x_start, y_start, x_size, y_size = fireplace
            self.source_term[x_start:x_start + x_size, y_start:y_start + y_size, :2] = 100000.0 # Temperatur pro Sekunde

    def run_simulation(self):
        self.tic = time.perf_counter()  # Start time
        for n in range(self.Nt):
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    for k in range(1, self.Nz - 1):
                        laplacian = (
                                (self.u[n, i + 1, j, k] - 2 * self.u[n, i, j, k] + self.u[
                                    n, i - 1, j, k]) / self.dx ** 2 +
                                (self.u[n, i, j + 1, k] - 2 * self.u[n, i, j, k] + self.u[
                                    n, i, j - 1, k]) / self.dy ** 2 +
                                (self.u[n, i, j, k + 1] - 2 * self.u[n, i, j, k] + self.u[
                                    n, i, j, k - 1]) / self.dz ** 2
                        )




                        self.u[n + 1, i, j, k] = self.u[n, i, j, k] + self.alpha * self.dt * laplacian + self.dt * self.source_term[i, j, k]

            # Update boundary conditions
            # For x boundaries
            self.u[n + 1, 0, :, :] = 20.0
            self.u[n + 1, self.Nx - 1, :, :] = 20.0

            # For y boundaries
            self.u[n + 1, :, 0, :] = 20.0
            self.u[n + 1, :, self.Ny - 1, :] = 20.0

            # For z boundaries
            self.u[n + 1, :, :, 0] = 20.0
            self.u[n + 1, :, :, self.Nz - 1] = 20.0

        self.tac = time.perf_counter()
        self.min_temp = np.min(self.u)
        self.max_temp = np.max(self.u)
            # make fireplace temperature constant for all timesteps
            #for fireplace in self.fireplaces:
            #    x_start, y_start, x_size, y_size = fireplace
            #    self.u[n, x_start:x_start + x_size, y_start:y_start + y_size, :2] = 1100.0

    def save_results(self):
        # Initialize a variable to hold information for all fireplaces
        all_fireplaces_info = ""
        # Extract the location and dimensions of the first fireplace
        if self.fireplaces:
            # Iterate over each fireplace in the list
            for i, fireplace in enumerate(self.fireplaces):
                x_start, y_start, x_size, y_size = fireplace
                # Append each fireplace's info to the string
                all_fireplaces_info += (f'Fireplace {i + 1}: Location: x={x_start}, y={y_start}, '
                                        f'Dimensions: width={x_size}, depth={y_size}\n')

                # Combine the fireplaces info with the simulation data
            simulation_info = (f'\nSimulated Time ={self.T}s, dt ={self.dt}s, '
                               f'amount of simulated timesteps ={self.Nt}, Alpha ={self.alpha}'
                               f'\nLx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, '
                               f'Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}'
                               f'\nmin Temperature={self.min_temp}, max Temperature={self.max_temp}'
                               f'\nProcessing time={self.tac - self.tic}s')
        else:
            location_info = 'No information'
        # Combine all information into one string
        full_info = all_fireplaces_info + simulation_info

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        experiment_folder = f'./data/forloops/testset/experiment_{self.experiment_number}_{current_time}'

        os.makedirs(experiment_folder, exist_ok=True)
        np.savez(f'{experiment_folder}/heat_equation_solution.npz', time=np.linspace(0, self.T, self.Nt + 1),
                 x=np.linspace(0, self.Lx, self.Nx), y=np.linspace(0, self.Ly, self.Ny),
                 z=np.linspace(0, self.Lz, self.Nz), temperature=self.u,minmax_temp=[self.min_temp, self.max_temp])
        # Here, you can save full_info to a file or handle it however you prefer
        # Example: Save to a text file
        with open(f'{experiment_folder}fireplace_simulation_results.txt', 'w') as file:
            file.write(full_info)


def place_fireplace(max_x, max_y):
    width = random.randint(1, 4)
    height = random.randint(1, 4)
    top_left_x = random.randint(0, max_x - width)
    top_left_y = random.randint(0, max_y - height)
    return (top_left_x, top_left_y, width, height)

def create_fireplace_experiments(num_experiments, num_fires):
    results = []
    max_x = 64
    max_y = 32
    for _ in range(num_experiments):
        fireplaces = [place_fireplace(max_x, max_y) for _ in range(num_fires)]
        results.append(fireplaces)
    return results

def run_experiments(num_experiments, num_fires):
    random_fireplaces = create_fireplace_experiments(num_experiments, num_fires)


    # Run and save results for single fireplace experiments
    for i, fireplaces in enumerate(random_fireplaces):
        simulation = HeatSimulation(experiment_number=i, fireplaces=fireplaces)
        simulation.run_simulation()
        simulation.save_results()



if __name__ == "__main__":


    num_experiments = 1
    num_fires = 6
    run_experiments(num_experiments, num_fires)

