import os
import numpy as np
import matplotlib.pyplot as plt




# Function to create a subfolder if it doesn't exist
def create_subfolder(folder_path, subfolder_name):
    subfolder_path = os.path.join(folder_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    return subfolder_path

# Load the .npz file
#experiment_number = 0  # Replace with the experiment number you want to plot
experiment_folder = f'./data/testset/experiment_1_20240229_125217'



npz_file_path = f'{experiment_folder}/heat_equation_solution.npz'

# Check if the .npz file exists
if not os.path.exists(npz_file_path):
    print(f'Error: The .npz file for experiment {experiment_number} does not exist.')
else:
    # Load data from the .npz file
    data = np.load(npz_file_path)
    time = data['time']
    x = data['x']
    y = data['y']
    z = data['z']
    temperature = data['temperature']

    # Find global minimum and maximum temperature values for chosing colors
    global_min_temp = np.min(temperature)
    global_max_temp = 500#np.max(temperature)

    # Search for the fire source y coordinate
    # Iterate through [x, y, 0] coordinates
    found_fire = False
    for x_coord in range(x.shape[0]):
        for y_coord in range(y.shape[0]):
            # Check if temperature at [x, y, 0] is greater than 500
            if temperature[0, x_coord, y_coord, 0] > 21: # here we check the y coordinate that indicates the firesource, since we chose the starting conditions at the fire source to be 22. degrees
                y_fire = y_coord  # Save the y coordinate
                found_fire = True
                break  # This breaks out of the inner loop
        if found_fire:
            break  # This breaks out of the outer loop


    # Create a subfolder for saving plots
    plots_folder = create_subfolder(experiment_folder, 'plots')

    # Plot and save the x-z slice for each timestep
    for timestep in range(data['temperature'].shape[0]):
        if timestep % 10 == 0:  # Check if the timestep is a multiple of 10
            plt.figure()
            plt.imshow(temperature[timestep, :, y_fire, :].T, cmap='hot', extent=(0, 64, 0, 16), origin='lower', vmin=global_min_temp, vmax=global_max_temp) #this is for stable experiments
            plt.colorbar(label='Temperature (°C)')
            plt.title(f'Timestep {time[timestep*100]}s') # times 100 because only every 100th timestep has been saved
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.savefig(f'{plots_folder}/timestep_{timestep}.png')
            plt.close()

    print('Plots have been saved in the "plots" subfolder.')