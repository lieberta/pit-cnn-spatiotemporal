import os
import json
import numpy as np
import torch
try:
    import zarr
except ImportError:
    zarr = None

SIM_TOTAL_SECONDS = 10.0
SIM_STEPS_PER_SECOND = 1000.0
SECONDS_PER_STEP = 1.0 / SIM_STEPS_PER_SECOND
# Create global normalization values:
def norm_values(base_path):
    global_max = 0.
    global_min = 200000.
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
               os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]
    for folder in folders:
        zarr_path = os.path.join(folder, 'heat_equation_solution.zarr')
        npz_file_path = os.path.join(folder, 'heat_equation_solution.npz')
        if os.path.exists(zarr_path):
            if zarr is None:
                raise ImportError("zarr is required to read heat_equation_solution.zarr")
            temperature = np.asarray(zarr.open_group(zarr_path, mode="r")["temperature"])
        elif os.path.exists(npz_file_path):
            temperature = np.load(npz_file_path)['temperature']
        else:
            continue
        local_max = np.max(temperature)
        local_min = np.min(temperature)
        if global_max < local_max:
            global_max = local_max
        if global_min > local_min:
            global_min = local_min

    # Now, save these global max and min values to a JSON file
    normalization_values = {'max_temp': float(global_max), 'min_temp': float(global_min)}
    with open(os.path.join(base_path, 'normalization_values.json'), 'w') as json_file:
        json.dump(normalization_values, json_file)



def normalization(base_path, delete_raw_after_normalization=True):
    # Load global normalization values
    with open(os.path.join(base_path, 'normalization_values.json'), 'r') as json_file:
        normalization_values = json.load(json_file)
        max_temp = normalization_values['max_temp']
        min_temp = normalization_values['min_temp']
    # create list of subfolders
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('experiment')]
    # Second pass to load, normalize data, and save if not already done
    for folder in folders:
        zarr_path = os.path.join(folder, 'heat_equation_solution.zarr')
        npz_file_path = os.path.join(folder, 'heat_equation_solution.npz')
        normalized_npz_filepath = os.path.join(folder, 'normalized_heat_equation_solution.npz')

        # Check if the normalized data file already exists
        if (os.path.exists(npz_file_path) or os.path.exists(zarr_path)) and not os.path.exists(normalized_npz_filepath):
            if os.path.exists(zarr_path):
                if zarr is None:
                    raise ImportError("zarr is required to read heat_equation_solution.zarr")
                data = np.asarray(zarr.open_group(zarr_path, mode="r")["temperature"])
            else:
                data = np.load(npz_file_path)['temperature']
            # Normalize the data using global min and max
            normalized_data = (data - min_temp) / (max_temp - min_temp)

            # Optional: Adjust data as needed
            #normalized_data = normalized_data[[0, -2], :, :]
            timesteps = normalized_data.shape[0]

            # Save the normalized version as .npz
            np.savez(
                normalized_npz_filepath,
                timesteps=timesteps,
                dt_seconds=SECONDS_PER_STEP,
                total_seconds=SIM_TOTAL_SECONDS,
                temperature=normalized_data,
            )
            print(f'Normalized data saved for {folder}')
            if delete_raw_after_normalization and os.path.exists(npz_file_path):
                os.remove(npz_file_path)
                print(f'Deleted raw data file: {npz_file_path}')
        elif os.path.exists(normalized_npz_filepath):
            print(f'Normalized data already exists for {folder}')
        else:
            print(f'Raw data file not found for {folder}')

if __name__ == '__main__':

    #base_path = './data/laplace_convolution/'
    #base_path = './data/testset/'
    base_path = './data/new_detailed_heat_sim'

    norm_values(base_path) # generates norm values 

    normalization(base_path) 
