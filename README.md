<<<<<<< HEAD
# Geospatial Time Series

Here we try to forecast temperature development in a simulated room via CNNs. Data used to train the model are simulated 
temperature grid meshes of burning rooms.

Preprocessing of new Experiment folders:

use "out_to_csv.py" to translate the *.out files of one experiment to
a *.csv-file

use then  "sort_data.py" sort these gridpoints in (x,y,z) directions
and to save a "temperature_matrices.npz" file for each experiment

Train Network:

Execute "train.py" to start the training of given method, the model will be
saved in the folder "saved_models" afterwards 




Explanation of files:

"import_normalize.py" are predefined functions, that import the *.npz files
and normalize the matrices in it.
import_all_rooms(abl = boolean) imports all rooms and is executed in "dataset.py"
if abl == True, the dataset will consist of derivation of the temperature
if abl == False, the dataset will consist of the actual temperature values


=======
# PIT-CNN

Surrogate modeling of transient 3D heat transfer using convolutional neural networks with physics-informed loss functions.
This repo contains tools to simulate heat transfer, preprocess data, train neural networks, and evaluate models.

# Workflow


# 1. Generate simulation data

Runs a 3D transient heat equation with random rectangular heat sources (“fireplaces”).

```bash
python heat_sim_class.py
```

Saves results in ./data/testset/experiment_* with: <br>

heat_equation_solution.npz → temperatures, grid, time <br>

fireplace_simulation_results.txt → metadata <br>


# 2. Preprocess experiments

Normalize temperature fields across all experiments.

```bash
python preprocess.py
```

Computes global min/max <br>

Saves normalized .npz files in each experiment folder <br>

Trained checkpoints saved in ./models/


# 3. Train models

Run static or dynamic CNNs with physics-informed loss.

```bash
python main.py
```

- Static models → PICNN_static

- Dynamic models → PECNN_dynamic
>>>>>>> 030630310e261920aa6eecd6588499cdf243ff92
