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


Newest update.

The newest models inherit their functions from a 'BaseModel'. The newest train_models.py file
is a class of said BaseModel that enables the main.py file to train and save multiple models
in a row.


Models:
PECNN_static = static Physics Enhanced Convolutional Network, each Model is trained on one timestep
only and is therefor static. For prognosis of 5 different timesteps into the future, you need 5
static models trained on these timesteps. For Loss calculation we need CombinedLoss

PECNN_dynamic = Similar architecture but this time has a additional input variable: time
give a prognosis of the behaviour of the data at given input time. The time is handled as 4
(the same as inputdimensions of the latent tensor) input
tensors of same size in the latent space with values 't' at every gridpoint. If you chose 1
input dimension for time, the network doesn't take time into consideration when calculating the output, i.e.
if the temperature distribution at t=0 is the same, the output distribution is for each input time the same.
For Loss calculation needs CombinedLoss_dynamic (time derivative is here approximated with (output-input)/t
extremely inprecise)

PECNN_timefirst = same but this time the timevalues are concatenated at the beginning of the network with
input dimension 1 (same input dimension as the input tensor)

PECNN_dynamic_batchnorm = replaces group norm with batchnorm in PECNN_dynamic

PECNN_autodiff = replaces the time derivative of CombinedLoss_dynamic with a autograd function that calculates the
derivative of output w.r.t. time


training.py
changes training class to a function