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