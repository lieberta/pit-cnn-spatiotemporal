import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
epochs = 10
a_list = [1]
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f32_final_straw"
run_comment = "fixed an error in physics loss -> laplacian(output) instead of laplacian(input), new dataset that has the same fineness as the simulation + f32"
