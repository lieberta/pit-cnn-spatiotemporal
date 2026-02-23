import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
epochs = 10
a_list = [0]
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f32_a=0"
run_comment = "fixed an error in physics loss -> laplacian(output) instead of laplacian(input) + f32"
