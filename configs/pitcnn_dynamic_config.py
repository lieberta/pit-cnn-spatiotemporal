import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
TRAIN_DTYPE = torch.float64
epochs = 10
a_list = [1]
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f64_noprmsource"
run_comment = "PITCNN_dynamic run from configs/pitcnn_dynamic.py"
