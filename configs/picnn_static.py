import torch

TRAIN_DTYPE = torch.float32
epochs = 10
a_list = [0, 1]
model_class_name = "PICNN_static"
model_name = model_class_name + "_f32_normsource"
run_comment = "float32 + normalized Heat  Source; PICNN_static run from configs/picnn_static.py"
