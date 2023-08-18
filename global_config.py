import torch.nn as nn

device_type = "cpu"     # The device used for the model and dataset
image_crop_size = 300    # The size of the image-crop used in training   (default = 80, corresponding to 80x80image)
batch_size = 25         # Batch size used in training

loss_function = nn.MSELoss()