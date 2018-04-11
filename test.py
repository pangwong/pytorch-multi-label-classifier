import torch
from torch.autograd import Variable
from torchvision import models

from data.loader import MultiLabelDataLoader
from options import Options
from models.lightcnn import LightCNN_29Layers_v2_templet 
from models.build_model import BuildMultiLabelModel

import time


# option 
op = Options()
opt = op.parse()

# load test data
opt.mode == "Test":
data_loader = MultiLabelDataLoader(opt)
test_set = data_loader.GetTestSet()
num_classes = data_loader.GetNumClasses()
dataset_size = len(test_set)

