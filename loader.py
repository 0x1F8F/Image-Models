import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from model import MNIST_Class,normalize,feature_extract
import matplotlib.pylab as plt

model_path = "./model.pth"
optim_path = "./optim.pth"
learning_rate = 1e-2

model = MNIST_Class()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters() , lr=learning_rate)

model.load_state_dict(torch.load(model_path))
optimizer.load_state_dict(torch.load(optim_path))