#!/bin/env python

import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
from torch import Tensor
from torch.utils.data import DataLoader
from model import MNIST_Class, normalize, feature_extract
from loader import model

batch_size = 32
learning_rate = 1e-2
epochs = 5
model_path = "./model.pth"
# optim_path = "./optim.pth"


transforms = tf.ToTensor()

test_data = datasets.MNIST(
    root="../tmp-py/data/", train=False, transform=transforms, download=False
)

loaded_data = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(f"Training script ...")
print(f"\tBatch size\t: {batch_size}")
print(f"\tModel path\t: {model_path}")
# print(f"\tOptimizer \t: {optim_path}\n")


model.eval()
with torch.no_grad():
    correct_prediction = 0
    total_prediction = 0
    for i, (image, label) in enumerate(loaded_data):
        x: Tensor = normalize(image.to(torch.float))
        y: Tensor = feature_extract(label, batch_size)
        output: Tensor = torch.argmax(model(x), dim=1)
        total_prediction += output.size(0)
        correct_prediction += (output == label).sum().item()
        print(f"\rTesting :  [ {i}/{len(loaded_data)} ]", end="", flush=True)
    accuracy = correct_prediction * 100 / total_prediction
    print(f"\nTest summary ______________")
    print(f"| Total  \t: {total_prediction}")
    print(f"| Correct\t: {correct_prediction}")
    print(f"| Accuracy score: {accuracy}")
