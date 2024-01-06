#!/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from model import MNIST_Class, normalize, feature_extract
import matplotlib.pylab as plt

batch_size = 32
learning_rate = 1e-3
epochs = 50
model_path = "./model.pth"
optim_path = "./optim.pth"


transforms = tf.ToTensor()


train_data = datasets.MNIST(
    root="../tmp-py/data/", train=True, transform=transforms, download=False
)
# test_data = datasets.MNIST(root="../tmp-py/data/", train=True, transform=transforms , download=False)
loaded_data = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

model = MNIST_Class()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)


loss_ = []
tloss_ = []

model.train()
for epoch in range(epochs):
    for x, y in loaded_data:
        # Pre-process
        x: torch.Tensor = x
        # x: torch.Tensor = normalize(x)
        y: torch.Tensor = feature_extract(y, batch_size)

        # Forward-pass
        output = model(x)
        loss = criterion(output, y)

        # Backward-pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_.append(loss.item())
        print(f"\rLoss: {loss_[-1] :.4f}   ", end="", flush=True)
    print(f"\nAvg. Loss: { sum(loss_)/len(loss_):.6f } \t@Epoch - { epoch }", flush=True)
    tloss_.extend(loss_)
    loss_ = []


print("Saving model checkpoint ...")
torch.save(model.state_dict(), model_path)
torch.save(optimizer.state_dict(), optim_path)

plt.figure(figsize=(10, 25))
plt.plot(tloss_[::500])
plt.show()

