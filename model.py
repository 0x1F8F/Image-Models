import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class MNIST_Class(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(MNIST_Class, self).__init__()
        self.fcl1 = nn.Linear(28 * 28, 512)
        self.fcl2 = nn.Linear(512, 10)
        self.activate = nn.Sigmoid()
        self.ot = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activate(self.fcl1(x.view(-1, 28 * 28)))
        x = self.ot( self.fcl2(x) )
        return x


def normalize(x: Tensor) -> Tensor:
    return (x - x.mean()) / x.std()


def feature_extract(y: Tensor, batch_size: int) -> Tensor:
    y_ = torch.zeros(batch_size, 10, dtype=torch.float)
    for i, j in enumerate(y):
        y_[i][j] = 1.0
    return y_
