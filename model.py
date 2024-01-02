import torch
import torch.nn as nn
from torch import Tensor

class MNIST_Class(nn.Module):
    def __init__(self,) -> None:
        super(MNIST_Class,self).__init__()
        self.fcl1 = nn.Linear(28*28 , 512)
        self.relu = nn.ReLU()
        self.fcl2 = nn.Linear(512,80)
        self.fcl3 = nn.Linear(80,10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x: Tensor) -> Tensor:
        x = self.relu( self.fcl1(x.view(-1,28*28)) )
        x = self.relu( self.fcl2(x) )
        x = self.softmax( self.fcl3(x) )
        return x


def normalize(x : Tensor) -> Tensor:
    return (x - x.mean()) / x.std()

def feature_extract(y : Tensor, batch_size : int) -> Tensor:
    y_ = torch.zeros(batch_size,10 , dtype=torch.float)
    for i , j in enumerate(y):
        y_[i][j] = 1.
    return y_