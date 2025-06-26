import torch
import torch as nn
import torch.nn.functional as F
import torch.nn as nn

class CenterLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x-x.mean()
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))#初始化权重偏置

    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)

layer = CenterLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenterLayer())

y = net(torch.rand(4, 8))
print(y.mean())

dense = MyLinear(5, 3)
print(dense.weight)
print(dense(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))