import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.rand(size=(2, 4))
net(x)
print(net[2].state_dict())
print(net[2].bias)
print(net)