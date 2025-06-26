import torch
import torch as nn
import torch.nn.functional as F
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
x = torch.randn(size=(2, 20))
y = net(x)
torch.save(net.state_dict(), 'mlp.params')#state_dict()里面是有所有的权重的
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
y_clone = clone(x)
print(y_clone == y)