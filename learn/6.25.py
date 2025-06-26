#对全连接层使用平移不变性和局部性得到卷积层
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()

    return y

'''x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(x, k))'''

x = torch.ones((6, 8))
x[:, 2:6] = 0
#print(x)

k = torch.tensor([[1.0, -1.0]])
#print(k)

y = corr2d(x, k)
#print(y)
#print(corr2d(x.t(), k))

# 定义卷积层
conv2d = nn.Conv2d(1, 1, (1, 2), bias=False)

# 初始化输入数据和目标输出
x = x.reshape((1, 1, 6, 8))  # 输入数据
y = y.reshape((1, 1, 6, 7))  # 目标输出


# 训练循环
for i in range(10):
    # 前向传播
    y_hat = conv2d(x)
    l = (y_hat - y) ** 2  # 计算损失

    # 反向传播
    conv2d.zero_grad()  # 梯度清零
    l.sum().backward()  # 计算梯度

    # 更新权重
    with torch.no_grad():
        conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad  # 使用梯度更新权重

    # 打印损失
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))#所学的卷积核的权重张量