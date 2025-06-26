import torch
from torch import nn

def comp_conv2d(conv2d, x):
    x = x.reshape((1, 1) + x.shape)
    y = conv2d(x)
    return y.reshape(y.shape[2:])

#conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))#8+2x2-5/1+1除步长
'''输出高度=（输入高度+填充*2-卷积核高度）/步长 + 1'''
conv2d = nn.Conv2d(1, 1,(3, 5),(3, 4),(0, 1))
x = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, x).shape)