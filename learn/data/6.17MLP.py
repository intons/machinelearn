import torch
from torch import nn
from d2l import torch as d2l
from model_utils import train_ch3, predict_ch3

batch_size = 256

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

def net(x):
    x = x.reshape((-1, num_inputs))
    h = relu(x @ w1 + b1)
    return (h @ w2 + b2)

loss = nn.CrossEntropyLoss()
if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256
    w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [w1, b1, w2, b2]
    num_epochs = 10
    lr = 0.1
    updater = torch.optim.SGD(params, lr=lr)
    loss = nn.CrossEntropyLoss()
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)