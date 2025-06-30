import torch
from IPython import display
from d2l import torch as d2l
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import time  # 用于演示延迟，非必需
from d2l import torch as d2l
from d2l.torch import set_axes  # 新增导入
import matplotlib
matplotlib.use('TkAgg')  # 可根据系统选择 'Qt5Agg' 或 'Agg' 等
import matplotlib.pyplot as plt

batch_size = 256
num_inputs = 784  # 修正变量名拼写
num_outputs = 10
lr = 0.1


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(10, 5)):
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.x, self.y, self.fmts = None, None, fmts

        # 启用交互模式
        plt.ion()
        self.config_axes()

    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if self.x is None:
            self.x = [[] for _ in range(n)]
        if self.y is None:
            self.y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.x[i].append(a)
                self.y[i].append(b)

        # 清除当前图表
        self.axes[0].cla()

        # 绘制新数据
        for i, (x_values, y_values, fmt) in enumerate(zip(self.x, self.y, self.fmts)):
            self.axes[0].plot(x_values, y_values, fmt, linewidth=2)

        # 应用配置
        self.config_axes()

        # 强制更新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 可选：添加短暂延迟以便观察更新
        plt.pause(0.1)


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition


def net(x):
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            l.sum().backward()
            updater(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        ylabel='loss/accuracy', legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    print(f'训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.3f}, 测试准确率: {test_acc:.3f}')
    # 关闭交互模式并保持图表打开
    plt.ioff()
    plt.show()


def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)


def predict_ch3(net, test_iter, n=6, save_path=None):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    # 显示图像
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

    # 确保显示图片
    plt.tight_layout()  # 调整布局
    if save_path:
        plt.savefig(save_path)  # 保存图片
    plt.show()  # 显示图片

    return X, y, titles  # 返回图像、标签和标题，方便后续处理

if __name__ == '__main__':

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    num_epochs = 10
    #train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)