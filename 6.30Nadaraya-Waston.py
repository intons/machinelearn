import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
#print(n_test)

def plot_kernel_reg(y_hat):
    # 绘制预测曲线与真实曲线
    plt.plot(x_test, y_truth, label='Truth')
    plt.plot(x_test, y_hat, label='Prediction')

    # 设置坐标轴范围
    plt.xlim(0, 5)
    plt.ylim(-1, 5)

    # 添加坐标轴标签
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示图例
    plt.legend()

    # 绘制训练数据散点图
    plt.scatter(x_train, y_train, color='red', marker='o', alpha=0.5)

    # 展示图形
    plt.show()

y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
