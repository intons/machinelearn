import random

import torch
import matplotlib.pyplot as plt
from torch.utils import data
from d2l import torch as d2l
from torch import nn
import torchvision
from torchvision import transforms

d2l.use_svg_display()
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, transform=trans)
mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, transform=trans)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    tans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, transform=trans)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axs = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axs.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
def get_dataloader_workers():
    return 4
x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
#plt.show()
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for x, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')