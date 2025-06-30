import matplotlib
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

def apply(img, aug, num_rows=2, num_cols=4, scale=0.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

apply(img, torchvision.transforms.RandomHorizontalFlip())#左右
apply(img, torchvision.transforms.RandomVerticalFlip())#上下
shape_aug = torchvision.transforms.RandomResizedCrop(#随机裁剪
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)
apply(img, torchvision.transforms.ColorJitter(#随机更改图像的亮度
    brightness=0.5, contrast=0.5, saturation=0, hue=0
))
apply(img, torchvision.transforms.ColorJitter(#随机更改图像的色调
    brightness=0, contrast=0, saturation=0.5, hue=0.5
))