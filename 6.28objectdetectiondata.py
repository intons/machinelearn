import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签。"""
    # 下载并解压香蕉检测数据集
    data_dir = d2l.download_extract('banana-detection')
    # 根据训练或测试模式，确定 CSV 文件名称
    csv_fname = os.path.join(
        data_dir,
        'bananas_train' if is_train else 'bananas_val',
        'label.csv'
    )
    # 读取 CSV 文件数据
    csv_data = pd.read_csv(csv_fname)
    # 将图像名称设置为索引，方便后续按图像名称遍历
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    # 遍历 CSV 数据，逐行读取图像和对应标签
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(
                    data_dir,
                    'bananas_train' if is_train else 'bananas_val',
                    'images',
                    f'{img_name}'
                )
            )
        )
        targets.append(list(target))

    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (
            f' training examples' if is_train else f' validation examples'
        ))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集。"""
    # 创建训练集 DataLoader，is_train=True 表示加载训练集，打乱数据
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True),
        batch_size=batch_size,
        shuffle=True
    )
    # 创建验证集 DataLoader，is_train=False 表示加载验证集
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size,
        shuffle=False  # 验证集一般不打乱
    )
    return train_iter, val_iter

batch_size = 32
edge_size = 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['red'])
d2l.plt.show()