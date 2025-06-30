import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# 1. 数据集处理
class VOCSegDataset(Dataset):
    """PASCAL VOC2012语义分割数据集处理类"""

    # VOC2012类别列表
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]

    # 类别对应的RGB颜色映射
    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ]

    def __init__(self, root_dir, split='train', crop_size=(320, 480), augment=True):
        """
        初始化VOC语义分割数据集

        参数:
            root_dir: 数据集根目录，应包含JPEGImages、SegmentationClass等文件夹
            split: 'train'或'val'，指定加载训练集还是验证集
            crop_size: 图像裁剪尺寸
            augment: 是否应用数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.augment = augment

        # 构建颜色到标签的映射
        self.colormap2label = self._build_colormap2label()

        # 读取图像和标签路径
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClass')
        self.images = self._read_image_list()

        # 定义预处理和增强转换
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 数据增强转换
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2), ratio=(0.9, 1.1))
        ])

    def _read_image_list(self):
        """读取图像列表"""
        split_file = os.path.join(
            self.root_dir, 'ImageSets', 'Segmentation', f'{self.split}.txt')
        with open(split_file, 'r') as f:
            with open(split_file, 'r') as f:
                image_names = f.read().splitlines()
        return image_names

    def _build_colormap2label(self):
        """构建RGB颜色到类别索引的映射"""
        colormap2label = np.zeros(256 ** 3, dtype=np.int64)
        for i, cm in enumerate(self.VOC_COLORMAP):
            colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return colormap2label

    def _rgb_to_label(self, rgb_image):
        """将RGB标签图像转换为类别索引图"""
        rgb_image = np.array(rgb_image, dtype=np.int64)
        idx = (rgb_image[:, :, 0] * 256 + rgb_image[:, :, 1]) * 256 + rgb_image[:, :, 2]
        return np.array(self.colormap2label[idx], dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
        label_path = os.path.join(self.label_dir, f'{img_name}.png')

        # 读取图像和标签
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        # 应用数据增强
        if self.augment:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.aug_transform(image)

            torch.manual_seed(seed)
            label = self.aug_transform(label)
        else:
            # 调整大小到固定尺寸
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)

        # 图像预处理
        image = self.img_transform(image)

        # 标签转换为类别索引
        label = self._rgb_to_label(label)
        label = torch.from_numpy(label).long()

        return image, label


# 2. 语义分割模型（U-Net）
class DoubleConv(nn.Module):
    """两次卷积操作的基本模块"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net语义分割模型"""

    def __init__(self, in_channels=3, num_classes=21):
        super(UNet, self).__init__()

        # 编码器路径
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器路径
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # 最终分类层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # 解码器路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        # 最终分类
        out = self.final_conv(dec1)
        return out


# 3. 评估指标计算
def calculate_iou(pred, target, num_classes):
    """计算IoU（交并比）"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # 忽略背景类（类别0）
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()

        if union == 0:
            ious.append(float('nan'))  # 如果没有该类，记为NaN
        else:
            ious.append(intersection / union)

    return ious


def calculate_miou(ious):
    """计算平均IoU（mIoU），忽略NaN值"""
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0


# 4. 训练和验证函数
def train(model, train_loader, criterion, optimizer, device):
    """训练模型一个epoch"""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device, num_classes=21):
    """验证模型性能"""
    model.eval()
    running_loss = 0.0
    all_ious = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # 计算预测结果
            _, preds = torch.max(outputs, 1)

            # 计算IoU
            for i in range(images.size(0)):
                ious = calculate_iou(preds[i], labels[i], num_classes)
                all_ious.append(ious)

    # 计算平均损失和mIoU
    avg_loss = running_loss / len(val_loader.dataset)
    miou = calculate_miou(np.array(all_ious).flatten())

    return avg_loss, miou


# 5. 可视化函数
def visualize_predictions(model, val_loader, device, num_samples=5):
    """可视化模型预测结果"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_samples:
                break

            image = images[0].unsqueeze(0).to(device)
            label = labels[0].cpu().numpy()

            # 预测
            output = model(image)
            _, pred = torch.max(output, 1)
            pred = pred[0].cpu().numpy()

            # 显示原图
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            # 显示标签和预测
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(label)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


# 6. 主函数
def main():
    # 设置参数
    data_dir = 'path/to/VOCdevkit/VOC2012'  # 请替换为你的数据集路径
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.001
    crop_size = (320, 480)
    num_classes = 21  # VOC2012有21个类别（包括背景）

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据集和数据加载器
    train_dataset = VOCSegDataset(data_dir, split='train', crop_size=crop_size, augment=True)
    val_dataset = VOCSegDataset(data_dir, split='val', crop_size=crop_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型、损失函数和优化器
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 忽略标签值为255的像素
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 训练循环
    best_miou = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # 训练阶段
        train_loss = train(model, train_loader, criterion, optimizer, device)

        # 验证阶段
        val_loss, miou = validate(model, val_loader, criterion, device, num_classes)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_unet_model.pth')

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f}')

    # 加载最佳模型并可视化预测结果
    model.load_state_dict(torch.load('best_unet_model.pth'))
    visualize_predictions(model, val_loader, device)


if __name__ == '__main__':
    main()