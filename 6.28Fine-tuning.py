import os
import torch
import torchvision
from torch import nn
from torchvision import transforms

# 手动下载并解压Hotdog数据集
# 下载地址: https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
# 解压到data_dir目录

# 定义数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
data_dir = 'D:\learn\data_dir\hotdog'  # 修改为实际解压路径
train_dataset = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train'), transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'test'), transform=test_transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False)

# 加载预训练模型
pretrained_net = torchvision.models.resnet18(pretrained=True)#加载预训练模型

# 修改最后一层
pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)#只对最后一层做初始化


# 定义训练函数
def train_fine_tuning(net, learning_rate, batch_size=32, num_epochs=5,
                      param_group=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    loss = nn.CrossEntropyLoss()
    if param_group:
        # 微调: 对不同层设置不同学习率
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}#希望最后一层学的更快，前面初始化好了不希望改变太多
        ], lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    weight_decay=0.001)
    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_val = loss(outputs, labels)
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {running_loss / len(train_iter):.4f}, '
              f'Train Acc: {correct / total:.4f}')

        # 在测试集上评估
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_iter:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        print(f'Test Acc: {test_correct / test_total:.4f}')


# 训练模型
train_fine_tuning(pretrained_net, 5e-5)