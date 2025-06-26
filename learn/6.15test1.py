import random
import torch
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))#均值，方差
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)#随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(x, w, b):
    return torch.matmul(x, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
'''print('features:', features[0],'\nlabels:', labels[0])
plt.figure(figsize=(5, 3))
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)#特征的第一列
plt.xlabel('Feature 1')      # x 轴标签
plt.ylabel('Label')          # y 轴标签
plt.title('Scatter Plot: Feature vs Label')  # 标题
plt.show()'''
batch_size = 10
'''for x,y in data_iter(batch_size, features, labels):
    print(x,'\n', y)
    break'''
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, train loss: {float(train_l.mean()):f}')
print(f'w的估计误差:{true_w-w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b-b}')