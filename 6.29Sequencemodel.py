import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成时间序列数据
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T, ))

'''# 绘制时间序列图
plt.figure(figsize=(10, 6))
plt.plot(time.numpy(), x.numpy())
plt.title('时间序列数据可视化')
plt.xlabel('时间')
plt.ylabel('x值')
plt.xlim([1, 1000])
plt.grid(True)  # 添加网格线，便于观察

# 显示图形
plt.show()'''
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch{epoch + 1},'
              f'loss{d2l.evaluate_loss(net, train_iter, loss)},')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
onestep_preds = net(features)
'''d2l.plot(
    [time, time[tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy()],'time', 'x',
    legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3)
)'''
plt.figure(figsize=(10, 6))

# 绘制原始数据
plt.plot(time.numpy(), x.detach().numpy(), label='data')

# 绘制一步预测
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds')

# 设置图表属性
plt.title('时间序列与一步预测')
plt.xlabel('时间')
plt.ylabel('x值')
plt.xlim([1, 1000])
plt.legend()  # 显示图例
plt.grid(True)  # 添加网格线

# 显示图形
plt.show()