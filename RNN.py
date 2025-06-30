import math
import torch
from IPython.extensions.autoreload import update_property
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from txtpre import load_corpus_time_machine, read_time_machine, load_data_time_machine, Vocab
import matplotlib.pyplot as plt

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
#print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
X = torch.arange(10).reshape((2, 5))
#print(F.one_hot(X.T, 28).shape)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size  # num_inputs应等于词汇表大小

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 输入到隐藏的权重矩阵形状应为 (num_inputs, num_hiddens)
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)

state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
#print(Y.shape, len(new_state), new_state[0].shape)
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    # 检查 prefix 长度，避免索引越界
    if len(prefix) == 0:
        raise ValueError("prefix should not be empty")
    # 获取 prefix 第一个字符的索引
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # 获取预测类别索引并处理形状
        pred_idx = y.argmax(dim=1).reshape(1).item()
        outputs.append(pred_idx)
    # 根据索引转换为 token 文本并拼接
    return ''.join([vocab.idx_to_token[i] for i in outputs])

#print(predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu()))

def grad_clipping(net, theta):
    if isinstance(theta, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad ]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 损失总和,  token数量

    for X, Y in train_iter:
        batch_size = X.shape[0]
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=batch_size, device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)  # 转换为(num_steps * batch_size,)
        X, y = X.to(device), y.to(device)

        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=batch_size)  # 传入实际批次大小

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_ite=False):
    loss = nn.CrossEntropyLoss()
    # 调整 xlim，让绘图更合理展示
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[1, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_ite)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs = 500
lr = 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())