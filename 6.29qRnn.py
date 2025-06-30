import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from txtpre import load_corpus_time_machine, read_time_machine, load_data_time_machine, Vocab
from RNN import predict_ch8

batch_size = 32
num_steps = 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # 假设是普通 RNN，隐藏状态形状：(num_directions * num_layers, batch_size, num_hiddens)
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers,
                 batch_size,
                 self.num_hiddens),
                device=device
            )
        else:
            # LSTM 有两个状态（h 和 c），形状同隐藏状态
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                     batch_size,
                     self.num_hiddens),
                    device=device
                ),
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                     batch_size,
                     self.num_hiddens),
                    device=device
                )
            )

device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(predict_ch8('time traveller', 10, net, vocab, device))