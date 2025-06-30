import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 修复：调整AdditiveAttention的参数，假设正确签名为key_size, query_size, num_hiddens, dropout
        self.attention = d2l.AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # 解码器当前时间步的查询向量
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 计算注意力上下文
            context = self.attention(
                query,
                enc_outputs,  # keys
                enc_outputs,  # values
                enc_valid_lens
            )
            # 将上下文与嵌入向量拼接
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 通过RNN层
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 最后通过全连接层生成预测
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

# 测试代码
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()

# 初始化注意力解码器
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()

# 构造输入张量
X = torch.zeros((4, 7), dtype=torch.long)

# 初始化解码器状态，enc_valid_lens设为None
state = decoder.init_state(encoder(X), None)

# 解码器前向传播
output, state = decoder(X, state)

# 打印输出及状态相关信息
print("output.shape:", output.shape)
print("len(state):", len(state))
print("state[0].shape:", state[0].shape)
if len(state) > 1:
    print("state[1].shape:", state[1].shape)