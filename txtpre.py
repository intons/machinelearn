import collections
import re
from d2l import torch as d2l
import torch
from torch.utils import data

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
#print(lines[0])
#print(lines[10])

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('unknown token' + token)

tokens = tokenize(lines)
'''for i in range(11):
    print(tokens[i])'''

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)#排序不是必要的，但有可能对计算性能较好
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):#给token返回index
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):#给index返回token
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

vocab = Vocab(tokens=tokens)
'''print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('words', tokens[i])
    print('indices', vocab[tokens[i]])'''

def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens=tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
'''corpus, vocab = load_corpus_time_machine()
print(len(corpus))
print(len(vocab))'''

def load_data_time_machine(batch_size, num_steps, max_tokens=10000):
    """加载时间机器数据集，返回数据迭代器和词汇表"""
    # 加载语料库和词汇表
    corpus, vocab = load_corpus_time_machine(max_tokens)

    # 将语料库转换为张量
    corpus = torch.tensor(corpus)

    # 计算可生成的样本数
    num_samples = (len(corpus) - 1) // num_steps

    # 创建特征和标签
    features = []
    labels = []
    for i in range(num_samples):
        features.append(corpus[i * num_steps: (i + 1) * num_steps])
        labels.append(corpus[i * num_steps + 1: (i + 1) * num_steps + 1])

    # 创建数据加载器
    dataset = data.TensorDataset(torch.stack(features), torch.stack(labels))
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)

    return dataloader, vocab