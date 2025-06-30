import torch
from d2l import torch as d2l
import random
from txtpre import read_time_machine, Vocab, load_corpus_time_machine
import matplotlib.pyplot as plt


tokens = d2l.tokenize(read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
#print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
'''plt.figure(figsize=[10, 6])
plt.plot(freqs)
plt.xlabel('token: x')
plt.ylabel('frequency:n(x)')
plt.xscale('log')
plt.yscale('log')
plt.show()'''
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]#得到相邻二元组
bigram_vocab = Vocab(bigram_tokens)
#print(bigram_vocab.token_freqs[:10])
trigram_tokens = [
    triple for triple in zip(corpus[:-2], corpus[1: -1], corpus[2:])#得到相邻三元组冒号前面按顺序排列，0,1,2
]
trigram_vocab = Vocab(trigram_tokens)
#print(trigram_vocab.token_freqs[:10])
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for toke, freq in trigram_vocab.token_freqs]
'''plt.figure(figsize=[8, 6])
plt.plot(freqs)
plt.plot(bigram_freqs)
plt.plot(trigram_freqs)
plt.xlabel('token: x')
plt.ylabel('frequency:n(x)')
plt.xscale('log')
plt.yscale('log')
plt.legend(['unigram', 'bigram', 'trigram'])
plt.show()'''
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
'''for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)'''

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
