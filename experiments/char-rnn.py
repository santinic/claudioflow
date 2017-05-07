import numpy as np
from collections import defaultdict

from numpy.testing import assert_array_equal

from layers import Softmax
from loss import CrossEntropyLoss, NLL
from optim import SGD, AdaGrad
from utils import sliding_window, chunks
from vanilla_rnn import VanillaRNN

import matplotlib.pyplot as plt

data = open('input.txt', 'r').read()
data_len = len(data)

chars = sorted(list(set(data)))
data_size = len(data)
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

one_hots = np.eye(vocab_size)


def sample(rnn, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros(vocab_size)
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        y = rnn.nodes[-1].forward(x, is_training=False)
        p = Softmax().forward(y)
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros(vocab_size)
        x[ix] = 1
        ixes.append(ix)
    return ixes


def sample_and_print(rnn, seed_ix, mean_loss, n=40):
    sample_ix = sample(rnn, seed_ix, n)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt,)
    print 'loss %f' % mean_loss


hidden_size = 100
seq_length = 25
rnn = VanillaRNN(seq_length, vocab_size, hidden_size)
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
# optimizer = AdaGrad(learning_rate=0.01)

W_names = ["Wxh", "Whh", 'Why', 'delta_Wxh', 'delta_Whh', 'delta_Why']
plot_matrix_magnitudes = defaultdict(list)


def magnitude(W):
    return np.sum(np.abs(W)) / np.sum(np.abs(W.shape))

def append_plot_magnitudes(rnn):
    plot_matrix_magnitudes['Wx'].append(magnitude(rnn.Wx.get()))
    plot_matrix_magnitudes['Wh'].append(magnitude(rnn.Wh.get()))
    plot_matrix_magnitudes['Wh'].append(magnitude(rnn.Wy.get()))
    plot_matrix_magnitudes['delta_Wx'].append(magnitude(rnn.Wx.delta))
    plot_matrix_magnitudes['delta_Wh'].append(magnitude(rnn.Wh.delta))
    plot_matrix_magnitudes['delta_Wy'].append(magnitude(rnn.Wy.delta))

def show_plots():
    plt.figure(1)
    Ws = ['Wx', 'Wh', 'Wy']
    for W in Ws:
        plt.plot(np.array(plot_matrix_magnitudes[W]))
    plt.legend(Ws)
    plt.figure(2)
    deltas = ['delta_Wx', 'delta_Wh', 'delta_Wy']
    for d in deltas:
        plt.plot(np.array(plot_matrix_magnitudes[d]))
    plt.legend(deltas)
    plt.show()

iter = 0
for _ in xrange(1000):
# while True:
    for x_str, target_str in sliding_window(data, seq_length):
        # print(x_str, target_str)
        inputs = [char_to_ix[ch] for ch in x_str]
        targets = [char_to_ix[ch] for ch in target_str]

        inputs_one_hot = [one_hots[x] for x in inputs]
        targets_one_hot = [one_hots[t] for t in targets]

        window = zip(inputs_one_hot, targets_one_hot)

        ys, dJdys, mean_loss = rnn.forward_window(window, loss)

        dhnext = np.zeros(hidden_size)
        for node, (x, target), dJdy in reversed(zip(rnn.nodes, window, dJdys)):
            dhnext = node.backward_through_time(dJdy, dhnext)

        append_plot_magnitudes(rnn)

        rnn.clip()
        rnn.update_weights(optimizer)

        if iter % 100 == 0:
            sample_and_print(rnn, inputs[0], mean_loss)

        iter += 1
    rnn.reset_h()
    print('############################################################################')

show_plots()