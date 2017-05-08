import numpy as np
from collections import defaultdict, deque

from numpy.testing import assert_array_equal

from gru import GRU
from layers import Softmax
from loss import CrossEntropyLoss, NLL
from lstm import LSTM
from optim import SGD, AdaGrad, RMSProp
from utils import sliding_window, chunks
from vanilla_rnn import VanillaRNN

import matplotlib.pyplot as plt

# data = open('input.txt', 'r').read()
# data = '''I couldn't find how to initialize the weight matrices (input to hidden, hidden to hidden). Are they initialized randomly? to zeros? are they initialized differently for each LSTM I create?
# Another motivation for this question is in pre-training some LSTMs and using their weights in a subsequent model. I don't currently know how to do that currently without saving all the states and restoring the entire model.'''
data = '''E' proprio bello il ballo del billo col birillo!'''
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


def sample_and_print(rnn, seed_ix, mean_loss, n=100):
    sample_ix = sample(rnn, seed_ix, n)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt,)
    print 'loss %f' % mean_loss



seq_length = 25
hidden_size = 50

# rnn = VanillaRNN(seq_length=25, in_size=vocab_size, hidden_size=50, scale=0.01)

rnn = LSTM(seq_length, vocab_size, vocab_size, initialize='randn', scale=0.01)

# rnn = GRU(seq_length, vocab_size, vocab_size, initialize='randn', scale=1)

loss = CrossEntropyLoss()
# optimizer = SGD(learning_rate=0.01)
optimizer = AdaGrad(learning_rate=0.1)
# optimizer = RMSProp(0.2, decay_rate=0.01)

plot_matrix_magnitudes = defaultdict(lambda: deque([], 1000))
plt.ion()

def magnitude(W):
    return np.sum(np.abs(W)) / np.sum(np.abs(W.shape))

def append_plot_magnitudes(rnn, mean_loss):
    for W_name in rnn.get_weights():
        W = rnn.__dict__[W_name]
        plot_matrix_magnitudes[W_name].append(magnitude(W.get()))
        plot_matrix_magnitudes['delta'+W_name].append(magnitude(W.delta))
    plot_matrix_magnitudes['mean_loss'].append(mean_loss)

def show_plots():
    plt.figure(1)
    plt.clf()
    cmap = plt.get_cmap('Set1')
    for i, W_name in enumerate(rnn.get_weights()):
        plt.plot(np.array(plot_matrix_magnitudes[W_name]), 'b', lw=2,
                 color=cmap(float(i)/len(rnn.get_weights())),
                 label=W_name)
    plt.legend(rnn.get_weights())
    plt.figure(2)
    plt.clf()
    deltas = ['delta'+W_name for W_name in rnn.get_weights()]
    for i, delta_name in enumerate(deltas):
        plt.plot(np.array(plot_matrix_magnitudes[delta_name]), 'b', lw=2,
                 color=cmap(float(i) / len(rnn.get_weights())),
                 label=delta_name)
    plt.legend(deltas)
    plt.figure(3)
    plt.clf()
    plt.plot(np.array(plot_matrix_magnitudes['mean_loss']), 'b', lw=2, label='mean_loss')
    plt.legend(['mean_loss'])
    plt.ylim([0, 5])
    plt.show()

iter = 0
# for _ in xrange(100):
while True:
    for x_str, target_str in sliding_window(data, seq_length):
        # print(x_str, target_str)
        inputs = [char_to_ix[ch] for ch in x_str]
        targets = [char_to_ix[ch] for ch in target_str]

        inputs_one_hot = [one_hots[x] for x in inputs]
        targets_one_hot = [one_hots[t] for t in targets]
        window = zip(inputs_one_hot, targets_one_hot)

        # mean_loss = rnn.learn_window(window, loss, optimizer)

        ys, dJdys, mean_loss = rnn.forward_window(window, loss)
        rnn.backward_window(dJdys)

        if iter % 100 == 0:
            append_plot_magnitudes(rnn, mean_loss)

        rnn.clip()
        rnn.update_weights(optimizer)

        if iter % 100 == 0:
            sample_and_print(rnn, inputs[0], mean_loss)
            show_plots()
            plt.pause(0.05)

        iter += 1

    rnn.reset_memory()

