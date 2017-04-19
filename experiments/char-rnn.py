import numpy as np

from layers import Softmax
from loss import CrossEntropyLoss
from optim import SGD, AdaGrad
from utils import sliding_window
from vanilla_rnn import VanillaRNN

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
        y = rnn.forward(x, is_training=False)
        p = Softmax().forward(y)
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros(vocab_size)
        x[ix] = 1
        ixes.append(ix)
    return ixes


def sample_and_print(rnn, seed_ix, n=100):
    sample_ix = sample(rnn, seed_ix, n)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt,)


hidden_size = 20
seq_length = 10
vanilla_rnn = VanillaRNN(vocab_size, vocab_size, hidden_size)
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
# optimizer = AdaGrad(learning_rate=0.1)

iter = 0
while True:
    for x_str, target_str in sliding_window(data, seq_length, step=1):
        inputs = [char_to_ix[ch] for ch in x_str]
        targets = [char_to_ix[ch] for ch in target_str]

        inputs_one_hot = [one_hots[x] for x in inputs]
        targets_one_hot = [one_hots[t] for t in targets]

        window = zip(inputs_one_hot, targets_one_hot)

        for x, target in window:
            y = vanilla_rnn.forward(x, is_training=True)
            J = loss.calc_loss(y, target)
            dJdy = loss.calc_gradient(y, target)

        dhnext = np.zeros(hidden_size)
        for x, target in reversed(window):
            dhnext = vanilla_rnn.backward_through_time(dJdy, dhnext)

        vanilla_rnn.clip()
        vanilla_rnn.update_weights(optimizer)
        vanilla_rnn.reset_h()

        if iter % 300 == 0:
            sample_and_print(vanilla_rnn, inputs[0])

        iter += 1