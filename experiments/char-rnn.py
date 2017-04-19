import numpy as np

from layers import Softmax
from loss import CrossEntropyLoss
from optim import RMSProp, SGD
from trainers import MinibatchTrainer
from vanilla_rnn import VanillaRNN

data = open('input.txt', 'r').read()
data_len = len(data)

# n = 0
# l = 3
# for _ in range(40):
#     print n
#     print data[n:n+l], data[n+1:n+l+1]
#     n = (n+1) % len(data)
# exit(1)

chars = sorted(list(set(data)))
data_size = len(data)
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_eye = np.eye(vocab_size)


def one_hot(v):
    return vocab_eye[v]


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
        # h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        # y = np.dot(Why, h) + by
        # p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros(vocab_size)
        x[ix] = 1
        ixes.append(ix)
    return ixes


def sample_and_print(rnn, seed_ix, n=100):
    sample_ix = sample(rnn, seed_ix, n)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt,)


in_size = vocab_size
out_size = vocab_size
hidden_size = 100
seq_length = 4
vanilla_rnn = VanillaRNN(in_size, out_size, hidden_size)

loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.1)

iter = 0
while True:
    print('iter = %d' % iter)
    # print('        %d:%d' % (iter, iter + seq_length))
    inputs = [char_to_ix[ch] for ch in data[iter : iter + seq_length]]
    targets = [char_to_ix[ch] for ch in data[iter + 1 : iter + seq_length + 1]]

    input_ch = ''.join([ix_to_char[iter] for iter in inputs])
    target_ch = ''.join([ix_to_char[iter] for iter in targets])
    print('         %s -> %s' % (input_ch, target_ch))

    # if len(inputs) != len(targets):
    #     print("DIVERSI", input_ch, target_ch)
    #     n = 0
    #     continue

    inputs_one_hot = [one_hot(x) for x in inputs]
    targets_one_hot = [one_hot(t) for t in targets]

    window = zip(inputs_one_hot, targets_one_hot)

    loss_sum = 0.
    for x, target in window:
        y = vanilla_rnn.forward(x, is_training=True)
        J = loss.calc_loss(y, target)
        dJdy = loss.calc_gradient(y, target)
        loss_sum += J
    mean_losses = loss_sum / seq_length
    batch_mean_loss = np.mean(mean_losses)
    # print('batch mean J %f' % batch_mean_loss)

    dhnext = np.zeros(hidden_size)
    for x, target in reversed(window):
        dhnext = vanilla_rnn.backward_through_time(dJdy, dhnext)

    vanilla_rnn.clip()
    vanilla_rnn.update_weights(optimizer)
    # vanilla_rnn.reset_h()

    # if iter % 100 == 0:
    #     sample_and_print(vanilla_rnn, inputs[0])

    # n = (n+1) % data_len

