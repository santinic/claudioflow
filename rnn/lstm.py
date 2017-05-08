import numpy as np

from layers import MatrixWeight, VectorWeight
from rnn import RNN
from shared import SharedMemory
from syntax import Sigmoid, Tanh, Var, Concat, WxBiasLinear, Dropout


class LSTM(RNN):
    def __init__(self, seq_length, in_size, hidden_size, initialize='random', scale=1):
        scale *= 1. / np.sqrt(in_size + hidden_size)

        x_and_h_size = in_size + hidden_size
        self.hidden_size = hidden_size

        self.Wf = MatrixWeight(x_and_h_size, hidden_size, initialize, scale)
        self.bf = VectorWeight(hidden_size, initialize, scale)

        self.Wi = MatrixWeight(x_and_h_size, hidden_size, initialize, scale)
        self.bi = VectorWeight(hidden_size, initialize, scale)

        self.Wc = MatrixWeight(x_and_h_size, hidden_size, initialize, scale)
        self.bc = VectorWeight(hidden_size, initialize, scale)

        self.Wo = MatrixWeight(x_and_h_size, hidden_size, initialize, scale)
        self.bo = VectorWeight(hidden_size, initialize, scale)

        self.last_h_store = SharedMemory(np.zeros(hidden_size))
        self.last_C_store = SharedMemory(np.zeros(hidden_size))

        self.nodes = []
        for i in range(seq_length):
            node = LSTMNode(in_size, hidden_size,
                            self.last_h_store, self.last_C_store,
                            self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo)
            self.nodes.append(node)

            # print 'C = ' + str(self.nodes[0].C_model)
            # print 'h = ' + str(self.nodes[0].h_model)

    def backward_window(self, dJdys):
        delta_future = {
            'last_h': np.zeros(self.hidden_size),
            'last_C': np.zeros(self.hidden_size)
        }
        for node, dJdy in reversed(zip(self.nodes, dJdys)):
            delta_future = node.backward_through_time(dJdy, delta_future)

    def clip(self):
        for W in [self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo]:
            np.clip(W.delta, -5, 5, out=W.delta)

    def reset_memory(self):
        self.last_h_store.get().fill(0.)
        self.last_C_store.get().fill(0.)


class LSTMNode:
    """
    Implements traditional Long Short Term Memory as described here (variable names should match):
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, in_size, hidden_size, last_h_store, last_C_store, Wf, bf, Wi, bi, Wc, bc, Wo, bo):
        # States
        self.last_h_store = last_h_store
        self.last_C_store = last_C_store

        # Input
        x = Var('x')
        last_h = Var('last_h')
        last_C = Var('last_C')
        xh = Concat(x, last_h)

        # C network
        f = Sigmoid(WxBiasLinear(in_size, hidden_size, initialize_W=Wf, initialize_b=bf, input=xh))
        i = Sigmoid(WxBiasLinear(in_size, hidden_size, initialize_W=Wi, initialize_b=bi, input=xh))
        C_tilde = Tanh(WxBiasLinear(in_size, hidden_size, initialize_W=Wc, initialize_b=bc, input=xh))
        self.C_model = (f * last_C) + (i * C_tilde)

        # h network
        new_C = Var("new_C")
        o = Sigmoid(WxBiasLinear(in_size, hidden_size, initialize_W=Wo, initialize_b=bo, input=xh))
        self.h_model = o * Tanh(new_C)

    def forward(self, x, is_training=False):
        new_C = self.C_model.forward_variables({
            'x': x,
            'last_h': self.last_h_store.get(),
            'last_C': self.last_C_store.get()
        })
        new_h = self.h_model.forward_variables({
            'x': x,
            'last_h': self.last_h_store.get(),
            'new_C': new_C
        })

        self.last_C_store.set(new_C)
        self.last_h_store.set(new_h)
        return new_h

    def backward_through_time(self, dJdy, delta_future):
        h_gradients = self.h_model.backward_variables(dJdy + delta_future['last_h'])
        C_gradients = self.C_model.backward_variables(h_gradients['new_C'] + delta_future['last_C'])

        return {
            'last_C': C_gradients['last_C'],
            'last_h': C_gradients['last_h'] + h_gradients['last_h'],
            'x': C_gradients['x'] + h_gradients['x']
        }

    def update_weights(self, optimizer):
        self.h_model.update_weights(optimizer)
        self.C_model.update_weights(optimizer)
