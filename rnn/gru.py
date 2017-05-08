import numpy as np

from layers import MatrixWeight
from rnn import RNN
from shared import SharedMemory
from syntax import Var, Concat, Sigmoid, Wx, Tanh, Const, WxBiasLinear, Linear


class GRU(RNN):
    def __init__(self, seq_length, in_size, hidden_size, initialize, scale=1):
        self.hidden_size = hidden_size
        x_and_h_size = in_size + hidden_size
        self.Wz = MatrixWeight(x_and_h_size+1, hidden_size, initialize, scale)
        self.Wr = MatrixWeight(x_and_h_size+1, hidden_size, initialize, scale)
        self.W = MatrixWeight(x_and_h_size+1, hidden_size, initialize, scale)
        self.last_h = SharedMemory(np.zeros(hidden_size))

        self.decoderW = MatrixWeight(hidden_size+1, in_size, initialize, scale)

        self.nodes = []
        for i in range(seq_length):
            node = GRUNode(self.Wz, self.Wr, self.W, self.last_h, self.decoderW)
            self.nodes.append(node)

    def get_weights(self):
        return ['Wz', 'Wr', 'W']

    def reset_memory(self):
        self.last_h.get().fill(0.)

    def clip(self):
        for W in [self.Wz, self.Wr, self.W]:
            np.clip(W.delta, -5, 5, out=W.delta)


class GRUNode:
    def __init__(self, Wz, Wr, W, last_h, decoderW):
        self.last_h = last_h

        h = Var('h')
        x = Var('x')
        hx = Concat(h, x)

        z = Sigmoid(Linear(0, 0, initialize=Wz, input=hx))
        r = Sigmoid(Linear(0, 0, initialize=Wr, input=hx))

        h_tilde = Tanh(Linear(0, 0, initialize=W, input=Concat(r * h, x)))

        self.h_model = ((Const(1) - z) * h) + (z * h_tilde)

        self.decoder = Linear(0, 0, initialize=decoderW, input=Var('h'))

    def forward(self, x, is_training=False):
        new_h = self.h_model.forward_variables({'x': x, 'h': self.last_h.get()})
        self.last_h.set(new_h)
        decoded = self.decoder.forward_variables({ 'h': new_h })
        return decoded

    def backward_through_time(self, dJdy, dhnext):
        h_grad = self.decoder.backward_variables(dJdy)['h']
        return self.h_model.backward_variables(h_grad + dhnext)['h']

    def update_weights(self, optimizer):
        self.h_model.update_weights(optimizer)
