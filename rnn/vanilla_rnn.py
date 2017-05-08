import numpy as np

import syntax
from layers import MatrixWeight, VectorWeight
from rnn import RNN
from shared import SharedMemory
from syntax import Var, Tanh


class VanillaRNN(RNN):
    def __init__(self, seq_length, in_size, hidden_size, _Wx=None, _Wh=None, _Wy=None, _by=None, scale=1):
        if _Wx is None:
            _Wx = np.random.randn(hidden_size, in_size) * scale
            _Wh = np.random.randn(hidden_size, hidden_size) * scale
            _Wy = np.random.randn(in_size, hidden_size) * scale
            _by = np.random.randn(in_size) * scale

        # print _Wx.shape, _Wh.shape, _Wy.shape, _by.shape
        self.Wx = MatrixWeight(in_size, hidden_size, _Wx)
        self.Wh = MatrixWeight(hidden_size, hidden_size, _Wh)
        self.Wy = MatrixWeight(hidden_size, in_size, _Wy)
        self.by = VectorWeight(hidden_size, _by)

        self.hidden_size = hidden_size
        # self.weights = [self.Wx, self.Wh, self.Wy, self.by]
        self.last_h = SharedMemory(np.zeros(hidden_size))

        self.nodes = []

        for i in xrange(seq_length):
            node = VanillaNode(self.last_h, in_size, hidden_size, self.Wx, self.Wh, self.Wy, self.by)
            self.nodes.append(node)

    def get_weights(self):
        return ['Wx', 'Wh', 'Wy', 'by']

    def reset_memory(self):
        self.last_h.get().fill(0.)

    def clip(self):
        for W in [self.Wx, self.Wh, self.Wy, self.by]:
            np.clip(W.delta, -5, 5, out=W.delta)


class VanillaNode:
    def __init__(self, last_h, in_size, hidden_size, Wx, Wh, Wy, by):
        self.last_h = last_h
        self.Wx = syntax.Wx(in_size, hidden_size, initialize=Wx, input=Var('x'))
        self.Wh = syntax.Wx(hidden_size, hidden_size, initialize=Wh, input=Var('last_h'))
        self.first = Tanh(self.Wx + self.Wh)
        self.second = syntax.WxBiasLinear(hidden_size, in_size, initialize_W=Wy, initialize_b=by, input=Var('h'))
        self.Wy = self.second

    def forward(self, x, is_training=False):
        h = self.first.forward_variables({'x': x, 'last_h': self.last_h.get()})
        self.last_h.set(h)
        return self.second.forward_variables({'h': h})

    def backward_through_time(self, dJdy, dhnext):
        delta_h = self.second.backward_variables(dJdy)['h']
        # assert delta_h.shape == dhnext.shape
        grads = self.first.backward_variables(delta_h + dhnext)
        return grads['last_h']

    def update_weights(self, optimizer):
        self.second.update_weights(optimizer)
        self.first.update_weights(optimizer)
