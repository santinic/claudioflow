import numpy as np
from numpy.testing import assert_array_equal

import syntax
from layers import MatrixWeight, VectorWeight
from network import Layer
from shared import SharedMemory
from syntax import Var, Tanh


class VanillaRNN(Layer):
    def __init__(self, seq_length, in_size, hidden_size, _Wx=None, _Wh=None, _Wy=None, _by=None):
        if _Wx is None:
            _Wx = np.random.rand(hidden_size, in_size)
            _Wh = np.random.rand(hidden_size, hidden_size)
            _Wy = np.random.rand(in_size, hidden_size)
            _by = np.random.rand(in_size)

        self.Wx = MatrixWeight(in_size, hidden_size, _Wx)
        self.Wh = MatrixWeight(hidden_size, hidden_size, _Wh)
        self.Wy = MatrixWeight(hidden_size, in_size, _Wy)
        self.by = VectorWeight(hidden_size, _by)

        self.weights = [self.Wx, self.Wh, self.Wy, self.by]
        self.last_h = SharedMemory(np.zeros(hidden_size))

        self.nodes = []

        for i in xrange(seq_length):
            node = VanillaNode(self.last_h, in_size, hidden_size, self.Wx, self.Wh, self.Wy, self.by)
            self.nodes.append(node)

    def update_weights(self, optimizer):
        # print('update weights')
        self.nodes[0].update_weights(optimizer)

    def reset_h(self):
        self.last_h.get().fill(0.)

    def forward_window(self, window, loss):
        loss_sum = 0.
        dJdys = []
        ys = []
        for node, (x, target) in zip(self.nodes, window):
            y = node.forward(x, is_training=True)
            ys.append(y)
            J = loss.calc_loss(y, target)
            dJdy = loss.calc_gradient(y, target)
            dJdys.append(dJdy)
            loss_sum += np.sum(J)
        mean_loss = loss_sum
        return ys, dJdys, mean_loss

    def clip(self):
        """Prevent exploding gradients"""
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
        dh = delta_h + dhnext
        grads = self.first.backward_variables(dh)
        return grads['last_h']

    def update_weights(self, optimizer):
        self.second.update_weights(optimizer)
        self.first.update_weights(optimizer)


# class VanillaRNN_Old(Layer):
#     def __init__(self, seq_length, in_size, out_size, hidden_size, Wx=None, Wh=None, Wy=None, by=None):
#         self.nodes = []
#
#         if Wx is None:
#             self.Wx = Wx = np.random.randn(hidden_size, in_size)
#             self.Wh = Wh = np.random.randn(hidden_size, hidden_size)
#             self.Wy = Wy = np.random.randn(out_size, hidden_size + 1)
#         else:
#             self.Wx, self.Wh, self.Wy = Wx, Wh, Wy
#             self.by = by
#
#         # self.last_h = np.zeros(hidden_size)
#         self.last_h = SharedMemory(np.zeros(hidden_size))
#
#         self.delta_Wh = np.zeros(Wh.shape)
#         self.delta_Wy = np.zeros(Wy.shape)
#         self.delta_by = np.zeros(out_size)
#         self.delta_Wx = np.zeros(Wx.shape)
#
#         for i in range(seq_length):
#             node = VanillaRNNNode(in_size, out_size, hidden_size)
#
#             node.last_h = self.last_h
#
#             node.Wh.layer.W = Wh
#             node.Wh.layer.delta_W = self.delta_Wh
#
#             # node.Wy.layer.W = Wy
#             # node.Wy.layer.delta_W = self.delta_Wy
#
#             node.Wy.layer.Wx.W = Wy
#             node.Wy.layer.Wx.delta_W = self.delta_Wy
#             node.Wy.layer.b.W = self.by
#             node.Wy.layer.b.delta_b = self.delta_by
#
#             node.Wx.layer.W = Wx
#             node.Wx.layer.delta_W = self.delta_Wx
#             self.nodes.append(node)
#
#             # scale = 0.01
#             # Wx *= scale
#             # Wh *= scale
#             # Wy *= scale
#
#     def clip(self):
#         """Prevent exploding gradients"""
#         for W in [self.delta_Wx, self.delta_Wh, self.delta_Wy]:
#             np.clip(W, -5, 5, out=W)
#
#     def reset_h(self):
#         for node in self.nodes:
#             node.reset_h()
#
#     def reset_deltas(self):
#         for node in self.nodes:
#             node.reset_deltas()
#
#     def forward_window(self, window, loss):
#         loss_sum = 0.
#         dJdys = []
#         ps = []
#         for node, (x, target) in zip(self.nodes, window):
#             p = node.forward(x, is_training=True)
#             ps.append(p)
#             J = loss.calc_loss(p, target)
#             dJdy = loss.calc_gradient(p, target)
#             dJdys.append(dJdy)
#             loss_sum += np.sum(J)
#         mean_loss = loss_sum
#         return ps, dJdys, mean_loss
#
#     def update_weights(self, optimizer):
#         for node in self.nodes:
#             node.update_weights(optimizer)
#
#
# class VanillaRNNNode(Layer):
#     """
#     Vanilla RNN, the simples Recurrent Neural Network (suffers vanishing gradients).
#
#         h_t = tanh(W * h_t-1 + W' * x_t)
#         y_t = h_t * W''
#
#     Karpathy's python implementation: https://gist.github.com/karpathy/d4dee566867f8291f086
#     Explanatory video: https://www.youtube.com/watch?v=iX5V1WpxxkY
#     """
#
#     def __init__(self, in_size, out_size, hidden_size):
#         self.hidden_size = hidden_size
#         # self.last_h = np.zeros(hidden_size)
#
#         self.Wx = syntax.Wx(in_size, hidden_size, initialize={}, input=Var('x_t'))
#         self.Wh = syntax.Wx(hidden_size, hidden_size, initialize={}, input=Var('last_h'))
#
#         # self.Wy = Linear(hidden_size, out_size, initialize='zeros', input=Var('h_t'))
#         self.Wy = syntax.WxBiasLinear(hidden_size, out_size, initialize={}, input=Var('h_t'))
#
#         self.h = Tanh(self.Wx + self.Wh)
#
#         # self.h = Tanh(PlusBias(hidden_size, input=self.Wx + self.Wh))
#
#         self.y = self.Wy
#
#     def reset_h(self):
#         self.last_h.get().fill(0.)
#
#     def forward(self, x, is_training=True):
#         h_t = self.h.forward_variables({'x_t': x, 'last_h': self.last_h.get()})
#         # print('my_h_t', h_t)
#         y = self.y.forward_variables({'h_t': h_t})
#         self.last_h.set(h_t)
#         return y
#
#     def backward_through_time(self, dJdy, dhnext, debug=False):
#         y_grads = self.y.backward_variables(dJdy, debug=debug)
#         delta_h = y_grads['h_t'] + dhnext
#
#         # print(y_grads)
#
#         h_grads = self.h.backward_variables(delta_h, debug=True)
#         dhprev = h_grads['last_h']
#         return y_grads['h_t'], dhprev
#
#     def update_weights(self, optimizer):
#         self.h.update_weights(optimizer)
#         self.y.update_weights(optimizer)
#
#     def reset_deltas(self):
#         for delta_W in [self.Wx.layer.delta_W, self.Wh.layer.delta_W, self.Wy.layer.delta_W]:
#             delta_W.fill(0.)
#
#
# class VanillaRNNCharTest(unittest.TestCase):
#     def test_chars(self):
#         x = np.random.rand(3)
#         model = VanillaRNN(3, 4, 5)
#         y = model.forward(x)
#         grad = model.backward(np.ones(3))
