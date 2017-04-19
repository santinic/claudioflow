import unittest

import numpy as np

from network import Layer
from syntax import Var, Wx, Tanh


class VanillaRNN(Layer):
    """
    Vanilla RNN, the simples Recurrent Neural Network (suffers vanishing gradients).

        h_t = tanh(W * h_t-1 + W' * x_t)
        y_t = h_t * W''

    Karpathy's python implementation: https://gist.github.com/karpathy/d4dee566867f8291f086
    Explanatory video: https://www.youtube.com/watch?v=iX5V1WpxxkY
    """

    def __init__(self, in_size, out_size, hidden_size):
        self.hidden_size = hidden_size
        self.delta_h_previous = np.zeros(hidden_size)
        self.reset_h()

        self.Wxh = Wx(in_size, hidden_size, input=Var('x_t'))
        self.Whh = Wx(hidden_size, hidden_size, input=Var('last_h'))
        self.Why = Wx(hidden_size, out_size, input=Var('h_t'))
        self.h = Tanh(self.Wxh + self.Whh)
        self.y = self.Why

        scale = 0.01
        self.Wxh.layer.W *= scale
        self.Whh.layer.W *= scale
        self.Why.layer.W *= scale


    def reset_h(self):
        self.last_h = np.zeros(self.hidden_size)

    def forward(self, x, is_training=True):
        h_t = self.h.forward_variables({'last_h': self.last_h, 'x_t': x})
        y = self.y.forward_variables({'h_t': h_t})
        self.last_h = h_t
        return y

    def backward_through_time(self, dJdy, dhnext):
        y_grads = self.y.backward_variables(dJdy)
        delta_h = y_grads['h_t'] + dhnext

        h_grads = self.h.backward_variables(delta_h + self.delta_h_previous)
        dhprev = h_grads['last_h']
        return dhprev

    def update_weights(self, optimizer):
        self.h.update_weights(optimizer)
        self.y.update_weights(optimizer)

    def clip(self):
        """Prevent exploding gradients"""
        for param in [self.Wxh, self.Whh, self.Why]:
            W = param.layer.W
            np.clip(W, -5, 5, out=W)


class VanillaRNNCharTest(unittest.TestCase):
    def test_chars(self):
        x = np.random.rand(3)
        model = VanillaRNN(3, 4, 5)
        y = model.forward(x)
        grad = model.backward(np.ones(3))
        # print grad
