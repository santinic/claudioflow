import unittest

import numpy as np

from network import Layer
from syntax import Sigmoid, Linear, Tanh, Var


class LSTM(Layer):
    """
    Implements traditional Long Short Term Memory as described here (variable names should match):
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    Torch implementation for comparison:
    https://github.com/Element-Research/rnn/blob/master/LSTM.lua
    """

    def __init__(self, in_size, out_size):
        x_and_h = Var("x_and_h")
        last_C = Var("last_C")

        f = Sigmoid(Linear(in_size, out_size, input=x_and_h))
        i = Sigmoid(Linear(in_size, out_size, input=x_and_h))
        C_tilde = Tanh(Linear(in_size, out_size, input=x_and_h))
        o = Sigmoid(Linear(in_size, out_size, input=x_and_h))

        fxC = f * last_C
        ixC_tilde = i * C_tilde
        C = fxC + ixC_tilde
        h = o * Tanh(C)

        print 'h = ', h
        print 'C = ', C
        print 'C_tilde = ', C_tilde

    def forward(self, x):
        x_and_h = np.hstack([x, self.last_h])

        # self.last_C = self.C.forward_variables({ 'x_and_h': x_and_h })

        return self.h.forward_variables({ 'x_and_h': x_and_h })

    def backward(self, dJdy):
        return self.h.backward(dJdy)


class LSTMTest(unittest.TestCase):
    def test_init(self):
        model = LSTM(2, 3)

