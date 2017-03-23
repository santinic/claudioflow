import numpy as np

from layers import Linear, Sigmoid, Tanh, Const, Store
from network import Seq, Par


class LSTM:
    """
    Implements traditional Long Short Term Memory as described here (variable names should match):
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    Torch implementation for comparison:
    https://github.com/Element-Research/rnn/blob/master/LSTM.lua
    """

    def __init__(self, in_size, out_size):

        # self.C_store = Store(in_size)

        f =         Seq(Linear(in_size, out_size), Sigmoid)
        i =         Seq(Linear(in_size, out_size), Sigmoid)
        C_tilde =   Seq(Linear(in_size, out_size), Tanh)
        o =         Seq(Linear(in_size, out_size), Sigmoid)

        fxC =       Seq(Par(f, Const(self.last_C)), Mul)
        # ixC_tilde = Mul(i, C_tilde)
        # C =         Sum(fxC, ixC_tilde)
        # h =         Mul(o, Seq(C, Tanh))

        self.C = C
        self.h = h

    def forward(self, x):
        x_and_h = np.hstack([x, self.last_h])

        self.last_C = self.C.forward(x_and_h)

        return self.h.forward(x_and_h)

    def backward(self, dJdy):
        return self.h.backward(dJdy)
