import numpy as np

from layers import Linear, Tanh, Sum, Store, Print
from network import Seq, Map


class VanillaRNN:
    """
    Vanilla RNN, the simples Recurrent Neural Network (suffers vanishing gradients).

        h_t = tanh(W * h_t-1 + W' * x_t)
        y_t = h_t * W''

    Karpathy's python implementation: https://gist.github.com/karpathy/d4dee566867f8291f086
    Explanatory video: https://www.youtube.com/watch?v=iX5V1WpxxkY
    """

    def __init__(self, input_size):
        self.h_store = Store(input_size=input_size)


        self.model = Seq(
            Map(Linear(input_size, input_size), Linear(input_size, input_size)),
            Sum,
            Tanh,
            self.h_store,
            Linear(input_size, input_size),
        )

        # Model expressed with syntax.py:
        # self.h_t = Var("h_t", size=(4, 3), dtype=np.float16)
        # self.x_h = Var("x_h", size=(4, 3), dtype=np.float16)
        # self.syntax_model = Linear(Tanh(Linear(h_t) + Linear(x_t))
        # y = self.syntax_model.forward(x)
        # self.synax_model.backward(dJdy)

    def forward(self, x):
        last_h = self.h_store.read()
        x_tilde = np.array([last_h, x])
        return self.model.forward(x_tilde)

    def backward(self, dJdy):
        return self.model.backward(dJdy)
