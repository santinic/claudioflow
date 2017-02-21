import unittest
from itertools import izip

import numpy as np
import matplotlib.pyplot as plt
import time

from main import SequentialModel, LinearLayer, SignLayer, SquaredLoss, PrintLayer, SigmoidLayer, ReluLayer

OUTPUT_A = 1.
OUTPUT_B = 0.
MIDDLE = 0.5

def gen_data(n=10):
    x1 = np.linspace(-1, 1, n)
    y1 = np.random.randn(n)
    x2 = np.linspace(1, 3, n)
    y2 = np.random.randn(n)

    plt.figure(1)
    plt.scatter(x1, y1, color='gray', marker="o", s=100)
    plt.scatter(x2, y2, color='gray', marker="x", s=100)

    data = zip(x1, y1) + zip(x2, y2)
    targets = [OUTPUT_A for i in range(n)] + [OUTPUT_B for i in range(n)]
    data = [np.array(x, dtype=float) for x in data]
    targets = [np.array(t, dtype=float) for t in targets]
    return data, targets


def scatter_data(data, target, model):
    plt.figure(1)
    color = lambda y: 'g' if y > MIDDLE else 'r'

    for x, target in izip(data, target):
        y = model.forward(x)
        plt.scatter(x[0], x[1], s=40, c=color(y))


class Perceptron(unittest.TestCase):

    def test_Perceptron(self):

        data, targets = gen_data()

        model = SequentialModel(2)
        model.add(LinearLayer(2, 1, initialize='random'))
        # model.add(LinearLayer(5, 1, initialize='random'))
        model.add(SigmoidLayer())
        # model.add(ReluLayer())
        # model.add(SignLayer())

        model.learn(data, targets, loss=SquaredLoss(), learning_rate=0.9, epochs=1000)

        test_data, test_targets = gen_data(10)
        scatter_data(test_data, test_targets, model)

        model.plot_errors_history()
        model.plot_loss_gradient_history()
        plt.show()
