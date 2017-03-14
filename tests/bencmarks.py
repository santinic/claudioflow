import timeit
import unittest
import numpy as np

from loss import CrossEntropyLoss


class BenchmarkTests(unittest.TestCase):
    def test_softmaxNLL(self):
        loss = CrossEntropyLoss()
        x = np.random.rand(10)
        y = np.random.rand(10)
        target_class = 3

        def forward():
            J = loss.calc_loss_slow(x, target_class)
            dJdy = loss.calc_gradient_slow(y, target_class)

        print timeit.timeit(forward, number=100000)
