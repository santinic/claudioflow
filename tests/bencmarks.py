import timeit
import unittest
import numpy as np

from layers import CheapTanh, Tanh
from loss import CrossEntropyLoss, ClaudioMaxNLL


class BenchmarkTests(unittest.TestCase):
    def test_ClaudioMaxNLL(self):
        x = np.random.rand(10)
        y = np.random.rand(10)
        target_class = 3

        loss = CrossEntropyLoss()
        def f1():
            J = loss.calc_loss(x, target_class)
            dJdy = loss.calc_gradient(y, target_class)

        cla_loss = ClaudioMaxNLL()
        def f2():
            J = cla_loss.calc_loss(x, target_class)
            dJdy = cla_loss.calc_gradient(y, target_class)

        t1 = timeit.timeit(f1, number=10000)
        t2 = timeit.timeit(f1, number=10000)
        print(t1, t2)
        self.assertGreater(t2, t1)

    def test_CheapTanh(self):
        x = np.random.rand(10)

        cheap_tanh = CheapTanh()
        def f1(): cheap_tanh.forward(x)

        tanh_layer = Tanh()
        def f2(): tanh_layer.forward(x)

        t1 = timeit.timeit(f1, number=10000)
        t2 = timeit.timeit(f2, number=10000)
        self.assertGreater(t2, t1)

